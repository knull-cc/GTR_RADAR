"""RADARWrapper: dependency-aware input repair placed *before* the backbone (and thus before its RevIN).

x (StandardScaler space) -> RADAR -> x̃ -> backbone(x̃, cycle_index) -> ŷ
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from radar.dependency import DependencyProfile, hutchinson_dependency, patch_dependency
from radar.modules import (CyclePriorExpert, LocalMedianExpert, ReliabilityNet, Repairer, Router,
                           from_patches, patch_to_points, point_features, robust_stats, to_patches)

N_EXPERTS = 3  # 0: Repairer, 1: LocalMedian, 2: CyclePrior


def _cfg(cfg, key, default):
    return getattr(cfg, key, default) if cfg is not None else default


class RADARWrapper(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.P = int(_cfg(cfg, 'radar_patch', 8))
        self.N = math.ceil(self.seq_len / self.P)
        self.pad = self.N * self.P - self.seq_len

        d = int(_cfg(cfg, 'radar_d', 64))
        heads = int(_cfg(cfg, 'radar_heads', 4))
        layers = int(_cfg(cfg, 'radar_layers', 2))
        d_ff = int(_cfg(cfg, 'radar_dff', 128))
        drop = float(_cfg(cfg, 'radar_dropout', 0.1))

        self.reliability = ReliabilityNet(self.P, self.N, d, heads, layers, d_ff, drop, head_bias=-4.0)
        self.repairer = Repairer(self.P, self.N, d, heads, layers, d_ff, drop)
        self.router = Router(d, self.P, N_EXPERTS)
        self.local_median = LocalMedianExpert(3)
        self.cycle_prior = CyclePriorExpert()

        q = getattr(backbone, 'Q', None)
        self.use_cycle_prior = isinstance(q, torch.Tensor) and q.dim() == 2 and bool(_cfg(cfg, 'use_cycle_prior', 1))
        self.register_buffer('expert_mask', torch.tensor([True, True, self.use_cycle_prior]), persistent=False)

        tau0 = float(_cfg(cfg, 'tau_init', 0.3))
        self.tau_raw = nn.Parameter(torch.tensor(math.log(tau0 / (1.0 - tau0)), dtype=torch.float32))
        self.gate_T = float(_cfg(cfg, 'gate_T', 0.05))
        self.u_clip = float(_cfg(cfg, 'u_clip', 10.0))

        self.dep_mode = str(_cfg(cfg, 'dep_mode', 'grad'))
        assert self.dep_mode in ('grad', 'ema')
        self.dep_probes = int(_cfg(cfg, 'dep_probes', 1))
        self.dep_profile = DependencyProfile(self.N, float(_cfg(cfg, 'dep_ema', 0.99)))

        self.force_gate_zero = bool(_cfg(cfg, 'force_gate_zero', False))
        self.chunk = int(_cfg(cfg, 'radar_chunk', 8192))
        self.grad_ckpt = bool(_cfg(cfg, 'radar_grad_ckpt', 0))
        self.probe_generator = None  # optional torch.Generator for reproducible Hutchinson probes (val/test)
        self.last_aux = None

    # ------------------------------------------------------------------ helpers
    @property
    def tau(self):
        return torch.sigmoid(self.tau_raw)

    def radar_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def radar_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}

    def _run(self, fn, inp):
        """Run fn over the B*C axis in chunks (avoids OOM for many-channel datasets such as Traffic)."""
        def call(z):
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                return checkpoint(fn, z, use_reentrant=False)
            return fn(z)

        n = inp.shape[0]
        if n <= self.chunk:
            return call(inp)
        outs = [call(inp[i:i + self.chunk]) for i in range(0, n, self.chunk)]
        if isinstance(outs[0], tuple):
            return tuple(torch.cat(z, dim=0) for z in zip(*outs))
        return torch.cat(outs, dim=0)

    def compute_dependency(self, x, cycle_index):
        """dbar (B, C, N), detached. Computed on the *unrepaired* input."""
        B, L, C = x.shape
        dbar = None
        if self.dep_mode == 'grad' or self.training:
            D = hutchinson_dependency(self.backbone, x, cycle_index, self.pred_len, self.dep_probes,
                                      generator=self.probe_generator)
            dbar, _ = patch_dependency(D, self.P)
            if self.training:
                self.dep_profile.update(dbar)
        if self.dep_mode == 'ema':
            dbar = self.dep_profile.profile().to(x.dtype).view(1, 1, -1).expand(B, C, self.N)
        return dbar.detach()

    # ------------------------------------------------------------------ forward
    def repair(self, x, cycle_index, rec_mask=None, compute_dep=True):
        """Returns x̃ and the aux dict. rec_mask: optional bool (B*C, N) of patches whose R is forced to 0
        in the Repairer input (masked-reconstruction training)."""
        B, L, C = x.shape
        if compute_dep:
            dbar_p = self.compute_dependency(x, cycle_index)
        else:
            dbar_p = x.new_ones(B, C, self.N)

        m, s = robust_stats(x)
        u = (x - m) / s
        uc = u.clamp(-self.u_clip, self.u_clip)  # bounded network inputs (flat windows give s≈1e-5)

        up, pad = to_patches(uc, self.P)  # (BC, N, P)
        BC = up.shape[0]
        feats = point_features(up.reshape(BC, -1)).reshape(BC, self.N, self.P * 3)
        logit_p = self._run(self.reliability, feats)  # (BC, N, P)
        R_p = 1.0 - torch.sigmoid(logit_p)

        R_in = R_p
        if rec_mask is not None:
            R_in = R_p * (~rec_mask).to(R_p.dtype).unsqueeze(-1)
        rep_tokens = torch.stack([R_in * up, 1.0 - R_in], dim=-1).reshape(BC, self.N, 2 * self.P)
        u_hat_p, h = self._run(self.repairer, rep_tokens)
        pi_p = self.router(h, self.expert_mask)  # (BC, N, P, K)

        logit = from_patches(logit_p, B, C, pad)
        R = from_patches(R_p, B, C, pad)
        u_hat = from_patches(u_hat_p, B, C, pad)
        pi = from_patches(pi_p, B, C, pad)  # (B, L, C, K)

        c1 = u_hat * s + m
        c2 = self.local_median(x)
        if self.use_cycle_prior:
            c3, _ = self.cycle_prior(x, cycle_index, self.backbone.Q, R)
        else:
            c3 = x.detach()  # placeholder; its routing weight is exactly 0
        c = torch.stack([c1, c2, c3], dim=-1)
        target = (pi * c).sum(dim=-1)

        dbar = patch_to_points(dbar_p, self.P, pad)  # (B, L, C)
        r = dbar * (1.0 - R)
        if self.force_gate_zero:
            alpha = torch.zeros_like(r)
            x_tilde = x
        else:
            alpha = torch.sigmoid((r - self.tau) / self.gate_T)
            x_tilde = x + alpha * (target - x)

        aux = dict(logit=logit, R=R, alpha=alpha, dbar=dbar, dbar_patch=dbar_p, u_hat=u_hat, pi=pi,
                   x_tilde=x_tilde, c=c, target=target, r=r, m=m, s=s)
        self.last_aux = aux
        return x_tilde, aux

    def forward(self, x, cycle_index, rec_mask=None, compute_dep=True, run_backbone=True):
        x_tilde, _ = self.repair(x, cycle_index, rec_mask=rec_mask, compute_dep=compute_dep)
        if not run_backbone:
            return None
        return self.backbone(x_tilde, cycle_index)
