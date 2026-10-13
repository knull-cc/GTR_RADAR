"""RADARWrapper: dependency-aware input repair placed *before* the backbone (and thus before its RevIN).

    x ─ RobustScaler (floored) ─┬─ ReliabilityNet → ℓ, R = 1 − σ(ℓ)        ┐ stage A: backbone-agnostic,
                                ├─ E1 Repairer (R-weighted context)       │ trained once per dataset
                                └─ E2 reliable-neighbour median + trend   ┘ (L_det PU + L_rec)
                Router(π) → candidate repair c = Σ_k π_k c_k
    D̃ = D_p / D̄   (EMA profile of this backbone, or per-sample Hutchinson gradient)       ┐ stage B: per backbone,
    α = σ( softplus(a)·logit(1−R) + softplus(b)·log(D̃ + ε) − τ )                           ┘ gate + Router only
    x̃ = x + α (c − x)  →  backbone closure f(x̃) = backbone(x̃, *args)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from radar.dependency import DependencyProfile, hutchinson_dependency, patch_sum
from radar.modules import (ReliabilityNet, ReliableNeighborExpert, Repairer, Router, from_patches,
                           patch_to_points, point_features, robust_stats, to_patches)

EXPERTS = ('Repairer', 'ReliableNeighbor')


def _cfg(cfg, key, default):
    return getattr(cfg, key, default) if cfg is not None else default


def _inv_softplus(y):
    return math.log(math.expm1(y))


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


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

        # ---- stage A modules (backbone-agnostic)
        self.reliability = ReliabilityNet(self.P, self.N, d, heads, layers, d_ff, drop, head_bias=-4.0)
        self.repairer = Repairer(self.P, self.N, d, heads, layers, d_ff, drop)
        self.neighbor = ReliableNeighborExpert(k=int(_cfg(cfg, 'e2_k', 3)), thr=float(_cfg(cfg, 'e2_thr', 0.5)),
                                               trend=bool(_cfg(cfg, 'e2_trend', 1)))
        self.std_frac = float(_cfg(cfg, 'scale_std_frac', 0.1))
        self.s_min = float(_cfg(cfg, 'scale_min', 1e-2))
        self.u_clip = float(_cfg(cfg, 'u_clip', 10.0))

        # ---- stage B modules (per backbone): Router + gate (a, b, τ)
        self.n_experts = len(EXPERTS)
        self.expert_names = EXPERTS
        self.router = Router(d, self.P, self.n_experts)
        self.gate_a = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
        self.gate_b = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
        self.tau_raw = nn.Parameter(torch.tensor(float(_cfg(cfg, 'tau_init', 1.0))))
        self.gate_eps = float(_cfg(cfg, 'gate_eps', 1e-2))
        self.logit_clip = 12.0

        # ---- dependency
        self.dep_mode = str(_cfg(cfg, 'dep_mode', 'ema'))
        assert self.dep_mode in ('grad', 'ema')
        self.dep_probes = int(_cfg(cfg, 'dep_probes', 4))
        self.dep_dist = str(_cfg(cfg, 'dep_probe_dist', 'rademacher'))
        self.dep_profile = DependencyProfile(self.N, float(_cfg(cfg, 'dep_ema', 0.99)))

        self.use_cycle_prior = False  # E3 removed (kept for reporting compatibility)
        self.force_gate_zero = bool(_cfg(cfg, 'force_gate_zero', False))
        self.chunk = int(_cfg(cfg, 'radar_chunk', 8192))
        self.grad_ckpt = bool(_cfg(cfg, 'radar_grad_ckpt', 0))
        self.probe_generator = None  # optional torch.Generator for reproducible probes (val/test)
        self.last_aux = None

    # ------------------------------------------------------------------ parameter groups
    @property
    def tau(self):
        return self.tau_raw

    def gate_parameters(self):
        """Stage B: Router + gate (a, b, τ)."""
        return list(self.router.parameters()) + [self.gate_a, self.gate_b, self.tau_raw]

    def stage_a_parameters(self):
        return list(self.reliability.parameters()) + list(self.repairer.parameters())

    def radar_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def radar_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}

    # ------------------------------------------------------------------ helpers
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

    def _closure(self, bb_args):
        return lambda z: _first(self.backbone(z, *bb_args))

    def compute_dependency(self, x, f):
        """D̃ (B, C, N) = D_p / D̄, detached, computed on the *unrepaired* input.

        dep_mode='ema': the backbone's EMA profile (no extra cost at inference once calibrated);
        dep_mode='grad': per-sample Hutchinson estimate (K probes). During training D_p is always estimated to
        update the EMA. Before any calibration, D̄ falls back to the batch mean.
        """
        B, L, C = x.shape
        ready = bool(self.dep_profile.initialized)
        Dp = None
        if self.dep_mode == 'grad' or self.training or not ready:
            D = hutchinson_dependency(self.backbone, x, None, self.pred_len, self.dep_probes,
                                      generator=self.probe_generator, dist=self.dep_dist, forward_fn=f)
            Dp = patch_sum(D, self.P)
            if self.training:
                self.dep_profile.update(Dp)
                ready = True
        if self.dep_mode == 'ema' and ready:
            Dt = self.dep_profile.normalized().to(x.dtype).view(1, 1, -1).expand(B, C, self.N)
        else:
            scale = self.dep_profile.scale() if ready else Dp.mean().clamp_min(1e-12)
            Dt = Dp / scale
        return Dt.detach()

    # ------------------------------------------------------------------ forward
    def repair(self, x, *bb_args, rec_mask=None, compute_dep=True):
        """Returns x̃ and the aux dict. rec_mask: optional bool (B*C, N) of patches whose R is forced to 0
        in the Repairer input (masked-reconstruction training)."""
        B, L, C = x.shape
        f = self._closure(bb_args)
        Dt_p = self.compute_dependency(x, f) if compute_dep else x.new_ones(B, C, self.N)

        m, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        u = (x - m) / s
        uc = u.clamp(-self.u_clip, self.u_clip)

        up, pad = to_patches(uc, self.P)  # (BC, N, P)
        BC = up.shape[0]
        feats = point_features(up.reshape(BC, -1)).reshape(BC, self.N, self.P * 3)
        logit_p = self._run(self.reliability, feats)  # unreliability logit ℓ = logit(1 − R)
        R_p = 1.0 - torch.sigmoid(logit_p)

        R_in = R_p
        if rec_mask is not None:
            R_in = R_p * (~rec_mask).to(R_p.dtype).unsqueeze(-1)
        rep_tokens = torch.stack([R_in * up, 1.0 - R_in], dim=-1).reshape(BC, self.N, 2 * self.P)
        u_hat_p, h = self._run(self.repairer, rep_tokens)
        pi_p = self.router(h)  # (BC, N, P, K)

        logit = from_patches(logit_p, B, C, pad)
        R = from_patches(R_p, B, C, pad)
        u_hat = from_patches(u_hat_p, B, C, pad)
        pi = from_patches(pi_p, B, C, pad)  # (B, L, C, K)

        c1 = u_hat * s + m
        c2 = self.neighbor(x, R)
        c = torch.stack([c1, c2], dim=-1)
        target = (pi * c).sum(dim=-1)

        Dt = patch_to_points(Dt_p, self.P, pad)  # (B, L, C)
        if self.force_gate_zero:
            gate_logit = torch.full_like(x, -float('inf'))
            alpha = torch.zeros_like(x)
            x_tilde = x
        else:
            gate_logit = (F.softplus(self.gate_a) * logit.clamp(-self.logit_clip, self.logit_clip)
                          + F.softplus(self.gate_b) * torch.log(Dt + self.gate_eps) - self.tau_raw)
            alpha = torch.sigmoid(gate_logit)
            x_tilde = x + alpha * (target - x)

        aux = dict(logit=logit, R=R, alpha=alpha, gate_logit=gate_logit, dbar=Dt, dbar_patch=Dt_p, u_hat=u_hat,
                   pi=pi, x_tilde=x_tilde, c=c, target=target, m=m, s=s)
        self.last_aux = aux
        return x_tilde, aux

    def forward(self, x, *bb_args, rec_mask=None, compute_dep=True, run_backbone=True):
        x_tilde, _ = self.repair(x, *bb_args, rec_mask=rec_mask, compute_dep=compute_dep)
        if not run_backbone:
            return None
        return _first(self.backbone(x_tilde, *bb_args))
