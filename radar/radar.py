"""RADARWrapper (v4): dependency-aware input repair placed *before* the backbone (and thus before its RevIN).

    x ─ RobustScaler (floored) ─┬─ ReliabilityNet(5 point features) → ℓ, R = 1 − σ(ℓ)      ┐ stage A: backbone-agnostic,
                                ├─ E1 Repairer (R-weighted context)                          │ trained once per dataset
                                ├─ E2 local linear (reliable neighbours, interpolation)      │ (L_det PU + L_rec)
                                └─ E3 periodic shape (same-phase template + residual interp) ┘ E2/E3 parameter-free
         Router(π): linear on named per-point features φ (gap length, support, distances, density, disagreements)
                    → candidate repair c = Σ_k π_k c_k   (unavailable experts masked, softmax temperature annealed)
    D̃ = D_p / D̄   (EMA profile of this backbone, per-sample Hutchinson gradient, or none)     ┐ stage B: per backbone
    α = σ( softplus(a)·logit(1−R) + softplus(b)·log(D̃ + ε) − τ )                               ┘
    x̃ = x + α (c − x)  →  backbone closure f(x̃) = backbone(x̃, *args)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from radar.dependency import DependencyProfile, hutchinson_dependency, patch_sum
from radar.modules import (N_POINT_FEATURES, ROUTE_FEATURES, LinearRouter, LocalLinearExpert, PeriodicShapeExpert,
                           ReliabilityNet, Repairer, Router, from_patches, from_series, patch_to_points,
                           point_features, reliable_context, robust_stats, route_features, to_patches, to_series)

EXPERTS = ('Repairer', 'LocalLinear', 'PeriodicShape')
EXPERT_KEYS = ('e1', 'e2', 'e3')


def _cfg(cfg, key, default):
    return getattr(cfg, key, default) if cfg is not None else default


def _inv_softplus(y):
    return math.log(math.expm1(y))


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def resolve_e3_period(cfg):
    """--e3_period: > 0 explicit; -1 (default) = the backbone's --cycle; 0 = E3 off."""
    p = int(_cfg(cfg, 'e3_period', -1))
    return int(_cfg(cfg, 'cycle', 0)) if p < 0 else p


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
        self.det_feats = str(_cfg(cfg, 'det_feats', 'v4'))
        self.n_point_feats = N_POINT_FEATURES[self.det_feats]
        self.reliability = ReliabilityNet(self.P, self.N, d, heads, layers, d_ff, drop, head_bias=-4.0,
                                          n_feat=self.n_point_feats)
        self.repairer = Repairer(self.P, self.N, d, heads, layers, d_ff, drop)
        self.e2_k = int(_cfg(cfg, 'e2_k', 3))
        self.e2_thr = float(_cfg(cfg, 'e2_thr', 0.5))
        self.neighbor = LocalLinearExpert()
        self.e3_period = resolve_e3_period(cfg)
        self.periodic = PeriodicShapeExpert(self.e3_period, self.seq_len, int(_cfg(cfg, 'e3_cycles', 3)))
        self.std_frac = float(_cfg(cfg, 'scale_std_frac', 0.1))
        self.s_min = float(_cfg(cfg, 'scale_min', 1e-2))
        self.u_clip = float(_cfg(cfg, 'u_clip', 10.0))

        # ---- experts / Router
        self.expert_names = EXPERTS
        self.n_experts = len(EXPERTS)
        keys = [k.strip().lower() for k in str(_cfg(cfg, 'experts', 'e1,e2,e3')).split(',') if k.strip()]
        assert 'e1' in keys and set(keys) <= set(EXPERT_KEYS), f'--experts must contain e1, got {keys}'
        self.register_buffer('expert_enabled', torch.tensor([k in keys for k in EXPERT_KEYS]), persistent=False)
        self.router_type = str(_cfg(cfg, 'router', 'linear'))
        assert self.router_type in ('linear', 'h'), self.router_type
        if self.router_type == 'linear':
            self.router = LinearRouter(len(ROUTE_FEATURES), self.n_experts)
        else:
            self.router = Router(d, self.P, self.n_experts)
        self.register_buffer('route_temp', torch.tensor(float(_cfg(cfg, 'route_temp_start', 1.0))))
        self.route_hard = False  # argmax routing (evaluation only)

        # ---- stage B gate (a, b, τ)
        # softplus(a) = 4 makes the gate close to a switch in the unreliability logit (v1's effective slope was 20)
        self.gate_a = nn.Parameter(torch.tensor(_inv_softplus(float(_cfg(cfg, 'gate_a_init', 4.0)))))
        self.gate_b = nn.Parameter(torch.tensor(_inv_softplus(float(_cfg(cfg, 'gate_b_init', 1.0)))))
        self.tau_raw = nn.Parameter(torch.tensor(float(_cfg(cfg, 'tau_init', 1.0))))
        self.gate_eps = float(_cfg(cfg, 'gate_eps', 1e-2))
        self.logit_clip = 12.0

        # ---- dependency
        self.dep_mode = str(_cfg(cfg, 'dep_mode', 'ema'))
        assert self.dep_mode in ('grad', 'ema', 'none'), self.dep_mode
        self.dep_probes = int(_cfg(cfg, 'dep_probes', 4))
        self.dep_dist = str(_cfg(cfg, 'dep_probe_dist', 'rademacher'))
        self.dep_profile = DependencyProfile(self.N, float(_cfg(cfg, 'dep_ema', 0.99)))

        self.use_cycle_prior = False  # the GTR-aligned CyclePrior of v1 is gone; E3 uses the window only
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
        return list(self.router.parameters()) + self.gate_scalars()

    def gate_scalars(self):
        return [self.gate_a, self.gate_b, self.tau_raw]

    def gate_values(self):
        return dict(a=float(F.softplus(self.gate_a)), b=float(F.softplus(self.gate_b)), tau=float(self.tau_raw),
                    route_temp=float(self.route_temp))

    def set_route_temp(self, value):
        self.route_temp.fill_(float(value))

    @torch.no_grad()
    def calibrate_dependency(self, x, *bb_args):
        """Update the EMA dependency profile from one batch (any train/eval mode)."""
        if self.dep_mode == 'none':
            return
        D = hutchinson_dependency(self.backbone, x, None, self.pred_len, self.dep_probes,
                                  generator=self.probe_generator, dist=self.dep_dist,
                                  forward_fn=self._closure(bb_args))
        self.dep_profile.update(patch_sum(D, self.P))

    @torch.no_grad()
    def gate_logit_without_tau(self, x, *bb_args):
        """softplus(a)·logit(1−R) + softplus(b)·log(D̃+ε) for every point (used to calibrate τ)."""
        _, aux = self.repair(x, *bb_args)
        return aux['gate_logit'] + self.tau_raw if not self.force_gate_zero else None

    def stage_a_parameters(self):
        return list(self.reliability.parameters()) + list(self.repairer.parameters())

    def radar_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def radar_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}

    def router_coefficients(self):
        if self.router_type != 'linear':
            return None
        return self.router.coefficients(self.expert_names)

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
        dep_mode='grad': per-sample Hutchinson estimate (K probes); dep_mode='none': D̃ ≡ 1 (gate = detector only).
        During training D_p is estimated to update the EMA. Before any calibration, D̄ falls back to the batch mean.
        """
        B, L, C = x.shape
        if self.dep_mode == 'none':
            return x.new_ones(B, C, self.N)
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

    def candidates(self, x, R, c1, s, h=None):
        """Expert candidates c (B, L, C, K), availability (bool, B, L, C, K), Router input and the reliable context.
        R is detached here (E2/E3/features only read the reliable mask R > e2_thr and R itself as a feature)."""
        B, L, C = x.shape
        ctx = reliable_context(to_series(R.detach()) > self.e2_thr, self.e2_k)
        c2 = self.neighbor(x, ctx)
        c3, a3, spread3 = self.periodic(x, ctx, fill=c2)
        a3 = a3 & self.expert_enabled[2]
        c3 = torch.where(a3, c3, c2)
        cand = torch.stack([c1, c2, c3], dim=-1)
        ones = torch.ones_like(a3)
        avail = torch.stack([ones, ones & self.expert_enabled[1], a3], dim=-1)
        if self.router_type == 'linear':
            route_in = route_features(ctx, R, cand, s, a3, spread3)
        else:
            route_in = h
        return cand, avail, route_in, ctx

    def route(self, route_in, avail, B, C, pad):
        """π (B, L, C, K) from the Router input; unavailable experts get probability 0."""
        if self.router_type == 'linear':
            logits = self.router(route_in)
        else:
            logits = from_patches(self._run(self.router, route_in), B, C, pad)
        logits = logits / self.route_temp.clamp_min(1e-3)
        logits = logits.masked_fill(~avail, float('-inf'))
        if self.route_hard:
            return F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(logits.dtype)
        return torch.softmax(logits, dim=-1)

    def route_probs(self, aux):
        """π recomputed from the *detached* Router input of a previous repair() call (gradients reach the Router
        only; used by L_route)."""
        return self.route(aux['route_in'].detach(), aux['avail'], aux['B'], aux['C'], aux['pad'])

    @torch.no_grad()
    def repairer_candidate(self, x, R):
        """E1 candidate c1 for a *given* reliability R (B, L, C) — e.g. the oracle R = 1 − mask in diag_v4.py."""
        B, L, C = x.shape
        m, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        uc = ((x - m) / s).clamp(-self.u_clip, self.u_clip)
        up, pad = to_patches(uc, self.P)
        Rp, _ = to_patches(R.to(x.dtype), self.P)
        tokens = torch.stack([Rp * up, 1.0 - Rp], dim=-1).reshape(up.shape[0], self.N, 2 * self.P)
        u_hat_p, _ = self._run(self.repairer, tokens)
        return from_patches(u_hat_p, B, C, pad) * s + m

    # ------------------------------------------------------------------ forward
    def repair(self, x, *bb_args, rec_mask=None, compute_dep=True, need_route=True):
        """Returns x̃ and the aux dict. rec_mask: optional bool (B*C, N) of patches whose R is forced to 0
        in the Repairer input (masked-reconstruction training). need_route=False skips E2/E3/Router/gate
        (stage-A training only needs ℓ and û)."""
        B, L, C = x.shape
        f = self._closure(bb_args)

        m, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        u = (x - m) / s
        uc = u.clamp(-self.u_clip, self.u_clip)

        up, pad = to_patches(uc, self.P)  # (BC, N, P)
        BC = up.shape[0]
        feats = point_features(up.reshape(BC, -1), self.det_feats).reshape(BC, self.N, self.P * self.n_point_feats)
        logit_p = self._run(self.reliability, feats)  # unreliability logit ℓ = logit(1 − R)
        R_p = 1.0 - torch.sigmoid(logit_p)

        R_in = R_p
        if rec_mask is not None:
            R_in = R_p * (~rec_mask).to(R_p.dtype).unsqueeze(-1)
        rep_tokens = torch.stack([R_in * up, 1.0 - R_in], dim=-1).reshape(BC, self.N, 2 * self.P)
        u_hat_p, h = self._run(self.repairer, rep_tokens)

        logit = from_patches(logit_p, B, C, pad)
        R = from_patches(R_p, B, C, pad)
        u_hat = from_patches(u_hat_p, B, C, pad)
        c1 = u_hat * s + m

        if not need_route:
            aux = dict(logit=logit, R=R, u_hat=u_hat, m=m, s=s, x_tilde=x, alpha=torch.zeros_like(x))
            self.last_aux = aux
            return x, aux

        Dt_p = self.compute_dependency(x, f) if compute_dep else x.new_ones(B, C, self.N)
        cand, avail, route_in, ctx = self.candidates(x, R, c1, s, h=h)
        pi = self.route(route_in, avail, B, C, pad)  # (B, L, C, K)
        target = (pi * cand).sum(dim=-1)

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
                   pi=pi, x_tilde=x_tilde, c=cand, avail=avail, route_in=route_in, target=target, m=m, s=s,
                   gap=from_series(ctx['gap'], B, C), B=B, C=C, pad=pad)
        self.last_aux = aux
        return x_tilde, aux

    def forward(self, x, *bb_args, rec_mask=None, compute_dep=True, run_backbone=True):
        x_tilde, _ = self.repair(x, *bb_args, rec_mask=rec_mask, compute_dep=compute_dep)
        if not run_backbone:
            return None
        return _first(self.backbone(x_tilde, *bb_args))
