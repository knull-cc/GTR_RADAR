"""RADARWrapper: dependency-aware input repair placed *before* the backbone (and thus before its RevIN).

v6 (default, `--arch v6`) = v5 plus
    * E4 cross-channel expert in stage 1 (the k most correlated reliable channels, per-window linear fit);
    * segment gate: points whose logit z exceeds τ − δ form runs; every point of a run takes the run's maximum
      logit, so a run is repaired as a whole and isolated weak alarms never fire (hysteresis thresholds);
      τ is calibrated to a window-level false-repair budget, δ = τ − (point-level threshold);
    * forecast-space mixing: ŷ = (1 − ρ) f(x) + ρ f(x̃), ρ = max_t α_t per window and channel, x̃ repairs the
      runs with α ≥ 0.5 completely (uncertain windows average the two forecasts instead of half-repairing x).

v5 (`--arch v5`): select, then refine
    x ─ RobustScaler ─┬─ ReliabilityNet(5 point features) → ℓ, R = 1 − σ(ℓ)                       stage A (S2)
                      ├─ stage 1  E2 local linear | E3 periodic shape  (parameter-free)
                      │           π = softmax( −log ê_k  +  wᵀφ )      ê_k: leave-one-out error of expert k on the
                      │           c_base = π_2 c_2 + π_3 c_3            nearest reliable points; wᵀφ: learned (Router)
                      └─ stage 2  E1 residual Repairer([R·u, 1−R, u_base]) → Δ (zero-initialised)
                                  c = c_base + Δ·s
    α = σ( softplus(a)·ℓ + softplus(b)·log(D̃ + ε) + softplus(g)·log(1 + |c − x|/s) − τ )      stage B (S3)
    x̃ = x + α (c − x)  →  backbone(x̃, *args)
v4 (`--arch v4`): E1 is a third competing expert, π = softmax(wᵀφ) over {E1, E2, E3}, no consistency term.

The Router (w) only ever receives forecast-loss gradients: the forecast MSE through c_base and L_route. The
reconstruction loss sees u_base detached, and the gate sees c detached.
"""
import contextlib
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from radar.dependency import DependencyProfile, hutchinson_dependency, patch_sum
from radar.modules import (N_POINT_FEATURES, ROUTE_FEATURES, ROUTE_FEATURES_V5, ROUTE_FEATURES_V6, CrossChannelExpert,
                           LinearRouter, LocalLinearExpert, PeriodicShapeExpert, ReliabilityNet, Repairer, Router,
                           from_patches, from_series, loo_prior, loo_prior_multi, patch_to_points, point_features,
                           reliable_context, robust_stats, route_features, route_features_v5, route_features_v6,
                           seasonal_reference,
                           to_patches, to_series)

EXPERTS = ('Repairer', 'LocalLinear', 'PeriodicShape')
EXPERT_KEYS = ('e1', 'e2', 'e3', 'e4')
ARCHS = ('v4', 'v5', 'v6')


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


def _auto_flag(cfg, key, default):
    v = int(_cfg(cfg, key, -1))
    return bool(default) if v < 0 else bool(v)


def resolve_gate_consistency(cfg):
    """--gate_consistency: 1 / 0, or -1 (default) = on for v5 / v6, off for v4."""
    return _auto_flag(cfg, 'gate_consistency', str(_cfg(cfg, 'arch', 'v6')) in ('v5', 'v6'))


def resolve_tau_recal(cfg):
    """--tau_recal: post-training recalibration of τ at evaluation for Ours-plugin; 1 / 0, -1 (default) = off.
    On ETTh1 the trained τ was the better operating point (recalibrating to 5% of clean windows helped Clean by
    0.5% but cost 1–12% on the corrupted tails), so the recalibrated gate is reported as 'Ours-recal' instead."""
    return _auto_flag(cfg, 'tau_recal', False)


def resolve_experts(cfg):
    """--experts: comma list, or 'auto' (default) = e1,e2,e3,e4 for v6 and e1,e2,e3 otherwise."""
    e = str(_cfg(cfg, 'experts', 'auto')).strip().lower()
    if e in ('', 'auto'):
        e = 'e1,e2,e3,e4' if str(_cfg(cfg, 'arch', 'v6')) == 'v6' else 'e1,e2,e3'
    return [k.strip() for k in e.split(',') if k.strip()]


class _Section:
    """Wall-clock timer of one section (CUDA-synchronised), accumulated into a dict (timing diagnostic only)."""

    def __init__(self, store, name, sync):
        self.store, self.name, self.sync = store, name, sync

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        if self.sync:
            torch.cuda.synchronize()
        self.store[self.name] = self.store.get(self.name, 0.0) + time.perf_counter() - self.t0
        return False


def masked_median(v, mask, dim=1):
    """Median of v over the points where mask is True (along dim); NaN where the mask is empty."""
    return torch.nanmedian(torch.where(mask, v, torch.full_like(v, float('nan'))), dim=dim).values


def segment_max(z, inrun):
    """z, inrun: (B, L, C). Every point of a run of consecutive inrun points (along time, per series) takes the
    maximum z of its run; points outside runs keep their own z. Differentiable w.r.t. z (max sub-gradient)."""
    B, L, C = z.shape
    zs, rs = to_series(z), to_series(inrun)
    S = zs.shape[0]
    prev = torch.cat([torch.zeros_like(rs[:, :1]), rs[:, :-1]], dim=1)
    seg = torch.cumsum((rs & ~prev).long(), dim=1)  # run number 1.. inside each series (0 before the first run)
    gid = torch.arange(S, device=z.device).view(S, 1) * (L + 1) + torch.where(rs, seg, torch.zeros_like(seg))
    flat = gid.reshape(-1)
    init = torch.full((S * (L + 1),), -1e4, dtype=zs.dtype, device=z.device)
    smax = init.scatter_reduce(0, flat, zs.reshape(-1), reduce='amax', include_self=False)
    zr = smax.gather(0, flat).reshape(S, L)
    return from_series(torch.where(rs, zr, zs), B, C)


class RADARWrapper(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.P = int(_cfg(cfg, 'radar_patch', 8))
        self.N = math.ceil(self.seq_len / self.P)
        self.pad = self.N * self.P - self.seq_len
        self.arch = str(_cfg(cfg, 'arch', 'v6'))
        assert self.arch in ARCHS, self.arch
        v6 = self.arch == 'v6'
        v5 = self.arch in ('v5', 'v6')  # select-then-refine family

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
        self.repairer = Repairer(self.P, self.N, d, heads, layers, d_ff, drop, in_ch=3 if v5 else 2, residual=v5)
        self.e2_k = int(_cfg(cfg, 'e2_k', 3))
        self.e2_thr = float(_cfg(cfg, 'e2_thr', 0.5))
        self.neighbor = LocalLinearExpert()
        self.e3_period = resolve_e3_period(cfg)
        self.periodic = PeriodicShapeExpert(self.e3_period, self.seq_len, int(_cfg(cfg, 'e3_cycles', 3)))
        self.cross = CrossChannelExpert(k=int(_cfg(cfg, 'e4_k', 3)))
        self.std_frac = float(_cfg(cfg, 'scale_std_frac', 0.1))
        self.s_min = float(_cfg(cfg, 'scale_min', 1e-2))
        self.u_clip = float(_cfg(cfg, 'u_clip', 10.0))
        # repair range: every candidate / the final repair is clamped to [min, max] of the window's reliable points
        # of that channel ± repair_clip·s (0 = off). Guards against extrapolation blow-ups (E4 least squares, E2 slope).
        self.repair_clip = float(_cfg(cfg, 'repair_clip', 3.0))

        # ---- experts / Router
        keys = resolve_experts(cfg)
        assert 'e1' in keys and set(keys) <= set(EXPERT_KEYS), f'--experts must contain e1, got {keys}'
        assert v6 or 'e4' not in keys, 'E4 (cross-channel) exists in v6 only'
        if v5:
            assert 'e2' in keys, 'v5/v6 stage 1 needs E2: --experts must contain e2'
        self.register_buffer('expert_enabled', torch.tensor([k in keys for k in EXPERT_KEYS]), persistent=False)
        self._e4_on = 'e4' in keys
        self.router_type = str(_cfg(cfg, 'router', 'linear'))
        assert self.router_type in ('linear', 'h'), self.router_type
        if v6:
            assert self.router_type == 'linear', 'v6 uses the linear stage-1 Router'
            self.expert_names = ('LocalLinear', 'PeriodicShape', 'CrossChannel')  # routed experts (π)
            self.cand_names = ('Refined', 'LocalLinear', 'PeriodicShape', 'CrossChannel')
            self.route_feature_names = ROUTE_FEATURES_V6
            self.router = LinearRouter(len(ROUTE_FEATURES_V6), 3)
        elif v5:
            assert self.router_type == 'linear', 'v5 uses the linear stage-1 Router'
            self.expert_names = ('LocalLinear', 'PeriodicShape')  # routed experts (π)
            self.cand_names = ('Refined', 'LocalLinear', 'PeriodicShape')  # c[..., k]: final, E2, E3
            self.route_feature_names = ROUTE_FEATURES_V5
            self.router = LinearRouter(len(ROUTE_FEATURES_V5), 2)
        else:
            self.expert_names = EXPERTS
            self.cand_names = EXPERTS
            self.route_feature_names = ROUTE_FEATURES
            self.router = (LinearRouter(len(ROUTE_FEATURES), 3) if self.router_type == 'linear'
                           else Router(d, self.P, 3))
        self.n_experts = len(self.expert_names)
        self.register_buffer('route_temp', torch.tensor(float(_cfg(cfg, 'route_temp_start', 1.0))))
        self.route_hard = False  # argmax routing (evaluation only)

        # ---- stage B gate (a, b, [g], τ)
        # softplus(a) = 4 makes the gate close to a switch in the unreliability logit (v1's effective slope was 20)
        # label-free mode (v6): no corruption is ever injected in training. The detector is the stage-1 leave-one-out
        # residual (no labels), E1 / Router learn from random masks on clean data, the gate is not trained: its
        # three evidence terms have unit weight and τ / δ / the anchor threshold are calibrated on clean windows
        self.label_free = bool(int(_cfg(cfg, 'label_free', 0)))
        assert not self.label_free or v6, '--label_free needs --arch v6'
        a_init = 1.0 if self.label_free else float(_cfg(cfg, 'gate_a_init', 4.0))
        self.gate_a = nn.Parameter(torch.tensor(_inv_softplus(a_init)))
        self.gate_b = nn.Parameter(torch.tensor(_inv_softplus(float(_cfg(cfg, 'gate_b_init', 1.0)))))
        self.gate_consistency = resolve_gate_consistency(cfg) or self.label_free
        if self.gate_consistency:
            self.gate_g = nn.Parameter(torch.tensor(_inv_softplus(float(_cfg(cfg, 'gate_g_init', 1.0)))))
        self.tau_raw = nn.Parameter(torch.tensor(float(_cfg(cfg, 'tau_init', 1.0))))
        self.gate_eps = float(_cfg(cfg, 'gate_eps', 1e-2))
        self.logit_clip = 12.0
        # v6: segment gate (hysteresis; δ = τ − point-level threshold, calibrated) and forecast-space mixing
        self.seg_gate = _auto_flag(cfg, 'seg_gate', v6)
        self.fc_mix = _auto_flag(cfg, 'fc_mix', v6)
        if self.seg_gate:  # registered only when used, so v4 / v5 checkpoints keep loading strictly
            self.register_buffer('seg_delta', torch.tensor(2.0))
        if self.label_free:  # residual level above which a point is not used as an anchor (calibrated, 1% clean)
            self.register_buffer('lf_q0', torch.tensor(float('inf')))
        self.oracle_gate = False  # evaluation diagnostic: α = 1 exactly on the corrupted points
        self.oracle_mask = None
        # tail-segment hypothesis (v5 / v6): "everything from the strongest tail alarm to the end is corrupted" is
        # tested against a stage-1 repair anchored before it; the evidence lowers R on that segment and enters the gate
        self.tail_hyp = bool(int(_cfg(cfg, 'tail_hyp', 0))) and v5
        self.tail_len = int(_cfg(cfg, 'tail_hyp_len', 16))
        if self.tail_hyp:
            self.th_k = nn.Parameter(torch.tensor(_inv_softplus(4.0)))  # slope of w = σ(sp(k)(E − β))
            self.th_b = nn.Parameter(torch.tensor(1.5))                 # β: evidence level of w = 0.5
            self.th_h = nn.Parameter(torch.tensor(_inv_softplus(1.0)))  # gate weight of the evidence
        self.timer = None  # dict -> per-section wall-clock seconds (diag_v4.py timing)
        self._tprefix = ''

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
        """Stage B: Router + gate scalars."""
        return list(self.router.parameters()) + self.gate_scalars()

    def gate_scalars(self):
        g = [self.gate_g] if self.gate_consistency else []
        th = [self.th_k, self.th_b, self.th_h] if self.tail_hyp else []
        return [self.gate_a, self.gate_b] + g + th + [self.tau_raw]

    def gate_values(self):
        out = dict(a=float(F.softplus(self.gate_a)), b=float(F.softplus(self.gate_b)), tau=float(self.tau_raw),
                   route_temp=float(self.route_temp))
        if self.gate_consistency:
            out['g'] = float(F.softplus(self.gate_g))
        if self.seg_gate:
            out['seg_delta'] = float(self.seg_delta)
        if self.tail_hyp:
            out.update(th_k=float(F.softplus(self.th_k)), th_b=float(self.th_b), th_h=float(F.softplus(self.th_h)))
        return out

    def _sec(self, name):
        if self.timer is None:
            return contextlib.nullcontext()
        return _Section(self.timer, self._tprefix + name, torch.cuda.is_available())

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
        """Point-level gate logit z (before τ and before segment pooling) for every point (calibration of τ, δ)."""
        if self.force_gate_zero:
            return None
        _, aux = self.repair(x, *bb_args)
        return aux['gate_z']

    def stage_a_parameters(self):
        return list(self.reliability.parameters()) + list(self.repairer.parameters())

    def radar_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def radar_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}

    def router_coefficients(self):
        if self.router_type != 'linear':
            return None
        return self.router.coefficients(self.expert_names, self.route_feature_names)

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

    def repair_bounds(self, x, R, s):
        """(lo, hi), each (B, 1, C), or None when repair_clip <= 0: range of the reliable points (R > e2_thr) of each
        window/channel widened by repair_clip·s; windows/channels without a reliable point use all points."""
        if self.repair_clip <= 0:
            return None
        rel = R.detach() > self.e2_thr
        has = rel.any(dim=1, keepdim=True)
        big = 1e30
        xmax = torch.where(rel, x, torch.full_like(x, -big)).amax(dim=1, keepdim=True)
        xmin = torch.where(rel, x, torch.full_like(x, big)).amin(dim=1, keepdim=True)
        xmax = torch.where(has, xmax, x.amax(dim=1, keepdim=True))
        xmin = torch.where(has, xmin, x.amin(dim=1, keepdim=True))
        k = self.repair_clip * s.detach()
        return xmin - k, xmax + k

    @staticmethod
    def clamp_repair(c, bounds):
        if bounds is None:
            return c
        lo, hi = bounds
        return torch.maximum(torch.minimum(c, hi), lo)

    def _param_free(self, x, R):
        """E2 / E3 candidates and the reliable context for reliability R (detached)."""
        with self._sec('stage1/context+E2'):
            ctx = reliable_context(to_series(R.detach()) > self.e2_thr, self.e2_k)
            c2 = self.neighbor(x, ctx)
        with self._sec('stage1/E3'):
            c3, a3, spread3 = self.periodic(x, ctx, fill=c2)
        a3 = a3 & self.expert_enabled[2]
        c3 = torch.where(a3, c3, c2)
        return ctx, c2, c3, a3, spread3

    def _softmax_route(self, logits, avail):
        logits = logits / self.route_temp.clamp_min(1e-3)
        logits = logits.masked_fill(~avail, float('-inf'))
        if self.route_hard:
            return F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(logits.dtype)
        return torch.softmax(logits, dim=-1)

    # ---- v5 / v6 stage 1
    def stage1(self, x, R, s):
        """Stage 1 (v5: E2 / E3; v6: E2 / E3 / E4) with π = softmax(−log ê + wᵀφ). Returns a dict; c_base keeps the
        Router gradient, everything else is detached."""
        ctx, c2, c3, a3, spread3 = self._param_free(x, R)
        bounds = self.repair_bounds(x, R, s)
        c2, c3 = self.clamp_repair(c2, bounds), self.clamp_repair(c3, bounds)
        ones = torch.ones_like(a3)
        if self.arch == 'v6':
            if self._e4_on:
                with self._sec('stage1/E4'):
                    c4, a4, corr4 = self.cross(x, R.detach() > self.e2_thr, fill=c2)
                a4 = a4 & self.expert_enabled[3]
                c4 = self.clamp_repair(torch.where(a4, c4, c2), bounds)
            else:  # E4 disabled (ablation): skip its computation entirely
                c4, a4, corr4 = c2, torch.zeros_like(a3), x.new_zeros(x.shape[0], x.shape[2])
            cands, avails = [c2, c3, c4], [ones, a3, a4]
            with self._sec('stage1/loo_prior'):
                prior, ehat = loo_prior_multi(x, ctx, cands, avails, s)
            with self._sec('stage1/features'):
                phi = route_features_v6(ctx, R, c2, c3, c4, s, a3, a4, spread3, corr4)
        else:
            cands, avails = [c2, c3], [ones, a3]
            with self._sec('stage1/loo_prior'):
                prior, ehat = loo_prior(x, ctx, c2, c3, a3, s)
            with self._sec('stage1/features'):
                phi = route_features_v5(ctx, R, c2, c3, s, a3, spread3)
        with self._sec('stage1/router'):
            avail = torch.stack(avails, dim=-1)
            corr = self.router(phi)  # (B, L, C, K), E2 = reference (0)
            pi = self._softmax_route(prior + corr, avail)
            c_base = (pi * torch.stack(cands, dim=-1)).sum(dim=-1)
        return dict(ctx=ctx, cands=cands, avails=avails, prior=prior, ehat=ehat, phi=phi, corr=corr, avail=avail,
                    pi=pi, c_base=c_base, bounds=bounds)

    # ---- v4 routing
    def _route_v4(self, route_in, avail, B, C, pad):
        if self.router_type == 'linear':
            logits = self.router(route_in)
        else:
            logits = from_patches(self._run(self.router, route_in), B, C, pad)
        return self._softmax_route(logits, avail)

    def route_candidate(self, aux):
        """Routed repair value recomputed with gradients reaching the Router only (used by L_route):
        v5: Σ_k π_k sg(c_k) + sg(Δ·s);  v4: Σ_k π_k sg(c_k)."""
        if self.arch in ('v5', 'v6'):
            pi = self._softmax_route(aux['route_prior'] + self.router(aux['route_in'].detach()), aux['avail'])
            cand = aux['c'][..., 1:].detach()
            return (pi * cand).sum(dim=-1) + aux['delta_x'].detach()
        pi = self._route_v4(aux['route_in'].detach(), aux['avail'], aux['B'], aux['C'], aux['pad'])
        return (pi * aux['c'].detach()).sum(dim=-1)

    def route_probs(self, aux):
        """(kept for compatibility) π recomputed from the detached Router input."""
        if self.arch in ('v5', 'v6'):
            return self._softmax_route(aux['route_prior'] + self.router(aux['route_in'].detach()), aux['avail'])
        return self._route_v4(aux['route_in'].detach(), aux['avail'], aux['B'], aux['C'], aux['pad'])

    def _e1(self, Rp, up, u_base=None):
        """E1 forward on patch layout. Rp, up: (BC, N, P). v4: returns (û, h); v5 (u_base (B, L, C), the detached
        stage-1 repair in robust units): returns (Δ, h)."""
        BC = up.shape[0]
        chans = [Rp * up, 1.0 - Rp]
        if u_base is not None:
            ub_p, _ = to_patches(u_base, self.P)
            chans.append(ub_p)
        tokens = torch.stack(chans, dim=-1).reshape(BC, self.N, len(chans) * self.P)
        return self._run(self.repairer, tokens)

    @torch.no_grad()
    def repairer_candidate(self, x, R):
        """E1 candidate for a *given* reliability R (B, L, C) — e.g. the oracle R = 1 − mask in diag_v4.py.
        v5: the refined repair c_base + Δ·s (stage 1 run with the same R)."""
        B, L, C = x.shape
        m, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        uc = ((x - m) / s).clamp(-self.u_clip, self.u_clip)
        up, pad = to_patches(uc, self.P)
        R = R.to(x.dtype)
        Rp, _ = to_patches(R, self.P)
        if self.arch in ('v5', 'v6'):
            st = self.stage1(x, R, s)
            u_base = ((st['c_base'] - m) / s).clamp(-self.u_clip, self.u_clip)
            delta_p, _ = self._e1(Rp, up, u_base)
            return self.clamp_repair(st['c_base'] + from_patches(delta_p, B, C, pad) * s, st['bounds'])
        u_hat_p, _ = self._e1(Rp, up)
        return from_patches(u_hat_p, B, C, pad) * s + m

    # ------------------------------------------------------------------ forward
    def repair(self, x, *bb_args, rec_mask=None, compute_dep=True, need_route=True, drop_mask=None):
        """Returns x̃ and the aux dict. rec_mask: optional bool (B*C, N) of patches whose R is forced to 0
        (masked-reconstruction training; in v5 those patches are also unreliable for stage 1). drop_mask: optional
        bool (B, L, C) of points whose R is forced to 0 (v5 / v6; label-free masked-forecast training). need_route=False
        skips the dependency and the gate (stage-A training only needs ℓ and û; v4 also skips E2/E3/Router)."""
        B, L, C = x.shape
        f = self._closure(bb_args)

        m, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        u = (x - m) / s
        uc = u.clamp(-self.u_clip, self.u_clip)

        up, pad = to_patches(uc, self.P)  # (BC, N, P)
        BC = up.shape[0]
        if self.label_free:
            with self._sec('detector'):
                logit, R = self._label_free_detect(x, s, need_route)
            R_p, _ = to_patches(R, self.P)
        else:
            with self._sec('detector'):
                feats = point_features(up.reshape(BC, -1), self.det_feats).reshape(BC, self.N,
                                                                                    self.P * self.n_point_feats)
                logit_p = self._run(self.reliability, feats)  # unreliability logit ℓ = logit(1 − R)
            R_p = 1.0 - torch.sigmoid(logit_p)
            logit = from_patches(logit_p, B, C, pad)
            R = from_patches(R_p, B, C, pad)
        R_in_p = R_p
        if rec_mask is not None:
            R_in_p = R_p * (~rec_mask).to(R_p.dtype).unsqueeze(-1)

        aux = dict(logit=logit, R=R, m=m, s=s, B=B, C=C, pad=pad)
        if self.arch in ('v5', 'v6'):
            R_in = from_patches(R_in_p, B, C, pad)
            if drop_mask is not None:
                R_in = R_in * (~drop_mask).to(R_in.dtype)
                R_in_p, _ = to_patches(R_in, self.P)
            th_E = th_mask = None
            if self.tail_hyp and need_route:
                with self._sec('tail_hyp'):
                    th_E, th_mask, th_w = self._tail_hypothesis(x, logit, R_in, s)
                R_in = R_in * (1.0 - th_w.unsqueeze(1) * th_mask)
                R_in_p, _ = to_patches(R_in, self.P)
                aux.update(tail_E=th_E, tail_w=th_w, tail_mask=th_mask)
            st = self.stage1(x, R_in, s)
            u_base = ((st['c_base'] - m) / s).clamp(-self.u_clip, self.u_clip).detach()
            with self._sec('E1'):
                delta_p, h = self._e1(R_in_p, up, u_base)
            delta = from_patches(delta_p, B, C, pad)
            aux['u_hat'] = u_base + delta  # reconstruction target frame; stage 1 detached
            target = self.clamp_repair(st['c_base'] + delta * s, st['bounds'])  # Router gradient through c_base
            cand = torch.stack([target] + st['cands'], dim=-1)
            av = st['avail'].to(x.dtype)
            if self.arch == 'v6':  # log-odds of E3 and E4 vs E2 (vector), 0 where unavailable
                lo_prior = (st['prior'][..., 1:] - st['prior'][..., :1]) * av[..., 1:]
                lo_corr = (st['corr'][..., 1:] - st['corr'][..., :1]) * av[..., 1:]
            else:
                lo_prior = (st['prior'][..., 1] - st['prior'][..., 0]) * av[..., 1]
                lo_corr = (st['corr'][..., 1] - st['corr'][..., 0]) * av[..., 1]
            aux.update(pi=st['pi'], c=cand, avail=st['avail'], route_in=st['phi'], route_prior=st['prior'],
                       ehat=st['ehat'], delta_u=delta, delta_x=delta * s, route_lo_prior=lo_prior,
                       route_lo_corr=lo_corr, gap=from_series(st['ctx']['gap'], B, C),
                       cand_avail=torch.cat([torch.ones_like(st['avail'][..., :1]), st['avail']], dim=-1))
            if not need_route:
                aux.update(x_tilde=x, alpha=torch.zeros_like(x), target=target)
                self.last_aux = aux
                return x, aux
        else:
            u_hat_p, h = self._e1(R_in_p, up)
            u_hat = from_patches(u_hat_p, B, C, pad)
            aux['u_hat'] = u_hat
            c1 = u_hat * s + m
            if not need_route:
                aux.update(x_tilde=x, alpha=torch.zeros_like(x))
                self.last_aux = aux
                return x, aux
            ctx, c2, c3, a3, spread3 = self._param_free(x, R)
            cand = torch.stack([c1, c2, c3], dim=-1)
            ones = torch.ones_like(a3)
            avail = torch.stack([ones, ones & self.expert_enabled[1], a3], dim=-1)
            route_in = route_features(ctx, R, cand, s, a3, spread3) if self.router_type == 'linear' else h
            pi = self._route_v4(route_in, avail, B, C, pad)
            target = (pi * cand).sum(dim=-1)
            aux.update(pi=pi, c=cand, avail=avail, cand_avail=avail, route_in=route_in,
                       gap=from_series(ctx['gap'], B, C))

        with self._sec('dependency'):
            Dt_p = self.compute_dependency(x, f) if compute_dep else x.new_ones(B, C, self.N)
            Dt = patch_to_points(Dt_p, self.P, pad)  # (B, L, C)
        gsec = self._sec('gate')
        gsec.__enter__()
        z = (F.softplus(self.gate_a) * logit.clamp(-self.logit_clip, self.logit_clip)
             + F.softplus(self.gate_b) * torch.log(Dt + self.gate_eps))
        if self.gate_consistency:  # reconstruction-consistency evidence (repair detached)
            z = z + F.softplus(self.gate_g) * torch.log1p((target.detach() - x).abs() / s)
        if self.arch in ('v5', 'v6') and th_E is not None:  # segment evidence of the tail hypothesis
            z = z + F.softplus(self.th_h) * th_E.unsqueeze(1) * th_mask
        oracle = self.oracle_gate and self.oracle_mask is not None
        if self.force_gate_zero:
            gate_logit = torch.full_like(x, -float('inf'))
            alpha = torch.zeros_like(x)
        elif oracle:  # diagnostic: perfect gate on the corrupted points
            alpha = self.oracle_mask.to(x.dtype)
            gate_logit = torch.where(alpha > 0.5, torch.full_like(x, 30.0), torch.full_like(x, -30.0))
        else:
            zeff = segment_max(z, (z > self.tau_raw - self.seg_delta).detach()) if self.seg_gate else z
            gate_logit = zeff - self.tau_raw
            alpha = torch.sigmoid(gate_logit)
        if self.fc_mix and not self.force_gate_zero:
            hard = (alpha >= 0.5).to(x.dtype).detach()  # runs with α ≥ 0.5 are repaired completely
            x_tilde = x + hard * (target - x)
            rho = alpha.amax(dim=1)  # (B, C) probability that this window/channel needs the repair
            aux['rho'] = rho
        else:
            x_tilde = x + alpha * (target - x)

        gsec.__exit__(None, None, None)
        aux.update(alpha=alpha, gate_logit=gate_logit, gate_z=z.detach(), dbar=Dt, dbar_patch=Dt_p, x_tilde=x_tilde,
                   target=target)
        self.last_aux = aux
        return x_tilde, aux

    @torch.no_grad()
    def label_free_score(self, x, s=None):
        """Pass 0 of the label-free detector (no corruption labels anywhere):
          local term     log(1 + |c_base − x| / s), stage 1 run on ALL points: every expert predicts x_t from other
                         points (E2 / E3 anchors exclude t, E4 uses the partner channels at t) -> leave-one-out error;
                         strong on spikes and on the edges of corrupted runs, blind inside a run (its neighbours are
                         corrupted too)
          non-local term log(1 + |c_ref − x| / s), c_ref = the window-level seasonal reference (other cycles with one
                         window-wide level offset each) where the window holds another cycle, else E4 (correlated
                         channels) where available, else 0 -> sees the inside of zeroed / held / shifted runs.
        ℓ0 = local + non-local."""
        if s is None:
            _, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        prefix, self._tprefix = self._tprefix, 'detector/'
        st = self.stage1(x, torch.ones_like(x), s)
        self._tprefix = prefix
        local = torch.log1p((st['c_base'] - x).abs() / s)
        c_seas, a_seas = seasonal_reference(x, self.periodic.offsets, self.periodic.enabled and bool(self.expert_enabled[2]))
        c4, a4 = st['cands'][2], st['avails'][2]
        ref = torch.where(a_seas, c_seas, c4)
        has = a_seas | a4
        nonlocal_ = torch.where(has, torch.log1p((ref - x).abs() / s), torch.zeros_like(x))
        return local + nonlocal_

    def _label_free_detect(self, x, s, need_route):
        """Returns (ℓ, R): ℓ = pass-0 leave-one-out evidence (the gate's first term), R = 0 where ℓ exceeds the
        calibrated anchor threshold lf_q0 (such points are not used as anchors / reliable inputs), 1 elsewhere.
        In S2 (need_route=False) there is nothing to detect: ℓ = 0, R = 1 (only the random masks act)."""
        if not need_route:
            return torch.zeros_like(x), torch.ones_like(x)
        ell = self.label_free_score(x, s)
        R = (ell <= self.lf_q0).to(x.dtype)
        return ell, R

    def _tail_hypothesis(self, x, logit, R, s):
        """H: the points from the strongest unreliability alarm among the last `tail_len` points to the end of the
        window are corrupted (a shifted / held / zeroed tail whose inner points look locally normal). Stage 1 is
        re-run with that segment excluded from the anchors (no gradient); the evidence E = log(1 + median over the
        segment of |c_H − x| / s) per (window, channel) is large when the prefix cannot explain the segment.
        Returns E (B, C), the segment mask (B, L, C) and w = σ(sp(k)(E − β)) (B, C), which scales R down on it."""
        B, L, C = x.shape
        K = min(self.tail_len, L)
        j = (L - K) + logit[:, L - K:, :].detach().argmax(dim=1)  # (B, C)
        t = torch.arange(L, device=x.device).view(1, L, 1)
        seg = t >= j.unsqueeze(1)  # (B, L, C)
        mask = seg.to(x.dtype)
        prefix = self._tprefix
        self._tprefix = 'tail_hyp/'
        with torch.no_grad():
            c_h = self.stage1(x, R.detach() * (1.0 - mask), s)['c_base']
            E = torch.log1p(masked_median((c_h - x).abs() / s, seg, dim=1))
            E = torch.nan_to_num(E, nan=0.0)
        self._tprefix = prefix
        w = torch.sigmoid(F.softplus(self.th_k) * (E - self.th_b))
        return E, mask, w

    def forward(self, x, *bb_args, rec_mask=None, compute_dep=True, run_backbone=True):
        x_tilde, aux = self.repair(x, *bb_args, rec_mask=rec_mask, compute_dep=compute_dep)
        if not run_backbone:
            return None
        if 'rho' not in aux:
            with self._sec('backbone'):
                return _first(self.backbone(x_tilde, *bb_args))
        # forecast-space mixing. Exact saving: a window none of whose points is repaired has x̃ = x, hence
        # f(x̃) = f(x); the second backbone pass runs only on the windows that were changed.
        B = x.shape[0]
        with self._sec('backbone'):
            idx = (aux['alpha'] >= 0.5).flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                return _first(self.backbone(x, *bb_args))
            args2 = [torch.cat([a, a[idx]], dim=0) if torch.is_tensor(a) and a.shape[:1] == (B,) else a
                     for a in bb_args]
            y = _first(self.backbone(torch.cat([x, x_tilde[idx]], dim=0), *args2))
            y0 = y[:B]
            y1 = y0.index_copy(0, idx, y[B:])
        rho = aux['rho']
        w = rho.unsqueeze(1) if y.shape[-1] == rho.shape[-1] else rho.amax(dim=-1).view(B, 1, 1)
        return y0 + w * (y1 - y0)
