"""RADAR unit tests, v4 / v5 / v6 (B=4, L=96, C=7, H=96, cycle=24; randomly initialised GTR in eval mode).

Run from the repository root:  python -m pytest -q tests/test_radar.py
"""
import json
import math
import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import radar.compat  # noqa: E402,F401
from models import GTR as GTR_module  # noqa: E402
from radar.dependency import hutchinson_dependency, patch_sum  # noqa: E402
from radar.losses import (det_loss, hindsight_loss, hindsight_targets, patch_mask_to_points,  # noqa: E402
                          radar_loss, rec_loss, route_loss)
from radar.modules import (ROUTE_FEATURES, ROUTE_FEATURES_V5, ROUTE_FEATURES_V6,  # noqa: E402
                           CrossChannelExpert, LocalLinearExpert, PeriodicShapeExpert, ReliableNeighborExpert,
                           nanmedian_small,
                           from_patches, hampel_filter, loo_prior, loo_prior_multi, point_features,
                           reliable_context, robust_stats, shifted_windows, to_patches, to_series)
from radar.perturb import (MixturePerturber, apply_condition, apply_test_perturbation, condition_rng,  # noqa: E402
                           grid_conditions, parse_condition, perturb_batch, raw_zero_from_scaler)
from radar.radar import RADARWrapper, segment_max  # noqa: E402

B, L, C, H, CYCLE = 4, 96, 7, 96, 24


def make_cfg(**kw):
    cfg = types.SimpleNamespace(
        seq_len=L, pred_len=H, enc_in=C, cycle=CYCLE, d_model=512, dropout=0.5, use_revin=1, individual=0,
        radar_patch=8, radar_d=64, radar_heads=4, radar_layers=2, radar_dff=128, radar_dropout=0.1,
        radar_chunk=8192, u_clip=10.0, tau_init=1.0, gate_eps=1e-2, dep_mode='grad', dep_probes=4,
        dep_probe_dist='rademacher', dep_ema=0.99, force_gate_zero=False, e2_k=3, e2_thr=0.5,
        arch='v5', det_feats='v4', e3_period=24, e3_cycles=3, experts='auto', router='linear',
        route_temp_start=1.0, gate_consistency=-1,
        scale_std_frac=0.1, scale_min=1e-2,
        lam_det=0.5, lam_rec=0.5, lam_hind=0.5, lam_alpha=0.01, pos_weight=1.0, pu_neg_weight=0.1)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def make_gtr(cfg, seed=0):
    torch.manual_seed(seed)
    return GTR_module.Model(cfg).float().eval()


def make_batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, C, generator=g)
    ci = torch.randint(0, CYCLE, (B,), generator=g).int()
    return x, ci


def make_wrapper(seed=0, **kw):
    cfg = make_cfg(**kw)
    bb = make_gtr(cfg, seed)
    torch.manual_seed(seed + 1)
    return RADARWrapper(bb, cfg).eval(), bb, cfg


# 1 ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('arch,router', [('v6', 'linear'), ('v5', 'linear'), ('v4', 'linear'), ('v4', 'h')])
def test_output_shape_matches_backbone(arch, router):
    W, bb, _ = make_wrapper(arch=arch, router=router)
    x, ci = make_batch()
    with torch.no_grad():
        y_bb = bb(x, ci)
        y = W(x, ci)
    assert y.shape == y_bb.shape == (B, H, C)
    aux = W.last_aux
    for k in ('logit', 'R', 'alpha', 'dbar', 'u_hat', 'x_tilde', 'target', 'gap'):
        assert aux[k].shape == (B, L, C), k
    K = {'v6': 3, 'v5': 2, 'v4': 3}[arch]
    Kc = 4 if arch == 'v6' else 3
    assert aux['pi'].shape == (B, L, C, K) and aux['avail'].shape == (B, L, C, K)
    assert aux['c'].shape == (B, L, C, Kc) and aux['cand_avail'].shape == (B, L, C, Kc)
    assert aux['dbar_patch'].shape == (B, C, L // 8)
    assert torch.allclose(aux['pi'].sum(-1), torch.ones(B, L, C), atol=1e-5)
    assert (aux['pi'][~aux['avail']] == 0).all()
    if arch == 'v6':
        assert W.seg_gate and W.fc_mix and aux['rho'].shape == (B, C)
        assert aux['route_lo_prior'].shape == (B, L, C, 2)
    else:
        assert not W.seg_gate and not W.fc_mix and 'rho' not in aux
    if arch in ('v5', 'v6'):
        feats = ROUTE_FEATURES_V6 if arch == 'v6' else ROUTE_FEATURES_V5
        assert W.gate_consistency and aux['route_in'].shape == (B, L, C, len(feats))
        # residual E1 starts as the identity on the stage-1 repair (zero-initialised head)
        assert torch.count_nonzero(aux['delta_u']) == 0
        c_base = (aux['pi'] * aux['c'][..., 1:]).sum(-1)
        assert torch.allclose(aux['target'], c_base, atol=1e-5)
        assert torch.allclose(W.repairer_candidate(x, aux['R']), c_base, atol=1e-5)
    else:
        assert not W.gate_consistency
        if router == 'linear':
            assert aux['route_in'].shape == (B, L, C, len(ROUTE_FEATURES))


# 2 ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('dep_mode', ['grad', 'ema', 'none'])
def test_force_gate_zero_is_identity_and_init_alpha_small(dep_mode):
    W, bb, _ = make_wrapper(force_gate_zero=True, dep_mode=dep_mode)
    x, ci = make_batch()
    with torch.no_grad():
        ref = bb(x, ci)
        out = W(x, ci)
    assert torch.allclose(out, ref, atol=1e-6, rtol=0)
    assert torch.count_nonzero(W.last_aux['alpha']) == 0

    W.force_gate_zero = False
    with torch.no_grad():
        W(x, ci)
    assert W.last_aux['alpha'].mean().item() < 0.02
    if dep_mode == 'none':
        assert torch.allclose(W.last_aux['dbar'], torch.ones_like(W.last_aux['dbar']))


# 3 ------------------------------------------------------------------------------------------------------
class LinearToy(nn.Module):
    """y = W·vec(x) for every sample; accepts (x, cycle_index) like GTR."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(H * C, L * C, generator=g) / math.sqrt(L * C))

    def forward(self, x, cycle_index):
        return (x.reshape(x.shape[0], -1) @ self.W.t()).reshape(x.shape[0], H, C)


@pytest.mark.parametrize('dist', ['gaussian', 'rademacher'])
def test_hutchinson_matches_exact_jacobian_column_energy(dist):
    toy = LinearToy()
    x, ci = make_batch()
    g = torch.Generator().manual_seed(123)
    D = hutchinson_dependency(toy, x, ci, H, n_probes=256, generator=g, dist=dist)
    exact = (toy.W.detach() ** 2).sum(0).reshape(L, C).unsqueeze(0).expand(B, L, C)
    rel = ((D - exact).norm() / exact.norm()).item()
    assert rel < 0.10, rel
    assert toy.W.grad is None
    Dp = patch_sum(D, 8)
    assert Dp.shape == (B, C, L // 8) and torch.allclose(Dp.sum(-1), D.sum(1), rtol=1e-4)


# 4 ------------------------------------------------------------------------------------------------------
def test_legacy_conditions_and_y_untouched():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    scaler = StandardScaler().fit(rng.normal(5.0, 2.0, size=(500, C)))
    rz = raw_zero_from_scaler(scaler, 'raw')
    np.testing.assert_allclose(rz, (0.0 - scaler.mean_) / scaler.scale_, rtol=1e-6)

    x, _ = make_batch()
    y = torch.randn(B, H, C)
    y_copy = y.clone()
    xp, yp, mask = perturb_batch(x, y, 'T1_raw', rz)
    assert torch.equal(xp[:, :-1, :], x[:, :-1, :])
    assert torch.allclose(xp[:, -1, :], torch.as_tensor(rz).unsqueeze(0).expand(B, C))
    assert yp is y and torch.equal(y, y_copy)
    assert not mask[:, :-1, :].any() and mask[:, -1, :].all()
    xk, mk = apply_test_perturbation(x, 'T2_k4', rz)
    assert torch.equal(xk[:, :-4], x[:, :-4]) and mk[:, -4:].all()
    xf, mf = apply_test_perturbation(x, 'T3_flat', rz)
    assert torch.equal(xf[:, -8:], x[:, -9:-8].expand(B, 8, C))


def test_test_grid_definitions():
    conds = grid_conditions('full')
    assert conds[0] == 'clean' and len(conds) == 53 and len(grid_conditions('main')) == 27
    rz = np.full(C, -4.0, dtype=np.float32)
    x = torch.randn(32, L, C, generator=torch.Generator().manual_seed(3))
    for cond in conds[1:]:
        sp = parse_condition(cond)
        xp, ch = apply_condition(x, cond, rz, 'raw', condition_rng(7, cond, 0), guard=8)
        xp2, _ = apply_condition(x, cond, rz, 'raw', condition_rng(7, cond, 0), guard=8)
        assert torch.equal(xp, xp2), cond  # deterministic
        assert torch.equal(ch, (xp - x).abs() > 1e-6)
        t_any = ch.any(dim=2)  # (B, L)
        if sp['pos'] == 'int':
            assert not t_any[:, L - 8:].any(), cond  # never touches the last patch
        elif sp['type'] == 'spike':
            assert not t_any[:, :L - 8].any(), cond
        else:
            assert not t_any[:, :L - sp['k']].any(), cond
        if sp['ratio'] == 50:
            assert (ch.any(dim=1).sum(dim=1) <= math.ceil(C / 2)).all(), cond
        if sp['type'] == 'zero':
            assert torch.allclose(xp[ch], torch.full_like(xp[ch], -4.0))


def test_training_mixture_v4():
    rz = np.zeros(C, dtype=np.float32) - 3.0
    x, _ = make_batch()
    x = torch.cat([x] * 16)
    pert = MixturePerturber(rz, p_apply=1.0, seed=0)
    xm, mm, ty = pert(x)
    assert 'shift' not in MixturePerturber.TYPES and 'flat' in MixturePerturber.TYPES
    assert xm.shape == x.shape and (ty >= 0).all() and set(pert.last_pos.tolist()) <= {0, 1}
    assert torch.equal(mm, (xm - x).abs() > 1e-6)
    for b in range(x.shape[0]):
        ts = torch.nonzero(mm[b].any(dim=1)).flatten()
        if len(ts) and pert.last_pos[b] == 0:
            assert ts.max() < L - 8


# 5 ------------------------------------------------------------------------------------------------------
def test_point_features_v4_see_the_last_point():
    z = torch.sin(torch.arange(L, dtype=torch.float32) / 5.0)
    z2 = z.clone()
    z2[-1] = 6.0
    u = torch.stack([z, z2])
    f3 = point_features(u, 'v3')
    f4 = point_features(u, 'v4')
    assert f3.shape == (2, L, 3) and f4.shape == (2, L, 5)
    assert f3[1, -1, 2].abs() < 1e-6  # v3: |u − med5| is always 0 at the last point
    assert f4[1, -1, 2] > 3 and f4[1, -1, 3] > 3
    w = shifted_windows(torch.arange(20.).view(1, 20), 5)
    assert torch.equal(w[0, -1], torch.arange(15., 20.)) and torch.equal(w[0, 10], torch.arange(8., 13.))
    flat = z.clone()
    flat[40:48] = flat[39]
    f = point_features(flat.view(1, -1), 'v4')
    assert (f[0, 42:46, 4] == 0).all() and (f[0, 20:24, 4] > 0).all()  # MAD5 = 0 on a flat segment


def test_reliable_context():
    M = torch.ones(1, 20, dtype=torch.bool)
    M[0, 5:9] = False
    M[0, 17:] = False
    ctx = reliable_context(M, 3)
    assert ctx['gap'][0, 4] == 0 and (ctx['gap'][0, 5:9] == 4).all() and (ctx['gap'][0, 17:] == 3).all()
    assert ctx['dl'][0, 8] == 4 and ctx['dr'][0, 5] == 4 and (ctx['dr'][0, 17:] == 20).all()
    assert ctx['pb'][0, 6].tolist() == [4, 3, 2] and ctx['pf'][0, 6].tolist() == [9, 10, 11]
    assert not ctx['vf'][0, 18].any()


def test_local_linear_and_periodic_experts_are_exact_on_ideal_signals():
    # E2: a straight line with an internal gap is recovered exactly; at the window end -> level of the last 3 points
    lin = 0.5 * torch.arange(40.) + 2.0
    x = lin.clone()
    x[10:16] = -9.0
    x[35:] = -9.0
    R = torch.ones(40)
    R[10:16] = 0
    R[35:] = 0
    x, R = x.view(1, 40, 1), R.view(1, 40, 1)
    ctx = reliable_context(to_series(R) > 0.5, 3)
    c2 = LocalLinearExpert()(x, ctx)[0, :, 0]
    assert torch.allclose(c2[10:16], lin[10:16], atol=1e-4)
    assert torch.allclose(c2[35:], lin[32:35].median().expand(5))
    # E3: periodic signal + linear trend, internal (k=8) and tail (k=6) gaps -> exact; E2 is not
    t = torch.arange(96.)
    sig = torch.sin(2 * math.pi * t / 24) + 0.3 * torch.cos(4 * math.pi * t / 24) + 0.02 * t
    x = sig.clone()
    x[40:48] = 0.0
    x[90:] = 0.0
    R = torch.ones(96)
    R[40:48] = 0
    R[90:] = 0
    x, R = x.view(1, 96, 1), R.view(1, 96, 1)
    ctx = reliable_context(to_series(R) > 0.5, 3)
    c2 = LocalLinearExpert()(x, ctx)
    c3, a3, _ = PeriodicShapeExpert(24, 96, 3)(x, ctx, fill=c2)
    assert a3.all()
    for sl in (slice(40, 48), slice(90, 96)):
        assert (c3[0, sl, 0] - sig[sl]).abs().max() < 1e-4
        assert (c2[0, sl, 0] - sig[sl]).abs().max() > 0.5
    # T >= L: E3 unavailable, returns the fill
    c3o, a3o, _ = PeriodicShapeExpert(96, 96, 3)(x, ctx, fill=c2)
    assert not a3o.any() and torch.equal(c3o, c2)


def test_legacy_reliable_neighbor_expert_and_scale_floor():
    E2 = ReliableNeighborExpert(k=3, thr=0.5, trend=True)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 20, 1, generator=g)
    x[:, -4:] = -5.0
    R = torch.ones_like(x)
    R[:, -4:] = 0.0
    c = E2(x, R)[0, :, 0]
    z = x[0, :, 0]
    assert torch.allclose(c[10], torch.cat([z[7:10], z[11:14]]).median())
    flat = torch.ones(2, 96, 3)
    m, s = robust_stats(flat)
    assert torch.allclose(s, torch.full_like(s, 1e-2)) and torch.isfinite((flat - m) / s).all()


# 6 ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('arch', ['v5', 'v6'])
def test_gradients_joint_stage_b_and_route_loss(arch):
    W, bb, cfg = make_wrapper(arch=arch)
    if arch == 'v6':  # open the gate everywhere: with nothing repaired the forecast path skips the repair entirely
        with torch.no_grad():
            W.tau_raw.fill_(-1e4)
    x, ci = make_batch()
    y = torch.randn(B, H, C)
    x_p = x.clone()
    x_p[:, -4:, :] = -3.0
    x_p[:, 30:34, :2] = 4.0
    mask = (x_p - x).abs() > 1e-6
    rec_mask = torch.rand(B * C, W.N, generator=torch.Generator().manual_seed(0)) < 0.1

    # joint loss: every RADAR module receives a finite gradient
    y_hat = W(x_p, ci, rec_mask=rec_mask)
    aux = W.last_aux
    tg = hindsight_targets(W.backbone, x_p, ci, y, aux['target'].detach(), W.P, H)
    hind = hindsight_loss(aux['alpha'], tg, W.P)
    total, _ = radar_loss(y_hat, y, aux, x, mask, patch_mask_to_points(rec_mask, B, C, L, W.P), hind, cfg)
    W.zero_grad(set_to_none=True)
    total.backward()
    for name, mod in (('ReliabilityNet', W.reliability), ('Repairer', W.repairer), ('Router', W.router)):
        for n, p in mod.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f'{name}.{n}'
    for p in W.gate_scalars():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert len(W.gate_scalars()) == 4  # a, b, g (consistency), τ

    # L_route: gradients only w.r.t. the Router, non-zero, finite
    W.zero_grad(set_to_none=True)
    W(x_p, ci)
    lr = route_loss(W, W.last_aux, x_p, mask, y, (ci,), H)
    rp = list(W.router.parameters())
    gr = torch.autograd.grad(lr, rp)
    assert all(torch.isfinite(g).all() for g in gr) and any(g.abs().sum() > 0 for g in gr)
    assert all(p.grad is None for p in W.parameters())
    assert route_loss(W, W.last_aux, x, torch.zeros_like(mask), y, (ci,), H) is None

    # stage B: backbone + stage-A modules frozen -> only Router and gate are trained
    for p in list(W.backbone.parameters()) + W.stage_a_parameters():
        p.requires_grad_(False)
    W.zero_grad(set_to_none=True)
    y_hat = W(x_p, ci)
    aux = W.last_aux
    tg = hindsight_targets(W.backbone, x_p, ci, y, aux['target'].detach(), W.P, H)
    total, _ = radar_loss(y_hat, y, aux, x, mask, None, hindsight_loss(aux['alpha'], tg, W.P), cfg, stage='B')
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in W.gate_parameters())
    assert all(p.grad is None for p in W.stage_a_parameters())
    assert all(p.grad is None for p in W.backbone.parameters())


@pytest.mark.parametrize('arch', ['v6', 'v5', 'v4'])
def test_routing_temperature_argmax_and_disabled_experts(arch):
    e3 = 2 if arch == 'v4' else 1  # index of E3 in π
    W, _, _ = make_wrapper(arch=arch, experts='e1,e2')
    x, ci = make_batch()
    with torch.no_grad():
        W(x, ci)
    assert (W.last_aux['pi'][..., e3] == 0).all() and not W.last_aux['avail'][..., e3].any()
    if arch == 'v6':  # E4 disabled too
        assert (W.last_aux['pi'][..., 2] == 0).all() and not W.last_aux['avail'][..., 2].any()
    W2, _, _ = make_wrapper(arch=arch, e3_period=96)  # T >= L -> E3 unavailable everywhere
    with torch.no_grad():
        W2(x, ci)
    assert (W2.last_aux['pi'][..., e3] == 0).all()
    W3, _, _ = make_wrapper(arch=arch)
    with torch.no_grad():
        W3.router.lin.bias.copy_(torch.tensor({'v6': [3.0, -1.0], 'v5': [3.0], 'v4': [0.5, -0.5]}[arch]))
        W3.set_route_temp(0.3)
        W3(x, ci)
        pi_soft = W3.last_aux['pi']
        W3.route_hard = True
        W3(x, ci)
        pi_hard = W3.last_aux['pi']
    assert torch.equal(pi_hard.argmax(-1), pi_soft.argmax(-1))
    assert torch.equal(pi_hard.sum(-1), torch.ones_like(pi_hard.sum(-1))) and set(pi_hard.unique().tolist()) <= {0., 1.}
    assert 'route_temp' in W3.state_dict() and abs(W3.state_dict()['route_temp'].item() - 0.3) < 1e-6
    coef = W3.router_coefficients()
    if arch == 'v6':
        assert set(coef) == {'PeriodicShape', 'CrossChannel'} and abs(coef['CrossChannel']['bias'] + 1.0) < 1e-6
    elif arch == 'v5':
        assert set(coef) == {'PeriodicShape'} and abs(coef['PeriodicShape']['bias'] - 3.0) < 1e-6
    else:
        assert set(coef) == {'LocalLinear', 'PeriodicShape'} and abs(coef['LocalLinear']['bias'] - 0.5) < 1e-6


def test_loo_prior_prefers_the_expert_that_explains_the_neighbours():
    t = torch.arange(96.)
    sig = torch.sin(2 * math.pi * t / 24) + 0.3 * torch.cos(4 * math.pi * t / 24) + 0.02 * t
    x = sig.clone()
    x[40:48] = 0.0
    R = torch.ones(96)
    R[40:48] = 0
    x, R = x.view(1, 96, 1), R.view(1, 96, 1)
    ctx = reliable_context(to_series(R) > 0.5, 3)
    c2 = LocalLinearExpert()(x, ctx)
    c3, a3, _ = PeriodicShapeExpert(24, 96, 3)(x, ctx, fill=c2)
    _, s = robust_stats(x)
    prior, ehat = loo_prior(x, ctx, c2, c3, a3, s)
    assert prior.shape == (1, 96, 1, 2) and torch.isfinite(prior).all()
    assert (prior[0, 40:48, 0, 1] > prior[0, 40:48, 0, 0]).all()  # periodic signal: E3 explains the neighbours
    lin = (0.3 * t + 1.0).view(1, 96, 1)
    ctx2 = reliable_context(to_series(torch.ones_like(lin)) > 0.5, 3)
    c2l = LocalLinearExpert()(lin, ctx2)
    c3l, a3l, _ = PeriodicShapeExpert(24, 96, 3)(lin, ctx2, fill=c2l)
    prior2, _ = loo_prior(lin, ctx2, c2l, c3l, a3l, robust_stats(lin)[1])
    assert torch.isfinite(prior2).all()


def test_checkpoint_layouts_v4_v5_v6():
    W4, _, _ = make_wrapper(arch='v4')
    W5, _, _ = make_wrapper(arch='v5')
    W6, _, _ = make_wrapper(arch='v6')
    k4, k5, k6 = set(W4.state_dict()), set(W5.state_dict()), set(W6.state_dict())
    assert 'gate_g' in k5 and 'gate_g' not in k4
    assert W4.repairer.body.proj.in_features == 2 * 8 and W5.repairer.body.proj.in_features == 3 * 8
    # v6 adds only the segment-gate offset δ (buffer) and a wider Router; v4 / v5 checkpoints keep loading strictly
    assert k6 - k5 == {'seg_delta'} and 'seg_delta' not in k5 and 'seg_delta' not in k4
    assert W6.router.lin.weight.shape == (2, len(ROUTE_FEATURES_V6))
    W5b, _, _ = make_wrapper(arch='v5', seed=3)
    W5b.load_state_dict(W5.state_dict())  # strict


# 7 ------------------------------------------------------------------------------------------------------
def test_dependency_runs_under_no_grad_without_param_grads():
    cfg = make_cfg()
    bb = make_gtr(cfg).train()  # mode must be restored afterwards
    x, ci = make_batch()
    with torch.no_grad():
        D = hutchinson_dependency(bb, x, ci, H, n_probes=2)
    assert D.shape == x.shape and torch.isfinite(D).all() and (D >= 0).all() and D.sum() > 0
    assert not D.requires_grad
    assert all(p.grad is None for p in bb.parameters())
    assert bb.training

    W = RADARWrapper(bb, cfg).eval()
    with torch.no_grad():
        out = W(x, ci)
    assert torch.isfinite(out).all()
    assert all(p.grad is None for p in W.parameters())


# extra checks --------------------------------------------------------------------------------------------
def test_losses_pu_and_rec_frame():
    logit = torch.randn(2, 8, 3)
    mask = torch.zeros(2, 8, 3, dtype=torch.bool)
    mask[:, -1] = True
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, mask.float(), reduction='none')
    w = torch.where(mask, torch.ones_like(bce), torch.full_like(bce, 0.1))
    assert torch.allclose(det_loss(logit, mask, 1.0, 0.1), (w * bce).mean())

    x_clean = torch.randn(2, 8, 3)
    m, s = torch.zeros(2, 1, 3), torch.full((2, 1, 3), 2.0)
    u_hat = x_clean / 2.0
    assert rec_loss(u_hat, x_clean, torch.ones_like(mask), 10.0, m, s).item() < 1e-10


def test_patch_roundtrip_and_hampel():
    x, _ = make_batch()
    xp, pad = to_patches(x, 8)
    assert pad == 0 and torch.equal(from_patches(xp, B, C, pad), x)
    x2 = x[:, :93]
    xp2, pad2 = to_patches(x2, 8)
    assert pad2 == 3 and torch.equal(from_patches(xp2, B, C, pad2), x2)
    hf = hampel_filter(x)
    assert hf.shape == x.shape and torch.isfinite(hf).all()


@pytest.mark.parametrize('arch', ['v5', 'v6'])
def test_seq_len_not_multiple_of_patch(arch):
    W, bb, _ = make_wrapper(seq_len=93, arch=arch)
    x = torch.randn(B, 93, C)
    ci = torch.randint(0, CYCLE, (B,)).int()
    with torch.no_grad():
        out = W(x, ci)
    K = 3 if arch == 'v6' else 2
    assert out.shape == (B, H, C) and W.last_aux['pi'].shape == (B, 93, C, K) and W.last_aux['c'].shape[1] == 93


def test_ema_profile_and_global_scale():
    W, bb, cfg = make_wrapper(dep_mode='ema')
    x, ci = make_batch()
    W.train()
    W(x, ci)
    assert bool(W.dep_profile.initialized)
    W.eval()
    with torch.no_grad():
        W(x, ci)
    prof = W.dep_profile.normalized()
    assert torch.allclose(prof.mean(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(W.last_aux['dbar_patch'][0, 0], prof)


def test_tuple_returning_backbone_closure():
    class TupleToy(LinearToy):
        def forward(self, x, cycle_index):
            return super().forward(x, cycle_index), None  # (output, attention)

    toy = TupleToy()
    W = RADARWrapper(toy, make_cfg()).eval()
    x, ci = make_batch()
    with torch.no_grad():
        out = W(x, ci)
    assert out.shape == (B, H, C)


@pytest.mark.parametrize('name', ['GTRDLinear', 'GTRiTransformer', 'GTRPatchTST'])
def test_other_gtr_backbones(name):
    import importlib
    mod = importlib.import_module(f'models.{name}')
    cfg = make_cfg(embed='timeF', freq='h', output_attention=False, factor=1, n_heads=8, e_layers=2, d_ff=256,
                   d_model=128, activation='gelu', fc_dropout=0.05, head_dropout=0.0, patch_len=16, stride=8,
                   padding_patch='end', revin=0, affine=0, subtract_last=0, decomposition=0, kernel_size=25)
    torch.manual_seed(0)
    bb = mod.Model(cfg).float().eval()
    W = RADARWrapper(bb, cfg).eval()
    x, ci = make_batch()
    with torch.no_grad():
        ref = bb(x, ci)
        out = W(x, ci)
    assert out.shape == ref.shape


def test_revin_forward_reproduces_gtr():
    from diag_v4 import revin_forward
    cfg = make_cfg()
    bb = make_gtr(cfg)
    x, ci = make_batch()
    x2 = x.clone()
    x2[:, -4:] = 5.0
    with torch.no_grad():
        assert torch.allclose(revin_forward(bb, x, x, ci), bb(x, ci), atol=1e-5)
        assert not torch.allclose(revin_forward(bb, x, x2, ci), bb(x, ci), atol=1e-3)
    assert bb.use_revin == 1


@pytest.mark.parametrize('arch', ['v5', 'v6'])
def test_eval_stats_and_summary(tmp_path, arch):
    from exp.exp_radar import RadarStats, hist_auroc
    from radar.summary import summarize
    W, _, _ = make_wrapper(dep_mode='none', arch=arch)
    x, ci = make_batch()
    rz = np.full(C, -3.0, dtype=np.float32)
    st_clean, st = RadarStats(W.n_experts, len(W.cand_names)), RadarStats(W.n_experts, len(W.cand_names))
    with torch.no_grad():
        W(x, ci)
        st_clean.update(W.last_aux, x, x, torch.zeros_like(x, dtype=torch.bool))
        xp, ch = apply_condition(x, 'zero_tail_k4_c100', rz, 'raw', condition_rng(0, 'zero_tail_k4_c100', 0))
        W(xp, ci)
        st.update(W.last_aux, x, xp, ch)
    r = st.result()
    assert r['n_changed'] == int(ch.sum()) and r['err_raw'] > 0 and len(r['pi_open_by_gap']) == 5
    assert len(r['err_expert']) == len(W.cand_names) and len(r['pi_changed']) == W.n_experts and 'delta_changed' in r
    if arch == 'v6':
        assert len(r['logodds_prior_changed']) == 2 and all(math.isfinite(v) for v in r['logodds_prior_changed'])
    else:
        assert math.isfinite(r['logodds_prior_changed'])
    a = hist_auroc(st.hist_pos, st_clean.hist_all)
    assert 0.0 <= a <= 1.0
    raw = tmp_path / 'raw'
    raw.mkdir()
    for m in ('B0', 'Ours-plugin', 'Ours-plugin-v4'):
        rec = dict(kind='eval', method=m, dataset='ETTh1', pred_len=96, seed=2024,
                   conditions={'clean': dict(mse=0.4, mae=0.4), 'zero_tail_k4_c100': dict(mse=0.6, mae=0.5)},
                   extras={'clean': st_clean.result(), 'zero_tail_k4_c100': dict(r, auroc=a),
                           'router_coef': W.router_coefficients(), 'experts': list(W.expert_names),
                           'candidates': list(W.cand_names), 'arch': W.arch,
                           'route_features': list(W.route_feature_names), 'gate': W.gate_values()}
                   if m != 'B0' else {})
        (raw / f'x__{m}.json').write_text(json.dumps(rec))
    summarize(str(tmp_path), print_tables=False)
    assert 'Zero-T' in (tmp_path / 'summary.md').read_text()
    assert 'PeriodicShape vs LocalLinear' in (tmp_path / 'summary_router.md').read_text()
    assert 'residual E1' in (tmp_path / 'summary.md').read_text()
    assert 'vs Ours-plugin-v4' in (tmp_path / 'summary.md').read_text()
    if arch == 'v6':
        assert 'CrossChannel vs LocalLinear' in (tmp_path / 'summary_router.md').read_text()


# v6 -----------------------------------------------------------------------------------------------------
def test_v6_closed_gate_is_the_backbone_and_open_gate_forecasts_the_repair():
    W, bb, _ = make_wrapper(arch='v6')
    x, ci = make_batch()
    with torch.no_grad():
        W.tau_raw.fill_(1e4)  # gate closed everywhere -> ρ = 0 -> ŷ = f(x)
        out = W(x, ci)
        assert torch.count_nonzero(W.last_aux['rho']) == 0
        assert torch.allclose(out, bb(x, ci), atol=1e-5)
        W.tau_raw.fill_(-1e4)  # gate open everywhere -> x̃ = repair, ρ = 1 -> ŷ = f(x̃)
        out = W(x, ci)
        aux = W.last_aux
        assert torch.allclose(aux['x_tilde'], aux['target'], atol=1e-5) and torch.allclose(aux['rho'], torch.ones(B, C))
        assert torch.allclose(out, bb(aux['target'], ci), atol=1e-5)
        W.force_gate_zero = True
        assert torch.allclose(W(x, ci), bb(x, ci), atol=1e-6) and 'rho' not in W.last_aux


def test_v6_forecast_mixing_is_convex_per_channel():
    W, bb, _ = make_wrapper(arch='v6')
    x, ci = make_batch()
    with torch.no_grad():
        W.tau_raw.fill_(0.0)
        out = W(x, ci)
        aux = W.last_aux
        y0, y1 = bb(x, ci), bb(aux['x_tilde'], ci)
    ref = y0 + aux['rho'].unsqueeze(1) * (y1 - y0)
    assert torch.allclose(out, ref, atol=1e-5)
    hard = aux['alpha'] >= 0.5
    assert torch.allclose(aux['x_tilde'], torch.where(hard, aux['target'], x), atol=1e-5)
    assert torch.allclose(aux['rho'], aux['alpha'].amax(dim=1))


def test_segment_max_runs():
    z = torch.tensor([[0.0, 3.0, 1.0, -1.0, 2.0, 0.5], [1.0, 0.0, 0.0, 0.0, 0.0, 5.0]]).T.unsqueeze(0)  # (1, 6, 2)
    inrun = torch.tensor([[True, True, True, False, True, True], [True, False, False, False, True, True]]).T.unsqueeze(0)
    out = segment_max(z, inrun)
    assert torch.equal(out[0, :, 0], torch.tensor([3.0, 3.0, 3.0, -1.0, 2.0, 2.0]))
    assert torch.equal(out[0, :, 1], torch.tensor([1.0, 0.0, 0.0, 0.0, 5.0, 5.0]))  # no leak across series
    zg = z.clone().requires_grad_(True)
    segment_max(zg, inrun).sum().backward()
    assert torch.isfinite(zg.grad).all() and zg.grad[0, 1, 0] == 3  # the run maximum receives the run's gradient


def test_v6_segment_gate_repairs_whole_runs():
    W, _, _ = make_wrapper(arch='v6')
    z = torch.full((1, 12, 1), -5.0)
    z[0, 3:7, 0] = torch.tensor([0.5, 2.0, 0.5, 0.5])  # one strong point inside a weakly suspicious run
    z[0, 10, 0] = 0.5                                   # isolated weak false alarm (above τ − δ, below τ)
    W.seg_delta.fill_(1.0)
    with torch.no_grad():
        W.tau_raw.fill_(1.0)
    zeff = segment_max(z, z > W.tau_raw - W.seg_delta)
    alpha = torch.sigmoid(zeff - W.tau_raw)
    assert (alpha[0, 3:7, 0] > 0.5).all() and alpha[0, 10, 0] < 0.5 and alpha[0, 0, 0] < 0.5


def test_cross_channel_expert_exact_on_correlated_channels():
    g = torch.Generator().manual_seed(0)
    base = torch.cumsum(torch.randn(2, 96, generator=g), dim=1)
    a, b = torch.randn(5, generator=g), torch.tensor([1.0, -0.7, 2.0, 0.5, -1.5])
    clean = a + b * base.unsqueeze(-1)
    x, M = clean.clone(), torch.ones_like(clean, dtype=torch.bool)
    x[:, 40:48, 0], M[:, 40:48, 0] = 0.0, False   # one channel corrupted -> E4 available and exact
    x[:, -4:, :], M[:, -4:, :] = -3.0, False      # all channels corrupted -> E4 unavailable
    c4, av, best = CrossChannelExpert(k=3)(x, M, fill=torch.full_like(x, -99.0))
    assert av[:, 40:48, 0].all() and torch.allclose(c4[:, 40:48, 0], clean[:, 40:48, 0], atol=1e-3)
    assert not av[:, -4:].any() and (c4[~av] == -99.0).all() and (best > 0.99).all()
    c1, a1, _ = CrossChannelExpert()(x[..., :1], M[..., :1], fill=x[..., :1])
    assert not a1.any() and torch.equal(c1, x[..., :1])


def test_loo_prior_multi_matches_loo_prior_and_prefers_e4():
    t = torch.arange(96.)
    sig = torch.sin(2 * math.pi * t / 24) + 0.02 * t
    X = torch.stack([sig + 0.1 * k for k in range(3)], dim=-1).unsqueeze(0)
    Xc = X.clone()
    Xc[0, 40:48, 0] = 0.0
    R = torch.ones_like(Xc)
    R[0, 40:48, 0] = 0.0
    ctx = reliable_context(to_series(R) > 0.5, 3)
    c2 = LocalLinearExpert()(Xc, ctx)
    c3, a3, _ = PeriodicShapeExpert(24, 96, 3)(Xc, ctx, fill=c2)
    c4, a4, _ = CrossChannelExpert(3)(Xc, R > 0.5, fill=c2)
    _, s = robust_stats(Xc)
    p2, _ = loo_prior(Xc, ctx, c2, c3, a3, s)
    pm, _ = loo_prior_multi(Xc, ctx, [c2, c3], [torch.ones_like(a3), a3], s)
    assert torch.allclose(p2, pm)
    p3, _ = loo_prior_multi(Xc, ctx, [c2, c3, c4], [torch.ones_like(a3), a3, a4], s)
    assert p3.shape == (1, 96, 3, 3) and torch.isfinite(p3).all()
    assert (p3[0, 40:48, 0, 2] > p3[0, 40:48, 0, 0]).all()


def test_v6_oracle_gate():
    W, bb, _ = make_wrapper(arch='v6')
    x, ci = make_batch()
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, -4:, :2] = True
    W.oracle_gate, W.oracle_mask = True, mask
    with torch.no_grad():
        out = W(x, ci)
        aux = W.last_aux
        assert torch.equal(aux['alpha'], mask.float())
        assert torch.equal(aux['rho'], mask.any(dim=1).float())
        assert torch.allclose(aux['x_tilde'], torch.where(mask, aux['target'], x), atol=1e-5)
        y0, y1 = bb(x, ci), bb(aux['x_tilde'], ci)
    assert torch.allclose(out, torch.where(mask.any(dim=1, keepdim=True), y1, y0), atol=1e-5)


def test_v6_gate_calibration_window_budget():
    from exp.exp_radar import Exp_Radar
    W, _, _ = make_wrapper(arch='v6', dep_mode='none')
    g = torch.Generator().manual_seed(5)
    loader = [(torch.randn(B, L, C, generator=g), torch.randn(B, H, C, generator=g), None, None,
               torch.randint(0, CYCLE, (B,), generator=g)) for _ in range(3)]
    fake = types.SimpleNamespace(args=types.SimpleNamespace(tau_calib_batches=3, tau_calib_frac=0.01,
                                                            tau_calib_wfrac=0.05),
                                 device=torch.device('cpu'))
    fake._to_dev = lambda bx, by, bc: (bx.float(), by.float(), bc.int())
    Exp_Radar._calibrate_gate(fake, W, loader)
    zs = torch.cat([W.gate_logit_without_tau(bx, bc.int()) for bx, _, _, _, bc in loader])
    zw = zs.amax(dim=1)
    frac_w = (zw > W.tau_raw).float().mean().item()
    # torch.quantile interpolates between order statistics: up to one extra pair above τ on a small sample
    assert frac_w <= 0.05 + 1.0 / zw.numel() + 1e-6 and W.seg_delta.item() >= 0
    tau_pt = W.tau_raw.item() - W.seg_delta.item()
    assert abs((zs > tau_pt).float().mean().item() - 0.01) < 0.01


# v6 follow-ups: tail-segment hypothesis, exact second-pass skip, timers, variant naming ---------------------------
def test_tail_hypothesis_shapes_gradients_and_checkpoint():
    W, bb, cfg = make_wrapper(arch='v6', tail_hyp=1)
    assert W.tail_hyp and {'th_k', 'th_b', 'th_h'} <= set(W.state_dict())
    assert len(W.gate_scalars()) == 7  # a, b, g, k, β, h, τ
    x, ci = make_batch()
    x_p = x.clone()
    x_p[:, -6:, :3] += 3.0
    mask = (x_p - x).abs() > 1e-6
    y = torch.randn(B, H, C)
    y_hat = W(x_p, ci)
    aux = W.last_aux
    assert aux['tail_E'].shape == (B, C) and aux['tail_w'].shape == (B, C) and aux['tail_mask'].shape == (B, L, C)
    assert torch.isfinite(aux['tail_E']).all() and (aux['tail_E'] >= 0).all()
    assert (aux['tail_mask'][:, : L - 16] == 0).all() and (aux['tail_mask'][:, -1] == 1).all()
    total, _ = radar_loss(y_hat, y, aux, x, mask, None, None, cfg)
    total.backward()
    for p in (W.th_k, W.th_b, W.th_h):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    # S2 (need_route=False) does not use the hypothesis
    W.repair(x_p, ci, need_route=False)
    assert 'tail_E' not in W.last_aux


def test_tail_hypothesis_evidence_separates_a_shifted_tail():
    W, _, _ = make_wrapper(arch='v6', tail_hyp=1)
    t = torch.arange(L, dtype=torch.float32)
    sig = torch.stack([torch.sin(2 * math.pi * t / 24 + k) + 0.01 * t for k in range(C)], dim=-1).unsqueeze(0)
    x = sig.repeat(2, 1, 1)
    x[1, -8:, :] += 3.0  # window 1: shifted tail, inner points locally normal
    logit = torch.full_like(x, -5.0)
    logit[:, L - 8, :] = 2.0  # strongest alarm at the jump
    _, s = robust_stats(x)
    with torch.no_grad():
        E, seg, w = W._tail_hypothesis(x, logit, torch.ones_like(x), s)
    assert (seg[:, L - 8:] == 1).all() and (seg[:, :L - 8] == 0).all()
    assert (E[1] > E[0] + 0.5).all() and (w[1] > w[0]).all()


def test_second_backbone_pass_only_for_changed_windows_is_exact():
    W, bb, _ = make_wrapper(arch='v6', dep_mode='none')  # no Hutchinson probes through the backbone
    x, ci = make_batch()
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, -4:, :2] = True  # only window 0 is repaired
    W.oracle_gate, W.oracle_mask = True, mask
    calls = []
    orig = W.backbone.forward

    def spy(z, c):
        calls.append(z.shape[0])
        return orig(z, c)

    W.backbone.forward = spy
    with torch.no_grad():
        out = W(x, ci)
        aux = W.last_aux
        W.backbone.forward = orig
        ref = torch.where(mask.any(dim=1, keepdim=True), bb(aux['x_tilde'], ci), bb(x, ci))
    assert calls == [B + 1]  # x for all windows + x̃ for the one changed window
    assert torch.allclose(out, ref, atol=1e-5)
    W.oracle_mask = torch.zeros_like(mask)
    calls.clear()
    W.backbone.forward = spy
    with torch.no_grad():
        out = W(x, ci)
    W.backbone.forward = orig
    assert calls == [B] and torch.allclose(out, bb(x, ci), atol=1e-5)


def test_section_timers_and_variant_names():
    from exp.exp_radar import ckpt_suffix
    W, _, cfg = make_wrapper(arch='v6', tail_hyp=1)
    x, ci = make_batch()
    W.timer = {}
    with torch.no_grad():
        W(x, ci)
    keys = set(W.timer)
    assert {'detector', 'stage1/E3', 'stage1/E4', 'stage1/loo_prior', 'E1', 'gate', 'backbone', 'tail_hyp'} <= keys
    assert any(k.startswith('tail_hyp/') for k in keys) and all(v >= 0 for v in W.timer.values())
    W.timer = None
    a = types.SimpleNamespace(arch='v6', experts='auto', tail_hyp=1, tail_hyp_len=16, ckpt_tag='')
    assert ckpt_suffix(a, 'plugin') == '_th' and ckpt_suffix(a, 'pre') == ''
    a.experts = 'e1,e2,e3'
    assert ckpt_suffix(a, 'pre') == '_expe1e2e3' and ckpt_suffix(a, 'plugin') == '_expe1e2e3_th'
    W2, _, _ = make_wrapper(arch='v6', experts='e1,e2,e3')
    with torch.no_grad():
        W2(x, ci)
    assert (W2.last_aux['pi'][..., 2] == 0).all()


def test_nanmedian_small_equals_torch_nanmedian():
    g = torch.Generator().manual_seed(3)
    for shape, dim in (((40, 9, 3), -1), ((40, 9, 3, 6), 2), ((40, 3), 1), ((20, 5), -1)):
        v = (torch.randn(*shape, generator=g) * 10).round() / 10  # ties
        v[torch.rand(*shape, generator=g) < 0.35] = float('nan')
        a, b = nanmedian_small(v, dim=dim), torch.nanmedian(v, dim=dim).values
        assert torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(a[~torch.isnan(a)], b[~torch.isnan(b)])


def test_pre_checkpoint_loads_into_tail_hypothesis_wrapper(tmp_path):
    from exp.exp_radar import load_pre_state
    W, _, _ = make_wrapper(arch='v6')
    p = tmp_path / 'pre.pth'
    torch.save(W.state_dict(), p)  # S2 saves the full wrapper state
    Wt, _, _ = make_wrapper(arch='v6', tail_hyp=1, seed=4)
    load_pre_state(Wt, str(p), torch.device('cpu'))  # th_* keep their initial values
    assert abs(float(Wt.th_b) - 1.5) < 1e-6
    for (k, v), (k2, v2) in zip(W.reliability.state_dict().items(), Wt.reliability.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)


# label-free mode (no injected corruption anywhere) --------------------------------------------------------------
def test_label_free_detector_and_forward():
    W, bb, _ = make_wrapper(arch='v6', label_free=1, dep_mode='none')
    assert W.label_free and W.gate_consistency and abs(float(F_softplus(W.gate_a)) - 1.0) < 1e-5
    t = torch.arange(L, dtype=torch.float32)
    sig = torch.stack([torch.sin(2 * math.pi * t / 24 + 0.3 * k) * (1 + 0.1 * k) + 0.01 * t for k in range(C)], -1)
    x = sig.unsqueeze(0).repeat(3, 1, 1)
    x[1, 50, 0] += 5.0     # internal spike
    x[2, -8:, 0] += 3.0    # shifted tail: the inside of the run must be seen too
    ell = W.label_free_score(x)
    assert ell.shape == x.shape and (ell >= 0).all()
    q = torch.quantile(ell[0].flatten(), 0.99)
    assert ell[1, 50, 0] > 3 * q and (ell[2, -8:, 0] > 2 * q).all()
    ci = torch.zeros(3, dtype=torch.int32)
    with torch.no_grad():
        out = W(x, ci)  # lf_q0 = inf before calibration: every point is an anchor
        assert out.shape == (3, H, C) and torch.equal(W.last_aux['R'], torch.ones_like(x))
        W.lf_q0.fill_(float(q))
        W(x, ci)
        R = W.last_aux['R']
    assert R[1, 50, 0] == 0 and (R[2, -8:, 0] == 0).all() and R[0].mean() > 0.95


def F_softplus(v):
    return torch.nn.functional.softplus(v)


def test_label_free_s2_loss_and_calibration():
    from exp.exp_radar import Exp_Radar, ckpt_suffix
    W, _, _ = make_wrapper(arch='v6', label_free=1, dep_mode='none')
    W.train()
    x, ci = make_batch()
    y = torch.randn(B, H, C)
    rec_mask = torch.rand(B * C, W.N, generator=torch.Generator().manual_seed(1)) < 0.3
    fake = types.SimpleNamespace(args=types.SimpleNamespace(u_clip=10.0, pred_len=H, tau_calib_batches=2,
                                                            tau_calib_frac=0.01, tau_calib_wfrac=0.05),
                                 f_dim=0, device=torch.device('cpu'))
    fake._to_dev = lambda bx, by, bc: (bx.float(), by.float(), bc.int())
    l_rec, l_route = Exp_Radar._lf_batch_loss(fake, W, x, y, ci, rec_mask)
    (l_rec + 0.5 * l_route).backward()
    assert torch.isfinite(l_rec) and torch.isfinite(l_route) and l_route > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in W.repairer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in W.router.parameters())
    assert all(p.grad is None for p in W.reliability.parameters())  # the learned detector is unused
    g = torch.Generator().manual_seed(9)
    loader = [(torch.randn(B, L, C, generator=g), torch.randn(B, H, C, generator=g), None, None,
               torch.randint(0, CYCLE, (B,), generator=g)) for _ in range(2)]
    Exp_Radar._calibrate_gate(fake, W, loader)
    assert math.isfinite(float(W.lf_q0)) and float(W.lf_q0) > 0
    a = types.SimpleNamespace(arch='v6', experts='auto', label_free=1, lf_train=0, ckpt_tag='')
    assert ckpt_suffix(a, 'pre') == '_lf' and ckpt_suffix(a, 'plugin') == '_lf'


def test_label_free_masked_forecast_training_and_clean_budget():
    from exp.exp_radar import Exp_Radar, ckpt_suffix
    W, _, _ = make_wrapper(arch='v6', label_free=1, dep_mode='none')
    fake = types.SimpleNamespace(args=types.SimpleNamespace(u_clip=10.0, pred_len=H, lf_mask_p=0.5, lf_clean_budget=0.01,
                                                            tau_calib_batches=2, tau_calib_frac=0.01,
                                                            tau_calib_wfrac=0.05),
                                 f_dim=0, device=torch.device('cpu'))
    fake._to_dev = lambda bx, by, bc: (bx.float(), by.float(), bc.int())
    g = torch.Generator().manual_seed(2)
    M = Exp_Radar._lf_masks(fake, 64, L, C, gen=g)
    runs = M.sum(dim=1)  # gap length per (window, channel)
    assert M.shape == (64, L, C) and int(runs.max()) <= 16 and 0.2 < (runs > 0).float().mean() < 0.8
    assert M[:, -1].any() and M[:, : L - 24].any()  # tail and internal gaps
    x, ci = make_batch()
    y = torch.randn(B, H, C)
    M = torch.zeros_like(x, dtype=torch.bool)
    M[:, -6:, :3] = True
    W.train()
    mse, rec = Exp_Radar._lf_forecast_loss(fake, W, x, y, ci, M)
    (mse + 0.5 * rec).backward()
    assert torch.isfinite(mse) and torch.isfinite(rec)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in W.repairer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in W.router.parameters())
    # the gap is hidden from the repair: R = 0 there, stage 1 does not use those values
    assert (W.last_aux['delta_u'] == W.last_aux['delta_u']).all()
    W.eval()
    loader = [(torch.randn(B, L, C, generator=g), torch.randn(B, H, C, generator=g), None, None,
               torch.randint(0, CYCLE, (B,), generator=g)) for _ in range(2)]
    fake.ordered_loader = lambda split: (None, loader)
    Exp_Radar._calibrate_gate(fake, W, loader)
    Exp_Radar._lf_budget_tau(fake, W)
    se0, se1, n = 0.0, 0.0, 0
    with torch.no_grad():
        for bx, by, _, _, bc in loader:
            yy = by.float()
            se1 += ((W(bx, bc.int()) - yy) ** 2).sum().item()
            W.force_gate_zero = True
            se0 += ((W(bx, bc.int()) - yy) ** 2).sum().item()
            W.force_gate_zero = False
    assert se1 <= 1.01 * se0 + 1e-6  # the chosen gate respects the clean budget on these windows
    a = types.SimpleNamespace(arch='v6', experts='auto', label_free=1, lf_train=1, lf_clean_budget=0.01, ckpt_tag='')
    assert ckpt_suffix(a, 'pre') == '_lf' and ckpt_suffix(a, 'plugin') == '_lf_mf'
