"""RADAR v2 unit tests (B=4, L=96, C=7, H=96, cycle=24; randomly initialised GTR in eval mode).

Run from the repository root:  python -m pytest -q tests/test_radar.py
"""
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
                          radar_loss, rec_loss)
from radar.modules import ReliableNeighborExpert, from_patches, hampel_filter, robust_stats, to_patches  # noqa: E402
from radar.perturb import MixturePerturber, apply_test_perturbation, perturb_batch, raw_zero_from_scaler  # noqa: E402
from radar.radar import RADARWrapper  # noqa: E402

B, L, C, H, CYCLE = 4, 96, 7, 96, 24


def make_cfg(**kw):
    cfg = types.SimpleNamespace(
        seq_len=L, pred_len=H, enc_in=C, cycle=CYCLE, d_model=512, dropout=0.5, use_revin=1, individual=0,
        radar_patch=8, radar_d=64, radar_heads=4, radar_layers=2, radar_dff=128, radar_dropout=0.1,
        radar_chunk=8192, u_clip=10.0, tau_init=1.0, gate_eps=1e-2, dep_mode='grad', dep_probes=4,
        dep_probe_dist='rademacher', dep_ema=0.99, force_gate_zero=False, e2_k=3, e2_thr=0.5, e2_trend=1,
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
def test_output_shape_matches_backbone():
    W, bb, _ = make_wrapper()
    x, ci = make_batch()
    with torch.no_grad():
        y_bb = bb(x, ci)
        y = W(x, ci)
    assert y.shape == y_bb.shape == (B, H, C)
    aux = W.last_aux
    for k in ('logit', 'R', 'alpha', 'dbar', 'u_hat', 'x_tilde', 'target'):
        assert aux[k].shape == (B, L, C), k
    assert aux['pi'].shape == (B, L, C, 2) and aux['c'].shape == (B, L, C, 2)
    assert aux['dbar_patch'].shape == (B, C, L // 8)


# 2 ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('dep_mode', ['grad', 'ema'])
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
    assert rel < 0.10, rel  # Gaussian ≈ sqrt(2/256) ≈ 0.088; Rademacher is lower
    assert toy.W.grad is None
    Dp = patch_sum(D, 8)
    assert Dp.shape == (B, C, L // 8) and torch.allclose(Dp.sum(-1), D.sum(1), rtol=1e-4)


# 4 ------------------------------------------------------------------------------------------------------
def test_T1_changes_only_last_point_to_raw_zero_and_not_y():
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
    assert torch.equal(xp[:, -1, :], torch.as_tensor(rz).unsqueeze(0).expand(B, C))
    assert yp is y and torch.equal(y, y_copy)
    assert not mask[:, :-1, :].any() and mask[:, -1, :].all()

    xk, mk = apply_test_perturbation(x, 'T2_k4', rz)
    assert torch.equal(xk[:, :-4], x[:, :-4]) and mk[:, -4:].all()
    assert 'flatline' not in MixturePerturber.TYPES
    pert = MixturePerturber(rz, p_apply=1.0, seed=0)
    xm, mm, types_ = pert(x)
    assert xm.shape == x.shape and (types_ >= 0).all()
    assert torch.equal(mm, (xm - x).abs() > 1e-6)


# 5 ------------------------------------------------------------------------------------------------------
def test_reliable_neighbor_expert_and_scale_floor():
    E2 = ReliableNeighborExpert(k=3, thr=0.5, trend=True)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 20, 1, generator=g)
    x[:, -4:] = -5.0  # contiguous corruption of the last 4 points (T2_k4-like)
    R = torch.ones_like(x)
    R[:, -4:] = 0.0
    c = E2(x, R)[0, :, 0]
    z = x[0, :, 0]
    # interior point: median of the 3 reliable points before and the 3 after (corrupted ones skipped)
    t = 10
    assert torch.allclose(c[t], torch.cat([z[t - 3:t], z[t + 1:t + 4]]).median())
    # last point: nothing reliable after -> level (median) + OLS trend of the last 3 reliable points (13..15)
    pts = z[13:16]
    tt = torch.tensor([13.0, 14.0, 15.0])
    slope = ((tt - tt.mean()) * (pts - pts.mean())).sum() / ((tt - tt.mean()) ** 2).sum()
    assert torch.allclose(c[19], pts.median() + slope * (19 - tt.mean()), atol=1e-5)
    assert not torch.isclose(c[19], torch.tensor(-5.0))  # a plain local median would return the zeroed value
    # no reliable point at all -> window median, finite
    c0 = E2(x, torch.zeros_like(x))
    assert torch.isfinite(c0).all() and torch.allclose(c0[0, :, 0], z.median().expand(20))

    flat = torch.ones(2, 96, 3)
    m, s = robust_stats(flat)
    assert torch.allclose(s, torch.full_like(s, 1e-2)) and torch.isfinite((flat - m) / s).all()


# 6 ------------------------------------------------------------------------------------------------------
def test_gradients_joint_and_stage_b():
    W, bb, cfg = make_wrapper()
    x, ci = make_batch()
    y = torch.randn(B, H, C)
    x_p = x.clone()
    x_p[:, -1, :] = -3.0
    mask = torch.zeros(B, L, C, dtype=torch.bool)
    mask[:, -1, :] = True
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
    for p in (W.gate_a, W.gate_b, W.tau_raw):
        assert p.grad is not None and torch.isfinite(p.grad).all()

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
    u_hat = x_clean / 2.0  # perfect reconstruction in the corrupted window's frame (m, s)
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
