"""RADAR unit tests (B=4, L=96, C=7, H=96, cycle=24, randomly initialised GTR in eval mode).

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
from radar.dependency import hutchinson_dependency, patch_dependency  # noqa: E402
from radar.losses import hindsight_loss, hindsight_targets, patch_mask_to_points, radar_loss  # noqa: E402
from radar.modules import (CyclePriorExpert, LocalMedianExpert, from_patches, gtr_gather_index,  # noqa: E402
                           hampel_filter, to_patches)
from radar.perturb import MixturePerturber, apply_test_perturbation, perturb_batch, raw_zero_from_scaler  # noqa: E402
from radar.radar import RADARWrapper  # noqa: E402

B, L, C, H, CYCLE = 4, 96, 7, 96, 24


def make_cfg(**kw):
    cfg = types.SimpleNamespace(
        seq_len=L, pred_len=H, enc_in=C, cycle=CYCLE, d_model=512, dropout=0.5, use_revin=1, individual=0,
        radar_patch=8, radar_d=64, radar_heads=4, radar_layers=2, radar_dff=128, radar_dropout=0.1,
        radar_chunk=8192, u_clip=10.0, tau_init=0.3, gate_T=0.05, dep_mode='grad', dep_probes=1, dep_ema=0.99,
        use_cycle_prior=1, force_gate_zero=False,
        lam_det=0.5, lam_rec=0.5, lam_hind=0.1, lam_alpha=0.01, pos_weight=10.0)
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
    for k in ('logit', 'R', 'alpha', 'dbar', 'u_hat', 'x_tilde'):
        assert aux[k].shape == (B, L, C), k
    assert aux['pi'].shape == (B, L, C, 3)
    assert aux['dbar_patch'].shape == (B, C, L // 8)


# 2 ------------------------------------------------------------------------------------------------------
def test_force_gate_zero_is_identity_and_init_alpha_small():
    W, bb, _ = make_wrapper(force_gate_zero=True)
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


def test_hutchinson_matches_exact_jacobian_column_energy():
    toy = LinearToy()
    x, ci = make_batch()
    g = torch.Generator().manual_seed(123)
    D = hutchinson_dependency(toy, x, ci, H, n_probes=256, generator=g)
    exact = (toy.W.detach() ** 2).sum(0).reshape(L, C).unsqueeze(0).expand(B, L, C)
    rel = ((D - exact).norm() / exact.norm()).item()
    assert rel < 0.10, rel  # expected ≈ sqrt(2/256) ≈ 0.088 for Gaussian probes
    assert toy.W.grad is None
    dbar, Dp = patch_dependency(D, 8)
    assert dbar.shape == (B, C, L // 8)
    assert torch.allclose(dbar.amax(-1), torch.ones(B, C))
    assert (dbar >= 0).all() and (dbar <= 1 + 1e-6).all()


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

    xs, _ = apply_test_perturbation(x, 'T1_scaled', rz)
    assert torch.equal(xs[:, :-1], x[:, :-1]) and torch.count_nonzero(xs[:, -1]) == 0
    xf, mf = apply_test_perturbation(x, 'T3_flat', rz)
    assert torch.equal(xf[:, :-8], x[:, :-8]) and torch.equal(xf[:, -8:], x[:, -9:-8].expand(B, 8, C))
    assert 'flatline' not in MixturePerturber.TYPES

    pert = MixturePerturber(rz, p_apply=1.0, seed=0)
    xm, mm, types_ = pert(x)
    assert xm.shape == x.shape and mm.shape == x.shape and (types_ >= 0).all()
    assert torch.equal(mm, (xm - x).abs() > 1e-6)


# 5 ------------------------------------------------------------------------------------------------------
def test_cycle_prior_zero_Q_and_gather_index():
    _, bb, _ = make_wrapper()
    x, ci = make_batch()
    w = torch.rand(B, L, C, generator=torch.Generator().manual_seed(5))
    expert = CyclePriorExpert()

    assert torch.count_nonzero(bb.Q) == 0  # GTR initialises Q with zeros
    c3, _ = expert(x, ci, bb.Q, w)
    wmean = (w * x).sum(1, keepdim=True) / w.sum(1, keepdim=True)
    assert torch.isfinite(c3).all()
    assert torch.allclose(c3, wmean.expand_as(x), atol=1e-5)
    c0, _ = expert(x, ci, bb.Q, torch.zeros_like(w))  # degenerate weights fall back to uniform
    assert torch.isfinite(c0).all()

    with torch.no_grad():
        bb.Q.copy_(torch.randn_like(bb.Q))
    captured = {}

    def grab(module, inputs, output):  # returns None -> output untouched
        captured['q'] = inputs[1].detach().clone()

    hook = bb.GTR.register_forward_hook(grab)
    with torch.no_grad():
        bb(x, ci)
    hook.remove()
    c3, q = expert(x, ci, bb.Q, w)
    assert torch.equal(q, captured['q'].permute(0, 2, 1))  # same Q rows GTR itself retrieved
    ref = (ci.view(-1, 1) + torch.arange(L).view(1, -1)) % CYCLE  # formula of models/GTR.py
    assert torch.equal(gtr_gather_index(ci, L, CYCLE), ref)
    assert torch.isfinite(c3).all()


# 6 ------------------------------------------------------------------------------------------------------
def test_gradients_reach_radar_modules_but_not_Q_through_cycle_prior():
    W, bb, cfg = make_wrapper()
    x, ci = make_batch()
    y = torch.randn(B, H, C)
    x_p = x.clone()
    x_p[:, -1, :] = -3.0
    mask = torch.zeros(B, L, C, dtype=torch.bool)
    mask[:, -1, :] = True
    rec_mask = torch.rand(B * C, W.N, generator=torch.Generator().manual_seed(0)) < 0.1

    y_hat = W(x_p, ci, rec_mask=rec_mask)
    aux = W.last_aux
    tg = hindsight_targets(W.backbone, x_p, ci, y, aux['target'].detach(), W.P, H)
    hind = hindsight_loss(aux['alpha'], tg, W.P)
    total, _ = radar_loss(y_hat, y, aux, x, mask, patch_mask_to_points(rec_mask, B, C, L, W.P), hind, cfg)
    W.zero_grad(set_to_none=True)
    total.backward()

    for name, mod in (('ReliabilityNet', W.reliability), ('Repairer', W.repairer), ('Router', W.router)):
        for n, p in mod.named_parameters():
            assert p.grad is not None, f'{name}.{n} has no grad'
            assert torch.isfinite(p.grad).all(), f'{name}.{n} grad not finite'
    assert W.tau_raw.grad is not None and torch.isfinite(W.tau_raw.grad).all()

    # CyclePrior uses Q.detach(): no gradient path from its output to Q (x made differentiable on purpose)
    x_req = x_p.clone().requires_grad_(True)
    c3b, _ = W.cycle_prior(x_req, ci, bb.Q, aux['R'])
    assert c3b.requires_grad
    assert torch.autograd.grad(c3b.sum(), bb.Q, allow_unused=True)[0] is None


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


# extra sanity checks --------------------------------------------------------------------------------------
def test_patch_roundtrip_and_local_median():
    x, _ = make_batch()
    xp, pad = to_patches(x, 8)
    assert pad == 0 and xp.shape == (B * C, L // 8, 8)
    assert torch.equal(from_patches(xp, B, C, pad), x)
    x2 = x[:, :93]  # L % P != 0 -> left replicate padding, cropped on output
    xp2, pad2 = to_patches(x2, 8)
    assert pad2 == 3 and torch.equal(from_patches(xp2, B, C, pad2), x2)

    lm = LocalMedianExpert(3)(x)
    t = 50
    nb = torch.cat([x[:, t - 3:t], x[:, t + 1:t + 4]], dim=1)
    assert torch.equal(lm[:, t], nb.median(dim=1).values)
    assert torch.equal(lm[:, -1], x[:, -4:-1].median(dim=1).values)  # last point uses t-3..t-1 only
    hf = hampel_filter(x)
    assert hf.shape == x.shape and torch.isfinite(hf).all()


def test_ema_mode_and_training_forward():
    W, bb, cfg = make_wrapper(dep_mode='ema')
    x, ci = make_batch()
    W.train()
    y = W(x, ci)
    assert bool(W.dep_profile.initialized)
    W.eval()
    with torch.no_grad():
        y2 = W(x, ci)
    assert y.shape == y2.shape == (B, H, C)
    prof = W.dep_profile.profile()
    assert torch.allclose(W.last_aux['dbar_patch'][0, 0], prof)


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
    assert W.use_cycle_prior
