"""RADAR losses.

Stage A (pretrain_radar): L_det + L_rec                       (ReliabilityNet, Repairer)
Stage B (plug-in)       : MSE(ŷ, y) + λ_hind·L_hind           (Router + gate a, b, τ)
Joint / plug-in 'all'   : MSE + λ_det·L_det + λ_rec·L_rec + λ_hind·L_hind + λ_α·mean(α)   (0.5, 0.5, 0.5, 0.01)
All S3 variants (v4)    : + λ_route·L_route (0.5), gradient w.r.t. the Router only (see route_loss)

L_det is a PU (positive-unlabeled) loss: injected points are positives; natural points are *unlabeled* and only
get a small negative weight (pu_neg_weight, default 0.1), so genuinely bad natural points can still be flagged.
L_rec targets live in the corrupted window's robust coordinates (same frame as the Repairer input/output).
"""
import torch
import torch.nn.functional as F

from radar.modules import points_to_patch_max, robust_stats


def det_loss(logit, mask, pos_weight=1.0, neg_weight=0.1):
    """PU detection loss: BCE(ℓ_t, m_t) with weight pos_weight on injected points (m_t = 1) and neg_weight on
    natural (unlabeled) points; averaged over all points."""
    t = mask.to(logit.dtype)
    w = t * pos_weight + (1.0 - t) * neg_weight
    return (w * F.binary_cross_entropy_with_logits(logit, t, reduction='none')).mean()


def clean_u(x_clean, u_clip=10.0, m=None, s=None):
    """Repairer target: clean values in the robust frame (m, s) of the window the Repairer actually saw
    (the corrupted one). Without (m, s) the clean window's own statistics are used."""
    if m is None or s is None:
        m, s = robust_stats(x_clean)
    return ((x_clean - m) / s).clamp(-u_clip, u_clip)


def rec_loss(u_hat, x_clean, select, u_clip=10.0, m=None, s=None):
    """MSE(û_t, (x_clean_t − m_corrupt)/s_corrupt) on the selected points (perturbed ∪ randomly masked)."""
    target = clean_u(x_clean, u_clip, m, s)
    sel = select.to(u_hat.dtype)
    return ((u_hat - target) ** 2 * sel).sum() / sel.sum().clamp_min(1.0)


def patch_mask_to_points(rec_mask, B, C, L, P):
    """bool (B*C, N) -> bool (B, L, C) (left padding cropped)."""
    N = rec_mask.shape[1]
    pad = N * P - L
    z = rec_mask.view(B, C, N, 1).expand(B, C, N, P).reshape(B, C, N * P)[:, :, pad:]
    return z.permute(0, 2, 1)


@torch.no_grad()
def hindsight_targets(backbone, x_in, cycle_index, y, repair, P, pred_len, f_dim=0, level='sample',
                      generator=None, forward_fn=None):
    """Counterfactual supervision for the gate.

    For every sample: the last patch and one random patch (≠ last). Under no_grad + backbone.eval():
      e0 = per-sample MSE(f(x_in), y);  e1 = same after replacing that patch (all channels) by the detached repair.
      h  = 1[e1 < 0.98 e0];  w = |e0 - e1| / (batch mean + 1e-8), clipped to [0, 5].
    level='sample' follows the spec (one label per sample/patch, shared by its channels);
    level='channel' keeps per-channel errors (exact for channel-independent backbones such as GTR).
    Returns dict(patches (B, 2), h (B, 2, C'), w (B, 2, C')).
    """
    if forward_fn is None:
        def forward_fn(z):
            out = backbone(z, cycle_index)
            return out[0] if isinstance(out, (tuple, list)) else out
    was_training = backbone.training
    backbone.eval()
    try:
        B, L, C = x_in.shape
        N = -(-L // P)
        pad = N * P - L

        def err(out):
            e = (out[:, -pred_len:, f_dim:] - y) ** 2  # (B, H, C')
            e = e.mean(dim=1)  # (B, C')
            if level == 'sample':
                e = e.mean(dim=1, keepdim=True)  # (B, 1)
            return e

        e0 = err(forward_fn(x_in))
        p_last = torch.full((B,), N - 1, dtype=torch.long)
        p_rand = torch.randint(0, max(N - 1, 1), (B,), generator=generator)
        patches = torch.stack([p_last, p_rand], dim=1).to(x_in.device)  # (B, 2)

        point_patch = (torch.arange(L, device=x_in.device) + pad) // P  # (L,)
        e1s = []
        for j in range(2):
            sel = (point_patch.view(1, L) == patches[:, j].view(B, 1)).unsqueeze(-1)  # (B, L, 1)
            x_rep = torch.where(sel, repair, x_in)
            e1s.append(err(forward_fn(x_rep)))
        e1 = torch.stack(e1s, dim=1)  # (B, 2, C' or 1)
        e0 = e0.unsqueeze(1).expand_as(e1)
        h = (e1 < 0.98 * e0).float()
        diff = (e0 - e1).abs()
        w = (diff / (diff.mean() + 1e-8)).clamp(0.0, 5.0)
    finally:
        backbone.train(was_training)
    return dict(patches=patches, h=h, w=w)


def hindsight_loss(alpha, targets, P, f_dim=0):
    """L_hind = mean(w · BCE(a_p, h)), a_p = max_{t∈p} α_t (per channel)."""
    a_patch = points_to_patch_max(alpha, P)  # (B, N, C)
    if f_dim != 0:
        a_patch = a_patch[:, :, f_dim:]
    B, N, C = a_patch.shape
    idx = targets['patches'].view(B, 2, 1).expand(B, 2, C)
    a_p = torch.gather(a_patch, 1, idx).clamp(1e-6, 1 - 1e-6)  # (B, 2, C)
    h = targets['h'].expand(B, 2, C)
    w = targets['w'].expand(B, 2, C)
    bce = F.binary_cross_entropy(a_p, h, reduction='none')
    return (w * bce).mean()


def route_loss(W, aux, x_in, pert_mask, y, bb_args, pred_len, f_dim=0):
    """L_route = MSE(f(x + m ⊙ (Σ_k π_k · sg(c_k) − x)), y) on the perturbed samples of the batch.

    The repair location is given by the perturbation mask m (not by the gate α), so the Router receives a forecast
    signal wherever a corruption is; the repair is recomputed by W.route_candidate from detached inputs (v5: stage-1
    mix of E2 / E3 plus the detached E1 correction), so only the Router depends on it. Callers take gradients w.r.t. the Router parameters only (see Exp_Radar.train_radar).
    Returns None when the batch holds no perturbed sample."""
    sel = pert_mask.flatten(1).any(dim=1)  # (B,)
    if not bool(sel.any()):
        return None
    c = W.route_candidate(aux)  # routed repair; only the Router carries gradients
    m = pert_mask.to(x_in.dtype)
    x_r = (x_in + m * (c - x_in))[sel]
    args = [a[sel] if torch.is_tensor(a) and a.shape[:1] == sel.shape else a for a in bb_args]
    out = W.backbone(x_r, *args)
    out = out[0] if isinstance(out, (tuple, list)) else out
    return F.mse_loss(out[:, -pred_len:, f_dim:], y[sel])


def radar_loss(y_hat, y, aux, x_clean, pert_mask, rec_mask_pts=None, hind=None, cfg=None, stage='joint'):
    """stage='joint': full loss; stage='B': MSE + λ_hind·L_hind only (stage-A modules frozen)."""
    lam_det = getattr(cfg, 'lam_det', 0.5)
    lam_rec = getattr(cfg, 'lam_rec', 0.5)
    lam_hind = getattr(cfg, 'lam_hind', 0.5)
    lam_alpha = getattr(cfg, 'lam_alpha', 0.01)
    pos_weight = getattr(cfg, 'pos_weight', 1.0)
    neg_weight = getattr(cfg, 'pu_neg_weight', 0.1)
    u_clip = getattr(cfg, 'u_clip', 10.0)

    mse = F.mse_loss(y_hat, y)
    if stage == 'B':
        l_hind = hind if hind is not None else torch.zeros((), device=y_hat.device)
        total = mse + lam_hind * l_hind
        return total, dict(mse=mse.item(), det=0.0, rec=0.0, hind=float(l_hind.detach()),
                           alpha=aux['alpha'].mean().item())
    l_det = det_loss(aux['logit'], pert_mask, pos_weight, neg_weight)
    select = pert_mask if rec_mask_pts is None else (pert_mask | rec_mask_pts)
    l_rec = rec_loss(aux['u_hat'], x_clean, select, u_clip, aux['m'], aux['s'])
    l_hind = hind if hind is not None else torch.zeros((), device=y_hat.device)
    l_alpha = aux['alpha'].mean()
    total = mse + lam_det * l_det + lam_rec * l_rec + lam_hind * l_hind + lam_alpha * l_alpha
    parts = dict(mse=mse.item(), det=l_det.item(), rec=l_rec.item(), hind=float(l_hind.detach()),
                 alpha=l_alpha.item())
    return total, parts
