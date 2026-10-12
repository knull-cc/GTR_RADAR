"""RADAR losses.

L = MSE(ŷ, y) + λ_det·L_det + λ_rec·L_rec + λ_hind·L_hind + λ_α·mean(α)
    (defaults 0.5, 0.5, 0.1, 0.01)
"""
import torch
import torch.nn.functional as F

from radar.modules import points_to_patch_max, robust_stats


def det_loss(logit, mask, pos_weight=10.0):
    """BCE(ℓ_t, m_t) over all points; m_t = 1 for perturbed points."""
    pw = torch.tensor(pos_weight, device=logit.device, dtype=logit.dtype)
    return F.binary_cross_entropy_with_logits(logit, mask.to(logit.dtype), pos_weight=pw)


def clean_u(x_clean, u_clip=10.0):
    """Clean-window robust coordinates (target of the Repairer)."""
    m, s = robust_stats(x_clean)
    return ((x_clean - m) / s).clamp(-u_clip, u_clip)


def rec_loss(u_hat, x_clean, select, u_clip=10.0):
    """MSE(û_t, u_t^clean) on the selected points (perturbed ∪ randomly masked)."""
    target = clean_u(x_clean, u_clip)
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
                      generator=None):
    """Counterfactual supervision for the gate.

    For every sample: the last patch and one random patch (≠ last). Under no_grad + backbone.eval():
      e0 = per-sample MSE(f(x_in), y);  e1 = same after replacing that patch (all channels) by the detached repair.
      h  = 1[e1 < 0.98 e0];  w = |e0 - e1| / (batch mean + 1e-8), clipped to [0, 5].
    level='sample' follows the spec (one label per sample/patch, shared by its channels);
    level='channel' keeps per-channel errors (exact for channel-independent backbones such as GTR).
    Returns dict(patches (B, 2), h (B, 2, C'), w (B, 2, C')).
    """
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

        e0 = err(backbone(x_in, cycle_index))
        p_last = torch.full((B,), N - 1, dtype=torch.long)
        p_rand = torch.randint(0, max(N - 1, 1), (B,), generator=generator)
        patches = torch.stack([p_last, p_rand], dim=1).to(x_in.device)  # (B, 2)

        point_patch = (torch.arange(L, device=x_in.device) + pad) // P  # (L,)
        e1s = []
        for j in range(2):
            sel = (point_patch.view(1, L) == patches[:, j].view(B, 1)).unsqueeze(-1)  # (B, L, 1)
            x_rep = torch.where(sel, repair, x_in)
            e1s.append(err(backbone(x_rep, cycle_index)))
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


def radar_loss(y_hat, y, aux, x_clean, pert_mask, rec_mask_pts=None, hind=None, cfg=None):
    lam_det = getattr(cfg, 'lam_det', 0.5)
    lam_rec = getattr(cfg, 'lam_rec', 0.5)
    lam_hind = getattr(cfg, 'lam_hind', 0.1)
    lam_alpha = getattr(cfg, 'lam_alpha', 0.01)
    pos_weight = getattr(cfg, 'pos_weight', 10.0)
    u_clip = getattr(cfg, 'u_clip', 10.0)

    mse = F.mse_loss(y_hat, y)
    l_det = det_loss(aux['logit'], pert_mask, pos_weight)
    select = pert_mask if rec_mask_pts is None else (pert_mask | rec_mask_pts)
    l_rec = rec_loss(aux['u_hat'], x_clean, select, u_clip)
    l_hind = hind if hind is not None else torch.zeros((), device=y_hat.device)
    l_alpha = aux['alpha'].mean()
    total = mse + lam_det * l_det + lam_rec * l_rec + lam_hind * l_hind + lam_alpha * l_alpha
    parts = dict(mse=mse.item(), det=l_det.item(), rec=l_rec.item(), hind=float(l_hind), alpha=l_alpha.item())
    return total, parts
