"""RADAR building blocks.

Shapes: x is (B, L, C) in StandardScaler space. Inside RADAR every channel is processed independently,
so per-point series are reshaped to (B*C, N, P) patches (P points per patch, N patches).
"""
import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Robust scaling
# --------------------------------------------------------------------------------------
def robust_stats(x, eps=1e-5, std_frac=0.1, s_min=1e-2):
    """Per-sample, per-channel median and a floored robust scale. x: (B, L, C) -> m, s: (B, 1, C).

    s = max(1.4826*MAD, std_frac*std_window, s_min): the floor keeps u bounded on (near-)constant windows
    (Weather rain channels, Solar nights) where MAD = 0.
    """
    m = x.median(dim=1, keepdim=True).values
    mad = (x - m).abs().median(dim=1, keepdim=True).values
    sd = x.std(dim=1, keepdim=True, unbiased=False)
    s = torch.maximum(1.4826 * mad, std_frac * sd).clamp_min(s_min)
    return m, s


class RobustScaler(nn.Module):
    """u = (x - median) / max(1.4826*MAD, 0.1*std, 1e-2), statistics over the time axis of each sample/channel."""

    def __init__(self, std_frac=0.1, s_min=1e-2):
        super().__init__()
        self.std_frac, self.s_min = std_frac, s_min

    def forward(self, x):
        m, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        return (x - m) / s, m, s

    @staticmethod
    def inverse(u, m, s):
        return u * s + m


# --------------------------------------------------------------------------------------
# Patching helpers (left replicate padding when L is not a multiple of P; cropped on output)
# --------------------------------------------------------------------------------------
def to_patches(z, P):
    """z: (B, L, C) -> (B*C, N, P), pad. Left-pads by replicating the first value."""
    B, L, C = z.shape
    pad = (-L) % P
    zc = z.permute(0, 2, 1).reshape(B * C, L)
    if pad:
        zc = torch.cat([zc[:, :1].expand(-1, pad), zc], dim=1)
    return zc.reshape(B * C, -1, P), pad


def from_patches(zp, B, C, pad):
    """(B*C, N, P[, K]) -> (B, L, C[, K]) with the left padding cropped."""
    tail = tuple(zp.shape[3:])
    z = zp.reshape(B, C, -1, *tail)[:, :, pad:]
    if tail:
        return z.permute(0, 2, 1, *range(3, 3 + len(tail)))
    return z.permute(0, 2, 1)


def patch_to_points(zp, P, pad):
    """Broadcast a per-patch quantity (B, C, N) to points (B, L, C)."""
    B, C, N = zp.shape
    z = zp.unsqueeze(-1).expand(B, C, N, P).reshape(B, C, N * P)[:, :, pad:]
    return z.permute(0, 2, 1)


def points_to_patch_max(z, P, fill=0.0):
    """(B, L, C) -> (B, N, C): max over the points of each patch (left padding filled with `fill`)."""
    B, L, C = z.shape
    pad = (-L) % P
    if pad:
        z = torch.cat([z.new_full((B, pad, C), fill), z], dim=1)
    return z.reshape(B, -1, P, C).amax(dim=2)


def rolling_median(z, k=5):
    """Centered rolling median along the last axis of z (M, T); border values replicated."""
    h = k // 2
    zp = torch.cat([z[:, :1].expand(-1, h), z, z[:, -1:].expand(-1, h)], dim=1)
    return zp.unfold(1, k, 1).median(dim=-1).values


def point_features(u):
    """u: (M, T) robust-scaled series -> (M, T, 3) = [u_t, Δu_t (0 at t=0), |u_t - median5_t|]."""
    du = torch.cat([torch.zeros_like(u[:, :1]), u[:, 1:] - u[:, :-1]], dim=1)
    dev = (u - rolling_median(u, 5)).abs()
    return torch.stack([u, du, dev], dim=-1)


# --------------------------------------------------------------------------------------
# Patch transformer used by ReliabilityNet and Repairer
# --------------------------------------------------------------------------------------
class PatchTransformer(nn.Module):
    def __init__(self, in_dim, n_patches, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, tokens):  # (M, N, in_dim) -> (M, N, d_model)
        return self.encoder(self.proj(tokens) + self.pos)


class ReliabilityNet(nn.Module):
    """Point-level unreliability logits ℓ_t; R_t = 1 - sigmoid(ℓ_t)."""

    def __init__(self, P, N, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, head_bias=-4.0):
        super().__init__()
        self.P = P
        self.body = PatchTransformer(3 * P, N, d_model, n_heads, n_layers, d_ff, dropout)
        self.head = nn.Linear(d_model, P)
        nn.init.constant_(self.head.bias, head_bias)

    def forward(self, feats):  # feats: (M, N, 3P) -> logits (M, N, P)
        return self.head(self.body(feats))


class Repairer(nn.Module):
    """Context reconstruction expert. Input per point [R_t*u_t, 1-R_t]; outputs û (M, N, P) and hidden (M, N, d)."""

    def __init__(self, P, N, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.P = P
        self.body = PatchTransformer(2 * P, N, d_model, n_heads, n_layers, d_ff, dropout)
        self.head = nn.Linear(d_model, P)

    def forward(self, tokens):  # (M, N, 2P)
        h = self.body(tokens)
        return self.head(h), h


class Router(nn.Module):
    """Per-point softmax over K experts from the Repairer hidden state."""

    def __init__(self, d_model, P, n_experts=3):
        super().__init__()
        self.P, self.K = P, n_experts
        self.proj = nn.Linear(d_model, n_experts * P)

    def forward(self, h, expert_mask=None):  # h: (M, N, d) -> π: (M, N, P, K)
        M, N, _ = h.shape
        logits = self.proj(h).reshape(M, N, self.P, self.K)
        if expert_mask is not None:  # bool (K,), False = disabled expert
            logits = logits.masked_fill(~expert_mask.view(1, 1, 1, -1), float('-inf'))
        return torch.softmax(logits, dim=-1)


# --------------------------------------------------------------------------------------
# Parameter-free experts
# --------------------------------------------------------------------------------------
class LocalMedianExpert(nn.Module):
    """c_t = median of the neighbours in [t-h, t+h] excluding t itself (NaN padding + nanmedian)."""

    def __init__(self, half_window=3):
        super().__init__()
        self.h = half_window

    def forward(self, x):  # (B, L, C)
        B, L, C = x.shape
        h = self.h
        nan = x.new_full((B, h, C), float('nan'))
        win = torch.cat([nan, x, nan], dim=1).unfold(1, 2 * h + 1, 1)  # (B, L, C, 2h+1)
        nb = torch.cat([win[..., :h], win[..., h + 1:]], dim=-1)
        return torch.nanmedian(nb, dim=-1).values


class ReliableNeighborExpert(nn.Module):
    """E2: median of the nearest *reliable* neighbours (R > thr; up to k before and k after t).

    Unlike a plain local median it keeps working under contiguous corruption (e.g. the last 4/8 points zeroed):
    corrupted neighbours are skipped. When no reliable point exists after t (end of the window), it falls back to
    the level of the last k reliable points plus their least-squares linear trend, extrapolated to t.
    Windows without any reliable point fall back to the window median. Parameter-free; R is detached.
    """

    def __init__(self, k=3, thr=0.5, trend=True):
        super().__init__()
        self.k, self.thr, self.trend = int(k), float(thr), bool(trend)

    def forward(self, x, R):  # x, R: (B, L, C) -> (B, L, C)
        B, L, C = x.shape
        K = self.k
        z = x.permute(0, 2, 1).reshape(B * C, L)
        M = (R.detach() > self.thr).permute(0, 2, 1).reshape(B * C, L)
        BC = z.shape[0]
        ar = torch.arange(L, device=x.device)
        order = torch.where(M, ar, ar + L).argsort(dim=1)  # reliable positions first, in time order
        n_rel = M.sum(dim=1, keepdim=True)  # (BC, 1)
        cs = M.long().cumsum(dim=1)  # reliable count in [0, t]
        before = cs - M.long()  # reliable count in [0, t)
        ks = torch.arange(1, K + 1, device=x.device)

        rb = before.unsqueeze(-1) - ks  # rank of the k-th reliable point before t
        vb = rb >= 0
        pb = torch.gather(order, 1, rb.clamp_min(0).reshape(BC, -1)).reshape(BC, L, K)
        xb = torch.gather(z, 1, pb.reshape(BC, -1)).reshape(BC, L, K)
        rf = cs.unsqueeze(-1) + ks - 1  # rank of the k-th reliable point after t
        vf = rf < n_rel.unsqueeze(-1)
        pf = torch.gather(order, 1, rf.clamp_max(L - 1).reshape(BC, -1)).reshape(BC, L, K)
        xf = torch.gather(z, 1, pf.reshape(BC, -1)).reshape(BC, L, K)

        nan = torch.full_like(xb, float('nan'))
        both = torch.nanmedian(torch.cat([torch.where(vb, xb, nan), torch.where(vf, xf, nan)], dim=-1), dim=-1).values
        level_b = torch.nanmedian(torch.where(vb, xb, nan), dim=-1).values
        end_val = level_b
        if self.trend:
            w = vb.to(z.dtype)
            n = w.sum(-1)
            tb = pb.to(z.dtype)
            tm = (w * tb).sum(-1) / n.clamp_min(1.0)
            xm = (w * xb).sum(-1) / n.clamp_min(1.0)
            var = (w * (tb - tm.unsqueeze(-1)) ** 2).sum(-1)
            cov = (w * (tb - tm.unsqueeze(-1)) * (xb - xm.unsqueeze(-1))).sum(-1)
            slope = torch.where(n >= 2, cov / var.clamp_min(1e-6), torch.zeros_like(cov))
            end_val = level_b + slope * (ar.to(z.dtype).view(1, L) - tm)

        has_f, has_b = vf[..., 0], vb[..., 0]
        fallback = z.median(dim=1, keepdim=True).values.expand(BC, L)
        out = torch.where(has_f, both, torch.where(has_b, end_val, fallback))
        out = torch.where(torch.isfinite(out), out, fallback)
        return out.reshape(B, C, L).permute(0, 2, 1)


def gtr_gather_index(cycle_index, seq_len, cycle_len):
    """Exactly the retrieval index used in models/GTR.py."""
    return (cycle_index.view(-1, 1) +
            torch.arange(seq_len, device=cycle_index.device).view(1, -1)) % cycle_len


class CyclePriorExpert(nn.Module):
    """Re-uses the backbone's global cycle matrix Q (detached; no new parameters).

    Weighted least squares per sample/channel: x_t ≈ a*q_t + b with weights w_t = R_t (detached).
    """

    def __init__(self, eps=1e-6, a_max=10.0):
        super().__init__()
        self.eps = eps
        self.a_max = a_max

    def forward(self, x, cycle_index, Q, w):
        Qd = Q.detach()
        gi = gtr_gather_index(cycle_index, x.shape[1], Qd.shape[0])
        q = Qd[gi]  # (B, L, C)
        w = w.detach()
        wsum = w.sum(dim=1, keepdim=True)
        w = torch.where(wsum < 1e-6, torch.ones_like(w), w)  # all-zero weights -> uniform
        wsum = w.sum(dim=1, keepdim=True)
        mq = (w * q).sum(1, keepdim=True) / wsum
        mx = (w * x).sum(1, keepdim=True) / wsum
        cov = (w * (q - mq) * (x - mx)).sum(1, keepdim=True) / wsum
        var = (w * (q - mq) ** 2).sum(1, keepdim=True) / wsum
        a = (cov / (var + self.eps)).clamp(0.0, self.a_max)
        a = torch.where(var < self.eps, torch.zeros_like(a), a)
        b = mx - a * mq
        return a * q + b, q


# --------------------------------------------------------------------------------------
# Baseline B2: Hampel filter (test-time preprocessing only)
# --------------------------------------------------------------------------------------
def hampel_filter(x, window=7, n_sigma=3.0, mad_scale=1.0):
    """Replace x_t by the window median when |x_t - med_t| > n_sigma * mad_scale * MAD_t.

    Window is centred (t-h..t+h, includes t) and truncated at the borders (NaN padding).
    mad_scale=1.0 implements the literal '3·MAD' threshold; use 1.4826 for the Gaussian-consistent variant.
    """
    B, L, C = x.shape
    h = window // 2
    nan = x.new_full((B, h, C), float('nan'))
    win = torch.cat([nan, x, nan], dim=1).unfold(1, 2 * h + 1, 1)  # (B, L, C, w)
    med = torch.nanmedian(win, dim=-1).values
    mad = torch.nanmedian((win - med.unsqueeze(-1)).abs(), dim=-1).values
    out = torch.where((x - med).abs() > n_sigma * mad_scale * mad, med, x)
    return out


def logit(p):
    return math.log(p / (1.0 - p))
