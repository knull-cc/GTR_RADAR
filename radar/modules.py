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
    """(v3) Centered rolling median along the last axis of z (M, T); border values replicated."""
    h = k // 2
    zp = torch.cat([z[:, :1].expand(-1, h), z, z[:, -1:].expand(-1, h)], dim=1)
    return zp.unfold(1, k, 1).median(dim=-1).values


def shifted_windows(z, w):
    """z: (M, T) -> (M, T, w). For every t the w-point window [start, start + w) with
    start = clamp(t − w//2, 0, T − w): centred inside the series, *shifted* (not padded) at both ends, so it always
    holds w real points and the last point only sees its left context."""
    M, T = z.shape
    w = min(int(w), T)
    win = z.unfold(1, w, 1)  # (M, T − w + 1, w)
    start = (torch.arange(T, device=z.device) - w // 2).clamp(0, T - w)
    return win[:, start]


N_POINT_FEATURES = {'v3': 3, 'v4': 5}
POINT_FEATURE_NAMES = {'v3': ('u', 'du', '|u-med5|'),
                       'v4': ('u', 'du', '|u-med5|', '|u-med17|', 'MAD5(u)')}


def point_features(u, version='v4'):
    """u: (M, T) robust-scaled series -> (M, T, F).

    v3 (F=3): [u_t, Δu_t, |u_t − med5_t|] with a centred, replicate-padded median (|·| ≡ 0 at the last point).
    v4 (F=5): [u_t, Δu_t, |u_t − med5_t|, |u_t − med17_t|, MAD5(u)_t] with shifted windows (see shifted_windows):
      * med17 covers corrupted runs of up to 8 points, where the 5-point median is dominated by the run itself;
      * MAD5 = median_j |u_j − med5| inside the same 5-point window (u is already divided by s): high on noise
        segments, ~0 on flat / constant segments.
    """
    du = torch.cat([torch.zeros_like(u[:, :1]), u[:, 1:] - u[:, :-1]], dim=1)
    if version == 'v3':
        dev = (u - rolling_median(u, 5)).abs()
        return torch.stack([u, du, dev], dim=-1)
    assert version == 'v4', version
    w5 = shifted_windows(u, 5)
    med5 = w5.median(dim=-1).values
    med17 = shifted_windows(u, 17).median(dim=-1).values
    mad5 = (w5 - med5.unsqueeze(-1)).abs().median(dim=-1).values
    return torch.stack([u, du, (u - med5).abs(), (u - med17).abs(), mad5], dim=-1)


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

    def __init__(self, P, N, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, head_bias=-4.0, n_feat=3):
        super().__init__()
        self.P = P
        self.body = PatchTransformer(n_feat * P, N, d_model, n_heads, n_layers, d_ff, dropout)
        self.head = nn.Linear(d_model, P)
        nn.init.constant_(self.head.bias, head_bias)

    def forward(self, feats):  # feats: (M, N, F·P) -> logits (M, N, P)
        return self.head(self.body(feats))


class Repairer(nn.Module):
    """Context reconstruction expert E1.

    v4 (in_ch=2): input per point [R_t·u_t, 1−R_t]; output û.
    v5 (in_ch=3, residual=True): input per point [R_t·u_t, 1−R_t, u_base_t] where u_base is the stage-1 repair
    (E2/E3 routed); output the correction Δ, with a zero-initialised head so that E1 starts as the identity on the
    stage-1 repair. Returns (output (M, N, P), hidden (M, N, d))."""

    def __init__(self, P, N, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, in_ch=2, residual=False):
        super().__init__()
        self.P = P
        self.body = PatchTransformer(in_ch * P, N, d_model, n_heads, n_layers, d_ff, dropout)
        self.head = nn.Linear(d_model, P)
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, tokens):  # (M, N, in_ch·P)
        h = self.body(tokens)
        return self.head(h), h


class Router(nn.Module):
    """(v3, `--router h`) per-point expert logits from the Repairer hidden state."""

    def __init__(self, d_model, P, n_experts=3):
        super().__init__()
        self.P, self.K = P, n_experts
        self.proj = nn.Linear(d_model, n_experts * P)

    def forward(self, h):  # h: (M, N, d) -> logits (M, N, P, K)
        M, N, _ = h.shape
        return self.proj(h).reshape(M, N, self.P, self.K)


# --------------------------------------------------------------------------------------
# Series layout helpers: (B, L, C, ...) <-> (B*C, L, ...)
# --------------------------------------------------------------------------------------
def to_series(x):
    B, L, C = x.shape[:3]
    return x.transpose(1, 2).reshape(B * C, L, *x.shape[3:])


def from_series(z, B, C):
    L = z.shape[1]
    return z.reshape(B, C, L, *z.shape[2:]).transpose(1, 2)


# --------------------------------------------------------------------------------------
# Reliable context (shared by E2, E3 and the Router features). M: bool (S, L), True = reliable (R > thr).
# --------------------------------------------------------------------------------------
def nearest_reliable(M, K):
    """Positions of the K nearest reliable points strictly before / strictly after every t.
    Returns pb, vb, pf, vf, each (S, L, K); pb/pf hold a valid index (clamped) wherever vb/vf is False."""
    S, L = M.shape
    ar = torch.arange(L, device=M.device)
    order = torch.where(M, ar, ar + L).argsort(dim=1)  # reliable positions first, in time order
    n_rel = M.sum(dim=1, keepdim=True)  # (S, 1)
    cs = M.long().cumsum(dim=1)  # reliable count in [0, t]
    before = cs - M.long()  # reliable count in [0, t)
    ks = torch.arange(1, K + 1, device=M.device)
    rb = before.unsqueeze(-1) - ks  # rank of the k-th reliable point before t
    vb = rb >= 0
    pb = torch.gather(order, 1, rb.clamp_min(0).reshape(S, -1)).reshape(S, L, K)
    rf = cs.unsqueeze(-1) + ks - 1  # rank of the k-th reliable point after t
    vf = rf < n_rel.unsqueeze(-1)
    pf = torch.gather(order, 1, rf.clamp_max(L - 1).reshape(S, -1)).reshape(S, L, K)
    return pb, vb, pf, vf


def reliable_context(M, K=3):
    """Everything the parameter-free experts and the Router need about the reliable points of each series.

    gap   : length of the unreliable run containing t (0 if t is reliable; window borders bound the run)
    dl/dr : distance to the nearest reliable point at/before (after) t (0 if t is reliable, L if there is none)
    has_l / has_r : a reliable point exists at/before (at/after) t
    pb, vb, pf, vf: K nearest reliable points strictly before / after t (see nearest_reliable)
    """
    S, L = M.shape
    ar = torch.arange(L, device=M.device).view(1, L).expand(S, L)
    last = torch.where(M, ar, torch.full_like(ar, -1)).cummax(dim=1).values
    nxt = torch.where(M, ar, torch.full_like(ar, L)).flip(1).cummin(dim=1).values.flip(1)
    has_l, has_r = last >= 0, nxt < L
    zero = torch.zeros_like(ar)
    big = torch.full_like(ar, L)
    gap = torch.where(M, zero, nxt - last - 1)
    dl = torch.where(M, zero, torch.where(has_l, ar - last, big))
    dr = torch.where(M, zero, torch.where(has_r, nxt - ar, big))
    pb, vb, pf, vf = nearest_reliable(M, K)
    return dict(M=M, gap=gap, dl=dl, dr=dr, has_l=has_l, has_r=has_r, pb=pb, vb=vb, pf=pf, vf=vf)


def _side(z, p, v):
    """Robust level of z at the anchor positions p (S, L, K) where v (and z finite).
    Returns (nanmedian level, mean anchor position, any-valid), each (S, L)."""
    S, L, K = p.shape
    vals = torch.gather(z, 1, p.reshape(S, -1)).reshape(S, L, K)
    v = v & torch.isfinite(vals)
    lev = torch.nanmedian(torch.where(v, vals, torch.full_like(vals, float('nan'))), dim=-1).values
    w = v.to(z.dtype)
    pos = (w * p.to(z.dtype)).sum(-1) / w.sum(-1).clamp_min(1.0)
    return lev, pos, v.any(dim=-1)


def _interp(t, lev_b, pos_b, has_b, lev_f, pos_f, has_f, fallback):
    """Linear interpolation between the left and right levels where both exist, one-sided level otherwise."""
    w = ((t - pos_b) / (pos_f - pos_b).clamp_min(1e-6)).clamp(0.0, 1.0)
    both = lev_b + w * (lev_f - lev_b)
    return torch.where(has_b & has_f, both, torch.where(has_b, lev_b, torch.where(has_f, lev_f, fallback)))


# --------------------------------------------------------------------------------------
# Parameter-free experts
# --------------------------------------------------------------------------------------
class LocalLinearExpert(nn.Module):
    """E2 (v4): "the gap is a straight line between its reliable neighbours" — no periodic information.

    Left / right level = median of the k nearest reliable points before / after t (anchored at their mean position):
      both sides  -> linear interpolation of the two levels at t
      one side    -> that side's level (no trend extrapolation)
      no reliable -> window median.
    Works on x directly; R is detached (through the context).
    """

    def forward(self, x, ctx):  # x: (B, L, C), ctx from reliable_context on to_series layout -> (B, L, C)
        B, L, C = x.shape
        z = to_series(x)
        t = torch.arange(L, device=x.device, dtype=z.dtype).view(1, L)
        lb, pb, hb = _side(z, ctx['pb'], ctx['vb'])
        lf, pf, hf = _side(z, ctx['pf'], ctx['vf'])
        fallback = z.median(dim=1, keepdim=True).values.expand_as(z)
        out = _interp(t, lb, pb, hb, lf, pf, hf, fallback)
        out = torch.where(torch.isfinite(out), out, fallback)
        return from_series(out, B, C)


class PeriodicShapeExpert(nn.Module):
    """E3 (v4): "the gap has the same shape as the same-phase segments of the other cycles in the window".

    Uses only the window itself (no backbone parameter, no cycle_index; the phase is aligned implicitly by t + jT).
    For every other cycle j = ±1..±n_cycles (offset jT):
      level offset  d_j = median over the k nearest reliable anchors a before (after) t of (x_a − x_{a+jT}),
                    interpolated linearly between the left and the right anchors (one-sided where only one exists)
      candidate     c_t^(j) = x_{t+jT} + d_j(t)          (seasonal naive from cycle j, corrected to the local level)
    c3_t = median_j c_t^(j)  over the cycles whose same-phase point is inside the window and reliable.
    The per-cycle level correction keeps the repair exact under a linear trend (a single median template is not).
    Unavailable where T ≥ L, where no cycle qualifies, or where no anchor pair exists; there `fill` is returned and
    avail = False. spread = median_j |c_t^(j) − c3_t| (disagreement between cycles).
    """

    def __init__(self, period, seq_len, n_cycles=3):
        super().__init__()
        self.T = int(period) if period else 0
        self.enabled = 0 < self.T < int(seq_len)
        offs = [j * self.T for j in range(-int(n_cycles), int(n_cycles) + 1) if j != 0] if self.enabled else [0]
        self.register_buffer('offsets', torch.tensor(offs, dtype=torch.long), persistent=False)

    @staticmethod
    def _take(z, Mu, idx):
        """z (S, L) values, Mu (S, L) uint8 reliable mask, idx (S, ...) positions (may leave the window)
        -> values and validity (inside the window and reliable), both shaped like idx."""
        S, L = z.shape
        inb = (idx >= 0) & (idx < L)
        flat = idx.clamp(0, L - 1).reshape(S, -1)
        v = torch.gather(z, 1, flat).reshape(idx.shape)
        ok = torch.gather(Mu, 1, flat).reshape(idx.shape).bool() & inb
        return v, ok

    def _level_offsets(self, z, Mu, p, v):
        """Anchors p, v (S, L, K) -> d_j (S, L, J), mean anchor position (S, L, J), any valid pair (S, L, J)."""
        S, L, K = p.shape
        J = self.offsets.shape[0]
        xa = torch.gather(z, 1, p.reshape(S, -1)).reshape(S, L, K, 1)
        xs, oks = self._take(z, Mu, p.unsqueeze(-1) + self.offsets.view(1, 1, 1, J))  # (S, L, K, J)
        ok = v.unsqueeze(-1) & oks
        diff = torch.where(ok, xa - xs, torch.full_like(xs, float('nan')))
        lev = torch.nanmedian(diff, dim=2).values
        w = ok.to(z.dtype)
        pos = (w * p.unsqueeze(-1).to(z.dtype)).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        return lev, pos, ok.any(dim=2)

    def forward(self, x, ctx, fill):
        """x, fill: (B, L, C). Returns c3 (B, L, C), avail (bool, B, L, C), spread (B, L, C) (0 where unavailable)."""
        B, L, C = x.shape
        if not self.enabled:
            return fill, torch.zeros_like(x, dtype=torch.bool), torch.zeros_like(x)
        z = to_series(x)
        S = z.shape[0]
        Mu = ctx['M'].to(torch.uint8)
        J = self.offsets.shape[0]
        ar = torch.arange(L, device=x.device)
        idx = (ar.view(L, 1) + self.offsets.view(1, J)).view(1, L, J).expand(S, L, J)
        v_t, ok_t = self._take(z, Mu, idx)  # same-phase values x_{t+jT}
        db, pb, hb = self._level_offsets(z, Mu, ctx['pb'], ctx['vb'])
        df, pf, hf = self._level_offsets(z, Mu, ctx['pf'], ctx['vf'])
        t = ar.to(z.dtype).view(1, L, 1)
        nan = torch.full_like(v_t, float('nan'))
        d = _interp(t, db, pb, hb, df, pf, hf, nan)
        cand = torch.where(ok_t, v_t + d, nan)  # (S, L, J), NaN where cycle j does not qualify
        c3 = torch.nanmedian(cand, dim=-1).values
        spread = torch.nanmedian((cand - c3.unsqueeze(-1)).abs(), dim=-1).values
        avail = torch.isfinite(c3)
        c3 = torch.where(avail, c3, to_series(fill))
        spread = torch.where(avail & torch.isfinite(spread), spread, torch.zeros_like(spread))
        return from_series(c3, B, C), from_series(avail, B, C), from_series(spread, B, C)


# --------------------------------------------------------------------------------------
# Interpretable Router (v4, `--router linear`)
# --------------------------------------------------------------------------------------
LENGTH_BINS = ('0', '1', '2-4', '5-8', '>8')
ROUTE_FEATURES = (
    tuple(f'gap={b}' for b in LENGTH_BINS[1:]) +  # reference: gap = 0 (reliable point)
    ('support=left only', 'support=right only', 'support=none') +  # reference: both sides
    tuple(f'dist_left={b}' for b in LENGTH_BINS[1:]) +  # reference: 0
    tuple(f'dist_right={b}' for b in LENGTH_BINS[1:]) +
    ('reliable density ±4', 'log1p Δ(c1,c2)/s', 'log1p Δ(c1,c3)/s', 'log1p Δ(c2,c3)/s',
     'E3 available', 'log1p E3 spread/s'))


def length_bin(v):
    """Long tensor ≥ 0 -> bin index 0..4 for {0, 1, 2–4, 5–8, > 8}."""
    return (v >= 1).long() + (v >= 2).long() + (v >= 5).long() + (v >= 9).long()


def _onehot_ref(idx, n):
    """One-hot without the reference category 0: (…,) -> (…, n − 1)."""
    return torch.nn.functional.one_hot(idx, n)[..., 1:]


def _context_parts(ctx, R):
    """Series-layout routing features shared by v4 and v5: gap / support / distance bins (reference coding) and the
    reliable density ±4. Returns a list of (S, L, ·) float tensors."""
    dt = R.dtype
    hl, hr = ctx['has_l'], ctx['has_r']
    support = (hl & ~hr).long() * 1 + (~hl & hr).long() * 2 + (~hl & ~hr).long() * 3  # 0 = both sides
    parts = [_onehot_ref(length_bin(ctx['gap']), 5), _onehot_ref(support, 4),
             _onehot_ref(length_bin(ctx['dl']), 5), _onehot_ref(length_bin(ctx['dr']), 5)]
    parts = [p.to(dt) for p in parts]
    Rs = to_series(R.detach())
    dens = torch.nn.functional.avg_pool1d(Rs.unsqueeze(1), 9, stride=1, padding=4,
                                          count_include_pad=False).squeeze(1)
    return parts + [dens.unsqueeze(-1)]


def route_features(ctx, R, c, s, avail3, spread3):
    """(v4) Named, detached per-point routing features φ (B, L, C, len(ROUTE_FEATURES)).

    ctx: reliable_context (series layout); R: (B, L, C); c: (B, L, C, 3) candidates; s: (B, 1, C) robust scale;
    avail3 / spread3: E3 availability and template spread (B, L, C)."""
    B, L, C = R.shape
    dt = R.dtype
    parts = _context_parts(ctx, R)
    cs = to_series(c.detach())  # (S, L, 3)
    ss = to_series(s.expand(B, L, C))  # (S, L)
    a3 = to_series(avail3).to(dt)
    d12 = torch.log1p((cs[..., 0] - cs[..., 1]).abs() / ss)
    d13 = torch.log1p((cs[..., 0] - cs[..., 2]).abs() / ss) * a3
    d23 = torch.log1p((cs[..., 1] - cs[..., 2]).abs() / ss) * a3
    sp = torch.log1p(to_series(spread3) / ss) * a3
    parts += [d12.unsqueeze(-1), d13.unsqueeze(-1), d23.unsqueeze(-1), a3.unsqueeze(-1), sp.unsqueeze(-1)]
    phi = torch.cat(parts, dim=-1)
    return from_series(phi, B, C).detach()


ROUTE_FEATURES_V5 = ROUTE_FEATURES[:16] + ('log1p Δ(c2,c3)/s', 'E3 available', 'log1p E3 spread/s')


def route_features_v5(ctx, R, c2, c3, s, avail3, spread3):
    """(v5) Stage-1 routing features (E2 vs E3 only), detached: φ (B, L, C, len(ROUTE_FEATURES_V5))."""
    B, L, C = R.shape
    dt = R.dtype
    parts = _context_parts(ctx, R)
    ss = to_series(s.expand(B, L, C))
    a3 = to_series(avail3).to(dt)
    d23 = torch.log1p((to_series(c2.detach()) - to_series(c3.detach())).abs() / ss) * a3
    sp = torch.log1p(to_series(spread3) / ss) * a3
    parts += [d23.unsqueeze(-1), a3.unsqueeze(-1), sp.unsqueeze(-1)]
    return from_series(torch.cat(parts, dim=-1), B, C).detach()


ROUTE_FEATURES_V6 = ROUTE_FEATURES[:16] + (
    'log1p Δ(c2,c3)/s', 'log1p Δ(c2,c4)/s', 'log1p Δ(c3,c4)/s', 'E3 available', 'E4 available',
    'log1p E3 spread/s', 'E4 partner |corr|')


def route_features_v6(ctx, R, c2, c3, c4, s, avail3, avail4, spread3, corr4):
    """(v6) Stage-1 routing features over E2 / E3 / E4, detached: φ (B, L, C, len(ROUTE_FEATURES_V6)).
    corr4: (B, C) |correlation| of the best E4 partner."""
    B, L, C = R.shape
    dt = R.dtype
    parts = _context_parts(ctx, R)
    ss = to_series(s.expand(B, L, C))
    a3, a4 = to_series(avail3).to(dt), to_series(avail4).to(dt)
    z2, z3, z4 = (to_series(v.detach()) for v in (c2, c3, c4))
    d23 = torch.log1p((z2 - z3).abs() / ss) * a3
    d24 = torch.log1p((z2 - z4).abs() / ss) * a4
    d34 = torch.log1p((z3 - z4).abs() / ss) * a3 * a4
    sp = torch.log1p(to_series(spread3) / ss) * a3
    cr = corr4.detach().reshape(B * C, 1).expand(B * C, L) * a4
    parts += [v.unsqueeze(-1) for v in (d23, d24, d34, a3, a4, sp, cr)]
    return from_series(torch.cat(parts, dim=-1), B, C).detach()


class CrossChannelExpert(nn.Module):
    """E4 (v6): "this variable moves with the variables it is most correlated with in this window".

    For every channel c: among the other channels j, the k with the largest |corr(x_c, x_j)| over the window
    points where both are reliable (≥ min_pairs points); per partner a least-squares fit x_c ≈ a + b·x_j on those
    points; c4_t = median over the partners that are reliable at t of a + b·x_j(t). Unavailable when no partner
    is reliable at t (e.g. all channels corrupted at once) or C = 1. Parameter-free; works on x and the detached
    reliable mask. Returns c4 (B, L, C) (fill where unavailable), avail (B, L, C), best |corr| (B, C)."""

    def __init__(self, k=3, min_pairs=8, eps=1e-6):
        super().__init__()
        self.k, self.min_pairs, self.eps = int(k), int(min_pairs), float(eps)

    def forward(self, x, M, fill):
        B, L, C = x.shape
        if C < 2 or self.k < 1:
            return fill, torch.zeros_like(x, dtype=torch.bool), x.new_zeros(B, C)
        x = x.detach()
        w = M.to(x.dtype)
        xw = x * w
        ein = torch.einsum
        n = ein('blc,blj->bcj', w, w)  # joint reliable points of (c, j)
        sx = ein('blc,blj->bcj', xw, w)
        sy = ein('blc,blj->bcj', w, xw)
        sxx = ein('blc,blj->bcj', xw * x, w)
        syy = ein('blc,blj->bcj', w, xw * x)
        sxy = ein('blc,blj->bcj', xw, xw)
        nn_ = n.clamp_min(1.0)
        mx, my = sx / nn_, sy / nn_
        cov = sxy / nn_ - mx * my
        vx = (sxx / nn_ - mx * mx).clamp_min(0.0)
        vy = (syy / nn_ - my * my).clamp_min(0.0)
        corr = cov / torch.sqrt(vx * vy + self.eps)
        eye = torch.eye(C, dtype=torch.bool, device=x.device).view(1, C, C)
        ok = (n >= self.min_pairs) & (vy > self.eps) & ~eye
        score = torch.where(ok, corr.abs(), torch.full_like(corr, -1.0))
        k = min(self.k, C - 1)
        sc, idx = score.topk(k, dim=-1)  # (B, C, k)
        slope = torch.gather(cov / vy.clamp_min(self.eps), 2, idx)
        inter = torch.gather(mx, 2, idx) - slope * torch.gather(my, 2, idx)
        flat = idx.reshape(B, 1, C * k).expand(B, L, C * k)
        xp = torch.gather(x, 2, flat).reshape(B, L, C, k)
        mp = torch.gather(M.to(torch.uint8), 2, flat).reshape(B, L, C, k).bool()
        valid = mp & (sc > 0).unsqueeze(1)
        pred = inter.unsqueeze(1) + slope.unsqueeze(1) * xp
        c4 = torch.nanmedian(torch.where(valid, pred, torch.full_like(pred, float('nan'))), dim=-1).values
        avail = torch.isfinite(c4)
        c4 = torch.where(avail, c4, fill)
        best = sc[..., 0].clamp_min(0.0)
        return c4, avail, best


def loo_prior_multi(x, ctx, cands, avails, s, eps=0.05):
    """Leave-one-out routing prior over K parameter-free experts (v6; v5 = K=2 via loo_prior).

    cands / avails: lists of (B, L, C) candidates and availability masks, the first one (E2) always available.
    ê_k(t) = median over the k nearest reliable anchors before and after t of |c_k(a) − x(a)| / s;
    prior_k = −log(ê_k + eps); where ê_k is missing the expert gets the prior of the first expert (neutral);
    if the first one is missing too, all priors are 0. Returns (prior (B, L, C, K), ê (B, L, C, K)), detached."""
    B, L, C = x.shape
    z = to_series(x.detach())
    ss = to_series(s.expand(B, L, C))
    p = torch.cat([ctx['pb'], ctx['pf']], dim=-1)
    v = torch.cat([ctx['vb'], ctx['vf']], dim=-1)
    S, _, K2 = p.shape
    errs = []
    for c, a in zip(cands, avails):
        e = torch.where(to_series(a), (to_series(c.detach()) - z).abs() / ss, torch.full_like(z, float('nan')))
        g = torch.gather(e, 1, p.reshape(S, -1)).reshape(S, L, K2)
        g = torch.where(v, g, torch.full_like(g, float('nan')))
        errs.append(torch.nanmedian(g, dim=-1).values)
    ehat = torch.stack(errs, dim=-1)
    pr = -torch.log(ehat + eps)
    p0 = torch.where(torch.isfinite(pr[..., 0]), pr[..., 0], torch.zeros_like(pr[..., 0]))
    pri = [p0] + [torch.where(torch.isfinite(pr[..., i]), pr[..., i], p0) for i in range(1, len(cands))]
    return from_series(torch.stack(pri, dim=-1), B, C), from_series(ehat, B, C)


def loo_prior(x, ctx, c2, c3, avail3, s, eps=0.05):
    """Leave-one-out routing prior of stage 1 (v5).

    E2 and E3 never use x_t to predict x_t, so at a reliable point a their outputs are leave-one-out predictions.
    ê_k(t) = median over the k nearest reliable anchors before and after t of |c_k(a) − x(a)| / s; prior logit
    −log(ê_k + eps). Where one expert has no finite error it gets the other's prior (neutral); none -> 0.
    Returns (prior (B, L, C, 2) for [E2, E3], ê (B, L, C, 2)), detached."""
    B, L, C = x.shape
    z = to_series(x.detach())
    ss = to_series(s.expand(B, L, C))
    e2 = (to_series(c2.detach()) - z).abs() / ss
    e3 = torch.where(to_series(avail3), (to_series(c3.detach()) - z).abs() / ss, torch.full_like(z, float('nan')))
    p = torch.cat([ctx['pb'], ctx['pf']], dim=-1)  # (S, L, 2K) anchors
    v = torch.cat([ctx['vb'], ctx['vf']], dim=-1)
    S, _, K2 = p.shape
    errs = []
    for e in (e2, e3):
        g = torch.gather(e, 1, p.reshape(S, -1)).reshape(S, L, K2)
        g = torch.where(v, g, torch.full_like(g, float('nan')))
        errs.append(torch.nanmedian(g, dim=-1).values)
    ehat = torch.stack(errs, dim=-1)  # (S, L, 2)
    pr = -torch.log(ehat + eps)
    f2, f3 = torch.isfinite(pr[..., 0]), torch.isfinite(pr[..., 1])
    p2 = torch.where(f2, pr[..., 0], torch.where(f3, pr[..., 1], torch.zeros_like(pr[..., 0])))
    p3 = torch.where(f3, pr[..., 1], p2)
    prior = torch.stack([p2, p3], dim=-1)
    return from_series(prior, B, C), from_series(ehat, B, C)


class LinearRouter(nn.Module):
    """logit_k = w_kᵀφ + b_k for the experts k = 2..K; E1 (Repairer) is the reference expert with logit 0, so every
    coefficient reads as a log-odds of "expert k instead of E1" relative to the reference bins. Zero-initialised
    (uniform routing over the available experts)."""

    def __init__(self, n_features=len(ROUTE_FEATURES), n_experts=3):
        super().__init__()
        self.K = n_experts
        self.lin = nn.Linear(n_features, n_experts - 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, phi):  # (..., F) -> logits (..., K)
        z = self.lin(phi)
        return torch.cat([torch.zeros_like(z[..., :1]), z], dim=-1)

    def coefficients(self, expert_names, feature_names=ROUTE_FEATURES):
        W = self.lin.weight.detach().cpu().numpy()
        b = self.lin.bias.detach().cpu().numpy()
        return {name: dict(bias=float(b[i]), **{f: float(W[i, j]) for j, f in enumerate(feature_names)})
                for i, name in enumerate(expert_names[1:])}


# --------------------------------------------------------------------------------------
# Legacy / reference experts (diagnostics and the v3 comparison)
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
    """(v3 E2, kept for the offline comparison) median of the nearest *reliable* neighbours (R > thr; up to k before and k after t).

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
