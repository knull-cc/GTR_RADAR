"""Perturbations in StandardScaler space. Only the input window batch_x is modified, never batch_y.

raw_zero[c] = (0 - scaler.mean_[c]) / scaler.scale_[c]  (a physical 0 expressed in scaled units), taken from the
*train* dataset scaler, so "zero" corruptions are physical zeros. zero_mode='scaled' uses 0 in scaled space instead.

RADAR v4 test protocol: a grid "type × position × length × channel ratio"; condition names are
    {type}_{pos}_k{k}_c{ratio}        e.g. zero_tail_k4_c100, flat_int_k8_c50, spike_tail_k2_c100
  type   zero  : segment set to the physical 0                       k ∈ {1, 4, 8}
         flat  : segment holds the last value before it               k ∈ {1, 4, 8}
         noise : segment + N(0, σ²), σ = window std (≥ 0.1)            k ∈ {8, 16}
         spike : k isolated points + U(3, 6)·σ with random sign        k ∈ {1, 2, 3} points
         shift : segment + U(2, 4)·σ (one sign per channel)  — OOD, never used in training, k ∈ {4, 8}
  pos    tail  : the segment ends at the last point (spikes fall in the last `guard` points = last RADAR patch)
         int   : the segment lies inside the first L − guard points (never touches the last patch), uniform start
  ratio  100   : every channel;  50: a random ceil(C/2) subset of channels, drawn per window
Per batch, the random draws come from rng([seed, crc32(condition), batch index]), so every method sees exactly the
same perturbed windows. The training mixture (MixturePerturber) uses the same corruption functions.
Legacy names are still accepted: T1_raw = zero_tail_k1_c100, T2_k{k} = zero_tail_k{k}_c100,
T3_flat = flat_tail_k8_c100, T1_scaled = last point set to 0 in scaled space.
"""
import math
import re
import zlib

import numpy as np
import torch

TYPES = ('zero', 'flat', 'noise', 'spike')  # in-distribution types (training mixture + main table)
OOD_TYPES = ('shift',)  # held out from training
POSITIONS = ('tail', 'int')
GRID_LENGTHS = {'zero': (1, 4, 8), 'flat': (1, 4, 8), 'noise': (8, 16), 'spike': (1, 2, 3), 'shift': (4, 8)}
CHANNEL_RATIOS = (100, 50)
TYPE_LABEL = {'zero': 'Zero', 'flat': 'Flat', 'noise': 'Noise', 'spike': 'Spike', 'shift': 'Shift (OOD)'}
POS_LABEL = {'tail': 'Tail', 'int': 'Internal'}

LEGACY_CONDITIONS = ('T1_raw', 'T1_scaled', 'T2_k4', 'T2_k8', 'T3_flat')
_LEGACY_MAP = {'T1_raw': ('zero', 'tail', 1, 100), 'T3_flat': ('flat', 'tail', 8, 100)}
_COND_RE = re.compile(r'^(zero|flat|noise|spike|shift)_(tail|int)_k(\d+)_c(\d+)$')


def grid_conditions(grid='full', ood=True):
    """'full': both channel ratios; 'main': all channels only. Always starts with 'clean'."""
    assert grid in ('full', 'main'), grid
    ratios = CHANNEL_RATIOS if grid == 'full' else (100,)
    conds = ['clean']
    for typ in TYPES + (OOD_TYPES if ood else ()):
        for pos in POSITIONS:
            for k in GRID_LENGTHS[typ]:
                for c in ratios:
                    conds.append(f'{typ}_{pos}_k{k}_c{c}')
    return conds


ALL_CONDITIONS = tuple(grid_conditions('full')) + LEGACY_CONDITIONS
TEST_CONDITIONS = tuple(grid_conditions('full'))


def parse_condition(cond):
    """-> dict(type, pos, k, ratio) for grid / legacy names, None for 'clean' (and 'T1_scaled', see below)."""
    if cond == 'clean':
        return None
    if cond in _LEGACY_MAP:
        t, p, k, c = _LEGACY_MAP[cond]
        return dict(type=t, pos=p, k=k, ratio=c)
    if cond.startswith('T2_k'):
        return dict(type='zero', pos='tail', k=int(cond[len('T2_k'):]), ratio=100)
    if cond == 'T1_scaled':
        return dict(type='zero_scaled', pos='tail', k=1, ratio=100)
    m = _COND_RE.match(cond)
    if m is None:
        raise ValueError(f'unknown test condition {cond}')
    return dict(type=m.group(1), pos=m.group(2), k=int(m.group(3)), ratio=int(m.group(4)))


def is_valid_condition(cond):
    try:
        parse_condition(cond)
        return True
    except ValueError:
        return False


def raw_zero_from_scaler(scaler, zero_mode='raw'):
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if zero_mode == 'scaled':
        return np.zeros_like(mean, dtype=np.float32)
    assert zero_mode == 'raw', zero_mode
    return ((0.0 - mean) / scale).astype(np.float32)


def condition_rng(seed, cond, batch_idx):
    return np.random.default_rng([int(seed), zlib.crc32(cond.encode()), int(batch_idx)])


# --------------------------------------------------------------------------------------------------------------
# one corruption on one window (numpy, in place). xb: (L, C) view to modify; orig: (L, C) clean copy.
# --------------------------------------------------------------------------------------------------------------
def _segment_start(rng, L, k, pos, guard, lo=0):
    if pos == 'tail':
        return L - k
    hi = L - guard - k  # last admissible start: segment ⊂ [0, L - guard)
    if hi < lo:  # very short windows (unit tests): allow touching the last patch, never the last point
        hi = max(lo, L - k - 1)
    return int(rng.integers(lo, hi + 1))


def corrupt_window(xb, orig, typ, pos, k, ch, rng, raw_zero, guard=8):
    """Apply one corruption of `typ` (length / number of points k) at `pos` to channels `ch` of window xb."""
    L = xb.shape[0]
    k = int(max(1, min(k, L - 1)))
    if typ == 'spike':
        if pos == 'tail':
            cand = np.arange(max(0, L - guard), L)
        else:
            cand = np.arange(0, max(1, L - guard))
        n = min(k, len(cand))
        ts = rng.choice(cand, size=n, replace=False)
        sig = np.maximum(orig[:, ch].std(axis=0), 0.1)
        mag = rng.uniform(3.0, 6.0, size=(n, len(ch))) * rng.choice([-1.0, 1.0], size=(n, len(ch)))
        xb[np.ix_(ts, ch)] += (mag * sig).astype(np.float32)
        return
    s = _segment_start(rng, L, k, pos, guard, lo=1 if typ == 'flat' else 0)
    e = s + k
    if typ == 'zero':
        xb[s:e, ch] = raw_zero[ch]
    elif typ == 'zero_scaled':
        xb[s:e, ch] = 0.0
    elif typ == 'flat':
        xb[s:e, ch] = orig[s - 1, ch]
    elif typ == 'noise':
        sig = np.maximum(orig[:, ch].std(axis=0), 0.1)
        xb[s:e, ch] += (rng.standard_normal((k, len(ch))) * sig).astype(np.float32)
    elif typ == 'shift':
        sig = np.maximum(orig[:, ch].std(axis=0), 0.1)
        mag = rng.uniform(2.0, 4.0, size=len(ch)) * rng.choice([-1.0, 1.0], size=len(ch))
        xb[s:e, ch] += (mag * sig).astype(np.float32)
    else:
        raise ValueError(typ)


def _grid_channels(rng, C, ratio):
    if ratio >= 100:
        return np.arange(C)
    n = max(1, int(math.ceil(C * ratio / 100.0)))
    return np.sort(rng.choice(C, size=n, replace=False))


def apply_condition(x, cond, raw_zero, zero_mode='raw', rng=None, guard=8):
    """x: (B, L, C) tensor. Returns (x_pert, changed_mask) on x's device. Deterministic given rng."""
    spec = parse_condition(cond)
    if spec is None:
        return x.clone(), torch.zeros_like(x, dtype=torch.bool)
    rng = np.random.default_rng(0) if rng is None else rng
    xn = x.detach().float().cpu().numpy().copy()
    orig = xn.copy()
    B, L, C = xn.shape
    rz = np.zeros(C, dtype=np.float32) if zero_mode == 'scaled' else np.asarray(raw_zero, dtype=np.float32)
    for b in range(B):
        ch = _grid_channels(rng, C, spec['ratio'])
        corrupt_window(xn[b], orig[b], spec['type'], spec['pos'], spec['k'], ch, rng, rz, guard)
    x_p = torch.from_numpy(xn).to(device=x.device, dtype=x.dtype)
    changed = torch.from_numpy(np.abs(xn - orig) > 1e-6).to(x.device)
    return x_p, changed


def apply_test_perturbation(x, cond, raw_zero, zero_mode='raw'):
    """Backward-compatible deterministic entry point (legacy tail conditions; grid names use rng(0))."""
    return apply_condition(x, cond, raw_zero, zero_mode)


def perturb_batch(batch_x, batch_y, cond, raw_zero, zero_mode='raw'):
    """Convenience wrapper: perturbs the input window only; batch_y is returned untouched."""
    x_p, mask = apply_test_perturbation(batch_x, cond, raw_zero, zero_mode)
    return x_p, batch_y, mask


class MixturePerturber:
    """Training / validation augmentation (RADAR v4). Returns point-level labels m_t (bool, changed points).

    Each sample is perturbed with prob p_apply; type ~ {zero .30, flat .20, noise .25, spike .25}; length uniform in
    zero/flat 1–8, noise 4–16, spike 1–3 points; position tail w.p. p_tail, otherwise internal (same definitions as
    the test grid). Channels: all with prob 0.5, otherwise each channel w.p. 0.5 (≥ 1).
    The OOD type 'shift' is deliberately NOT part of the mixture.
    """
    TYPES = TYPES
    PROBS = (0.30, 0.20, 0.25, 0.25)
    LENGTHS = {'zero': (1, 8), 'flat': (1, 8), 'noise': (4, 16), 'spike': (1, 3)}

    def __init__(self, raw_zero, p_apply=0.5, seed=0, p_tail=0.5, guard=8, probs=None):
        self.raw_zero = np.asarray(raw_zero, dtype=np.float32)
        self.p_apply = float(p_apply)
        self.p_tail = float(p_tail)
        self.guard = int(guard)
        self.probs = tuple(probs) if probs is not None else self.PROBS
        assert len(self.probs) == len(self.TYPES) and abs(sum(self.probs) - 1.0) < 1e-6, self.probs
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.last_pos = None

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.seed if seed is None else int(seed))

    def _channels(self, C):
        if self.rng.random() < 0.5:
            return np.arange(C)
        sel = self.rng.random(C) < 0.5
        if not sel.any():
            sel[self.rng.integers(C)] = True
        return np.nonzero(sel)[0]

    def __call__(self, x):
        """x: (B, L, C) tensor (CPU or GPU). Returns x_pert (float32, same device), mask (bool), types (B,)
        (index into TYPES, -1 = untouched). self.last_pos[b] = 1 tail, 0 internal, -1 untouched."""
        device = x.device
        xn = x.detach().float().cpu().numpy().copy()
        orig = xn.copy()
        B, L, C = xn.shape
        types = np.full(B, -1, dtype=np.int64)
        pos_flag = np.full(B, -1, dtype=np.int64)
        for b in range(B):
            if self.rng.random() >= self.p_apply:
                continue
            k_type = int(self.rng.choice(len(self.TYPES), p=self.probs))
            typ = self.TYPES[k_type]
            lo, hi = self.LENGTHS[typ]
            k = int(self.rng.integers(lo, hi + 1))
            pos = 'tail' if self.rng.random() < self.p_tail else 'int'
            ch = self._channels(C)
            corrupt_window(xn[b], orig[b], typ, pos, k, ch, self.rng, self.raw_zero, self.guard)
            types[b] = k_type
            pos_flag[b] = 1 if pos == 'tail' else 0
        self.last_pos = pos_flag
        mask = np.abs(xn - orig) > 1e-6  # points whose value actually changed
        return torch.from_numpy(xn).to(device), torch.from_numpy(mask).to(device), types
