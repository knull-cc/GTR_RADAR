"""Perturbations in StandardScaler space. Only the input window batch_x is modified, never batch_y.

raw_zero[c] = (0 - scaler.mean_[c]) / scaler.scale_[c]  (a physical 0 expressed in scaled units),
taken from the *train* dataset scaler. zero_mode='scaled' uses 0 in scaled space instead.
"""
import numpy as np
import torch

TEST_CONDITIONS = ('clean', 'T1_raw', 'T1_scaled', 'T2_k4', 'T2_k8', 'T3_flat')


def raw_zero_from_scaler(scaler, zero_mode='raw'):
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if zero_mode == 'scaled':
        return np.zeros_like(mean, dtype=np.float32)
    assert zero_mode == 'raw', zero_mode
    return ((0.0 - mean) / scale).astype(np.float32)


def apply_test_perturbation(x, cond, raw_zero, zero_mode='raw'):
    """x: (B, L, C) tensor. Returns (x_pert, changed_mask).

    T1_raw / T1_scaled : x[:, -1, :] = raw_zero / 0
    T2_k{k}            : x[:, -k:, :] = zero (raw_zero, or 0 if zero_mode == 'scaled')
    T3_flat            : x[:, -8:, :] = x[:, -9, :]   (held-out type, never used in training)
    """
    x_p = x.clone()
    rz = torch.as_tensor(np.asarray(raw_zero), dtype=x.dtype, device=x.device).view(1, 1, -1)
    if cond == 'clean':
        pass
    elif cond == 'T1_raw':
        x_p[:, -1:, :] = rz
    elif cond == 'T1_scaled':
        x_p[:, -1:, :] = 0.0
    elif cond.startswith('T2_k'):
        k = int(cond[len('T2_k'):])
        x_p[:, -k:, :] = 0.0 if zero_mode == 'scaled' else rz
    elif cond == 'T3_flat':
        x_p[:, -8:, :] = x[:, -9:-8, :]
    else:
        raise ValueError(f'unknown test condition {cond}')
    return x_p, (x_p != x)


def perturb_batch(batch_x, batch_y, cond, raw_zero, zero_mode='raw'):
    """Convenience wrapper: perturbs the input window only; batch_y is returned untouched."""
    x_p, mask = apply_test_perturbation(batch_x, cond, raw_zero, zero_mode)
    return x_p, batch_y, mask


class MixturePerturber:
    """Training/validation augmentation mixture. Returns point-level labels m_t (bool, changed points).

    Each sample is perturbed with prob p_apply; type ~ {last_zero .35, point_zero .15, spike .20,
    dropout_seg .15, noise_burst .15}. Channels: all with prob 0.5, otherwise each channel w.p. 0.5 (≥1).
    flatline is deliberately NOT part of the mixture (held-out test type T3).
    """
    TYPES = ('last_zero', 'point_zero', 'spike', 'dropout_seg', 'noise_burst')
    PROBS = (0.35, 0.15, 0.20, 0.15, 0.15)

    def __init__(self, raw_zero, p_apply=0.5, seed=0, recency=0.1):
        self.raw_zero = np.asarray(raw_zero, dtype=np.float32)
        self.p_apply = float(p_apply)
        self.seed = int(seed)
        self.recency = float(recency)
        self.rng = np.random.default_rng(self.seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.seed if seed is None else int(seed))

    def _recent_pos(self, L):
        t = np.arange(L)
        w = np.exp(self.recency * (t - L + 1))
        return int(self.rng.choice(L, p=w / w.sum()))

    def _channels(self, C):
        if self.rng.random() < 0.5:
            return np.arange(C)
        sel = self.rng.random(C) < 0.5
        if not sel.any():
            sel[self.rng.integers(C)] = True
        return np.nonzero(sel)[0]

    def __call__(self, x):
        """x: (B, L, C) tensor (CPU or GPU). Returns x_pert (float32, same device), mask (bool), types (B,)"""
        device = x.device
        xn = x.detach().float().cpu().numpy().copy()
        orig = xn.copy()
        B, L, C = xn.shape
        types = np.full(B, -1, dtype=np.int64)
        for b in range(B):
            if self.rng.random() >= self.p_apply:
                continue
            k = int(self.rng.choice(len(self.TYPES), p=self.PROBS))
            ch = self._channels(C)
            xb = xn[b]  # view (L, C)
            if k == 0:  # last_zero
                n = 1 if self.rng.random() < 0.5 else int(self.rng.integers(2, 5))
                xb[L - n:, ch] = self.raw_zero[ch]
            elif k == 1:  # point_zero
                t = self._recent_pos(L)
                xb[t, ch] = self.raw_zero[ch]
            elif k == 2:  # spike
                n = int(self.rng.integers(1, 4))
                ts = self.rng.choice(L, size=n, replace=False)
                sig = np.maximum(orig[b][:, ch].std(axis=0), 0.1)
                mag = self.rng.uniform(3.0, 6.0, size=(n, len(ch))) * self.rng.choice([-1.0, 1.0], size=(n, len(ch)))
                xb[np.ix_(ts, ch)] += (mag * sig).astype(np.float32)
            elif k == 3:  # dropout_seg
                n = int(self.rng.integers(2, 9))
                e = self._recent_pos(L)
                s = max(0, e - n + 1)
                xb[s:e + 1, ch] = self.raw_zero[ch]
            else:  # noise_burst
                n = int(self.rng.integers(4, 17))
                s = int(self.rng.integers(0, L - n + 1))
                sig = np.maximum(orig[b][:, ch].std(axis=0), 0.1)
                xb[s:s + n, ch] += (self.rng.standard_normal((n, len(ch))) * sig).astype(np.float32)
            types[b] = k
        mask = np.abs(xn - orig) > 1e-6  # points whose value actually changed
        return torch.from_numpy(xn).to(device), torch.from_numpy(mask).to(device), types
