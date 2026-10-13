"""Forecast dependency of each input point, estimated with Hutchinson probes.

D_t = E_v[(∂(vᵀ f(x)) / ∂x_t)²] == Σ_j J_{j,t}²  (column energy of the Jacobian), for v with E[vvᵀ] = I.
Rademacher probes (default) have lower variance than Gaussian ones.

RADAR uses a *global-scale* normalisation: D̃_p = D_p / D̄, where D̄ is a scalar EMA of the backbone's mean patch
dependency on the training set. This keeps absolute size and cross-channel differences (unlike a per-sample max
normalisation, which always maps the most-used patch to 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _probe(shape, device, dtype, generator, dist):
    if dist == 'gaussian':
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    v = torch.randint(0, 2, shape, device=device, generator=generator)
    return (v * 2 - 1).to(dtype)


def hutchinson_dependency(backbone, x, cycle_index, pred_len, n_probes=4, generator=None, dist='rademacher',
                          forward_fn=None):
    """Returns D with the shape of x (B, L, C), detached.

    * forward_fn(z) -> forecast; defaults to backbone(z, cycle_index) (tuple outputs: first element);
    * gradients are taken w.r.t. the input only (torch.autograd.grad) -> no parameter .grad is touched;
    * the backbone is temporarily switched to eval() (dropout off) and its mode is restored;
    * works inside an outer torch.no_grad() thanks to torch.enable_grad().
    """
    if forward_fn is None:
        def forward_fn(z):
            out = backbone(z, cycle_index)
            return out[0] if isinstance(out, (tuple, list)) else out
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            out = forward_fn(x_req)[:, -pred_len:, :]
            D = torch.zeros_like(x_req)
            for k in range(n_probes):
                v = _probe(out.shape, out.device, out.dtype, generator, dist)
                (g,) = torch.autograd.grad((out * v).sum(), x_req, retain_graph=k < n_probes - 1)
                D = D + g.detach() ** 2
            D = D / n_probes
    finally:
        backbone.train(was_training)
    return D.detach()


def patch_sum(D, P):
    """D: (B, L, C) -> D_p (B, C, N): sum over each patch (left zero padding when L % P != 0)."""
    pad = (-D.shape[1]) % P
    Dc = D.permute(0, 2, 1)
    if pad:
        Dc = F.pad(Dc, (pad, 0))
    B, C, Lp = Dc.shape
    return Dc.reshape(B, C, Lp // P, P).sum(-1)


def patch_dependency(D, P):
    """(legacy) per-sample max-normalised profile: returns (D_p / max_p D_p, D_p)."""
    Dp = patch_sum(D, P)
    return Dp / (Dp.amax(dim=-1, keepdim=True) + 1e-12), Dp


class DependencyProfile(nn.Module):
    """EMA of the absolute patch dependency (N,) of one backbone, updated from batch means during training."""

    def __init__(self, n_patches, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('ema', torch.ones(n_patches))
        self.register_buffer('initialized', torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, Dp):
        m = Dp.mean(dim=(0, 1)).to(self.ema.dtype)
        if not bool(self.initialized):
            self.ema.copy_(m)
            self.initialized.fill_(True)
        else:
            self.ema.mul_(self.momentum).add_(m, alpha=1.0 - self.momentum)

    def scale(self):
        """D̄: scalar mean patch dependency of this backbone."""
        return self.ema.mean().clamp_min(1e-12)

    def normalized(self):
        """EMA profile in units of D̄ (mean 1)."""
        return self.ema / self.scale()

    def profile(self):  # kept for backwards compatibility
        return self.normalized()
