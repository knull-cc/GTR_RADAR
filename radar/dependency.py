"""Forecast dependency of each input point, estimated with Hutchinson probes.

D_t = E_v[(∂(vᵀ f(x, cycle_index)) / ∂x_t)²],  v ~ N(0, I)  ==  Σ_j J_{j,t}²  (column energy of the Jacobian).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def hutchinson_dependency(backbone, x, cycle_index, pred_len, n_probes=1, generator=None):
    """Returns D with the shape of x (B, L, C), detached.

    * gradients are taken w.r.t. the input only (torch.autograd.grad) -> no parameter .grad is touched;
    * the backbone is temporarily switched to eval() (GTR uses dropout 0.5) and its mode is restored;
    * works inside an outer torch.no_grad() thanks to torch.enable_grad().
    """
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            out = backbone(x_req, cycle_index)[:, -pred_len:, :]
            D = torch.zeros_like(x_req)
            for k in range(n_probes):
                v = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
                (g,) = torch.autograd.grad((out * v).sum(), x_req, retain_graph=k < n_probes - 1)
                D = D + g.detach() ** 2
            D = D / n_probes
    finally:
        backbone.train(was_training)
    return D.detach()


def patch_dependency(D, P):
    """D: (B, L, C) -> (dbar (B, C, N) in [0, 1], D_p (B, C, N)).

    D_p = Σ_{t in patch p} D_t (left zero padding when L % P != 0); dbar = D_p / max_p D_p per sample & channel.
    """
    pad = (-D.shape[1]) % P
    Dc = D.permute(0, 2, 1)
    if pad:
        Dc = F.pad(Dc, (pad, 0))
    B, C, Lp = Dc.shape
    Dp = Dc.reshape(B, C, Lp // P, P).sum(-1)
    dbar = Dp / (Dp.amax(dim=-1, keepdim=True) + 1e-12)
    return dbar, Dp


class DependencyProfile(nn.Module):
    """EMA dependency profile over patches (N,), updated from batch means of dbar during training."""

    def __init__(self, n_patches, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('ema', torch.ones(n_patches))
        self.register_buffer('initialized', torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, dbar):
        m = dbar.mean(dim=(0, 1)).to(self.ema.dtype)
        if not bool(self.initialized):
            self.ema.copy_(m)
            self.initialized.fill_(True)
        else:
            self.ema.mul_(self.momentum).add_(m, alpha=1.0 - self.momentum)

    def profile(self):
        return self.ema / self.ema.max().clamp_min(1e-12)
