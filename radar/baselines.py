"""B3: a generic learned denoising front-end in front of the frozen backbone.

It is the same-information control for the RADAR plug-in: trained on exactly the same corruption mixture, with the same
clean targets and the same frozen backbone, at the same scale as RADAR's Repairer (patch Transformer, d = radar_d,
radar_layers layers) — but without detection, experts, routing or a gate. Robust scaling -> channel-independent patch
Transformer -> residual correction in robust units (zero-initialised head: identity at the start) -> frozen backbone.
"""
import torch
import torch.nn as nn

from radar.modules import from_patches, robust_stats, to_patches


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class DenoiseFront(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        g = lambda k, d: getattr(cfg, k, d)  # noqa: E731
        self.backbone = backbone
        self.P = int(g('radar_patch', 8))
        self.N = -(-int(cfg.seq_len) // self.P)
        d = int(g('radar_d', 64))
        self.std_frac = float(g('scale_std_frac', 0.1))
        self.s_min = float(g('scale_min', 1e-2))
        self.u_clip = float(g('u_clip', 10.0))
        self.embed = nn.Linear(self.P, d)
        self.pos = nn.Parameter(torch.zeros(1, self.N, d))
        layer = nn.TransformerEncoderLayer(d, int(g('radar_heads', 4)), int(g('radar_dff', 128)),
                                           float(g('radar_dropout', 0.1)), batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(g('radar_layers', 2)))
        self.head = nn.Linear(d, self.P)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def front_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def clean(self, x):
        """x (B, L, C) -> denoised x̂ (B, L, C) = x + Δ·s (Δ in robust units of the window)."""
        B, L, C = x.shape
        m, s = robust_stats(x, std_frac=self.std_frac, s_min=self.s_min)
        u = ((x - m) / s).clamp(-self.u_clip, self.u_clip)
        up, pad = to_patches(u, self.P)  # (BC, N, P)
        h = self.encoder(self.embed(up) + self.pos)
        du = self.head(h)
        return x + from_patches(du, B, C, pad) * s, s

    def forward(self, x, *bb_args):
        x_hat, _ = self.clean(x)
        return _first(self.backbone(x_hat, *bb_args))
