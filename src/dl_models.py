"""Phase 2 models. Both map (B, in_ch, H, W) -> (B, 1, H, W) logits (no sigmoid).

Use: from src.dl_models import build_model
     model = build_model("unet", depth=3, width=32)   # or build_model("cnn")
H and W must be divisible by 2**(depth-1) for the U-Net (64 and 32 work up to depth 4/5).
"""
import torch
from torch import nn

IN_CH = 12


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    """Plain conv stack, no downsampling. Receptive field is only 2*layers+1 pixels (9 by default): local context only."""

    def __init__(self, in_ch=IN_CH, width=32, layers=4):
        super().__init__()
        chans = [in_ch] + [width] * layers
        self.body = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
            for a, b in zip(chans, chans[1:])
        ])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        return self.head(self.body(x))


class UNet(nn.Module):
    """U-Net (Ronneberger 2015): depth levels, channels width * 2**level, skip connections."""

    def __init__(self, in_ch=IN_CH, width=32, depth=3):
        super().__init__()
        widths = [width * 2**i for i in range(depth)]
        self.down = nn.ModuleList([block(a, b) for a, b in zip([in_ch] + widths[:-1], widths)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(depth - 1)])
        self.dec = nn.ModuleList([block(2 * widths[i], widths[i]) for i in range(depth - 1)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else self.pool(x))
            skips.append(x)
        for i in reversed(range(len(self.up))):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)


def build_model(name, **kw):
    return {"cnn": CNN, "unet": UNet}[name](**kw)


if __name__ == "__main__":
    for size in (64, 32):
        x = torch.randn(2, IN_CH, size, size)
        for name, kw in [("cnn", {}), ("unet", {"depth": 2}), ("unet", {"depth": 3, "width": 16}), ("unet", {"depth": 4})]:
            m = build_model(name, **kw)
            assert m(x).shape == (2, 1, size, size), (name, kw, size)
    n = lambda m: sum(p.numel() for p in m.parameters())
    print("params cnn", n(build_model("cnn")), "unet", n(build_model("unet")), "dl_models self-check ok")
