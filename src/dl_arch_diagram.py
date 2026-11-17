"""Draw the Phase 2 architectures: CNN, the final U-Net (4 levels, width 32, skip connections) and the hybrid U-Net.

Run: python -m src.dl_arch_diagram   ->  reports/dl/figures/dl_architectures.png
Shapes, parameter counts and receptive fields come from src/dl_models.py (built and measured below), so the
figure cannot drift from the code. The hybrid is the same U-Net with a 13th input channel: the Phase 1 XGBoost
fire probability for each pixel (out-of-fold on train). It brings no measurable gain
(reports/dl/hybrid_ensemble_summary.md) and is not the final model.
"""
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.dl_models import IN_CH, build_model

OUT = Path(__file__).resolve().parent.parent / "reports" / "dl" / "figures" / "dl_architectures.png"
INK, MUTED, LINE = "#212529", "#6C757D", "#868E96"
ENC, DEC, INP, GBM = "#DCEAF5", "#F9E0D0", "#EEF0F2", "#FCEFC7"   # light fills; text stays in ink


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def box(ax, x, y, w, h, text, fill, size=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=LINE, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, a, b, style="-|>", ls="-", color=LINE, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, lw=0.9, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def title(ax, x, y, text, sub):
    ax.text(x, y, text, fontsize=8, fontweight="bold", color=INK, va="bottom")
    ax.text(x, y - 0.22, sub, fontsize=6.4, color=MUTED, va="top")


def receptive_field(m, size=256):
    """Smallest and largest side of the input window one output pixel depends on, measured from input
    gradients at 16 neighbouring positions (it varies with the position relative to the pooling grid)."""
    torch.manual_seed(0)
    m.eval()
    x = torch.randn(1, IN_CH, size, size, requires_grad=True)
    y = m(x)
    sides = []
    for p in range(size // 2 - 8, size // 2 + 8):
        x.grad = None
        y[0, 0, p, p].backward(retain_graph=True)
        rows = (x.grad.abs().sum((0, 1)) > 0).any(1).nonzero()
        sides.append(int(rows.max() - rows.min() + 1))
    lo, hi = min(sides), max(sides)
    return f"{lo}x{lo}" if lo == hi else f"{lo}-{hi}"


def unet(ax, x0, y0, in_label, in_fill, widths, size=64):
    """U-Net drawing with its lower-left corner at (x0, y0); one encoder block per entry of widths."""
    depth = len(widths)
    w, h, dx, dy = 1.25, 0.5, 0.8, 0.78
    top = y0 + (depth - 1) * dy
    box(ax, x0, top, 1.1, h, in_label, in_fill, size=6.2)
    enc = [(x0 + 1.35 + i * dx, top - i * dy, f"{c} @ {size >> i}x{size >> i}") for i, c in enumerate(widths)]
    dec = [(x0 + 1.35 + (depth - 1 + j) * dx, top - (depth - 1 - j) * dy,
            f"{widths[depth - 1 - j]} @ {size >> (depth - 1 - j)}x{size >> (depth - 1 - j)}") for j in range(1, depth)]
    for bx, by, t in enc:
        box(ax, bx, by, w, h, f"conv block\n{t}", ENC)
    for bx, by, t in dec:
        box(ax, bx, by, w, h, f"conv block\n{t}", DEC)
    hx = dec[-1][0] + w + 0.2
    box(ax, hx, top - 0.06, 1.0, h + 0.12, "1x1 conv\nlogits\n1 @ 64x64", INP, size=6.0)
    arrow(ax, (x0 + 1.1, top + h / 2), (enc[0][0], top + h / 2))
    arrow(ax, (dec[-1][0] + w, top + h / 2), (hx, top + h / 2))
    for (ax0, ay0, _), (bx0, by0, _) in zip(enc, enc[1:]):            # down: max-pool 2x2
        arrow(ax, (ax0 + w / 2, ay0), (bx0 + w / 2, by0 + h))
    for (ax0, ay0, _), (bx0, by0, _) in zip([enc[-1]] + dec[:-1], dec):  # up: transposed conv 2x2
        arrow(ax, (ax0 + w / 2, ay0 + h), (bx0 + w / 2, by0))
    for (ex, ey, _), (qx, qy, _) in zip(enc[:-1], dec[::-1]):          # skip connections (concatenate)
        arrow(ax, (ex + w, ey + h / 2), (qx, qy + h / 2), ls=(0, (3, 2)))
    ax.text(enc[1][0] - 0.05, enc[1][1] - 0.2, "max-pool 2x2", fontsize=5.8, color=MUTED, ha="right")
    ax.text(dec[-2][0] + w + 0.05, dec[-2][1] - 0.2, "up-conv 2x2", fontsize=5.8, color=MUTED, ha="left")
    ax.text((enc[0][0] + w + dec[-1][0]) / 2, top + h / 2 + 0.06, "skip: concatenate", fontsize=5.8,
            color=MUTED, ha="center", va="bottom")


def main():
    depth, width = 4, 32                                   # final Phase 2 model (unet_final_s0..s2)
    cnn = build_model("cnn")
    un = build_model("unet", depth=depth, width=width)
    hyb = build_model("unet", depth=depth, width=width, in_ch=IN_CH + 1)
    widths = [blk[0].out_channels for blk in un.down]
    assert IN_CH == 12 and widths == [32, 64, 128, 256] and hyb.down[0][0].in_channels == 13
    rf_cnn, rf_un = receptive_field(cnn), receptive_field(un)

    fig, ax = plt.subplots(figsize=(7.0, 5.9))
    ax.set(xlim=(0, 9.0), ylim=(-0.5, 8.6))
    ax.axis("off")

    # (a) CNN
    title(ax, 0.1, 8.25, "(a) CNN baseline", f"4 conv layers, no downsampling; receptive field {rf_cnn} px; "
                                             f"{n_params(cnn):,} parameters")
    y = 7.0
    box(ax, 0.3, y, 1.1, 0.62, "input\n12 @ 64x64", INP, size=6.2)
    xs = [1.75 + i * 1.45 for i in range(4)]
    for i, x in enumerate(xs):
        box(ax, x, y, 1.15, 0.62, "conv 3x3\nBN, ReLU\n32 @ 64x64", ENC, size=6.0)
        arrow(ax, ((xs[i - 1] + 1.15) if i else 1.4, y + 0.31), (x, y + 0.31))
    box(ax, xs[-1] + 1.5, y, 1.0, 0.62, "1x1 conv\nlogits\n1 @ 64x64", INP, size=6.0)
    arrow(ax, (xs[-1] + 1.15, y + 0.31), (xs[-1] + 1.5, y + 0.31))

    # (b) U-Net
    title(ax, 0.1, 6.35, f"(b) U-Net, {depth} levels, base width {width} (final model)",
          f"conv block = 2 x (conv 3x3, BN, ReLU); receptive field {rf_un} px (whole 64x64 patch around the "
          f"fire); {n_params(un) / 1e6:.2f} M parameters")
    unet(ax, 0.3, 2.97, "input\n12 @ 64x64", INP, widths)

    # (c) Hybrid
    title(ax, 0.1, 2.35, "(c) Hybrid U-Net (tested, no measurable gain)",
          f"same U-Net with a 13th input channel: the Phase 1 XGBoost fire probability per pixel; "
          f"{n_params(hyb) / 1e6:.2f} M parameters")
    y = 0.8
    box(ax, 0.3, y + 0.45, 1.6, 0.5, "12 input channels\n12 @ 64x64", INP, size=6.2)
    box(ax, 0.3, y - 0.35, 1.6, 0.6, "XGBoost probability\n(out-of-fold on train)\n1 @ 64x64", GBM, size=6.0)
    box(ax, 2.35, y + 0.05, 1.15, 0.5, "concatenate\n13 @ 64x64", INP, size=6.2)
    arrow(ax, (1.9, y + 0.7), (2.35, y + 0.4))
    arrow(ax, (1.9, y - 0.05), (2.35, y + 0.2))
    box(ax, 3.95, y + 0.05, 2.6, 0.5, "U-Net as in (b),\nfirst block takes 13 channels", ENC, size=6.2)
    arrow(ax, (3.5, y + 0.3), (3.95, y + 0.3))
    box(ax, 7.0, y + 0.05, 1.3, 0.5, "logits\n1 @ 64x64", INP, size=6.2)
    arrow(ax, (6.55, y + 0.3), (7.0, y + 0.3))
    ax.text(0.3, 0.1, "Out-of-fold: each training patch gets its probability from an XGBoost fitted on the "
                       "other 4 of 5 fire-event-grouped folds,\nso the U-Net never sees a probability the GBM learned "
                       "from that same fire event. Eval and test use the GBM fitted on all of train.",
            fontsize=6.0, color=MUTED, va="top", linespacing=1.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(OUT.parents[3])}")


if __name__ == "__main__":
    main()
