"""Draw the Phase 2 architectures: CNN, U-Net (3 levels, width 32, skip connections) and the hybrid U-Net.

Run: python -m src.dl_arch_diagram   ->  reports/dl/figures/dl_architectures.png
Shapes and parameter counts come from src/dl_models.py (checked below), so the figure cannot drift from the code.
The hybrid is the same U-Net with a 13th input channel: the Phase 1 XGBoost fire probability for each pixel
(out-of-fold on train, src/dl_gbm_oof.py on branch sushant/gbm-oof).
"""
from pathlib import Path

import matplotlib

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


def unet(ax, x0, y0, in_label, in_fill):
    """U-Net drawing with its lower-left corner at (x0, y0); returns nothing."""
    w, h, dx, dy = 1.25, 0.5, 1.45, 0.78
    box(ax, x0, y0 + 2 * dy, 1.1, h, in_label, in_fill, size=6.2)
    enc = [(x0 + 1.35, y0 + 2 * dy, "32 @ 64x64"), (x0 + 1.35 + dx * 0.55, y0 + dy, "64 @ 32x32"),
           (x0 + 1.35 + dx * 1.1, y0, "128 @ 16x16")]
    dec = [(x0 + 1.35 + dx * 1.65, y0 + dy, "64 @ 32x32"), (x0 + 1.35 + dx * 2.2, y0 + 2 * dy, "32 @ 64x64")]
    for bx, by, t in enc:
        box(ax, bx, by, w, h, f"conv block\n{t}", ENC)
    for bx, by, t in dec:
        box(ax, bx, by, w, h, f"conv block\n{t}", DEC)
    hx = dec[1][0] + w + 0.2
    box(ax, hx, y0 + 2 * dy - 0.06, 1.0, h + 0.12, "1x1 conv\nlogits\n1 @ 64x64", INP, size=6.0)
    arrow(ax, (x0 + 1.1, y0 + 2 * dy + h / 2), (enc[0][0], y0 + 2 * dy + h / 2))
    arrow(ax, (dec[1][0] + w, y0 + 2 * dy + h / 2), (hx, y0 + 2 * dy + h / 2))
    for (ax0, ay0, _), (bx0, by0, _) in zip(enc, enc[1:]):            # down: max-pool 2x2
        arrow(ax, (ax0 + w / 2, ay0), (bx0 + w / 2, by0 + h))
    for (ax0, ay0, _), (bx0, by0, _) in zip([enc[2], dec[0]], dec):    # up: transposed conv 2x2
        arrow(ax, (ax0 + w / 2, ay0 + h), (bx0 + w / 2, by0))
    for (ex, ey, _), (qx, qy, _) in zip(enc[:2], dec[::-1]):           # skip connections (concatenate)
        arrow(ax, (ex + w, ey + h / 2), (qx, qy + h / 2), ls=(0, (3, 2)))
    ax.text(enc[1][0] - 0.05, y0 + dy - 0.2, "max-pool 2x2", fontsize=5.8, color=MUTED, ha="right")
    ax.text(dec[0][0] + w + 0.05, y0 + dy - 0.2, "up-conv 2x2", fontsize=5.8, color=MUTED, ha="left")
    ax.text((enc[0][0] + w + dec[1][0]) / 2, y0 + 2 * dy + h / 2 + 0.06, "skip: concatenate", fontsize=5.8,
            color=MUTED, ha="center", va="bottom")


def main():
    cnn, un, hyb = build_model("cnn"), build_model("unet", depth=3, width=32), build_model("unet", in_ch=IN_CH + 1)
    assert IN_CH == 12 and len(un.down) == 3 and un.down[0][0].out_channels == 32 and hyb.down[0][0].in_channels == 13

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.set(xlim=(0, 9.0), ylim=(0.6, 8.6))
    ax.axis("off")

    # (a) CNN
    title(ax, 0.1, 8.25, "(a) CNN baseline", f"4 conv layers, no downsampling; receptive field 9x9 px; "
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
    title(ax, 0.1, 6.35, "(b) U-Net, 3 levels, base width 32",
          f"conv block = 2 x (conv 3x3, BN, ReLU); receptive field 44x44 px; {n_params(un):,} parameters")
    unet(ax, 0.3, 3.75, "input\n12 @ 64x64", INP)

    # (c) Hybrid
    title(ax, 0.1, 3.45, "(c) Hybrid U-Net", f"same U-Net with a 13th input channel: the Phase 1 XGBoost fire "
                                            f"probability per pixel; {n_params(hyb):,} parameters")
    y = 1.9
    box(ax, 0.3, y + 0.45, 1.6, 0.5, "12 input channels\n12 @ 64x64", INP, size=6.2)
    box(ax, 0.3, y - 0.35, 1.6, 0.6, "XGBoost probability\n(out-of-fold on train)\n1 @ 64x64", GBM, size=6.0)
    box(ax, 2.35, y + 0.05, 1.15, 0.5, "concatenate\n13 @ 64x64", INP, size=6.2)
    arrow(ax, (1.9, y + 0.7), (2.35, y + 0.4))
    arrow(ax, (1.9, y - 0.05), (2.35, y + 0.2))
    box(ax, 3.95, y + 0.05, 2.6, 0.5, "U-Net as in (b),\nfirst block takes 13 channels", ENC, size=6.2)
    arrow(ax, (3.5, y + 0.3), (3.95, y + 0.3))
    box(ax, 7.0, y + 0.05, 1.3, 0.5, "logits\n1 @ 64x64", INP, size=6.2)
    arrow(ax, (6.55, y + 0.3), (7.0, y + 0.3))
    ax.text(0.3, 1.2, "Out-of-fold: each training patch gets its probability from an XGBoost fitted on the "
                       "other 4 of 5 patch-grouped folds,\nso the U-Net never sees a probability the GBM learned "
                       "from that same patch. Eval and test use the GBM fitted on all of train.",
            fontsize=6.0, color=MUTED, va="top", linespacing=1.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(OUT.parents[3])}")


if __name__ == "__main__":
    main()
