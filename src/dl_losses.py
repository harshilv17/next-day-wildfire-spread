"""Phase 2 losses. Each takes logits (B, 1, H, W), labels y (B, H, W) in {-1, 0, 1} and pos_weight,
and ignores pixels labelled -1.

Use: from src.dl_losses import LOSSES
     loss = LOSSES["focal"](logits, y, pos_weight)

bce:      binary cross-entropy, fire pixels weighted by pos_weight (the Phase 2 default).
focal:    the same weighted BCE, each pixel scaled by (1 - p_t)**2, where p_t is the probability given
          to the true class (Lin et al. 2017, gamma = 2). Pixels the model already gets right count less.
          This is alpha-balanced focal loss with alpha set by the class ratio, so it differs from bce
          only in the focusing term.
bce_dice: weighted BCE + soft Dice loss (Milletari et al. 2016), the sum used by nnU-Net. Dice is
          computed over all labelled pixels of the batch, not per patch, because many patches have no
          fire on day t+1 and per-patch Dice would be 0/0 there.
"""
import torch
import torch.nn.functional as F

GAMMA = 2.0
SMOOTH = 1.0  # Dice smoothing; keeps the loss defined for a batch without fire


def _labelled(logits, y):
    keep = y != -1
    return logits.squeeze(1)[keep], y[keep]


def bce(logits, y, pos_weight):
    z, t = _labelled(logits, y)
    return F.binary_cross_entropy_with_logits(z, t, pos_weight=torch.tensor(pos_weight, device=y.device))


def focal(logits, y, pos_weight, gamma=GAMMA):
    z, t = _labelled(logits, y)
    ce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    p_t = torch.exp(-ce)  # probability given to the true class
    weight = 1 + (pos_weight - 1) * t  # pos_weight on fire pixels, 1 elsewhere
    return (weight * (1 - p_t) ** gamma * ce).mean()


def dice(logits, y):
    z, t = _labelled(logits, y)
    p = torch.sigmoid(z)
    return 1 - (2 * (p * t).sum() + SMOOTH) / (p.sum() + t.sum() + SMOOTH)


def bce_dice(logits, y, pos_weight):
    return bce(logits, y, pos_weight) + dice(logits, y)


LOSSES = {"bce": bce, "focal": focal, "bce_dice": bce_dice}


if __name__ == "__main__":
    torch.manual_seed(0)
    y = (torch.rand(4, 64, 64) < 0.05).float()
    y[:, :8] = -1
    logits = torch.randn(4, 1, 64, 64, requires_grad=True)
    w = 90.0
    for name, loss in LOSSES.items():
        value = loss(logits, y, w)
        value.backward()
        assert torch.isfinite(value), name
        assert logits.grad[:, 0][y == -1].abs().max() == 0, f"{name} uses pixels labelled -1"
        changed = logits.detach().clone()
        changed[:, 0][y == -1] = 50.0
        assert torch.equal(loss(changed, y, w), value.detach()), f"{name} depends on pixels labelled -1"
        logits.grad = None
    assert torch.allclose(focal(logits, y, w, gamma=0), bce(logits, y, w)), "focal with gamma 0 must equal bce"
    assert focal(logits, y, w) < bce(logits, y, w)
    sure = (2 * y - 1).clamp(min=-1).unsqueeze(1) * 20  # confident and correct
    assert dice(sure, y) < 1e-3 and dice(-sure, y) > 0.99
    no_fire = torch.zeros(2, 64, 64)
    assert all(torch.isfinite(loss(torch.randn(2, 1, 64, 64), no_fire, w)) for loss in LOSSES.values())
    print("dl_losses self-check ok")
