import torch
from torch import nn
from torch.nn import functional as F


class GroupBalancedASL(nn.Module):
    """Genre、instrument、mood/theme 分组等权 ASL。"""

    GROUPS = {"genre": slice(0, 87), "instrument": slice(87, 127), "mood": slice(127, 183)}

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1.0e-8) -> None:
        super().__init__()
        if gamma_pos < 0 or gamma_neg < 0 or clip < 0 or eps <= 0: raise ValueError("ASL 参数错误")
        self.gamma_pos, self.gamma_neg, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps

    def asl(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = logits.float().sigmoid()
        neg = 1.0 - pos
        if self.clip: neg = (neg + self.clip).clamp(max=1.0)

        targets = targets.float()
        probability = pos * targets + neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        log_probability = targets * pos.clamp_min(self.eps).log() + (1.0 - targets) * neg.clamp_min(self.eps).log()
        return -(log_probability * (1.0 - probability).pow(gamma)).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, return_components: bool = False):
        if logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != 183:
            raise ValueError(f"logits 和 targets 应为 [B,183]，实际为 {tuple(logits.shape)}、{tuple(targets.shape)}")

        losses = {name: self.asl(logits[:, group], targets[:, group]) for name, group in self.GROUPS.items()}
        total = torch.stack(tuple(losses.values())).mean()
        return {"total": total, **losses} if return_components else total


class VALoss(nn.Module):
    """Valence 和 Arousal 等权 Smooth L1。"""

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        if beta < 0: raise ValueError("beta 不能小于 0")
        self.beta = beta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, return_components: bool = False):
        if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
            raise ValueError(f"predictions 和 targets 应为 [B,2]，实际为 {tuple(predictions.shape)}、{tuple(targets.shape)}")

        axis = F.smooth_l1_loss(predictions.float(), targets.float(), beta=self.beta, reduction="none").mean(0)
        total = axis.mean()
        return {"total": total, "valence": axis[0], "arousal": axis[1]} if return_components else total