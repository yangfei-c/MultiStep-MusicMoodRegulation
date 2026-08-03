import torch

from src.losses import GroupBalancedASL, VALoss
from src.models import EnhancedBaselineModel


def build_model(config: dict, device: torch.device) -> EnhancedBaselineModel:
    model = config["model"]
    return EnhancedBaselineModel(
        layer_indices=config["feature"]["layer_indices"],
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        pooling=str(config["feature"].get("pooling", "mean_std")),
        pooling_eps=float(model.get("pooling_eps", 1.0e-5)),
    ).to(device)


def build_tag_loss(config: dict) -> GroupBalancedASL:
    loss = config["loss"]["tag"]
    if str(loss.get("name", "asymmetric_loss")).lower() not in {"asymmetric_loss", "asl"}:
        raise ValueError("当前标签主训练只支持 group-balanced ASL")
    return GroupBalancedASL(
        gamma_pos=float(loss["gamma_pos"]),
        gamma_neg=float(loss["gamma_neg"]),
        clip=float(loss["clip"]),
        eps=float(loss["eps"]),
    )


def build_va_loss(config: dict) -> torch.nn.Module:
    loss = config["loss"]["va"]
    name = str(loss.get("name", "smooth_l1")).lower()
    if name in {"smooth_l1", "huber"}:
        return VALoss(beta=float(loss.get("beta", 0.1)))
    if name == "mse":
        return torch.nn.MSELoss()
    if name in {"mae", "l1"}:
        return torch.nn.L1Loss()
    raise ValueError("VA loss 只支持 smooth_l1/mse/mae")


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer = config["optimizer"]
    if str(optimizer.get("name", "adamw")).lower() != "adamw":
        raise ValueError("当前训练基础设施只支持 AdamW")
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
    )
