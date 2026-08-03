import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.data import MTGDataset, VADataset, build_dataloader
from src.losses import GroupBalancedASL, VALoss
from src.models import EnhancedBaselineModel


DATASET_CONFIG = ROOT / "configs" / "dataset.yaml"
TRAIN_CONFIG = ROOT / "configs" / "train.yaml"


def load_yaml(path: Path) -> dict:
    if not path.is_file(): raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file: config = yaml.safe_load(file)
    if not isinstance(config, dict): raise ValueError(f"YAML 格式错误：{path}")
    return config


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(f"[FAIL] {message}")


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，改用 CPU")
        return torch.device("cpu")
    return torch.device(name)


def check_gradients(model: EnhancedBaselineModel, mtg: bool) -> None:
    routes = {
        "segment_projection": True,
        "song_fusion": True,
        "genre_head": mtg,
        "instrument_head": mtg,
        "mood_head": mtg,
        "va_head": not mtg,
    }

    for name, expected in routes.items():
        gradients = [parameter.grad for parameter in getattr(model, name).parameters()]

        if not expected:
            require(all(gradient is None for gradient in gradients), f"{name} 不应产生梯度")
            continue

        require(all(gradient is not None for gradient in gradients), f"{name} 缺少梯度")
        require(all(torch.isfinite(gradient).all().item() for gradient in gradients), f"{name} 梯度包含 NaN 或 Inf")
        require(sum(gradient.float().square().sum() for gradient in gradients).item() > 0, f"{name} 梯度为 0")


def check_batch(name: str, loader, model: EnhancedBaselineModel, tag_loss: GroupBalancedASL, va_loss: VALoss, optimizer, device: torch.device) -> None:
    batch = next(iter(loader))
    features, mask, targets = batch["features"].to(device), batch["segment_mask"].to(device), batch["targets"].to(device)
    outputs = model(features, mask)
    mtg = name == "mtg"

    components = tag_loss(outputs["tag_logits"], targets, True) if mtg else va_loss(outputs["va_predictions"], targets, True)
    loss = components["total"]
    require(loss.ndim == 0 and torch.isfinite(loss).item(), f"{name} loss 不是有限标量")

    before = model.segment_projection[1].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    check_gradients(model, mtg)
    optimizer.step()
    require(not torch.equal(before, model.segment_projection[1].weight.detach()), f"{name} 参数未更新")

    if mtg:
        require(torch.allclose(loss, torch.stack((components["genre"], components["instrument"], components["mood"])).mean()), "MTG 分组平均错误")
        detail = f"genre={components['genre'].item():.6f} | instrument={components['instrument'].item():.6f} | mood={components['mood'].item():.6f}"
    else:
        require(torch.allclose(loss, torch.stack((components["valence"], components["arousal"])).mean()), f"{name} 两轴平均错误")
        detail = f"valence={components['valence'].item():.6f} | arousal={components['arousal'].item():.6f}"

    print(f"[PASS] {name.upper()} | input={tuple(features.shape)} | total={loss.item():.6f} | {detail}")


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    data_config, model_config = train_config["data"], train_config["model"]
    tag_config, va_config = train_config["loss"]["tag"], train_config["loss"]["va"]
    device = resolve_device(str(train_config.get("device", "cpu")))

    model = EnhancedBaselineModel(
        layer_indices=train_config["feature"]["layer_indices"],
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        pooling=str(train_config["feature"].get("pooling", "mean_std")),
        pooling_eps=float(model_config.get("pooling_eps", 1.0e-5)),
    ).to(device).train()

    tag_loss = GroupBalancedASL(
        float(tag_config["gamma_pos"]),
        float(tag_config["gamma_neg"]),
        float(tag_config["clip"]),
        float(tag_config["eps"]),
    )
    va_loss = VALoss(float(va_config["beta"]))

    optimizer_config = train_config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_config["learning_rate"]), weight_decay=float(optimizer_config["weight_decay"]))

    datasets = {
        "mtg": MTGDataset(dataset_config["mtg"], "train"),
        "deam": VADataset(dataset_config["deam"], "deam", "train"),
        "pmemo": VADataset(dataset_config["pmemo"], "pmemo", "train"),
    }
    loaders = {
        name: build_dataloader(dataset, int(data_config["batch_size"]), True, int(data_config["num_workers"]), bool(data_config["pin_memory"]))
        for name, dataset in datasets.items()
    }

    print(f"device={device} | ASL=({tag_loss.gamma_pos},{tag_loss.gamma_neg},{tag_loss.clip}) | SmoothL1={va_loss.beta}")
    for name in ("mtg", "deam", "pmemo"): check_batch(name, loaders[name], model, tag_loss, va_loss, optimizer, device)
    print("[PASS] loss、梯度路由和参数更新检查全部通过")


if __name__ == "__main__":
    main()
