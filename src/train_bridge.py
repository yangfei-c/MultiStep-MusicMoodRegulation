"""冻结教师上的交叉拟合有界 Tag→VA 残差桥实验入口。"""

import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import VADataset, build_dataloader
from src.metrics import compute_va_metrics
from src.models import EnhancedBaselineModel
from src.models.bridge import apply_bridge, fit_cross_fitted_bridge
from src.training.common import atomic_json_save, get_device, load_yaml, set_seed


SEED = 42
FOLDS = 5
RIDGE_ALPHA = 1.0
DELTA_MAX = 0.25
RHO_MAX = 0.5
TEMPERATURE = 0.08
CONTROLS = ("none", "mood_permutation", "va_permutation")
EVALUATE_TEST = False
MTG_CHECKPOINT = None
VA_CHECKPOINT = None
DATASET_CONFIG = ROOT / "configs/dataset.yaml"


def latest_best(roots: tuple[Path, ...]) -> Path:
    candidates = [path for root in roots for path in root.glob("*/best.pt") if "smoke" not in path.parent.name.lower()]
    if not candidates:
        raise FileNotFoundError("没有找到正式 best.pt：" + ", ".join(map(str, roots)))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint(explicit, roots: tuple[Path, ...]) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return latest_best(roots)


def build_from_checkpoint(checkpoint: dict, device: torch.device) -> EnhancedBaselineModel:
    config, model_config = checkpoint["train_config"], checkpoint["train_config"]["model"]
    model = EnhancedBaselineModel(
        layer_indices=config["feature"]["layer_indices"],
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        pooling=str(config["feature"].get("pooling", "mean_std")),
        pooling_eps=float(model_config.get("pooling_eps", 1.0e-5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


@torch.no_grad()
def collect(dataset, tag_teacher, va_model, device, batch_size: int, workers: int) -> dict:
    loader = build_dataloader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    mood, direct, targets, ids = [], [], [], []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=device.type == "cuda")
        mask = batch["segment_mask"].to(device, non_blocking=device.type == "cuda")
        mood.append(tag_teacher(features, mask)["mood_logits"].float().sigmoid().cpu())
        direct.append(va_model(features, mask)["va_predictions"].float().cpu())
        targets.append(batch["targets"].float())
        ids.extend(batch["ids"])
    return {
        "mood": torch.cat(mood).numpy(),
        "direct": np.clip(torch.cat(direct).numpy(), -1, 1),
        "targets": torch.cat(targets).numpy(),
        "ids": np.asarray(ids),
    }


def evaluate_split(parameters: dict, data: dict) -> tuple[dict, dict]:
    prediction, gate, delta = apply_bridge(parameters, data["mood"], data["direct"])
    report = {
        "direct": compute_va_metrics(data["direct"], data["targets"]),
        "bridge": compute_va_metrics(prediction, data["targets"]),
        "mean_gate": float(gate.mean()),
        "coverage_gate_gt_0_25": float(np.mean(gate > 0.25)),
    }
    payload = {"ids": data["ids"], "targets": data["targets"], "direct": data["direct"], "bridge": prediction, "gate": gate, "delta": delta}
    return report, payload


def main() -> None:
    set_seed(SEED)
    dataset_config = load_yaml(DATASET_CONFIG)
    mtg_path = resolve_checkpoint(MTG_CHECKPOINT, (ROOT / "outputs/tag/mtg", ROOT / "outputs/mtg"))
    va_path = resolve_checkpoint(VA_CHECKPOINT, (ROOT / "outputs/va/joint", ROOT / "outputs/va_domains/joint"))
    tag_checkpoint = torch.load(mtg_path, map_location="cpu", weights_only=False)
    va_checkpoint = torch.load(va_path, map_location="cpu", weights_only=False)
    device = get_device(str(va_checkpoint["train_config"].get("device", "cpu")))
    tag_teacher = build_from_checkpoint(tag_checkpoint, device)
    va_model = build_from_checkpoint(va_checkpoint, device)
    data_config = va_checkpoint["train_config"]["data"]
    datasets = {
        name: {split: VADataset(dataset_config[name], name, split) for split in ("train", "validation", "test")}
        for name in ("deam", "pmemo")
    }
    collected = {
        name: {
            split: collect(dataset, tag_teacher, va_model, device, int(data_config["batch_size"]), int(data_config["num_workers"]))
            for split, dataset in sets.items()
        }
        for name, sets in datasets.items()
    }
    train_mood = np.concatenate([collected[name]["train"]["mood"] for name in ("deam", "pmemo")])
    train_direct = np.concatenate([collected[name]["train"]["direct"] for name in ("deam", "pmemo")])
    train_targets = np.concatenate([collected[name]["train"]["targets"] for name in ("deam", "pmemo")])
    domains = np.concatenate([np.full(len(collected[name]["train"]["targets"]), name) for name in ("deam", "pmemo")])

    output_root = ROOT / "outputs/bridge/cfrbr" / f"s{SEED}_{time.strftime('%Y%m%d-%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=False)
    atomic_json_save(output_root / "sources.json", {"mtg_checkpoint": str(mtg_path), "va_checkpoint": str(va_path), "test_locked": not EVALUATE_TEST})
    for control in CONTROLS:
        output_dir = output_root / control
        output_dir.mkdir()
        parameters, oof = fit_cross_fitted_bridge(
            train_mood, train_direct, train_targets, domains,
            folds=FOLDS, alpha=RIDGE_ALPHA, delta_max=DELTA_MAX, rho_max=RHO_MAX,
            temperature=TEMPERATURE, seed=SEED, control=control,
        )
        np.savez_compressed(output_dir / "bridge_parameters.npz", **{key: value for key, value in parameters.items() if key != "control"})
        report = {
            "oof_train": {
                "direct": oof["direct"],
                "bridge": oof["bridge_oof"],
                "selected_tau": oof["selected_tau"],
                "mean_gate": oof["mean_gate"],
                "coverage_gate_gt_0_25": oof["coverage_gate_gt_0_25"],
            }
        }
        for name in ("deam", "pmemo"):
            metrics, payload = evaluate_split(parameters, collected[name]["validation"])
            report[f"validation_{name}"] = metrics
            np.savez_compressed(output_dir / f"validation_predictions_{name}.npz", **payload)
            if EVALUATE_TEST:
                metrics, payload = evaluate_split(parameters, collected[name]["test"])
                report[f"test_{name}"] = metrics
                np.savez_compressed(output_dir / f"test_predictions_{name}.npz", **payload)
        atomic_json_save(output_dir / "metrics.json", report)
        print(f"[{control}] OOF CCC {oof['direct']['va_score']:.4f}->{oof['bridge_oof']['va_score']:.4f} | tau={oof['selected_tau']:.2f}")
    if not EVALUATE_TEST:
        print("[TEST LOCKED] Bridge 仅用 train 拟合并在 validation 比较。")


if __name__ == "__main__":
    main()
