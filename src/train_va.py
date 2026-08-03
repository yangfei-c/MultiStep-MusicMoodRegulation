from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.data import VADataset, build_dataloader
from src.losses import VALoss
from src.metrics import compute_va_metrics
from src.model import EnhancedBaselineModel
from src.training.common import get_device, load_yaml, logged_output, set_seed
from src.training.engine import RunOptions, TaskSpec, prepare_run, run_experiment


# 日常运行只改这里；DATASET_NAME 可选 "deam" 或 "pmemo"。
DATASET_NAME = "pmemo"
RUN = RunOptions(smoke_test=False, target_epochs=None, resume_from=None, init_weights=None, output_dir=None, max_train_batches=None, max_eval_batches=None, print_every=None)
DATASET_CONFIG, TRAIN_CONFIG = ROOT / "configs/dataset.yaml", ROOT / "configs/train.yaml"


def prediction_payload(predictions, targets, _) -> dict:
    return {"predictions": predictions, "targets": targets, "predictions_1_9": predictions * 4 + 5, "targets_1_9": targets * 4 + 5}


def task_spec(dataset_name: str) -> TaskSpec:
    return TaskSpec(
        name="va", dataset_name=dataset_name, output_key="va_predictions", score_key="va_score",
        metric_fn=compute_va_metrics, metric_description="CCC/RMSE/MAE", default_print_every=50,
        metric_fields=(("CCC_v", "valence_ccc"), ("CCC_a", "arousal_ccc"), ("VA_score", "va_score"), ("RMSE", "mean_rmse"), ("MAE", "mean_mae")),
        prediction_payload=prediction_payload,
    )


def build_data(dataset_config: dict, train_config: dict, dataset_name: str) -> tuple[dict, dict]:
    data = train_config["data"]
    datasets = {split: VADataset(dataset_config[dataset_name], dataset_name, split) for split in ("train", "validation", "test")}
    loaders = {split: build_dataloader(dataset, batch_size=int(data["batch_size"]), shuffle=split == "train", num_workers=int(data["num_workers"]), pin_memory=bool(data["pin_memory"])) for split, dataset in datasets.items()}
    return datasets, loaders


def build_criterion(config: dict) -> torch.nn.Module:
    name = str(config.get("name", "smooth_l1")).lower()
    if name in {"smooth_l1", "huber"}: return VALoss(beta=float(config.get("beta", .1)))
    if name == "mse": return torch.nn.MSELoss()
    if name in {"mae", "l1"}: return torch.nn.L1Loss()
    raise ValueError(f"不支持的 VA loss：{name}；可选 smooth_l1、mse、mae")


def build_objects(train_config: dict, device: torch.device):
    model_config, optimizer_config = train_config["model"], train_config["optimizer"]
    model = EnhancedBaselineModel(layer_indices=train_config["feature"]["layer_indices"], hidden_dim=int(model_config["hidden_dim"]), dropout=float(model_config["dropout"]), pooling_eps=float(model_config.get("pooling_eps", 1e-5))).to(device)
    for head in (model.genre_head, model.instrument_head, model.mood_head):
        for parameter in head.parameters(): parameter.requires_grad_(False)
    criterion = build_criterion(train_config["loss"]["va"])
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(optimizer_config["learning_rate"]), weight_decay=float(optimizer_config["weight_decay"]))
    return model, criterion, optimizer


def target_range(dataset: VADataset) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray([sample[2] for sample in dataset.samples], dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all(): raise ValueError("VA targets 不是有限的 [N,2] 数组")
    if (targets < -1).any() or (targets > 1).any(): raise ValueError("VA targets 超出 (y-5)/4 后的 [-1,1] 范围")
    return targets.min(0), targets.max(0)


def main() -> None:
    dataset_name = DATASET_NAME.lower()
    if dataset_name not in {"deam", "pmemo"}: raise ValueError('DATASET_NAME 只能是 "deam" 或 "pmemo"')
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    seed = int(train_config.get("seed", 42)); set_seed(seed); device = get_device(str(train_config.get("device", "cpu")))
    output_dir, resume_path, checkpoint = prepare_run(ROOT / "outputs/va" / dataset_name, RUN, seed)

    with logged_output(output_dir / "train.log"):
        try:
            print("\n" + "=" * 72); print(time.strftime("[RUN] %Y-%m-%d %H:%M:%S"))
            if resume_path: print(f"[RUN] resume={resume_path}")
            datasets, loaders = build_data(dataset_config, train_config, dataset_name)
            minimum, maximum = target_range(datasets["train"]); model, criterion, optimizer = build_objects(train_config, device)
            setup = (f"dataset={dataset_name}", f"normalization=(y-5)/4 | train v=[{minimum[0]:.4f},{maximum[0]:.4f}] | a=[{minimum[1]:.4f},{maximum[1]:.4f}]")
            run_config = {"active_dataset": dataset_name, "va_normalization": "(y - 5) / 4", "dataset": dataset_config, "train": train_config}
            run_experiment(task=task_spec(dataset_name), options=RUN, output_dir=output_dir, resume_checkpoint=checkpoint, dataset_config=dataset_config, train_config=train_config, run_config=run_config, datasets=datasets, loaders=loaders, model=model, criterion=criterion, optimizer=optimizer, device=device, setup_lines=setup)
        except Exception:
            print("\n[FAILED] 训练异常终止。", flush=True)
            last_path = output_dir / "last.pt"
            print(f'将 RUN.resume_from 设为 r"{last_path}" 后重新运行。' if last_path.is_file() else "尚无 last.pt，只能重新训练或设置 RUN.init_weights。", flush=True)
            raise


if __name__ == "__main__": main()
