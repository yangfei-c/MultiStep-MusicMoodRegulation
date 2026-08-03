from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.data import MTGDataset, build_dataloader
from src.losses import GroupBalancedASL
from src.metrics import compute_mtg_metrics
from src.model import EnhancedBaselineModel
from src.training.common import get_device, load_yaml, logged_output, set_seed
from src.training.engine import RunOptions, TaskSpec, prepare_run, run_experiment


# 日常运行只改这里；模型、损失和优化器参数位于 configs/train.yaml。
RUN = RunOptions(smoke_test=False, target_epochs=None, resume_from=None, init_weights=None, output_dir=None, max_train_batches=None, max_eval_batches=None, print_every=None)
DATASET_CONFIG, TRAIN_CONFIG = ROOT / "configs/dataset.yaml", ROOT / "configs/train.yaml"


def prediction_payload(logits, targets, metrics) -> dict:
    return {"logits": logits, "targets": targets, "per_label_ap": metrics["per_label_ap"], "per_label_roc_auc": metrics["per_label_roc_auc"]}


TASK = TaskSpec(
    name="mtg", output_key="tag_logits", score_key="tag_score", metric_fn=compute_mtg_metrics,
    metric_description="mAP/ROC-AUC", default_print_every=100,
    metric_fields=(("genre_mAP", "genre_map"), ("instrument_mAP", "instrument_map"), ("mood_mAP", "mood_map"), ("tag_score", "tag_score"), ("overall_mAP", "overall_map"), ("overall_AUC", "overall_roc_auc")),
    prediction_payload=prediction_payload,
)


def build_data(dataset_config: dict, train_config: dict) -> tuple[dict, dict]:
    data = train_config["data"]
    datasets = {split: MTGDataset(dataset_config["mtg"], split) for split in ("train", "validation", "test")}
    loaders = {split: build_dataloader(dataset, batch_size=int(data["batch_size"]), shuffle=split == "train", num_workers=int(data["num_workers"]), pin_memory=bool(data["pin_memory"])) for split, dataset in datasets.items()}
    return datasets, loaders


def build_objects(train_config: dict, device: torch.device):
    model_config, loss_config, optimizer_config = train_config["model"], train_config["loss"]["tag"], train_config["optimizer"]
    model = EnhancedBaselineModel(layer_indices=train_config["feature"]["layer_indices"], hidden_dim=int(model_config["hidden_dim"]), dropout=float(model_config["dropout"]), pooling_eps=float(model_config.get("pooling_eps", 1e-5))).to(device)
    criterion = GroupBalancedASL(gamma_pos=float(loss_config["gamma_pos"]), gamma_neg=float(loss_config["gamma_neg"]), clip=float(loss_config["clip"]), eps=float(loss_config["eps"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_config["learning_rate"]), weight_decay=float(optimizer_config["weight_decay"]))
    return model, criterion, optimizer


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    seed = int(train_config.get("seed", 42)); set_seed(seed); device = get_device(str(train_config.get("device", "cpu")))
    output_dir, resume_path, checkpoint = prepare_run(ROOT / "outputs/mtg", RUN, seed)

    with logged_output(output_dir / "train.log"):
        try:
            print("\n" + "=" * 72); print(time.strftime("[RUN] %Y-%m-%d %H:%M:%S"))
            if resume_path: print(f"[RUN] resume={resume_path}")
            datasets, loaders = build_data(dataset_config, train_config)
            model, criterion, optimizer = build_objects(train_config, device)
            run_experiment(task=TASK, options=RUN, output_dir=output_dir, resume_checkpoint=checkpoint, dataset_config=dataset_config, train_config=train_config, run_config={"dataset": dataset_config, "train": train_config}, datasets=datasets, loaders=loaders, model=model, criterion=criterion, optimizer=optimizer, device=device)
        except Exception:
            print("\n[FAILED] 训练异常终止。", flush=True)
            last_path = output_dir / "last.pt"
            print(f'将 RUN.resume_from 设为 r"{last_path}" 后重新运行。' if last_path.is_file() else "尚无 last.pt，只能重新训练或设置 RUN.init_weights。", flush=True)
            raise


if __name__ == "__main__": main()
