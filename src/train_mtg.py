import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.data import MTGDataset, build_dataloader
from src.losses import GroupBalancedASL
from src.metrics import compute_mtg_metrics
from src.model import EnhancedBaselineModel


DATASET_CONFIG = ROOT / "configs" / "dataset.yaml"
TRAIN_CONFIG = ROOT / "configs" / "train.yaml"


def load_yaml(path: Path) -> dict:
    if not path.is_file(): raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file: config = yaml.safe_load(file)
    if not isinstance(config, dict): raise ValueError(f"YAML 格式错误：{path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动改用 CPU", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return (
        batch["features"].to(device, non_blocking=non_blocking),
        batch["segment_mask"].to(device, non_blocking=non_blocking),
        batch["targets"].to(device, non_blocking=non_blocking),
    )


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr: float):
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_steps = int(total_steps * warmup_ratio)
    min_ratio = min(1.0, min_lr / base_lr)

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps: return max((step + 1) / warmup_steps, 1.0e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def scalar_metrics(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray): continue
        if isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            result[key] = float(value) if math.isfinite(float(value)) else None
    return result


def save_json(path: Path, content) -> None:
    with path.open("w", encoding="utf-8") as file: json.dump(content, file, ensure_ascii=False, indent=2)


def gpu_memory(device: torch.device) -> str:
    if device.type != "cuda": return "CPU"
    return f"{torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f}GB"


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    amp_enabled: bool,
    clip_norm: float,
    max_batches: int | None,
    print_every: int,
) -> dict:
    model.train()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    loss_sum = sample_count = 0
    start = time.perf_counter()

    print(f"  [TRAIN] 开始训练，共 {total_batches} batches，正在读取第一个 batch...", flush=True)

    for step, batch in enumerate(loader, 1):
        if step > total_batches: break

        features, mask, targets = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            losses = criterion(model(features, mask)["tag_logits"], targets, return_components=True)
            loss = losses["total"]

        if not torch.isfinite(loss).item(): raise RuntimeError(f"训练 loss 为 NaN/Inf：{loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if not torch.isfinite(gradient_norm).item(): raise RuntimeError("梯度范数为 NaN/Inf")

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = features.shape[0]
        loss_sum += loss.detach().item() * batch_size
        sample_count += batch_size

        if step == 1 or step % print_every == 0 or step == total_batches:
            elapsed = time.perf_counter() - start
            print(
                f"  [TRAIN] {step:4d}/{total_batches} | "
                f"loss={loss.item():.6f} | avg={loss_sum / sample_count:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"{sample_count / elapsed:.1f} songs/s | gpu_mem={gpu_memory(device)}",
                flush=True,
            )

    return {
        "loss": loss_sum / sample_count,
        "samples": sample_count,
        "seconds": time.perf_counter() - start,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(
    stage: str,
    model,
    loader,
    criterion,
    device,
    amp_enabled: bool,
    max_batches: int | None,
    print_every: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    loss_sum = sample_count = 0
    logits_list, targets_list = [], []
    start = time.perf_counter()

    print(f"  [{stage}] 开始评估，共 {total_batches} batches，正在读取第一个 batch...", flush=True)

    for step, batch in enumerate(loader, 1):
        if step > total_batches: break

        features, mask, targets = move_batch(batch, device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(features, mask)["tag_logits"]
            loss = criterion(logits, targets)

        batch_size = features.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        logits_list.append(logits.float().cpu())
        targets_list.append(targets.float().cpu())

        if step == 1 or step % print_every == 0 or step == total_batches:
            elapsed = time.perf_counter() - start
            print(
                f"  [{stage}] {step:4d}/{total_batches} | "
                f"avg_loss={loss_sum / sample_count:.6f} | "
                f"{sample_count / elapsed:.1f} songs/s",
                flush=True,
            )

    logits = torch.cat(logits_list).numpy()
    targets = torch.cat(targets_list).numpy()

    print(f"  [{stage}] 正在计算 {sample_count} 首歌曲的 mAP/ROC-AUC...", flush=True)
    metrics = compute_mtg_metrics(logits, targets)
    metrics["loss"], metrics["samples"] = loss_sum / sample_count, sample_count
    return metrics, logits, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="只跑少量 batch，检查完整流程")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    data_config, model_config = train_config["data"], train_config["model"]
    loss_config, optimizer_config = train_config["loss"]["tag"], train_config["optimizer"]
    training_config, scheduler_config = train_config["training"], train_config.get("scheduler", {})

    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    device = get_device(str(train_config.get("device", "cpu")))

    epochs = args.epochs or int(training_config["max_epochs"])
    max_train_batches, max_eval_batches = args.max_train_batches, args.max_eval_batches
    print_every = args.print_every or 100

    if args.smoke_test:
        epochs = args.epochs or 2
        max_train_batches = args.max_train_batches or 5
        max_eval_batches = args.max_eval_batches or 5
        print_every = args.print_every or 1

    run_type = "smoke" if args.smoke_test else "full"
    run_time = time.strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / "mtg" / f"{run_type}_seed{seed}_{run_time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[SETUP] 正在建立 MTG train/validation/test 数据集...", flush=True)
    datasets = {
        "train": MTGDataset(dataset_config["mtg"], "train"),
        "validation": MTGDataset(dataset_config["mtg"], "validation"),
        "test": MTGDataset(dataset_config["mtg"], "test"),
    }

    loaders = {
        split: build_dataloader(
            dataset,
            batch_size=int(data_config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"]),
        )
        for split, dataset in datasets.items()
    }

    print("[SETUP] 正在建立模型、损失、优化器和调度器...", flush=True)
    model = EnhancedBaselineModel(
        layer_indices=train_config["feature"]["layer_indices"],
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        pooling_eps=float(model_config.get("pooling_eps", 1.0e-5)),
    ).to(device)

    criterion = GroupBalancedASL(
        gamma_pos=float(loss_config["gamma_pos"]),
        gamma_neg=float(loss_config["gamma_neg"]),
        clip=float(loss_config["clip"]),
        eps=float(loss_config["eps"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )

    train_batches = min(len(loaders["train"]), max_train_batches) if max_train_batches else len(loaders["train"])
    total_steps = epochs * train_batches
    scheduler = make_scheduler(
        optimizer,
        total_steps,
        float(scheduler_config.get("warmup_ratio", 0.05)),
        float(scheduler_config.get("min_learning_rate", 1.0e-6)),
    )

    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    patience = int(training_config.get("early_stopping_patience", 10))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.001))

    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    print("\n================ MTG TRAINING ================", flush=True)
    print(f"mode={run_type} | device={device} | gpu={gpu_name}", flush=True)
    print(f"seed={seed} | parameters={parameters:,} | amp={amp_enabled}", flush=True)
    print(f"train/val/test={len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}", flush=True)
    print(f"batch={data_config['batch_size']} | workers={data_config['num_workers']} | epochs={epochs}", flush=True)
    print(f"train_batches/epoch={train_batches} | output={output_dir}", flush=True)

    if int(data_config["num_workers"]) == 0:
        print("[WARN] num_workers=0，主进程将串行读取 MERT .npy 文件，GPU 可能等待数据。", flush=True)

    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump({"dataset": dataset_config, "train": train_config}, file, allow_unicode=True, sort_keys=False)

    best_score, bad_epochs = -math.inf, 0
    best_path = output_dir / "best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n[EPOCH {epoch}/{epochs}] 开始", flush=True)

        train_metrics = train_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, scaler, device,
            amp_enabled, clip_norm, max_train_batches, print_every,
        )

        val_metrics, _, _ = evaluate(
            "VAL", model, loaders["validation"], criterion, device,
            amp_enabled, max_eval_batches, max(1, print_every * 2),
        )

        # smoke test 的少量验证样本不用于正式 mAP 选模
        score = -val_metrics["loss"] if args.smoke_test else float(val_metrics["tag_score"])
        if not math.isfinite(score): raise RuntimeError("validation 选模指标为 NaN/Inf")

        improved = score > best_score + min_delta
        if improved:
            best_score, bad_epochs = score, 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "score": score,
                    "validation": scalar_metrics(val_metrics),
                    "train_config": train_config,
                },
                best_path,
            )
            print(f"[BEST] epoch={epoch}，已保存最佳 checkpoint：{best_path}", flush=True)
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": scalar_metrics(val_metrics),
            "best_score": best_score,
        }
        history.append(record)
        save_json(output_dir / "history.json", history)

        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"genre_mAP={val_metrics['genre_map']:.4f} | "
            f"instrument_mAP={val_metrics['instrument_map']:.4f} | "
            f"mood_mAP={val_metrics['mood_map']:.4f} | "
            f"tag_score={val_metrics['tag_score']:.4f} | "
            f"bad_epochs={bad_epochs}/{patience}",
            flush=True,
        )

        if bad_epochs >= patience:
            print(f"[EARLY STOP] 连续 {patience} 个 epoch 未提升。", flush=True)
            break

    if not best_path.is_file(): raise RuntimeError("没有生成最佳 checkpoint")

    print("\n[TEST] 正在加载最佳 checkpoint...", flush=True)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_logits, test_targets = evaluate(
        "TEST", model, loaders["test"], criterion, device,
        amp_enabled, max_eval_batches, max(1, print_every * 2),
    )

    save_json(output_dir / "test_metrics.json", scalar_metrics(test_metrics))
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        logits=test_logits,
        targets=test_targets,
        per_label_ap=test_metrics["per_label_ap"],
        per_label_roc_auc=test_metrics["per_label_roc_auc"],
    )

    print("\n================ FINAL TEST ================", flush=True)
    print(f"best_epoch={checkpoint['epoch']} | loss={test_metrics['loss']:.6f}", flush=True)
    print(f"genre_mAP={test_metrics['genre_map']:.4f}", flush=True)
    print(f"instrument_mAP={test_metrics['instrument_map']:.4f}", flush=True)
    print(f"mood_mAP={test_metrics['mood_map']:.4f}", flush=True)
    print(f"tag_score={test_metrics['tag_score']:.4f}", flush=True)
    print(f"overall_mAP={test_metrics['overall_map']:.4f}", flush=True)
    print(f"[PASS] 结果已保存：{output_dir}", flush=True)


if __name__ == "__main__":
    main()