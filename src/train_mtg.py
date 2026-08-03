import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data import MTGDataset, build_dataloader
from src.losses import GroupBalancedASL
from src.metrics import compute_mtg_metrics
from src.model import EnhancedBaselineModel


DATASET_CONFIG = root / "configs" / "dataset.yaml"
TRAIN_CONFIG = root / "configs" / "train.yaml"


class Tee:
    """同时把 stdout/stderr 写到控制台和日志文件。"""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


# ------------------------------ 基础工具 ------------------------------


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"YAML 格式错误：{path}")
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
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def scalar_metrics(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            number = float(value)
            result[key] = number if math.isfinite(number) else None
    return result


def atomic_json_save(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(content, temporary)
    os.replace(temporary, path)


def gpu_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    return f"{torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f}GB"


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        print("[WARN] checkpoint 没有 RNG 状态，续训结果不能做到严格复现。", flush=True)
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError as error:
            print(f"[WARN] CUDA RNG 状态未恢复：{error}", flush=True)


def nonfinite_batch_message(batch: dict, features: torch.Tensor, targets: torch.Tensor) -> str:
    problems = []
    if not torch.isfinite(features).all().item():
        problems.append("features 含 NaN/Inf")
    if not torch.isfinite(targets).all().item():
        problems.append("targets 含 NaN/Inf")
    ids = batch.get("ids", [])
    return f"歌曲 IDs={ids}; " + ("；".join(problems) if problems else "输入本身为有限值")


def nonfinite_gradient_names(model: torch.nn.Module, limit: int = 8) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            names.append(name)
            if len(names) >= limit:
                break
    return names


# ------------------------------ checkpoint ------------------------------


def resolve_resume_path(value: str | None) -> Path | None:
    if value is None:
        return None

    if value.lower() == "latest":
        candidates = list((root / "outputs" / "mtg").glob("*/last.pt"))
        if not candidates:
            raise FileNotFoundError("没有找到 outputs/mtg/*/last.pt")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(f"断点文件不存在：{path}")
    return path


def build_checkpoint(
    *,
    epoch: int,
    target_epochs: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score: float,
    bad_epochs: int,
    history: list[dict],
    dataset_config: dict,
    train_config: dict,
) -> dict:
    return {
        "format_version": 2,
        "epoch": epoch,
        "target_epochs": target_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_score": best_score,
        "bad_epochs": bad_epochs,
        "history": history,
        "rng_state": capture_rng_state(),
        "dataset_config": dataset_config,
        "train_config": train_config,
    }


def load_resume_state(
    checkpoint: dict,
    model,
    optimizer,
    scheduler,
    scaler,
) -> tuple[int, float, int, list[dict]]:
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "best_score",
        "bad_epochs",
        "history",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "该文件不是完整断点，缺少："
            + ", ".join(missing)
            + "。旧版 best.pt 只能配合 --init-weights 使用，不能精确 --resume。"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"))

    start_epoch = int(checkpoint["epoch"]) + 1
    return (
        start_epoch,
        float(checkpoint["best_score"]),
        int(checkpoint["bad_epochs"]),
        list(checkpoint["history"]),
    )


# ------------------------------ train / eval ------------------------------


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
    max_consecutive_amp_skips: int,
    max_batches: int | None,
    print_every: int,
) -> dict:
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    loss_sum = 0.0
    sample_count = 0
    skipped_steps = 0
    consecutive_skips = 0
    start = time.perf_counter()

    print(f"  [TRAIN] 开始训练，共 {total_batches} batches，正在读取第一个 batch...", flush=True)

    for step, batch in enumerate(loader, 1):
        if step > total_batches:
            break

        features, mask, targets = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(features, mask)["tag_logits"]
            loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            detail = nonfinite_batch_message(batch, features, targets)
            raise RuntimeError(f"训练 loss 为 NaN/Inf：{loss.item()}；{detail}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        if not torch.isfinite(gradient_norm).item() and not amp_enabled:
            names = nonfinite_gradient_names(model)
            detail = nonfinite_batch_message(batch, features, targets)
            raise RuntimeError(f"非 AMP 训练出现 NaN/Inf 梯度：{names}；{detail}")

        scale_before = scaler.get_scale()
        scaler.step(optimizer)  # AMP 梯度异常时由 GradScaler 自动跳过 optimizer.step()
        scaler.update()
        scale_after = scaler.get_scale()

        step_skipped = amp_enabled and scale_after < scale_before
        if step_skipped:
            skipped_steps += 1
            consecutive_skips += 1
            names = nonfinite_gradient_names(model)
            detail = nonfinite_batch_message(batch, features, targets)
            print(
                f"  [WARN] step={step} 检测到 AMP 梯度溢出，已跳过参数更新；"
                f"scale {scale_before:g} -> {scale_after:g}；参数={names}；{detail}",
                flush=True,
            )
            if consecutive_skips >= max_consecutive_amp_skips:
                raise RuntimeError(
                    f"连续 {consecutive_skips} 个 batch 发生 AMP 梯度溢出，"
                    "已停止训练，避免长期静默跳步。"
                )
        else:
            consecutive_skips = 0
            scheduler.step()  # 只有参数真正更新时才推进学习率

        batch_size = features.shape[0]
        loss_sum += loss.detach().item() * batch_size
        sample_count += batch_size

        if step == 1 or step % print_every == 0 or step == total_batches:
            elapsed = time.perf_counter() - start
            print(
                f"  [TRAIN] {step:4d}/{total_batches} | "
                f"loss={loss.item():.6f} | avg={loss_sum / sample_count:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"skipped={skipped_steps} | "
                f"{sample_count / elapsed:.1f} songs/s | gpu_mem={gpu_memory(device)}",
                flush=True,
            )

    return {
        "loss": loss_sum / sample_count,
        "samples": sample_count,
        "seconds": time.perf_counter() - start,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "amp_skipped_steps": skipped_steps,
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
    loss_sum = 0.0
    sample_count = 0
    logits_list, targets_list = [], []
    start = time.perf_counter()

    print(f"  [{stage}] 开始评估，共 {total_batches} batches，正在读取第一个 batch...", flush=True)

    for step, batch in enumerate(loader, 1):
        if step > total_batches:
            break

        features, mask, targets = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(features, mask)["tag_logits"]
            loss = criterion(logits, targets)

        if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
            detail = nonfinite_batch_message(batch, features, targets)
            raise RuntimeError(f"{stage} 出现 NaN/Inf 输出；{detail}")

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
    metrics["loss"] = loss_sum / sample_count
    metrics["samples"] = sample_count
    return metrics, logits, targets


# ------------------------------ main ------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="只跑少量 batch，检查完整流程")
    parser.add_argument("--epochs", type=int, default=None, help="目标总 epoch 数，不是额外 epoch 数")
    parser.add_argument("--resume", type=str, default=None, help="last.pt、运行目录，或 latest")
    parser.add_argument("--init-weights", type=str, default=None, help="只加载模型权重并开始一次新训练")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    args = parser.parse_args()

    if args.resume and args.init_weights:
        parser.error("--resume 与 --init-weights 不能同时使用")
    return args


def run(args: argparse.Namespace, output_dir: Path, resume_checkpoint: dict | None) -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    data_config, model_config = train_config["data"], train_config["model"]
    loss_config, optimizer_config = train_config["loss"]["tag"], train_config["optimizer"]
    training_config, scheduler_config = train_config["training"], train_config.get("scheduler", {})

    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    device = get_device(str(train_config.get("device", "cpu")))

    saved_target_epochs = resume_checkpoint.get("target_epochs") if resume_checkpoint else None
    epochs = args.epochs or saved_target_epochs or int(training_config["max_epochs"])
    max_train_batches, max_eval_batches = args.max_train_batches, args.max_eval_batches
    print_every = args.print_every or 100

    if args.smoke_test:
        epochs = args.epochs or 2
        max_train_batches = args.max_train_batches or 5
        max_eval_batches = args.max_eval_batches or 5
        print_every = args.print_every or 1

    if resume_checkpoint and args.epochs and saved_target_epochs and args.epochs != saved_target_epochs:
        print(
            f"[WARN] 断点原目标为 {saved_target_epochs} epochs，本次改为 {args.epochs}；"
            "学习率余弦轨迹将按新总步数构建，不能视为严格等价续训。",
            flush=True,
        )

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
    scheduler = make_scheduler(
        optimizer,
        epochs * train_batches,
        float(scheduler_config.get("warmup_ratio", 0.05)),
        float(scheduler_config.get("min_learning_rate", 1.0e-6)),
    )

    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    max_amp_skips = int(training_config.get("max_consecutive_amp_skips", 8))
    patience = int(training_config.get("early_stopping_patience", 10))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.001))

    start_epoch, best_score, bad_epochs, history = 1, -math.inf, 0, []

    if resume_checkpoint is not None:
        start_epoch, best_score, bad_epochs, history = load_resume_state(
            resume_checkpoint, model, optimizer, scheduler, scaler
        )
        print(
            f"[RESUME] 已恢复 epoch={start_epoch - 1}，将从 epoch={start_epoch} 继续；"
            f"best_score={best_score:.6f}，bad_epochs={bad_epochs}",
            flush=True,
        )
    elif args.init_weights:
        weights_path = Path(args.init_weights).expanduser().resolve()
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"[INIT] 已从 {weights_path} 加载模型权重。优化器、调度器、AMP 和历史均重新开始。",
            flush=True,
        )

    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    run_type = "smoke" if args.smoke_test else "full"

    print("\n================ MTG TRAINING ================", flush=True)
    print(f"mode={run_type} | device={device} | gpu={gpu_name}", flush=True)
    print(f"seed={seed} | parameters={parameters:,} | amp={amp_enabled}", flush=True)
    print(f"train/val/test={len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}", flush=True)
    print(f"batch={data_config['batch_size']} | workers={data_config['num_workers']} | epochs={epochs}", flush=True)
    print(f"train_batches/epoch={train_batches} | output={output_dir}", flush=True)

    if int(data_config["num_workers"]) == 0:
        print("[WARN] num_workers=0，主进程将串行读取 MERT .npy 文件，GPU 可能等待数据。", flush=True)

    config_path = output_dir / "run_config.yaml"
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {"dataset": dataset_config, "train": train_config},
                file,
                allow_unicode=True,
                sort_keys=False,
            )

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    if start_epoch > epochs:
        print(f"[INFO] checkpoint 已完成 {start_epoch - 1} epochs，目标总数仅为 {epochs}，跳过训练。", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[EPOCH {epoch}/{epochs}] 开始", flush=True)

        train_metrics = train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            amp_enabled,
            clip_norm,
            max_amp_skips,
            max_train_batches,
            print_every,
        )
        val_metrics, _, _ = evaluate(
            "VAL",
            model,
            loaders["validation"],
            criterion,
            device,
            amp_enabled,
            max_eval_batches,
            max(1, print_every * 2),
        )

        score = -val_metrics["loss"] if args.smoke_test else float(val_metrics["tag_score"])
        if not math.isfinite(score):
            raise RuntimeError("validation 选模指标为 NaN/Inf")

        improved = score > best_score + min_delta
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": scalar_metrics(val_metrics),
            "best_score": best_score,
        }
        history.append(record)
        atomic_json_save(output_dir / "history.json", history)

        checkpoint = build_checkpoint(
            epoch=epoch,
            target_epochs=epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_score=best_score,
            bad_epochs=bad_epochs,
            history=history,
            dataset_config=dataset_config,
            train_config=train_config,
        )
        atomic_torch_save(last_path, checkpoint)

        if improved:
            atomic_torch_save(best_path, checkpoint)
            print(f"[BEST] epoch={epoch}，已保存最佳 checkpoint：{best_path}", flush=True)

        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"genre_mAP={val_metrics['genre_map']:.4f} | "
            f"instrument_mAP={val_metrics['instrument_map']:.4f} | "
            f"mood_mAP={val_metrics['mood_map']:.4f} | "
            f"tag_score={val_metrics['tag_score']:.4f} | "
            f"AMP_skips={train_metrics['amp_skipped_steps']} | "
            f"bad_epochs={bad_epochs}/{patience}",
            flush=True,
        )
        print(f"[CHECKPOINT] 已保存最近完整断点：{last_path}", flush=True)

        if bad_epochs >= patience:
            print(f"[EARLY STOP] 连续 {patience} 个 epoch 未提升。", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("没有生成最佳 checkpoint")

    print("\n[TEST] 正在加载最佳 checkpoint...", flush=True)
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics, test_logits, test_targets = evaluate(
        "TEST",
        model,
        loaders["test"],
        criterion,
        device,
        amp_enabled,
        max_eval_batches,
        max(1, print_every * 2),
    )
    atomic_json_save(output_dir / "test_metrics.json", scalar_metrics(test_metrics))
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        logits=test_logits,
        targets=test_targets,
        per_label_ap=test_metrics["per_label_ap"],
        per_label_roc_auc=test_metrics["per_label_roc_auc"],
    )

    print("\n================ FINAL TEST ================", flush=True)
    print(f"best_epoch={best_checkpoint['epoch']} | loss={test_metrics['loss']:.6f}", flush=True)
    print(f"genre_mAP={test_metrics['genre_map']:.4f}", flush=True)
    print(f"instrument_mAP={test_metrics['instrument_map']:.4f}", flush=True)
    print(f"mood_mAP={test_metrics['mood_map']:.4f}", flush=True)
    print(f"tag_score={test_metrics['tag_score']:.4f}", flush=True)
    print(f"overall_mAP={test_metrics['overall_map']:.4f}", flush=True)
    print(f"[PASS] 结果已保存：{output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    resume_path = resolve_resume_path(args.resume)
    resume_checkpoint = (
        torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None
    )

    if resume_path:
        output_dir = resume_path.parent
    else:
        train_config = load_yaml(TRAIN_CONFIG)
        seed = int(train_config.get("seed", 42))
        run_type = "smoke" if args.smoke_test else "full"
        run_time = time.strftime("%Y%m%d_%H%M%S")
        output_dir = root / "outputs" / "mtg" / f"{run_type}_seed{seed}_{run_time}"
        output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print("\n" + "=" * 72, flush=True)
            print(time.strftime("[RUN] %Y-%m-%d %H:%M:%S"), flush=True)
            if resume_path:
                print(f"[RUN] resume={resume_path}", flush=True)
            run(args, output_dir, resume_checkpoint)
        except Exception:
            print("\n[FAILED] 训练异常终止。", flush=True)
            last_path = output_dir / "last.pt"
            if last_path.is_file():
                print(
                    f"[RESUME COMMAND] python src/train_mtg.py --resume \"{last_path}\"",
                    flush=True,
                )
            else:
                print("[RESUME] 尚无 last.pt，只能重新训练或用 --init-weights 加载旧权重。", flush=True)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()