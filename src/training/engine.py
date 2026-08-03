from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.training.checkpoint import atomic_torch_save, build_checkpoint, load_resume_state, resolve_resume_path
from src.training.common import (
    atomic_json_save,
    gpu_memory,
    make_scheduler,
    move_batch,
    nonfinite_batch_message,
    nonfinite_gradient_names,
    save_yaml_once,
    scalar_metrics,
)


@dataclass
class RunOptions:
    """脚本顶部变量式运行选项；不使用命令行参数解析。"""

    smoke_test: bool = False
    target_epochs: int | None = None
    resume_from: str | Path | None = None
    init_weights: str | Path | None = None
    output_dir: str | Path | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    print_every: int | None = None
    evaluate_test: bool = False

    def validate(self) -> None:
        if self.resume_from and self.init_weights:
            raise ValueError("resume_from 和 init_weights 不能同时设置")
        if self.resume_from and self.output_dir:
            raise ValueError("续训沿用原目录，不能同时设置 output_dir")
        for name in ("target_epochs", "max_train_batches", "max_eval_batches", "print_every"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} 必须大于 0")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    output_key: str
    score_key: str
    metric_fn: Callable[[np.ndarray, np.ndarray], dict]
    metric_description: str
    metric_fields: tuple[tuple[str, str], ...]
    prediction_payload: Callable[[np.ndarray, np.ndarray, dict], dict]
    dataset_name: str | None = None
    default_print_every: int = 100


def prepare_run(outputs_root: Path, options: RunOptions, seed: int) -> tuple[Path, Path | None, dict | None]:
    options.validate()
    resume_path = resolve_resume_path(options.resume_from, outputs_root)
    if resume_path:
        return resume_path.parent, resume_path, torch.load(resume_path, map_location="cpu", weights_only=False)
    if options.output_dir:
        output_dir = Path(options.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        prefix = "smoke_" if options.smoke_test else ""
        output_dir = outputs_root / f"{prefix}s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, None, None


def train_epoch(
    *, model, loader, criterion, output_key: str, optimizer, scheduler, scaler, device,
    amp_enabled: bool, clip_norm: float, max_consecutive_amp_skips: int,
    max_batches: int | None, print_every: int,
) -> dict:
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total = min(len(loader), max_batches) if max_batches else len(loader)
    loss_sum = sample_count = skipped_steps = consecutive_skips = 0
    started = time.perf_counter()
    print(f"  [TRAIN] 共 {total} batches", flush=True)
    for step, batch in enumerate(loader, 1):
        if step > total:
            break
        features, mask, targets = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss = criterion(model(features, mask)[output_key], targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"训练 loss 为 {loss.item()}；{nonfinite_batch_message(batch, features, targets)}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if not amp_enabled and not torch.isfinite(gradient_norm).item():
            raise RuntimeError(f"梯度含 NaN/Inf：{nonfinite_gradient_names(model)}")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = amp_enabled and scaler.get_scale() < scale_before
        if skipped:
            skipped_steps += 1
            consecutive_skips += 1
            print(f"  [WARN] step={step} AMP overflow，scale {scale_before:g}->{scaler.get_scale():g}", flush=True)
            if consecutive_skips >= max_consecutive_amp_skips:
                raise RuntimeError(f"连续 {consecutive_skips} 个 batch 发生 AMP overflow")
        else:
            consecutive_skips = 0
            scheduler.step()
        batch_size = features.shape[0]
        loss_sum += loss.detach().item() * batch_size
        sample_count += batch_size
        if step == 1 or step % print_every == 0 or step == total:
            elapsed = time.perf_counter() - started
            print(
                f"  [TRAIN] {step:4d}/{total} | loss={loss.item():.6f} | avg={loss_sum / sample_count:.6f} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e} | skipped={skipped_steps} "
                f"| {sample_count / elapsed:.1f} songs/s | gpu_mem={gpu_memory(device)}",
                flush=True,
            )
    return {
        "loss": loss_sum / sample_count,
        "samples": sample_count,
        "seconds": time.perf_counter() - started,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "amp_skipped_steps": skipped_steps,
    }


@torch.no_grad()
def evaluate(
    *, stage: str, model, loader, criterion, task: TaskSpec, device,
    amp_enabled: bool, max_batches: int | None, print_every: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    total = min(len(loader), max_batches) if max_batches else len(loader)
    loss_sum = sample_count = 0
    predictions, targets = [], []
    started = time.perf_counter()
    print(f"  [{stage}] 共 {total} batches", flush=True)
    for step, batch in enumerate(loader, 1):
        if step > total:
            break
        features, mask, target = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(features, mask)[task.output_key]
            loss = criterion(prediction, target)
        if not torch.isfinite(loss).item() or not torch.isfinite(prediction).all().item():
            raise RuntimeError(f"{stage} 出现 NaN/Inf；{nonfinite_batch_message(batch, features, target)}")
        batch_size = features.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        predictions.append(prediction.float().cpu())
        targets.append(target.float().cpu())
        if step == 1 or step % print_every == 0 or step == total:
            elapsed = time.perf_counter() - started
            print(f"  [{stage}] {step:4d}/{total} | avg_loss={loss_sum / sample_count:.6f} | {sample_count / elapsed:.1f} songs/s", flush=True)
    prediction_array, target_array = torch.cat(predictions).numpy(), torch.cat(targets).numpy()
    metrics = task.metric_fn(prediction_array, target_array)
    metrics.update(loss=loss_sum / sample_count, samples=sample_count)
    return metrics, prediction_array, target_array


def metric_text(metrics: dict, fields: tuple[tuple[str, str], ...]) -> str:
    return " | ".join(f"{label}={metrics[key]:.4f}" for label, key in fields)


def run_experiment(
    *, task: TaskSpec, options: RunOptions, output_dir: Path, resume_checkpoint: dict | None,
    dataset_config: dict, train_config: dict, run_config: dict, datasets: dict, loaders: dict,
    model, criterion, optimizer, device: torch.device, setup_lines: tuple[str, ...] = (),
) -> dict:
    data_config, training_config = train_config["data"], train_config["training"]
    saved_epochs = resume_checkpoint.get("target_epochs") if resume_checkpoint else None
    epochs = int(options.target_epochs or saved_epochs or training_config["max_epochs"])
    max_train, max_eval = options.max_train_batches, options.max_eval_batches
    print_every = options.print_every or task.default_print_every
    if options.smoke_test:
        epochs, max_train, max_eval, print_every = options.target_epochs or 2, max_train or 3, max_eval or 2, options.print_every or 1
    if resume_checkpoint and resume_checkpoint.get("task_name") not in (None, task.name):
        raise ValueError(f"checkpoint 不是 {task.name} 训练断点")
    if resume_checkpoint and task.dataset_name and resume_checkpoint.get("dataset_name") not in (None, task.dataset_name):
        raise ValueError(f"checkpoint 数据集不是 {task.dataset_name}")

    train_batches = min(len(loaders["train"]), max_train) if max_train else len(loaders["train"])
    scheduler_config = train_config.get("scheduler", {})
    scheduler = make_scheduler(
        optimizer,
        epochs * train_batches,
        float(scheduler_config.get("warmup_ratio", 0.05)),
        float(scheduler_config.get("min_learning_rate", 1.0e-6)),
    )
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    max_skips = int(training_config.get("max_consecutive_amp_skips", 8))
    patience = int(training_config.get("early_stopping_patience", 8))
    min_delta = float(training_config.get("early_stopping_min_delta", 5.0e-4))

    start_epoch, best_score, bad_epochs, history = 1, -math.inf, 0, []
    if resume_checkpoint:
        if options.target_epochs and saved_epochs and options.target_epochs != saved_epochs:
            print(f"[WARN] 目标轮数从 {saved_epochs} 改为 {options.target_epochs}，余弦轨迹不再严格等价。", flush=True)
        start_epoch, best_score, bad_epochs, history = load_resume_state(
            resume_checkpoint, model, optimizer, scheduler, scaler
        )
        print(f"[RESUME] epoch={start_epoch - 1} -> {start_epoch} | best={best_score:.6f}", flush=True)
    elif options.init_weights:
        weights_path = Path(options.init_weights).expanduser().resolve()
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False)["model_state_dict"])
        print(f"[INIT] 已加载模型权重：{weights_path}", flush=True)

    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"\n================ {task.name.upper()} TRAINING ================", flush=True)
    print(f"device={device} | gpu={gpu} | parameters={parameters:,} | amp={amp_enabled}", flush=True)
    print(f"train/val/test={len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}", flush=True)
    for line in setup_lines:
        print(line, flush=True)
    print(f"batch={data_config['batch_size']} | epochs={epochs} | output={output_dir}", flush=True)
    save_yaml_once(output_dir / "run_config.yaml", run_config)

    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[EPOCH {epoch}/{epochs}]", flush=True)
        train_metrics = train_epoch(
            model=model, loader=loaders["train"], criterion=criterion, output_key=task.output_key,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device,
            amp_enabled=amp_enabled, clip_norm=clip_norm, max_consecutive_amp_skips=max_skips,
            max_batches=max_train, print_every=print_every,
        )
        val_metrics, _, _ = evaluate(
            stage="VAL", model=model, loader=loaders["validation"], criterion=criterion, task=task,
            device=device, amp_enabled=amp_enabled, max_batches=max_eval, print_every=max(1, print_every * 2),
        )
        score = -val_metrics["loss"] if options.smoke_test else float(val_metrics[task.score_key])
        if not math.isfinite(score):
            raise RuntimeError("validation 选模指标为 NaN/Inf")
        improved = score > best_score + min_delta
        best_score, bad_epochs = (score, 0) if improved else (best_score, bad_epochs + 1)
        history.append({"epoch": epoch, "train": train_metrics, "validation": scalar_metrics(val_metrics), "best_score": best_score})
        atomic_json_save(output_dir / "history.json", history)
        checkpoint = build_checkpoint(
            epoch=epoch, target_epochs=epochs, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, best_score=best_score, bad_epochs=bad_epochs, history=history,
            dataset_config=dataset_config, train_config=train_config,
            task_name=task.name, dataset_name=task.dataset_name,
        )
        atomic_torch_save(last_path, checkpoint)
        if improved:
            atomic_torch_save(best_path, checkpoint)
            print(f"[BEST] epoch={epoch}", flush=True)
        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.6f} | val_loss={val_metrics['loss']:.6f} "
            f"| {metric_text(val_metrics, task.metric_fields)} | AMP_skips={train_metrics['amp_skipped_steps']} "
            f"| bad_epochs={bad_epochs}/{patience}",
            flush=True,
        )
        if bad_epochs >= patience:
            print(f"[EARLY STOP] 连续 {patience} 个 epoch 未提升。", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("没有生成最佳 checkpoint")
    result = {"output_dir": output_dir, "best_path": best_path, "last_path": last_path, "history": history}
    if not options.evaluate_test:
        print("[TEST LOCKED] 已保存验证集最佳模型；测试集未读取。", flush=True)
        return result

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics, predictions, targets = evaluate(
        stage="TEST", model=model, loader=loaders["test"], criterion=criterion, task=task,
        device=device, amp_enabled=amp_enabled, max_batches=max_eval, print_every=max(1, print_every * 2),
    )
    atomic_json_save(output_dir / "test_metrics.json", scalar_metrics(test_metrics))
    np.savez_compressed(output_dir / "test_predictions.npz", **task.prediction_payload(predictions, targets, test_metrics))
    print(f"[TEST] best_epoch={best_checkpoint['epoch']} | loss={test_metrics['loss']:.6f} | {metric_text(test_metrics, task.metric_fields)}", flush=True)
    result["test_metrics"] = scalar_metrics(test_metrics)
    return result
