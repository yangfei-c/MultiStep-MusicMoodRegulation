from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def atomic_torch_save(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(content, temporary); os.replace(temporary, path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state: print("[WARN] checkpoint 没有 RNG 状态，续训不能严格复现。", flush=True); return
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        try: torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError as error: print(f"[WARN] CUDA RNG 状态未恢复：{error}", flush=True)


def resolve_resume_path(value: str | Path | None, outputs_root: Path) -> Path | None:
    if value is None: return None
    if str(value).lower() == "latest":
        candidates = list(outputs_root.glob("*/last.pt"))
        if not candidates: raise FileNotFoundError(f"没有找到 {outputs_root}/*/last.pt")
        return max(candidates, key=lambda path: path.stat().st_mtime)
    path = Path(value).expanduser().resolve()
    if path.is_dir(): path /= "last.pt"
    if not path.is_file(): raise FileNotFoundError(f"断点文件不存在：{path}")
    return path


def build_checkpoint(
    *, epoch: int, target_epochs: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler, scaler: torch.amp.GradScaler, best_score: float, bad_epochs: int,
    history: list[dict], dataset_config: dict, train_config: dict, task_name: str | None = None,
    dataset_name: str | None = None,
) -> dict:
    checkpoint = {
        "format_version": 2, "epoch": epoch, "target_epochs": target_epochs,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
        "best_score": best_score, "bad_epochs": bad_epochs, "history": history,
        "rng_state": capture_rng_state(), "dataset_config": dataset_config, "train_config": train_config,
    }
    if task_name is not None: checkpoint["task_name"] = task_name
    if dataset_name is not None: checkpoint["dataset_name"] = dataset_name
    return checkpoint


def load_resume_state(checkpoint: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, scaler: torch.amp.GradScaler) -> tuple[int, float, int, list[dict]]:
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "best_score", "bad_epochs", "history"}
    missing = sorted(required - checkpoint.keys())
    if missing: raise ValueError("该文件不是完整断点，缺少：" + ", ".join(missing) + "。旧版 best.pt 只能作为初始权重。")
    model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"]); scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_score"]), int(checkpoint["bad_epochs"]), list(checkpoint["history"])
