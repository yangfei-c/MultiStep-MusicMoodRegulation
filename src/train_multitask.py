"""MTG 标签与 DEAM/PMEmo VA 的 selective-update 基线。"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MTGDataset, VADataset, build_dataloader, domain_balanced_sampler
from src.training.checkpoint import atomic_torch_save, build_checkpoint, load_resume_state, resolve_resume_path
from src.training.common import atomic_json_save, get_device, load_yaml, logged_output, make_scheduler, move_batch, save_yaml_once, scalar_metrics, set_seed
from src.training.engine import evaluate
from src.training.factory import build_model, build_optimizer, build_tag_loss, build_va_loss
from src.training.tasks import MTG_TASK, make_va_task, tag_prediction_payload, va_prediction_payload


RESUME_FROM = None
EVALUATE_TEST = False
DATASET_CONFIG = ROOT / "configs/dataset.yaml"
TRAIN_CONFIG = ROOT / "configs/train.yaml"
DOMAINS = ("deam", "pmemo")
REFERENCE = {"tag": 0.127718, "deam": 0.744557, "pmemo": 0.781203}


def optimize_batch(model, batch, output_key, criterion, optimizer, scheduler, scaler, device, amp, clip):
    features, mask, targets = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp):
        loss = criterion(model(features, mask)[output_key], targets)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"{output_key} loss 为 NaN/Inf")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = amp and scaler.get_scale() < scale
    if not skipped:
        scheduler.step()
    return float(loss.detach()), int(features.shape[0]), skipped


def train_epoch(model, mtg_loader, va_loader, mtg_loss, va_loss, optimizer, scheduler, scaler, device, amp, clip, mtg_per_va):
    model.train()
    va_iterator = iter(va_loader)
    sums, counts, skipped = {"mtg": 0.0, "va": 0.0}, {"mtg": 0, "va": 0}, 0
    started = time.perf_counter()
    for step, mtg_batch in enumerate(mtg_loader, 1):
        loss, size, was_skipped = optimize_batch(
            model, mtg_batch, "tag_logits", mtg_loss, optimizer, scheduler, scaler, device, amp, clip
        )
        sums["mtg"] += loss * size
        counts["mtg"] += size
        skipped += int(was_skipped)
        if step % mtg_per_va == 0:
            try:
                va_batch = next(va_iterator)
            except StopIteration:
                va_iterator, va_batch = iter(va_loader), None
                va_batch = next(va_iterator)
            loss, size, was_skipped = optimize_batch(
                model, va_batch, "va_predictions", va_loss, optimizer, scheduler, scaler, device, amp, clip
            )
            sums["va"] += loss * size
            counts["va"] += size
            skipped += int(was_skipped)
        if step == 1 or step % 100 == 0 or step == len(mtg_loader):
            print(
                f"  [TRAIN] {step:4d}/{len(mtg_loader)} | mtg={sums['mtg']/counts['mtg']:.6f} "
                f"| va={sums['va']/max(1, counts['va']):.6f} | lr={optimizer.param_groups[0]['lr']:.2e} | skipped={skipped}",
                flush=True,
            )
    return {
        "mtg_loss": sums["mtg"] / counts["mtg"],
        "va_loss": sums["va"] / counts["va"],
        "mtg_samples": counts["mtg"],
        "va_samples": counts["va"],
        "seconds": time.perf_counter() - started,
        "amp_skipped_steps": skipped,
    }


def joint_score(mtg: dict, deam: dict, pmemo: dict) -> float:
    return float(np.mean((mtg["tag_score"] / REFERENCE["tag"], deam["va_score"] / REFERENCE["deam"], pmemo["va_score"] / REFERENCE["pmemo"])))


def main() -> None:
    dataset_config, config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = get_device(str(config.get("device", "cpu")))
    data, training = config["data"], config["training"]
    multitask = config.get("multitask", {})
    mtg_per_va = int(multitask.get("mtg_per_va", 2))

    mtg = {split: MTGDataset(dataset_config["mtg"], split) for split in ("train", "validation", "test")}
    va = {
        name: {split: VADataset(dataset_config[name], name, split) for split in ("train", "validation", "test")}
        for name in DOMAINS
    }
    loader_args = {
        "batch_size": int(data["batch_size"]),
        "num_workers": int(data["num_workers"]),
        "pin_memory": bool(data["pin_memory"]),
    }
    mtg_loaders = {
        split: build_dataloader(dataset, shuffle=split == "train", **loader_args)
        for split, dataset in mtg.items()
    }
    va_train = ConcatDataset([va[name]["train"] for name in DOMAINS])
    sampler = domain_balanced_sampler([len(va[name]["train"]) for name in DOMAINS], seed)
    va_train_loader = build_dataloader(va_train, sampler=sampler, **loader_args)
    va_loaders = {
        name: {split: build_dataloader(dataset, **loader_args) for split, dataset in sets.items() if split != "train"}
        for name, sets in va.items()
    }

    model = build_model(config, device)
    mtg_loss, va_loss = build_tag_loss(config), build_va_loss(config)
    optimizer = build_optimizer(model, config)
    epochs = int(training["max_epochs"])
    updates = len(mtg_loaders["train"]) + math.ceil(len(mtg_loaders["train"]) / mtg_per_va)
    scheduler = make_scheduler(
        optimizer, epochs * updates, float(config["scheduler"]["warmup_ratio"]), float(config["scheduler"]["min_learning_rate"])
    )
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    clip = float(training["gradient_clip_norm"])
    patience = int(training["early_stopping_patience"])
    min_delta = float(training["early_stopping_min_delta"])

    outputs_root = ROOT / "outputs/multitask/mtg_deam_pmemo"
    resume_path = resolve_resume_path(RESUME_FROM, outputs_root)
    output_dir = resume_path.parent if resume_path else outputs_root / f"s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=bool(resume_path))
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None
    start_epoch, best_score, bad_epochs, history = 1, -math.inf, 0, []
    if checkpoint:
        start_epoch, best_score, bad_epochs, history = load_resume_state(checkpoint, model, optimizer, scheduler, scaler)

    va_task = make_va_task("va")
    with logged_output(output_dir / "train.log"):
        save_yaml_once(output_dir / "run_config.yaml", {"dataset": dataset_config, "train": config, "mtg_per_va": mtg_per_va, "reference": REFERENCE})
        for epoch in range(start_epoch, epochs + 1):
            print(f"\n[EPOCH {epoch}/{epochs}]")
            train = train_epoch(model, mtg_loaders["train"], va_train_loader, mtg_loss, va_loss, optimizer, scheduler, scaler, device, amp, clip, mtg_per_va)
            mtg_val, _, _ = evaluate(stage="VAL-MTG", model=model, loader=mtg_loaders["validation"], criterion=mtg_loss, task=MTG_TASK, device=device, amp_enabled=amp, max_batches=None, print_every=200)
            domain_val = {
                name: evaluate(stage=f"VAL-{name.upper()}", model=model, loader=va_loaders[name]["validation"], criterion=va_loss, task=va_task, device=device, amp_enabled=amp, max_batches=None, print_every=100)[0]
                for name in DOMAINS
            }
            score = joint_score(mtg_val, domain_val["deam"], domain_val["pmemo"])
            improved = score > best_score + min_delta
            best_score, bad_epochs = (score, 0) if improved else (best_score, bad_epochs + 1)
            history.append({
                "epoch": epoch,
                "train": train,
                "validation": {"mtg": scalar_metrics(mtg_val), **{name: scalar_metrics(value) for name, value in domain_val.items()}},
                "joint_score": score,
                "best_score": best_score,
            })
            atomic_json_save(output_dir / "history.json", history)
            state = build_checkpoint(epoch=epoch, target_epochs=epochs, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, best_score=best_score, bad_epochs=bad_epochs, history=history, dataset_config=dataset_config, train_config=config, task_name="multitask", dataset_name="mtg_deam_pmemo")
            atomic_torch_save(output_dir / "last.pt", state)
            if improved:
                atomic_torch_save(output_dir / "best.pt", state)
            print(f"[EPOCH {epoch}] joint={score:.4f} | tag={mtg_val['tag_score']:.4f} | deam={domain_val['deam']['va_score']:.4f} | pmemo={domain_val['pmemo']['va_score']:.4f} | bad={bad_epochs}/{patience}")
            if bad_epochs >= patience:
                break

        if not EVALUATE_TEST:
            print("[TEST LOCKED] 多任务实验尚未读取测试集。")
            return
        best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        evaluations = {"mtg": (mtg_loaders["test"], mtg_loss, MTG_TASK), **{name: (va_loaders[name]["test"], va_loss, va_task) for name in DOMAINS}}
        for name, (loader, criterion, task) in evaluations.items():
            metrics, predictions, targets = evaluate(stage=f"TEST-{name.upper()}", model=model, loader=loader, criterion=criterion, task=task, device=device, amp_enabled=amp, max_batches=None, print_every=200)
            atomic_json_save(output_dir / f"test_metrics_{name}.json", scalar_metrics(metrics))
            payload = tag_prediction_payload(predictions, targets, metrics) if name == "mtg" else va_prediction_payload(predictions, targets, metrics)
            np.savez_compressed(output_dir / f"test_predictions_{name}.npz", **payload)


if __name__ == "__main__":
    main()
