from collections.abc import Callable

import numpy as np

from src.metrics import compute_mtg_metrics, compute_va_metrics
from src.training.engine import TaskSpec


def tag_prediction_payload(logits: np.ndarray, targets: np.ndarray, metrics: dict) -> dict:
    return {
        "logits": logits,
        "targets": targets,
        "per_label_ap": metrics["per_label_ap"],
        "per_label_roc_auc": metrics["per_label_roc_auc"],
    }


MTG_TASK = TaskSpec(
    name="mtg",
    dataset_name="mtg",
    output_key="tag_logits",
    score_key="tag_score",
    metric_fn=compute_mtg_metrics,
    metric_description="mAP/ROC-AUC",
    default_print_every=100,
    metric_fields=(
        ("genre_mAP", "genre_map"),
        ("instrument_mAP", "instrument_map"),
        ("mood_mAP", "mood_map"),
        ("tag_score", "tag_score"),
        ("overall_mAP", "overall_map"),
        ("overall_AUC", "overall_roc_auc"),
    ),
    prediction_payload=tag_prediction_payload,
)


def va_prediction_payload(predictions: np.ndarray, targets: np.ndarray, _: dict) -> dict:
    return {
        "predictions": predictions,
        "targets": targets,
        "predictions_1_9": predictions * 4 + 5,
        "targets_1_9": targets * 4 + 5,
    }


def make_va_task(dataset_name: str, metric_fn: Callable = compute_va_metrics) -> TaskSpec:
    return TaskSpec(
        name="va",
        dataset_name=dataset_name,
        output_key="va_predictions",
        score_key="va_score",
        metric_fn=metric_fn,
        metric_description="CCC/R²/RMSE/MAE",
        default_print_every=50,
        metric_fields=(
            ("CCC_v", "valence_ccc"),
            ("CCC_a", "arousal_ccc"),
            ("CCC", "va_score"),
            ("R²", "mean_r2"),
            ("RMSE", "mean_rmse"),
            ("MAE", "mean_mae"),
        ),
        prediction_payload=va_prediction_payload,
    )


def make_domain_balanced_va_metrics(lengths: tuple[int, ...], names: tuple[str, ...]) -> Callable:
    if len(lengths) != len(names):
        raise ValueError("lengths 与 names 数量不一致")

    def metric_fn(predictions: np.ndarray, targets: np.ndarray) -> dict:
        metrics, offset = {}, 0
        domain_scores = []
        for name, length in zip(names, lengths):
            domain = compute_va_metrics(predictions[offset : offset + length], targets[offset : offset + length])
            domain_scores.append(domain["va_score"])
            metrics.update({f"{name}_{key}": value for key, value in domain.items()})
            offset += length
        overall = compute_va_metrics(predictions, targets)
        metrics.update(overall)
        metrics["sample_weighted_va_score"] = overall["va_score"]
        metrics["va_score"] = float(np.mean(domain_scores))
        return metrics

    return metric_fn
