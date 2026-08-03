"""交叉拟合的有界 Tag→VA 残差桥；不参与基础模型训练。"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from src.metrics import compute_va_metrics


def domain_balanced_weights(domains: np.ndarray) -> np.ndarray:
    domains = np.asarray(domains)
    if domains.ndim != 1 or len(domains) == 0:
        raise ValueError("domains 应为非空一维数组")
    weights = np.zeros(len(domains), dtype=np.float64)
    unique = np.unique(domains)
    for domain in unique:
        mask = domains == domain
        weights[mask] = 1.0 / (len(unique) * mask.sum())
    return weights * len(weights)


def mood_confidence(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    entropy = -(probabilities * np.log(probabilities) + (1 - probabilities) * np.log(1 - probabilities)).mean(1)
    return probabilities.max(1) * (1 - entropy / np.log(2))


def fit_ridge(features: np.ndarray, residuals: np.ndarray, weights: np.ndarray, alpha: float) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    mean = np.sum(features * weights[:, None], axis=0)
    variance = np.sum((features - mean) ** 2 * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1.0e-8))
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    root_weight = np.sqrt(weights)[:, None]
    weighted_x, weighted_y = design * root_weight, residuals * root_weight
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y)
    return {"mean": mean, "scale": scale, "bias": beta[0], "coef": beta[1:]}


def predict_ridge(model: dict, features: np.ndarray) -> np.ndarray:
    normalized = (features - model["mean"]) / model["scale"]
    return normalized @ model["coef"] + model["bias"]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def apply_bridge(parameters: dict, mood_probabilities: np.ndarray, direct_predictions: np.ndarray):
    delta = np.clip(
        predict_ridge(parameters, mood_probabilities), -parameters["delta_max"], parameters["delta_max"]
    )
    confidence = mood_confidence(mood_probabilities)
    rho = parameters["rho_max"] * sigmoid(
        (confidence - parameters["tau"]) / parameters["temperature"]
    )
    prediction = np.clip(direct_predictions + rho[:, None] * delta, -1, 1)
    return prediction, rho, delta


def fit_cross_fitted_bridge(
    mood_probabilities: np.ndarray,
    direct_predictions: np.ndarray,
    targets: np.ndarray,
    domains: np.ndarray,
    *,
    folds: int = 5,
    alpha: float = 1.0,
    delta_max: float = 0.25,
    rho_max: float = 0.5,
    temperature: float = 0.08,
    seed: int = 42,
    control: str = "none",
) -> tuple[dict, dict]:
    mood = np.asarray(mood_probabilities, dtype=np.float64)
    direct = np.asarray(direct_predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    domains = np.asarray(domains)
    if mood.ndim != 2 or mood.shape[1] != 56 or direct.shape != targets.shape or direct.shape[1] != 2:
        raise ValueError("bridge 输入应为 mood=[N,56]、direct/targets=[N,2]")
    if len(domains) != len(targets) or not np.isfinite(mood).all() or not np.isfinite(direct).all() or not np.isfinite(targets).all():
        raise ValueError("bridge 输入数量不一致或含 NaN/Inf")
    if not 2 <= folds <= len(targets):
        raise ValueError("folds 必须位于 [2,N]")
    if alpha < 0 or delta_max < 0 or not 0 <= rho_max <= 1 or temperature <= 0:
        raise ValueError("bridge 超参数范围错误")

    rng, fit_mood, fit_targets = np.random.default_rng(seed), mood.copy(), targets.copy()
    if control == "mood_permutation":
        fit_mood = fit_mood[rng.permutation(len(fit_mood))]
    elif control == "va_permutation":
        fit_targets = fit_targets[rng.permutation(len(fit_targets))]
    elif control != "none":
        raise ValueError("control 只支持 none/mood_permutation/va_permutation")

    if np.unique(domains).size > 1:
        split_iterator = StratifiedKFold(folds, shuffle=True, random_state=seed).split(fit_mood, domains)
    else:
        split_iterator = KFold(folds, shuffle=True, random_state=seed).split(fit_mood)
    oof_delta = np.zeros_like(targets)
    for train_index, held_index in split_iterator:
        model = fit_ridge(
            fit_mood[train_index],
            fit_targets[train_index] - direct[train_index],
            domain_balanced_weights(domains[train_index]),
            alpha,
        )
        oof_delta[held_index] = predict_ridge(model, fit_mood[held_index])
    oof_delta = np.clip(oof_delta, -delta_max, delta_max)

    confidence = mood_confidence(fit_mood)
    selection_targets = fit_targets if control == "va_permutation" else targets
    best_tau, best_score = 0.5, -np.inf
    for tau in np.linspace(0.05, 0.95, 37):
        rho = rho_max * sigmoid((confidence - tau) / temperature)
        candidate = np.clip(direct + rho[:, None] * oof_delta, -1, 1)
        score = compute_va_metrics(candidate, selection_targets)["va_score"]
        if score > best_score:
            best_tau, best_score = float(tau), float(score)

    parameters = {
        **fit_ridge(fit_mood, fit_targets - direct, domain_balanced_weights(domains), alpha),
        "tau": best_tau,
        "alpha": alpha,
        "delta_max": delta_max,
        "rho_max": rho_max,
        "temperature": temperature,
        "control": control,
    }
    rho = rho_max * sigmoid((confidence - best_tau) / temperature)
    oof_prediction = np.clip(direct + rho[:, None] * oof_delta, -1, 1)
    report = {
        "direct": compute_va_metrics(direct, targets),
        "bridge_oof": compute_va_metrics(oof_prediction, targets),
        "selected_tau": best_tau,
        "mean_gate": float(rho.mean()),
        "coverage_gate_gt_0_25": float(np.mean(rho > 0.25)),
    }
    return parameters, report
