from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats


EPS = 1e-12


@dataclass
class BootstrapCI:
    mean: float
    low: float
    high: float
    num_valid_samples: int


def to_numpy_1d(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr


def finite_pair(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x_arr = to_numpy_1d(x)
    y_arr = to_numpy_1d(y)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def safe_mean(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def safe_median(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def pearsonr_safe(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> float:
    x_arr, y_arr = finite_pair(x, y)
    if x_arr.size < 2:
        return float("nan")
    if np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return float("nan")
    return float(stats.pearsonr(x_arr, y_arr)[0])


def spearmanr_safe(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> float:
    x_arr, y_arr = finite_pair(x, y)
    if x_arr.size < 2:
        return float("nan")
    if np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return float("nan")
    corr = stats.spearmanr(x_arr, y_arr, nan_policy="omit").correlation
    return float(corr) if corr is not None else float("nan")


def brier_score(confidence: np.ndarray | Sequence[float], target: np.ndarray | Sequence[float]) -> float:
    conf_arr, tgt_arr = finite_pair(confidence, target)
    if conf_arr.size == 0:
        return float("nan")
    return float(np.mean((conf_arr - tgt_arr) ** 2))


def compute_binary_auroc(scores: np.ndarray | Sequence[float], labels: np.ndarray | Sequence[int]) -> float:
    score_arr, label_arr = finite_pair(scores, labels)
    if score_arr.size == 0:
        return float("nan")
    label_arr = label_arr.astype(np.int64)
    n_pos = int(np.sum(label_arr == 1))
    n_neg = int(np.sum(label_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score_arr, method="average")
    rank_sum_pos = float(np.sum(ranks[label_arr == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_binary_auprc(scores: np.ndarray | Sequence[float], labels: np.ndarray | Sequence[int]) -> float:
    score_arr, label_arr = finite_pair(scores, labels)
    if score_arr.size == 0:
        return float("nan")
    label_arr = label_arr.astype(np.int64)
    n_pos = int(np.sum(label_arr == 1))
    if n_pos == 0 or n_pos == label_arr.size:
        return float("nan")
    order = np.argsort(-score_arr, kind="mergesort")
    y_true = label_arr[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, EPS)
    recall = tp / max(n_pos, 1)
    precision = np.concatenate([[1.0], precision.astype(np.float64)])
    recall = np.concatenate([[0.0], recall.astype(np.float64)])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_ece_mce(
    confidence: np.ndarray | Sequence[float],
    correctness: np.ndarray | Sequence[float],
    num_bins: int = 20,
) -> dict:
    conf_arr, corr_arr = finite_pair(confidence, correctness)
    if conf_arr.size == 0:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "bin_edges": np.linspace(0.0, 1.0, num_bins + 1),
            "bin_counts": np.zeros(num_bins, dtype=np.int64),
            "bin_confidence": np.full(num_bins, np.nan),
            "bin_accuracy": np.full(num_bins, np.nan),
            "bin_gap": np.full(num_bins, np.nan),
        }

    conf_arr = np.clip(conf_arr, 0.0, 1.0)
    corr_arr = np.clip(corr_arr, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(conf_arr, bin_edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.int64)

    bin_conf = np.full(num_bins, np.nan, dtype=np.float64)
    bin_acc = np.full(num_bins, np.nan, dtype=np.float64)
    bin_gap = np.full(num_bins, np.nan, dtype=np.float64)

    total = max(conf_arr.size, 1)
    ece = 0.0
    mce = 0.0
    for idx in range(num_bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        mean_conf = float(np.mean(conf_arr[mask]))
        mean_acc = float(np.mean(corr_arr[mask]))
        gap = abs(mean_conf - mean_acc)
        bin_conf[idx] = mean_conf
        bin_acc[idx] = mean_acc
        bin_gap[idx] = gap
        ece += gap * (np.count_nonzero(mask) / total)
        mce = max(mce, gap)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_edges": bin_edges,
        "bin_counts": counts,
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_gap": bin_gap,
    }


def topk_binary_mask(values: np.ndarray | Sequence[float], ratio: float) -> np.ndarray:
    arr = to_numpy_1d(values)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(np.ceil(arr.size * ratio)))
    kth = np.partition(arr, arr.size - k)[arr.size - k]
    return arr >= kth


def quantile_top_mask(values: np.ndarray | Sequence[float], q_percent: float) -> np.ndarray:
    arr = to_numpy_1d(values)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = np.quantile(arr, 1.0 - q_percent / 100.0)
    return arr >= threshold


def binary_iou(mask_a: np.ndarray | Sequence[bool], mask_b: np.ndarray | Sequence[bool]) -> float:
    a = np.asarray(mask_a, dtype=bool).reshape(-1)
    b = np.asarray(mask_b, dtype=bool).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union == 0:
        return float("nan")
    return float(inter / union)


def overlap_at_k(mask_a: np.ndarray | Sequence[bool], mask_b: np.ndarray | Sequence[bool]) -> float:
    a = np.asarray(mask_a, dtype=bool).reshape(-1)
    b = np.asarray(mask_b, dtype=bool).reshape(-1)
    denom = np.count_nonzero(a)
    if denom == 0:
        return float("nan")
    inter = np.count_nonzero(a & b)
    return float(inter / denom)


def sample_points(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx], y[idx]


def bootstrap_episode_metric(
    episodes: Sequence[dict],
    metric_fn: Callable[[Sequence[dict]], float],
    num_bootstrap: int,
    seed: int = 0,
    ci: float = 95.0,
) -> BootstrapCI:
    if not episodes:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(episodes)
    for _ in range(num_bootstrap):
        indices = rng.integers(0, n, size=n)
        sampled = [episodes[int(idx)] for idx in indices]
        value = float(metric_fn(sampled))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), 0)
    alpha = (100.0 - ci) / 2.0
    arr = np.asarray(values, dtype=np.float64)
    return BootstrapCI(
        mean=float(np.mean(arr)),
        low=float(np.percentile(arr, alpha)),
        high=float(np.percentile(arr, 100.0 - alpha)),
        num_valid_samples=int(arr.size),
    )
