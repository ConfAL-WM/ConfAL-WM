from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .metrics import (
    binary_iou,
    compute_binary_auprc,
    compute_binary_auroc,
    overlap_at_k,
    pearsonr_safe,
    quantile_top_mask,
    safe_mean,
    safe_median,
    safe_std,
    spearmanr_safe,
    topk_binary_mask,
)


def aggregate_map_per_frame(
    value_map: np.ndarray,
    mode: str = "mean",
    topk_ratio: float = 0.1,
    quantile: float = 0.95,
) -> np.ndarray:
    arr = np.asarray(value_map, dtype=np.float64)
    flat = arr.reshape(arr.shape[0], -1)
    mode = str(mode).lower()
    if mode == "mean":
        return flat.mean(axis=1)
    if mode == "max":
        return flat.max(axis=1)
    if mode == "p95":
        return np.quantile(flat, float(quantile), axis=1)
    if mode == "topk_mean":
        k = max(1, int(np.ceil(flat.shape[1] * float(topk_ratio))))
        topk = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
        return topk.mean(axis=1)
    raise ValueError(f"Unsupported frame aggregation mode: {mode}")


def aggregate_frame_values(
    risk_map: np.ndarray,
    error_map: np.ndarray,
    frame_agg_mode: str = "mean",
    frame_topk_ratio: float = 0.1,
    frame_quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    risk = np.asarray(risk_map, dtype=np.float64)
    err = np.asarray(error_map, dtype=np.float64)
    return (
        aggregate_map_per_frame(risk, mode=frame_agg_mode, topk_ratio=frame_topk_ratio, quantile=frame_quantile),
        aggregate_map_per_frame(err, mode=frame_agg_mode, topk_ratio=frame_topk_ratio, quantile=frame_quantile),
    )


def build_high_error_mask(error_map: np.ndarray, q_percent: float, scope: str = "episode") -> np.ndarray:
    err = np.asarray(error_map, dtype=np.float64)
    if scope == "episode":
        return quantile_top_mask(err.reshape(-1), q_percent).reshape(err.shape)
    if scope == "frame":
        masks = []
        for frame_err in err:
            masks.append(quantile_top_mask(frame_err.reshape(-1), q_percent).reshape(frame_err.shape))
        return np.stack(masks, axis=0)
    raise ValueError(f"Unsupported high_error_scope: {scope}")


def compute_episode_signal_metrics(
    risk_map: np.ndarray,
    error_map: np.ndarray,
    high_error_qs: Sequence[float],
    high_error_scope: str = "episode",
    topk_overlap_enabled: bool = True,
    frame_agg_mode: str = "mean",
    frame_topk_ratio: float = 0.1,
    frame_quantile: float = 0.95,
) -> dict:
    risk = np.asarray(risk_map, dtype=np.float64)
    err = np.asarray(error_map, dtype=np.float64)
    patch_risk = risk.reshape(-1)
    patch_err = err.reshape(-1)
    frame_risk, frame_err = aggregate_frame_values(
        risk, err,
        frame_agg_mode=frame_agg_mode,
        frame_topk_ratio=frame_topk_ratio,
        frame_quantile=frame_quantile,
    )

    metrics = {
        "patch_spearman": spearmanr_safe(patch_risk, patch_err),
        "patch_pearson": pearsonr_safe(patch_risk, patch_err),
        "frame_spearman": spearmanr_safe(frame_risk, frame_err),
        "frame_pearson": pearsonr_safe(frame_risk, frame_err),
        "trajectory_mean_risk": float(np.mean(frame_risk)),
        "trajectory_mean_error": float(np.mean(frame_err)),
    }

    for q in high_error_qs:
        labels = build_high_error_mask(err, q_percent=float(q), scope=high_error_scope).reshape(-1).astype(np.int32)
        metrics[f"high_error_top{int(q)}_auroc"] = compute_binary_auroc(patch_risk, labels)
        metrics[f"high_error_top{int(q)}_auprc"] = compute_binary_auprc(patch_risk, labels)
        if topk_overlap_enabled:
            ratio = float(q) / 100.0
            frame_ious = []
            frame_overlaps = []
            for frame_idx in range(risk.shape[0]):
                risk_mask = topk_binary_mask(risk[frame_idx].reshape(-1), ratio)
                err_mask = topk_binary_mask(err[frame_idx].reshape(-1), ratio)
                frame_ious.append(binary_iou(risk_mask, err_mask))
                frame_overlaps.append(overlap_at_k(risk_mask, err_mask))
            metrics[f"high_error_top{int(q)}_topk_iou"] = safe_mean(frame_ious)
            metrics[f"high_error_top{int(q)}_topk_overlap"] = safe_mean(frame_overlaps)

    return metrics


def compute_global_signal_summary(
    episodes: Sequence[dict],
    high_error_qs: Sequence[float],
    high_error_scope: str = "episode",
    topk_overlap_enabled: bool = True,
) -> dict:
    if not episodes:
        return {}

    pooled_patch_risk = np.concatenate([np.asarray(ep["patch_risk"], dtype=np.float64) for ep in episodes], axis=0)
    pooled_patch_err = np.concatenate([np.asarray(ep["patch_error"], dtype=np.float64) for ep in episodes], axis=0)
    pooled_frame_risk = np.concatenate([np.asarray(ep["frame_risk"], dtype=np.float64) for ep in episodes], axis=0)
    pooled_frame_err = np.concatenate([np.asarray(ep["frame_error"], dtype=np.float64) for ep in episodes], axis=0)
    traj_risk = np.asarray([ep["trajectory_risk"] for ep in episodes], dtype=np.float64)
    traj_err = np.asarray([ep["trajectory_error"] for ep in episodes], dtype=np.float64)

    patch_spearman_per_traj = [ep["episode_metrics"]["patch_spearman"] for ep in episodes]
    patch_pearson_per_traj = [ep["episode_metrics"]["patch_pearson"] for ep in episodes]
    frame_spearman_per_traj = [ep["episode_metrics"]["frame_spearman"] for ep in episodes]
    frame_pearson_per_traj = [ep["episode_metrics"]["frame_pearson"] for ep in episodes]

    summary = {
        "pooled_patch_spearman": spearmanr_safe(pooled_patch_risk, pooled_patch_err),
        "pooled_patch_pearson": pearsonr_safe(pooled_patch_risk, pooled_patch_err),
        "pooled_frame_spearman": spearmanr_safe(pooled_frame_risk, pooled_frame_err),
        "pooled_frame_pearson": pearsonr_safe(pooled_frame_risk, pooled_frame_err),
        "trajectory_spearman": spearmanr_safe(traj_risk, traj_err),
        "trajectory_pearson": pearsonr_safe(traj_risk, traj_err),
        "patch_spearman_per_traj_mean": safe_mean(patch_spearman_per_traj),
        "patch_spearman_per_traj_median": safe_median(patch_spearman_per_traj),
        "patch_spearman_per_traj_std": safe_std(patch_spearman_per_traj),
        "patch_pearson_per_traj_mean": safe_mean(patch_pearson_per_traj),
        "patch_pearson_per_traj_median": safe_median(patch_pearson_per_traj),
        "patch_pearson_per_traj_std": safe_std(patch_pearson_per_traj),
        "frame_spearman_per_traj_mean": safe_mean(frame_spearman_per_traj),
        "frame_spearman_per_traj_median": safe_median(frame_spearman_per_traj),
        "frame_spearman_per_traj_std": safe_std(frame_spearman_per_traj),
        "frame_pearson_per_traj_mean": safe_mean(frame_pearson_per_traj),
        "frame_pearson_per_traj_median": safe_median(frame_pearson_per_traj),
        "frame_pearson_per_traj_std": safe_std(frame_pearson_per_traj),
    }

    for q in high_error_qs:
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_ious: list[float] = []
        all_overlaps: list[float] = []
        for episode in episodes:
            risk_map = np.asarray(episode["risk_map"], dtype=np.float64)
            err_map = np.asarray(episode["oracle_error"], dtype=np.float64)
            labels = build_high_error_mask(err_map, q_percent=float(q), scope=high_error_scope).reshape(-1).astype(np.int32)
            all_scores.append(risk_map.reshape(-1))
            all_labels.append(labels)
            if topk_overlap_enabled:
                ratio = float(q) / 100.0
                for frame_idx in range(risk_map.shape[0]):
                    risk_mask = topk_binary_mask(risk_map[frame_idx].reshape(-1), ratio)
                    err_mask = topk_binary_mask(err_map[frame_idx].reshape(-1), ratio)
                    all_ious.append(binary_iou(risk_mask, err_mask))
                    all_overlaps.append(overlap_at_k(risk_mask, err_mask))
        merged_scores = np.concatenate(all_scores, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        summary[f"high_error_top{int(q)}_auroc"] = compute_binary_auroc(merged_scores, merged_labels)
        summary[f"high_error_top{int(q)}_auprc"] = compute_binary_auprc(merged_scores, merged_labels)
        if topk_overlap_enabled:
            summary[f"high_error_top{int(q)}_topk_iou_mean"] = safe_mean(all_ious)
            summary[f"high_error_top{int(q)}_topk_overlap_mean"] = safe_mean(all_overlaps)

    return summary
