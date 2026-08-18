from __future__ import annotations

from typing import Sequence

import numpy as np

from .correlation import aggregate_map_per_frame
from .metrics import binary_iou, pearsonr_safe, safe_mean, safe_median, safe_std, spearmanr_safe, topk_binary_mask


def _lagged_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag == 0:
        return pearsonr_safe(x, y)
    if lag > 0:
        return pearsonr_safe(x[:-lag], y[lag:])
    return pearsonr_safe(x[-lag:], y[:lag])


def compute_process_validity_episode(
    risk_map: np.ndarray,
    error_map: np.ndarray,
    lag_max: int = 5,
    risk_topk_ratio: float = 0.1,
    low_error_quantile: float = 0.3,
    stable_delta_quantile: float = 0.3,
    frame_agg_mode: str = "mean",
    frame_topk_ratio: float = 0.1,
    frame_quantile: float = 0.95,
) -> dict:
    risk = np.asarray(risk_map, dtype=np.float64)
    err = np.asarray(error_map, dtype=np.float64)
    frame_risk = aggregate_map_per_frame(risk, mode=frame_agg_mode, topk_ratio=frame_topk_ratio, quantile=frame_quantile)
    frame_err = aggregate_map_per_frame(err, mode=frame_agg_mode, topk_ratio=frame_topk_ratio, quantile=frame_quantile)
    delta_risk = np.diff(frame_risk)
    delta_err = np.diff(frame_err)

    low_err_thresh = np.quantile(frame_err, low_error_quantile) if frame_err.size else float("nan")
    abs_delta_err = np.abs(delta_err)
    stable_delta_thresh = np.quantile(abs_delta_err, stable_delta_quantile) if abs_delta_err.size else float("nan")
    stable_mask = (
        (frame_err[:-1] <= low_err_thresh)
        & (frame_err[1:] <= low_err_thresh)
        & (abs_delta_err <= stable_delta_thresh)
    ) if frame_err.size >= 2 else np.zeros(0, dtype=bool)
    flicker = float(np.mean(np.abs(delta_risk)[stable_mask])) if np.any(stable_mask) else float("nan")

    adjacent_ious = []
    spatial_match_ious = []
    for frame_idx in range(max(risk.shape[0] - 1, 0)):
        current = topk_binary_mask(risk[frame_idx].reshape(-1), risk_topk_ratio)
        nxt = topk_binary_mask(risk[frame_idx + 1].reshape(-1), risk_topk_ratio)
        adjacent_ious.append(binary_iou(current, nxt))
    for frame_idx in range(risk.shape[0]):
        risk_mask = topk_binary_mask(risk[frame_idx].reshape(-1), risk_topk_ratio)
        err_mask = topk_binary_mask(err[frame_idx].reshape(-1), risk_topk_ratio)
        spatial_match_ious.append(binary_iou(risk_mask, err_mask))

    lags = list(range(-lag_max, lag_max + 1))
    lag_corrs = [_lagged_corr(frame_risk, frame_err, lag) for lag in lags]
    if np.all(~np.isfinite(np.asarray(lag_corrs, dtype=np.float64))):
        peak_lag = float("nan")
        peak_corr = float("nan")
    else:
        peak_idx = int(np.nanargmax(np.asarray(lag_corrs, dtype=np.float64)))
        peak_lag = float(lags[peak_idx])
        peak_corr = float(lag_corrs[peak_idx])

    return {
        "temporal_alignment_pearson": pearsonr_safe(delta_risk, delta_err),
        "temporal_alignment_spearman": spearmanr_safe(delta_risk, delta_err),
        "flicker_score": flicker,
        "adjacent_frame_topk_risk_iou": safe_mean(adjacent_ious),
        "risk_topk_vs_error_topk_iou": safe_mean(spatial_match_ious),
        "lag_peak_corr": peak_corr,
        "lag_peak_offset": peak_lag,
        "lagged_corr_lags": lags,
        "lagged_corr_values": lag_corrs,
    }


def summarize_process_validity(episodes: Sequence[dict], lag_max: int) -> dict:
    if not episodes:
        return {}
    summary = {
        "temporal_alignment_pearson_mean": safe_mean([ep["process_metrics"]["temporal_alignment_pearson"] for ep in episodes]),
        "temporal_alignment_pearson_median": safe_median([ep["process_metrics"]["temporal_alignment_pearson"] for ep in episodes]),
        "temporal_alignment_pearson_std": safe_std([ep["process_metrics"]["temporal_alignment_pearson"] for ep in episodes]),
        "temporal_alignment_spearman_mean": safe_mean([ep["process_metrics"]["temporal_alignment_spearman"] for ep in episodes]),
        "temporal_alignment_spearman_median": safe_median([ep["process_metrics"]["temporal_alignment_spearman"] for ep in episodes]),
        "temporal_alignment_spearman_std": safe_std([ep["process_metrics"]["temporal_alignment_spearman"] for ep in episodes]),
        "flicker_score_mean": safe_mean([ep["process_metrics"]["flicker_score"] for ep in episodes]),
        "flicker_score_median": safe_median([ep["process_metrics"]["flicker_score"] for ep in episodes]),
        "flicker_score_std": safe_std([ep["process_metrics"]["flicker_score"] for ep in episodes]),
        "adjacent_frame_topk_risk_iou_mean": safe_mean([ep["process_metrics"]["adjacent_frame_topk_risk_iou"] for ep in episodes]),
        "risk_topk_vs_error_topk_iou_mean": safe_mean([ep["process_metrics"]["risk_topk_vs_error_topk_iou"] for ep in episodes]),
        "lag_peak_corr_mean": safe_mean([ep["process_metrics"]["lag_peak_corr"] for ep in episodes]),
        "lag_peak_offset_mean": safe_mean([ep["process_metrics"]["lag_peak_offset"] for ep in episodes]),
    }
    all_curves = []
    for episode in episodes:
        values = np.asarray(episode["process_metrics"]["lagged_corr_values"], dtype=np.float64)
        all_curves.append(values)
    curve = np.stack(all_curves, axis=0)
    lags = list(range(-lag_max, lag_max + 1))
    mean_curve = np.full(curve.shape[1], np.nan, dtype=np.float64)
    std_curve = np.full(curve.shape[1], np.nan, dtype=np.float64)
    for idx in range(curve.shape[1]):
        values = curve[:, idx]
        values = values[np.isfinite(values)]
        if values.size > 0:
            mean_curve[idx] = np.mean(values)
            std_curve[idx] = np.std(values)
    summary["lag_curve_lags"] = lags
    summary["lag_curve_mean"] = mean_curve.tolist()
    summary["lag_curve_std"] = std_curve.tolist()
    return summary
