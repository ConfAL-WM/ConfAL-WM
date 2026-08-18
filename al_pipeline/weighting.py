from __future__ import annotations

from typing import Any

import numpy as np


def normalize_risk(risk: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stop-gradient risk normalization helper.

    The returned array is a NumPy value used as a guide only. In retraining it
    must be converted to a tensor detached from the C3 probe graph; EVAC losses
    should never backpropagate into the C3 probe.
    """
    risk = np.asarray(risk, dtype=np.float32)
    lo = float(risk.min())
    hi = float(risk.max())
    return (risk - lo) / max(hi - lo, eps)


def confidence_weight_from_map(
    conf_map: np.ndarray,
    *,
    lambda_conf: float,
    weight_clip_min: float,
    weight_clip_max: float,
) -> np.ndarray:
    """Compute patch/time weights from C3 confidence for future patch loss."""
    risk = 1.0 - np.clip(np.asarray(conf_map, dtype=np.float32), 0.0, 1.0)
    norm = normalize_risk(risk)
    weights = 1.0 + float(lambda_conf) * norm
    return np.clip(weights, float(weight_clip_min), float(weight_clip_max)).astype(np.float32)


def sample_weight_from_stats(
    stats: dict[str, Any],
    *,
    score_key: str = "tail_risk_top5",
    lambda_conf: float,
    weight_clip_min: float,
    weight_clip_max: float,
) -> float:
    """First-version sample-level weighting fallback.

    The current EVAC Lightning dataset does not consume per-sample weights.
    The retraining wrapper therefore approximates sample weighting by optional
    dataset oversampling and writes these scalar weights for future native
    trainer integration.
    """
    risk = float(stats.get(score_key, stats.get("mean_risk", 0.0)))
    weight = 1.0 + float(lambda_conf) * max(0.0, min(1.0, risk))
    return float(np.clip(weight, float(weight_clip_min), float(weight_clip_max)))

