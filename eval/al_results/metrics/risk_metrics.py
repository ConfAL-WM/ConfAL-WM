from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_cvar(errors: np.ndarray | Sequence[float], q: float = 0.9) -> float | None:
    """Conditional Value at Risk: mean of the top (1-q) fraction of errors."""
    if isinstance(errors, (list, tuple)):
        errors = np.asarray(errors, dtype=np.float32)
    if errors.size == 0:
        return None
    threshold = np.quantile(errors, q)
    return float(errors[errors >= threshold].mean())


def compute_episode_error(
    latent_loss: np.ndarray | Sequence[float],
    pixel_mae: np.ndarray | Sequence[float],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Per-episode scalar error = alpha * latent_loss + beta * pixel_mae."""
    latent = np.asarray(latent_loss, dtype=np.float32)
    pixel = np.asarray(pixel_mae, dtype=np.float32)
    return alpha * latent + beta * pixel


def compute_risk_reduction(
    current_risk: float | None,
    base_risk: float | None = None,
    prev_risk: float | None = None,
) -> dict[str, float | None]:
    """Compute risk reduction relative to a base or previous round."""
    result: dict[str, float | None] = {
        "risk_cvar90": current_risk,
        "risk_reduction_vs_base": None,
        "risk_reduction_vs_prev": None,
    }
    if current_risk is not None and base_risk is not None and base_risk > 0:
        result["risk_reduction_vs_base"] = float((base_risk - current_risk) / base_risk)
    if current_risk is not None and prev_risk is not None and prev_risk > 0:
        result["risk_reduction_vs_prev"] = float((prev_risk - current_risk) / prev_risk)
    return result
