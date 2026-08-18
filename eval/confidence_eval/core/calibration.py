from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .metrics import brier_score, compute_ece_mce, finite_pair


def resolve_operational_tau(
    episode_records: Sequence[dict],
    mode: str,
    fixed_value: float | None = None,
    quantile: float | None = None,
    quantile_source_split: str = "train",
) -> float:
    """
    Resolve the operational tau threshold used for calibration.

    Notes:
    1. This tau is an evaluation definition; it does not imply the probe conditions on tau at inference.
    2. When training uses random-threshold BCE and inference does not input tau,
       treating tau as part of the evaluation protocol (rather than interpreting
       model output as q(x, tau)) is more faithful.
    """
    if mode == "fixed":
        if fixed_value is None:
            raise ValueError("tau_eval.mode=fixed requires fixed_value to be provided")
        return float(fixed_value)

    if mode != "quantile":
        raise ValueError(f"Unsupported tau resolution mode: {mode}")

    if quantile is None:
        raise ValueError("tau_eval.mode=quantile requires quantile to be provided")

    pooled_errors: list[np.ndarray] = []
    for episode in episode_records:
        if str(episode.get("split", "")) != str(quantile_source_split):
            continue
        oracle_error = np.asarray(episode["oracle_error"], dtype=np.float64).reshape(-1)
        oracle_error = oracle_error[np.isfinite(oracle_error)]
        if oracle_error.size > 0:
            pooled_errors.append(oracle_error)

    if not pooled_errors:
        raise ValueError(
            f"No oracle error found for split={quantile_source_split} , cannot estimate quantile tau"
        )

    merged = np.concatenate(pooled_errors, axis=0)
    return float(np.quantile(merged, float(quantile)))


def build_correctness(
    oracle_error: np.ndarray,
    tau: float,
    correctness_mode: str = "hard",
    soft_correctness_mode: str = "none",
) -> np.ndarray:
    """
    Generate correctness targets from oracle error.

    Main mode:
      correctness_mode="hard":
        correctness = 1[error <= tau]

    Diagnostic/supplementary mode:
      correctness_mode="soft" and soft_correctness_mode="linear_ratio":
        correctness = clip(1 - error / tau, 0, 1)

    Note:
    Soft correctness is for supplementary analysis only and does not replace
    the operational calibration definition used in the main text.
    """
    error_arr = np.asarray(oracle_error, dtype=np.float64)
    tau = max(float(tau), 1e-8)
    if correctness_mode == "hard":
        return (error_arr <= tau).astype(np.float32)
    if correctness_mode != "soft":
        raise ValueError(f"Unsupported correctness_mode: {correctness_mode}")
    if soft_correctness_mode != "linear_ratio":
        raise ValueError(
            "Currently only soft_correctness_mode=linear_ratio is supported; "
            "keep hard mode unless soft targets are explicitly needed."
        )
    return np.clip(1.0 - error_arr / tau, 0.0, 1.0).astype(np.float32)


def compute_calibration_stats(
    confidence: np.ndarray,
    oracle_error: np.ndarray,
    tau: float,
    num_bins: int = 20,
    correctness_mode: str = "hard",
    soft_correctness_mode: str = "none",
) -> dict:
    conf_arr, err_arr = finite_pair(confidence, oracle_error)
    correctness = build_correctness(
        oracle_error=err_arr,
        tau=tau,
        correctness_mode=correctness_mode,
        soft_correctness_mode=soft_correctness_mode,
    )
    calib = compute_ece_mce(conf_arr, correctness, num_bins=num_bins)
    calib["brier"] = brier_score(conf_arr, correctness)
    calib["tau"] = float(tau)
    calib["correctness_mode"] = correctness_mode
    calib["soft_correctness_mode"] = soft_correctness_mode
    calib["mean_confidence"] = float(np.mean(conf_arr)) if conf_arr.size else float("nan")
    calib["mean_correctness"] = float(np.mean(correctness)) if correctness.size else float("nan")
    calib["total_samples"] = int(conf_arr.size)
    return calib


def compute_threshold_sweep(
    confidence: np.ndarray,
    oracle_error: np.ndarray,
    tau_values: Iterable[float],
    num_bins: int = 20,
    correctness_mode: str = "hard",
    soft_correctness_mode: str = "none",
) -> list[dict]:
    results: list[dict] = []
    for tau in tau_values:
        stats = compute_calibration_stats(
            confidence=confidence,
            oracle_error=oracle_error,
            tau=float(tau),
            num_bins=num_bins,
            correctness_mode=correctness_mode,
            soft_correctness_mode=soft_correctness_mode,
        )
        results.append(stats)
    return results

