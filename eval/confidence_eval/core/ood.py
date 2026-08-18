from __future__ import annotations

from typing import Sequence

import numpy as np

from .metrics import compute_binary_auprc, compute_binary_auroc


def is_ood_record(record: dict) -> bool:
    split = str(record.get("split", "")).lower()
    ood_type = record.get("ood_type")
    return split == "ood" or split.startswith("ood_") or (ood_type is not None and str(ood_type) != "")


def compute_frame_scores(risk_map: np.ndarray, mode: str = "topk_mean_risk", topk_ratio: float = 0.1) -> np.ndarray:
    risk = np.asarray(risk_map, dtype=np.float64)
    if mode == "mean_risk":
        return risk.mean(axis=(1, 2))
    if mode != "topk_mean_risk":
        raise ValueError(f"Unsupported OOD frame_score_mode: {mode}")
    flat = risk.reshape(risk.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * topk_ratio)))
    part = np.partition(flat, flat.shape[1] - k, axis=1)
    topk = part[:, -k:]
    return topk.mean(axis=1)


def compute_trajectory_score(frame_scores: np.ndarray, mode: str = "mean") -> float:
    scores = np.asarray(frame_scores, dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "max":
        return float(np.max(scores))
    raise ValueError(f"Unsupported OOD trajectory_score_mode: {mode}")


def compute_ood_summary(
    episodes: Sequence[dict],
    frame_score_mode: str = "topk_mean_risk",
    frame_topk_ratio: float = 0.1,
    trajectory_score_mode: str = "mean",
) -> dict:
    if not episodes:
        return {}
    frame_scores_all: list[np.ndarray] = []
    frame_labels_all: list[np.ndarray] = []
    traj_scores_all: list[float] = []
    traj_labels_all: list[int] = []

    for episode in episodes:
        label = 1 if bool(episode["is_ood"]) else 0
        frame_scores = compute_frame_scores(
            risk_map=episode["risk_map"],
            mode=frame_score_mode,
            topk_ratio=frame_topk_ratio,
        )
        traj_score = compute_trajectory_score(frame_scores, mode=trajectory_score_mode)
        frame_scores_all.append(frame_scores)
        frame_labels_all.append(np.full(frame_scores.shape, label, dtype=np.int32))
        traj_scores_all.append(traj_score)
        traj_labels_all.append(label)

    merged_frame_scores = np.concatenate(frame_scores_all, axis=0)
    merged_frame_labels = np.concatenate(frame_labels_all, axis=0)
    traj_scores_arr = np.asarray(traj_scores_all, dtype=np.float64)
    traj_labels_arr = np.asarray(traj_labels_all, dtype=np.int32)

    return {
        "frame_ood_auroc": compute_binary_auroc(merged_frame_scores, merged_frame_labels),
        "frame_ood_auprc": compute_binary_auprc(merged_frame_scores, merged_frame_labels),
        "trajectory_ood_auroc": compute_binary_auroc(traj_scores_arr, traj_labels_arr),
        "trajectory_ood_auprc": compute_binary_auprc(traj_scores_arr, traj_labels_arr),
        "frame_scores": merged_frame_scores,
        "frame_labels": merged_frame_labels,
        "trajectory_scores": traj_scores_arr,
        "trajectory_labels": traj_labels_arr,
    }

