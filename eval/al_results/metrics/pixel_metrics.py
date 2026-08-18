from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils.video_utils import load_frames, normalize_frames


def summarize_pixel_errors(errors: np.ndarray | Sequence[float]) -> dict[str, float | None]:
    """Mean / median / cvar90 of a 1-D error array."""
    if isinstance(errors, (list, tuple)):
        errors = np.asarray(errors, dtype=np.float32)
    if errors.size == 0:
        return {"mean": None, "median": None, "cvar90": None}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "cvar90": float(errors[errors >= np.quantile(errors, 0.9)].mean()),
    }


def compute_pixel_mae(
    pred_video: np.ndarray,
    gt_video: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Per-frame pixel MAE between predicted and ground-truth videos.

    pred_video / gt_video: [T, H, W, C] in [0,1] float32.
    mask: optional [T, H, W] boolean to restrict MAE to action regions.

    Returns summary dict with 'mean', 'median', 'cvar90'.
    If mask is given, also returns 'masked_mean', 'masked_median', 'masked_cvar90'.
    """
    if pred_video.shape != gt_video.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_video.shape} vs gt {gt_video.shape}"
        )
    per_frame_mae = np.abs(pred_video - gt_video).mean(axis=(1, 2, 3))  # [T]
    result = summarize_pixel_errors(per_frame_mae)

    if mask is not None:
        if mask.shape[:2] != pred_video.shape[:2]:
            mask = _resize_mask_batch(mask, pred_video.shape[1:3])
        masked_errors = []
        for t in range(len(pred_video)):
            m = mask[t] if t < len(mask) else mask[-1]
            diff = np.abs(pred_video[t] - gt_video[t]).mean(axis=-1)  # [H,W]
            masked_errors.append(float(diff[m].mean()))
        masked_summary = summarize_pixel_errors(masked_errors)
        for k, v in masked_summary.items():
            result[f"masked_{k}"] = v

    return result


def _resize_mask_batch(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize [T, H, W] mask to target (H, W) using nearest-neighbour."""
    import cv2

    h, w = target_shape
    out = np.zeros((mask.shape[0], h, w), dtype=mask.dtype)
    for t in range(mask.shape[0]):
        out[t] = cv2.resize(mask[t].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(mask.dtype)
    return out


def compute_pixel_mae_from_paths(
    pred_path: str | Path,
    gt_path: str | Path,
    mask_path: str | Path | None = None,
    max_frames: int | None = None,
    pred_start_frame: int = 0,
    gt_start_frame: int = 0,
) -> dict[str, float | None]:
    """Load videos from disk and compute pixel MAE.

    If pred and GT have mismatched spatial shapes (e.g. cross-embodiment eval),
    GT is resized to match the prediction resolution with bilinear interpolation.
    """
    pred_read_max = None if max_frames is None else max_frames + max(0, pred_start_frame)
    gt_read_max = None if max_frames is None else max_frames + max(0, gt_start_frame)
    pred = normalize_frames(load_frames(pred_path, pred_read_max))
    gt = normalize_frames(load_frames(gt_path, gt_read_max))
    if pred_start_frame > 0:
        pred = pred[pred_start_frame:]
    if gt_start_frame > 0:
        gt = gt[gt_start_frame:]
    if max_frames is not None:
        pred = pred[:max_frames]
        gt = gt[:max_frames]

    if pred.shape[0] != gt.shape[0]:
        t = min(pred.shape[0], gt.shape[0])
        pred = pred[:t]
        gt = gt[:t]

    # Align spatial shapes: resize GT to match pred
    if pred.shape[1:3] != gt.shape[1:3]:
        import cv2

        t = pred.shape[0]
        ph, pw = pred.shape[1], pred.shape[2]
        gt_resized = np.zeros((t, ph, pw, 3), dtype=np.float32)
        for i in range(t):
            gt_resized[i] = cv2.resize(gt[i], (pw, ph), interpolation=cv2.INTER_LINEAR)
        gt = gt_resized

    mask = None
    if mask_path is not None:
        mask_data = np.load(mask_path)
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, ...]
        mask = mask_data.astype(bool)
    return compute_pixel_mae(pred, gt, mask)
