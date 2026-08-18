from __future__ import annotations

from typing import Sequence

import numpy as np


def summarize_losses(losses: np.ndarray | Sequence[float]) -> dict[str, float | None]:
    """Mean / median / cvar90 of a 1-D loss array."""
    if isinstance(losses, (list, tuple)):
        losses = np.asarray(losses, dtype=np.float32)
    if losses.size == 0:
        return {"mean": None, "median": None, "cvar90": None}
    return {
        "mean": float(np.mean(losses)),
        "median": float(np.median(losses)),
        "cvar90": float(losses[losses >= np.quantile(losses, 0.9)].mean()),
    }


def compute_latent_loss(
    pred_latents: np.ndarray,
    gt_latents: np.ndarray,
) -> dict[str, float | None]:
    """Per-episode MSE between predicted and ground-truth latent sequences.

    pred_latents / gt_latents: [T, D] float32 arrays.
    Returns mean across time steps, summarized via mean/median/cvar90.
    """
    if pred_latents.shape != gt_latents.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_latents.shape} vs gt {gt_latents.shape}"
        )
    loss_per_step = np.square(pred_latents - gt_latents).mean(axis=-1)  # [T]
    return summarize_losses(loss_per_step)
