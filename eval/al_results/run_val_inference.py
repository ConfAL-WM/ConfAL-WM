#!/usr/bin/env python3
"""Run EVAC inference on validation episodes using a retrained checkpoint.

Generates predicted frames for each episode and saves them to
{output_dir}/{episode_id}/pred_frames/, ready for evaluate_al_round.py.

Usage:
  python eval/al_results/run_val_inference.py \\
    --checkpoint al_runs/robotwin_al/retrain/c3_diverse_oversampling/logs/checkpoints/epoch=5-step=4000.ckpt \\
    --config configs/agibotworld/al_robotwin.yaml \\
    --manifest al_runs/robotwin_al/manifests/al_val.json \\
    --output al_runs/robotwin_al/retrain/c3_diverse_oversampling/val_infer \\
    --num_shards 2 --workers_per_gpu 4 --gpus 0,1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(1, str(_REPO / "evac"))

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from al_pipeline.utils import ensure_dir, flatten_manifest_items, load_json, load_yaml
from evac.utils.general_utils import instantiate_from_config
from eval.al_results.utils.video_utils import save_video

# Reuse validated data-loading functions from the scoring pipeline
from al_pipeline.score_pool_with_c3 import (
    _count_available_gt_frames,
    _load_future_gt_frames,
    _load_external_action_arrays,
    _load_external_camera,
    _n_frames_cli_value,
    _parse_n_frames_value,
    _record_to_episode,
    _read_condition_frames,
    _resolve_n_frames_to_generate,
    _trim_generated_frame_dir,
)
from eval.confidence_eval.probe_inference import (
    encode_rgb_frames_to_latents,
    internal_samples_to_latent_array,
    load_action_h5,
    load_caminfo_json,
    load_generated_frames,
    load_model_with_probe,
)
from lvdm.data.utils import get_transformation_matrix_from_quat

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_frame_dir(frame_dir: Path) -> np.ndarray:
    files = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in files]
    return np.stack(frames, axis=0)


def _count_frame_files(frame_dir: Path) -> int:
    if not frame_dir.is_dir():
        return 0
    return sum(1 for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _to_video_frames(frames: Any) -> np.ndarray:
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 1)
    return frames


def _apply_traj_conditioning_overrides(model: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply pipeline traj geometry to an already-instantiated EVAC model.

    Retraining writes RoboTwin geometry into the generated training config, but
    val inference instantiates the base EVAC config and overlays checkpoint
    weights. Without this patch, inference silently falls back to EVAC's AgiBot
    default traj_gripper_z_offset=0.23.
    """
    applied: dict[str, Any] = {}
    for key in ("traj_gripper_z_offset", "traj_keypoint_scale", "traj_radius"):
        if key not in cfg:
            continue
        value = cfg[key]
        if key == "traj_radius":
            value = int(value)
        else:
            value = float(value)
        setattr(model, key, value)
        applied[key] = value
    return applied


def _traj_conditioning_from_pipeline_cfg(pipeline_cfg: dict[str, Any]) -> dict[str, Any]:
    traj_conditioning_cfg = dict((pipeline_cfg.get("retraining", {}) or {}).get("traj_conditioning", {}) or {})
    traj_conditioning_cfg.update(dict((pipeline_cfg.get("scoring", {}) or {}).get("traj_conditioning", {}) or {}))
    return traj_conditioning_cfg


def _array_stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().cpu().float()
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return {"finite": 0, "total": int(x.numel()), "min": None, "max": None, "mean": None}
    xf = x[finite]
    return {
        "finite": int(finite.sum().item()),
        "total": int(x.numel()),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
    }


def _mask_first_last(mask: torch.Tensor) -> dict[str, int | None]:
    mask = mask.detach().cpu().bool().flatten()
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        return {"first": None, "last": None}
    return {"first": int(idx[0].item()), "last": int(idx[-1].item())}


def _mask_windows(mask: torch.Tensor, windows: tuple[int, ...] = (64, 128, 200)) -> dict[str, int]:
    mask = mask.detach().cpu().bool().flatten()
    return {f"first_{w}": int(mask[: min(w, mask.numel())].sum().item()) for w in windows}


def _projection_debug_stats(
    *,
    action: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    sample_size: tuple[int, int],
    original_size: tuple[int, int],
    traj_gripper_z_offset: float = 0.23,
    traj_keypoint_scale: float = 0.1,
) -> dict[str, Any]:
    """Project EE keypoints like ddpm3d.get_traj and report why maps are blank."""
    action = action.detach().cpu().float()
    w2c = w2c.detach().cpu().float()
    intrinsic = intrinsic.detach().cpu().float()
    if w2c.dim() == 3:
        w2c = w2c.unsqueeze(0)
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)

    H, W = sample_size
    h_ori, w_ori = original_size
    K = intrinsic.clone()
    K[:, 0, 0] *= float(W) / float(w_ori)
    K[:, 0, 2] *= float(W) / float(w_ori)
    K[:, 1, 1] *= float(H) / float(h_ori)
    K[:, 1, 2] *= float(H) / float(h_ori)

    action = action[: w2c.shape[1]]
    ee_key_pts = torch.tensor(
        [
            [0, 0, 0, 1],
            [traj_keypoint_scale, 0, 0, 1],
            [0, traj_keypoint_scale, 0, 1],
            [0, 0, traj_keypoint_scale, 1],
        ],
        dtype=torch.float32,
    ).view(1, 1, 4, 4).permute(0, 1, 3, 2)
    cvt = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4)
    cvt[:, :, 2, 3] = float(traj_gripper_z_offset)
    pose_l = get_transformation_matrix_from_quat(action[:, 0:7]).unsqueeze(0)
    pose_r = get_transformation_matrix_from_quat(action[:, 8:15]).unsqueeze(0)

    out: dict[str, Any] = {
        "sample_size": [int(H), int(W)],
        "original_size": [int(h_ori), int(w_ori)],
        "num_action_frames": int(action.shape[0]),
        "num_camera_frames": int(w2c.shape[1]),
        "intrinsic_scaled": K[0].tolist(),
        "traj_gripper_z_offset": float(traj_gripper_z_offset),
        "traj_keypoint_scale": float(traj_keypoint_scale),
    }
    for name, pose in [("left", pose_l), ("right", pose_r)]:
        ee2cam = torch.matmul(torch.matmul(w2c, pose), cvt)
        pts = torch.matmul(ee2cam, ee_key_pts)
        proj = torch.matmul(K.unsqueeze(1), pts[:, :, :3, :])
        uv = (proj / pts[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2)
        base_uv = uv[:, :, 0, :]
        base_z = pts[:, :, 2, 0]
        point_z = pts[:, :, 2, :]
        point_in_frame = (
            torch.isfinite(uv).all(dim=-1)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] < W)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < H)
            & torch.isfinite(point_z)
        )
        in_frame = (
            torch.isfinite(base_uv).all(dim=-1)
            & (base_uv[..., 0] >= 0)
            & (base_uv[..., 0] < W)
            & (base_uv[..., 1] >= 0)
            & (base_uv[..., 1] < H)
        )
        center_h = torch.ones((action.shape[0], 4), dtype=action.dtype)
        if name == "left":
            center_h[:, :3] = action[:, 0:3]
        else:
            center_h[:, :3] = action[:, 8:11]
        center_cam = torch.matmul(w2c, center_h.view(1, -1, 4, 1))
        center_proj = torch.matmul(K.unsqueeze(1), center_cam[:, :, :3, :])
        center_uv = (center_proj / center_cam[:, :, 2:3, :])[:, :, :2, 0]
        center_z = center_cam[:, :, 2, 0]
        center_in_frame = (
            torch.isfinite(center_uv).all(dim=-1)
            & (center_uv[..., 0] >= 0)
            & (center_uv[..., 0] < W)
            & (center_uv[..., 1] >= 0)
            & (center_uv[..., 1] < H)
            & torch.isfinite(center_z)
        )
        any_keypoint_in_frame = point_in_frame.any(dim=-1)
        out[name] = {
            "base_in_frame": int(in_frame.sum().item()),
            "base_total": int(in_frame.numel()),
            "base_in_frame_ratio": float(in_frame.float().mean().item()),
            "base_windows": _mask_windows(in_frame),
            "base_first_last": _mask_first_last(in_frame),
            "center_in_frame": int(center_in_frame.sum().item()),
            "center_total": int(center_in_frame.numel()),
            "center_in_frame_ratio": float(center_in_frame.float().mean().item()),
            "center_windows": _mask_windows(center_in_frame),
            "center_first_last": _mask_first_last(center_in_frame),
            "any_keypoint_in_frame": int(any_keypoint_in_frame.sum().item()),
            "any_keypoint_total": int(any_keypoint_in_frame.numel()),
            "any_keypoint_in_frame_ratio": float(any_keypoint_in_frame.float().mean().item()),
            "any_keypoint_windows": _mask_windows(any_keypoint_in_frame),
            "any_keypoint_first_last": _mask_first_last(any_keypoint_in_frame),
            "u": _array_stats(base_uv[..., 0]),
            "v": _array_stats(base_uv[..., 1]),
            "z": _array_stats(base_z),
            "center_u": _array_stats(center_uv[..., 0]),
            "center_v": _array_stats(center_uv[..., 1]),
            "center_z": _array_stats(center_z),
            "xyz_world": {
                "x": _array_stats(action[:, 0] if name == "left" else action[:, 8]),
                "y": _array_stats(action[:, 1] if name == "left" else action[:, 9]),
                "z": _array_stats(action[:, 2] if name == "left" else action[:, 10]),
            },
        }
    return out


def _episode_has_latents(ep_out: Path) -> bool:
    return (ep_out / "latent_pred.npy").exists() and (ep_out / "latent_gt.npy").exists()


def _save_episode_latents(
    *,
    model: torch.nn.Module,
    model_cfg: Any,
    device: torch.device,
    ep_record: dict[str, Any],
    ep_out: Path,
    pred_dir: Path,
    n_previous: int,
    n_frames_to_generate: int,
    internal_pred_latents_raw: Any | None = None,
) -> bool:
    """Save latent_pred.npy / latent_gt.npy aligned with val inference frames.

    Step-3 scoring saves latents under pool_scores, but retrained AL eval reads
    them from val_infer/{episode_id}.  This helper lets val inference produce
    the same files, and can also backfill them from existing pred_frames without
    rerunning EVAC.
    """
    if _episode_has_latents(ep_out):
        return False

    pred_frames = load_generated_frames(pred_dir)
    if pred_frames is None or pred_frames.shape[0] == 0:
        raise FileNotFoundError(f"No generated frames found in {pred_dir}")
    n_frames = min(int(pred_frames.shape[0]), int(n_frames_to_generate))
    pred_frames = pred_frames[:n_frames]

    sample_size = tuple(model_cfg.data.params.train.params.sample_size)
    pred_latents = None
    if internal_pred_latents_raw is not None:
        pred_latents = internal_samples_to_latent_array(
            internal_pred_latents_raw,
            n_previous=n_previous,
            n_frames=n_frames,
        )
        if pred_latents is not None and int(pred_latents.shape[0]) != n_frames:
            pred_latents = None

    if not (ep_out / "latent_pred.npy").exists():
        if pred_latents is None:
            pred_latents = encode_rgb_frames_to_latents(
                model,
                pred_frames,
                device,
                sample_size=sample_size,
            )
        np.save(ep_out / "latent_pred.npy", pred_latents.astype(np.float32))

    if not (ep_out / "latent_gt.npy").exists():
        gt_frames = _load_future_gt_frames(ep_record, n_previous=n_previous, n_frames=n_frames)
        if gt_frames is None or int(gt_frames.shape[0]) != n_frames:
            gt_shape = None if gt_frames is None else tuple(gt_frames.shape)
            raise RuntimeError(
                f"Could not load GT frames for latent_gt: expected {n_frames}, got {gt_shape}"
            )
        latent_gt = encode_rgb_frames_to_latents(
            model,
            gt_frames,
            device,
            sample_size=sample_size,
        )
        np.save(ep_out / "latent_gt.npy", latent_gt.astype(np.float32))

    meta_path = ep_out / "inference_meta.json"
    meta = {
        "episode_id": ep_record.get("episode_id") or ep_out.name,
        "n_condition_frames": int(n_previous),
        "n_generated_frames": int(n_frames),
        "latent_pred_path": str(ep_out / "latent_pred.npy"),
        "latent_gt_path": str(ep_out / "latent_gt.npy"),
        "latent_saved_by": "run_val_inference",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return True


def _unlink_derived_episode_outputs(ep_out: Path, *, include_traj_debug: bool = False) -> None:
    names = ["latent_pred.npy", "latent_gt.npy", "inference_meta.json", "generation_meta.json"]
    if include_traj_debug:
        names.extend(["traj_condition.mp4", "traj_projection_debug.json"])
    for name in names:
        path = ep_out / name
        if path.exists():
            path.unlink()


def _write_generation_meta(
    *,
    ep_out: Path,
    ep_id: str,
    n_previous: int,
    n_frames_setting: int | str | None,
    n_frames_to_generate: int,
    n_frames_mode: str,
    action_frames: int,
    gt_frames: int | None,
    generated_frames: int,
    source: str,
    traj_conditioning: dict[str, Any] | None = None,
) -> None:
    with (ep_out / "generation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_id": ep_id,
                "n_condition_frames": int(n_previous),
                "n_frames_to_generate_requested": n_frames_setting,
                "n_frames_to_generate_resolved": int(n_frames_to_generate),
                "n_frames_to_generate_mode": n_frames_mode,
                "n_generated_frames": int(generated_frames),
                "action_frames_total": int(action_frames),
                "gt_frames_total": int(gt_frames) if gt_frames is not None else None,
                "meta_source": source,
                "traj_conditioning": traj_conditioning or {},
            },
            f,
            indent=2,
        )


def _frame_indices(n_frames: int, count: int = 5) -> list[int]:
    if n_frames <= 0:
        return []
    if n_frames == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, count)})


def _resize_rgb(frame: np.ndarray, width: int, height: int) -> Image.Image:
    img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    return img.resize((width, height), Image.Resampling.BILINEAR)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, fill=(255, 255, 255)) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, font=font, fill=fill)


def _make_compare_sheet(
    *,
    group: list[dict[str, Any]],
    out_path: Path,
    tile_w: int = 320,
    tile_h: int = 200,
    samples_per_row: int = 5,
) -> None:
    row_h = tile_h
    label_w = 420
    total_w = 170
    gap = 8
    header_h = 28
    canvas_w = label_w + samples_per_row * tile_w + total_w + gap * (samples_per_row + 2)
    canvas_h = header_h + len(group) * 2 * row_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        title_font = ImageFont.load_default()

    y = header_h
    for item in group:
        ep_id = item["episode_id"]
        n_previous = int(item.get("n_condition_frames", 0) or 0)
        for row_name, frames in (("GT", item["gt_frames"]), ("Pred", item["pred_frames"])):
            n_total = int(frames.shape[0])
            wrapped_ep = "\n".join(textwrap.wrap(ep_id, width=38))
            row_label = f"{wrapped_ep}\n{row_name}" if row_name == "GT" else row_name
            draw.text((8, y + 8), row_label, font=title_font, fill=(255, 255, 255))
            for tile_idx, frame_idx in enumerate(_frame_indices(n_total, samples_per_row)):
                x = label_w + gap + tile_idx * (tile_w + gap)
                img = _resize_rgb(frames[frame_idx], tile_w, tile_h)
                canvas.paste(img, (x, y))
                if row_name == "GT":
                    label = f"gt raw {n_previous + frame_idx} / future {frame_idx}"
                else:
                    label = f"pred {frame_idx}/{n_total - 1}"
                _draw_label(draw, (x + 6, y + 6), label)
            total_x = label_w + gap + samples_per_row * (tile_w + gap)
            total_text = f"future\n{n_total} frames"
            if row_name == "GT" and n_total > 0:
                total_text = f"future\n{n_total} frames\nraw {n_previous}-{n_previous + n_total - 1}"
            draw.text(
                (total_x, y + tile_h // 2 - 10),
                total_text,
                font=title_font,
                fill=(255, 255, 255),
            )
            y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def export_compare_outputs(
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    config_path: str | Path | None = None,
    compare_output_dir: str | Path | None = None,
    compare_count: int = 3,
    fps: int = 8,
) -> dict[str, Any]:
    out_root = Path(output_dir)
    compare_root = Path(compare_output_dir) if compare_output_dir else out_root.parent / f"{out_root.name}_compare"
    compare_root.mkdir(parents=True, exist_ok=True)
    episodes = flatten_manifest_items(load_json(manifest_path))
    selected = episodes[: max(0, int(compare_count))]
    data_format = "agibot"
    if config_path is not None:
        cfg = load_yaml(config_path)
        scoring_cfg = cfg.get("scoring", {})
        data_format = scoring_cfg.get("data_format") or cfg.get("evac", {}).get("data_format", "agibot")
    exported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    sheet_items: list[dict[str, Any]] = []
    for ep in selected:
        ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
        if not ep_id:
            continue
        ep_out = out_root / ep_id
        pred_dir = out_root / ep_id / "pred_frames"
        try:
            pred_frames = _read_frame_dir(pred_dir)
            video_path = compare_root / f"{ep_id}.mp4"
            save_video(pred_frames, video_path, fps=fps)
            exported.append({"episode_id": ep_id, "video": str(video_path)})

            meta = load_json(ep_out / "generation_meta.json") if (ep_out / "generation_meta.json").exists() else {}
            n_previous = int(meta.get("n_condition_frames", 4))
            ep_record = _record_to_episode(ep, data_format)
            gt_frames = _load_future_gt_frames(ep_record, n_previous=n_previous, n_frames=int(pred_frames.shape[0]))
            if gt_frames is None or int(gt_frames.shape[0]) == 0:
                raise RuntimeError("could not load GT frames for compare sheet")
            if gt_frames.dtype != np.uint8:
                gt_frames = (np.clip(gt_frames, 0, 1) * 255).astype(np.uint8)
            sheet_items.append(
                {
                    "episode_id": ep_id,
                    "gt_frames": gt_frames,
                    "pred_frames": pred_frames,
                    "n_condition_frames": n_previous,
                }
            )
        except Exception as exc:
            skipped.append({"episode_id": ep_id, "error": str(exc)})

    sheets: list[dict[str, Any]] = []
    for group_idx in range(0, len(sheet_items), 3):
        group = sheet_items[group_idx : group_idx + 3]
        sheet_path = compare_root / f"compare_{group_idx // 3:03d}.png"
        _make_compare_sheet(group=group, out_path=sheet_path)
        sheets.append(
            {
                "path": str(sheet_path),
                "episodes": [item["episode_id"] for item in group],
            }
        )
    summary = {
        "source_output_dir": str(out_root),
        "compare_output_dir": str(compare_root),
        "compare_count": int(compare_count),
        "fps": int(fps),
        "exported": exported,
        "compare_sheets": sheets,
        "skipped": skipped,
    }
    with (compare_root / "compare_export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[infer] exported {len(exported)} compare videos and {len(sheets)} sheets → {compare_root}")
    if skipped:
        print(f"[infer] WARNING: skipped {len(skipped)} compare exports; see {compare_root / 'compare_export_summary.json'}")
    return summary


def export_prediction_videos(
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    video_output_dir: str | Path | None = None,
    video_count: int = 3,
    fps: int = 8,
) -> dict[str, Any]:
    return export_compare_outputs(
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_path=None,
        compare_output_dir=video_output_dir,
        compare_count=video_count,
        fps=fps,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_retrained_state(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> tuple[str, int, int]:
    """Overlay a retrained Lightning/plain checkpoint onto an already-loaded EVAC model."""
    # Load large Lightning checkpoints on CPU first.  Mapping the whole 8GB+
    # checkpoint directly to CUDA duplicates model weights in GPU memory and can
    # OOM before load_state_dict has a chance to stream/copy the matched tensors.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()

    def _strip_model_prefix(k: str) -> str:
        return k[len("model."):] if k.startswith("model.") else k

    def _add_model_prefix(k: str) -> str:
        return k if k.startswith("model.") else f"model.{k}"

    variants = {
        "as_is": dict(state),
        "strip_model": {_strip_model_prefix(k): v for k, v in state.items()},
        "add_model": {_add_model_prefix(k): v for k, v in state.items()},
    }

    def _score(candidate: dict[str, torch.Tensor]) -> tuple[int, int]:
        exact = 0
        shape_ok = 0
        for k, v in candidate.items():
            if k in model_state:
                exact += 1
                if hasattr(v, "shape") and tuple(v.shape) == tuple(model_state[k].shape):
                    shape_ok += 1
        return shape_ok, exact

    prefix_mode, new_state = max(variants.items(), key=lambda item: _score(item[1]))
    shape_ok, exact = _score(new_state)
    # Filter out diffusion schedule buffers (betas, alphas_cumprod, etc.) —
    # these are harmless and get recomputed by register_schedule.
    _schedule_keys = {"betas", "alphas_cumprod", "alphas_cumprod_prev",
                      "posterior_variance", "posterior_log_variance_clipped",
                      "posterior_mean_coef1", "posterior_mean_coef2", "logvar"}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    missing_real = [k for k in missing if not any(sk in k for sk in _schedule_keys)]
    unexpected_real = [k for k in unexpected if not any(sk in k for sk in _schedule_keys)]
    if missing_real:
        s = str(missing_real[:3]) if len(missing_real) > 3 else str(missing_real)
        print(f"[infer] missing keys: {s}")
    if unexpected_real:
        s = str(unexpected_real[:3]) if len(unexpected_real) > 3 else str(unexpected_real)
        print(f"[infer] unexpected keys: {s}")
    return prefix_mode, shape_ok, exact


def _load_evac_model(checkpoint_path: str, evac_config_path: str, c3_probe_path: str | None, device: torch.device):
    """Load EVAC through the same model path as score_pool_with_c3, then overlay retrained weights."""
    if c3_probe_path:
        model, cfg, _probe_step = load_model_with_probe(evac_config_path, c3_probe_path, device)
    else:
        cfg = OmegaConf.load(evac_config_path)
        model_cfg = cfg.model
        model = instantiate_from_config(model_cfg)
        from evac.utils.general_utils import load_checkpoints

        model = load_checkpoints(model, model_cfg, ignore_mismatched_sizes=True)
        model = model.to(device).eval()

    prefix_mode, shape_ok, exact = _load_retrained_state(model, checkpoint_path, device)
    print(f"[infer] retrained checkpoint key mode: {prefix_mode} ({shape_ok} shape matches, {exact} key matches)")
    print(f"[infer] loaded retrained checkpoint: {checkpoint_path}")
    return model, cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(
    checkpoint_path: str,
    config_path: str,
    manifest_path: str,
    output_dir: str,
    *,
    max_episodes: int | None = None,
    device_str: str = "cuda:0",
    worker_id: int = 0,
    total_workers: int = 1,
    overwrite: bool = False,
    save_traj_videos: bool = False,
    save_latents: bool = True,
    n_frames_override: int | str | None = "auto",
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Parse pipeline config. Use the same EVAC config/probe path as
    # score_pool_with_c3.py, then overlay the retrained checkpoint.
    pipeline_cfg = load_yaml(config_path)
    scoring_cfg = pipeline_cfg.get("scoring", {})
    model_cfg_section = pipeline_cfg.get("model", {})
    c3_probe_cfg = pipeline_cfg.get("c3_probe", {})
    evac_config = (
        scoring_cfg.get("evac_config")
        or model_cfg_section.get("evac_config")
        or pipeline_cfg.get("retraining", {}).get("evac_train_config")
        or model_cfg_section.get("evac_train_config")
    )
    if bool(c3_probe_cfg.get("train_on_external", False)):
        c3_probe_path = (
            scoring_cfg.get("external_c3_probe_checkpoint")
            or c3_probe_cfg.get("external_checkpoint")
            or scoring_cfg.get("c3_probe_checkpoint")
            or model_cfg_section.get("c3_probe_checkpoint")
        )
    else:
        c3_probe_path = (
            scoring_cfg.get("c3_probe_checkpoint")
            or c3_probe_cfg.get("checkpoint")
            or model_cfg_section.get("c3_probe_checkpoint")
        )
    if not evac_config:
        raise ValueError("Config must define scoring/model.evac_config or retraining/model.evac_train_config")
    print(f"[infer] EVAC model config: {evac_config}")
    if c3_probe_path:
        print(f"[infer] C3 probe checkpoint for model init: {c3_probe_path}")

    model, model_cfg = _load_evac_model(checkpoint_path, evac_config, c3_probe_path, device)
    traj_conditioning_cfg = _traj_conditioning_from_pipeline_cfg(pipeline_cfg)
    applied_traj_conditioning = _apply_traj_conditioning_overrides(model, traj_conditioning_cfg)
    if applied_traj_conditioning:
        print(f"[infer] traj_conditioning overrides: {applied_traj_conditioning}")
    else:
        print(
            "[infer WARNING] no traj_conditioning override found in config; "
            f"using model defaults: traj_gripper_z_offset={getattr(model, 'traj_gripper_z_offset', None)}, "
            f"traj_keypoint_scale={getattr(model, 'traj_keypoint_scale', None)}, "
            f"traj_radius={getattr(model, 'traj_radius', None)}"
        )

    payload = load_json(manifest_path)
    episodes = flatten_manifest_items(payload)
    original_episode_count = len(episodes)
    if max_episodes is not None and original_episode_count < max_episodes and worker_id == 0:
        print(
            f"[infer] WARNING: --max_episodes={max_episodes} requested, "
            f"but manifest only has {original_episode_count} episodes"
        )
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if total_workers > 1:
        episodes = [ep for i, ep in enumerate(episodes) if i % total_workers == worker_id]
        print(f"[infer] worker {worker_id}/{total_workers}: {len(episodes)} episodes")

    data_format = scoring_cfg.get("data_format") or pipeline_cfg.get("evac", {}).get("data_format", "agibot")

    chunk = int(model_cfg.chunk)
    n_previous = int(model_cfg.n_previous)
    n_frames_setting = _parse_n_frames_value(
        n_frames_override if n_frames_override is not None else scoring_cfg.get("n_frames_to_generate", "auto")
    )
    cfg_val = float(scoring_cfg.get("cfg", 1.0))
    gr_val = float(scoring_cfg.get("gr", 0.7))
    ddim_steps = int(scoring_cfg.get("ddim_steps", 27))

    out_root = ensure_dir(output_dir)
    count = 0
    skipped = 0
    latents_backfilled = 0
    latent_failures = 0

    for ep in tqdm(episodes, desc="[infer]", unit="ep"):
        ep_id = ep.get("episode_id", "")
        if not ep_id:
            continue
        ep_out = ensure_dir(out_root / ep_id)
        pred_dir = ep_out / "pred_frames"

        try:
            ep_record = _record_to_episode(ep, data_format)

            # Load condition frames (scorer's _read_condition_frames takes a path)
            if data_format == "external_worldmodel":
                cond_path = ep_record.get("frames_dir", "")
            else:
                cond_path = ep_record["video_mp4"]
            img = _read_condition_frames(cond_path, n_previous)

            if data_format == "external_worldmodel":
                allow_incompat = bool(scoring_cfg.get("allow_incompatible_actions_for_debug", False))
                action, delta_action = _load_external_action_arrays(
                    ep_record,
                    n_previous,
                    chunk,
                    allow_incompatible=allow_incompat,
                )
                n = int(action.shape[0])
                c2w, w2c, intrinsic = _load_external_camera(ep_record, n)
            else:
                action, delta_action = load_action_h5(
                    ep_record["action_h5"],
                    int(scoring_cfg.get("n_chunk", -1)),
                    chunk,
                    n_previous,
                )
                n = int(action.shape[0])
                c2w, w2c, intrinsic = load_caminfo_json(
                    ep_record["extrinsic"],
                    ep_record["intrinsic"],
                    n,
                )

            gt_frame_count = _count_available_gt_frames(ep_record)
            n_frames_to_generate, n_frames_mode = _resolve_n_frames_to_generate(
                n_frames_setting,
                action_frames=n,
                n_previous=n_previous,
                gt_frames=gt_frame_count,
            )
            existing_frame_count = _count_frame_files(pred_dir)
            if existing_frame_count > 0:
                if overwrite or existing_frame_count < n_frames_to_generate:
                    reason = "overwrite" if overwrite else f"only {existing_frame_count}/{n_frames_to_generate} frames"
                    print(f"[infer] {ep_id}: regenerating pred_frames ({reason})")
                    shutil.rmtree(pred_dir)
                    _unlink_derived_episode_outputs(ep_out, include_traj_debug=True)
                else:
                    current_frame_count = existing_frame_count
                    if existing_frame_count > n_frames_to_generate:
                        print(
                            f"[infer] {ep_id}: trimming cached pred_frames "
                            f"{existing_frame_count}->{n_frames_to_generate} for n_frames={n_frames_mode}"
                        )
                        _trim_generated_frame_dir(pred_dir, n_frames_to_generate)
                        current_frame_count = n_frames_to_generate
                        _unlink_derived_episode_outputs(ep_out, include_traj_debug=True)
                    if save_latents and not _episode_has_latents(ep_out):
                        try:
                            if _save_episode_latents(
                                model=model,
                                model_cfg=model_cfg,
                                device=device,
                                ep_record=ep_record,
                                ep_out=ep_out,
                                pred_dir=pred_dir,
                                n_previous=n_previous,
                                n_frames_to_generate=n_frames_to_generate,
                            ):
                                latents_backfilled += 1
                        except Exception as exc:
                            latent_failures += 1
                            print(f"[infer WARNING] {ep_id}: latent backfill failed: {exc}")
                    _write_generation_meta(
                        ep_out=ep_out,
                        ep_id=ep_id,
                        n_previous=n_previous,
                        n_frames_setting=n_frames_setting,
                        n_frames_to_generate=n_frames_to_generate,
                        n_frames_mode=n_frames_mode,
                        action_frames=n,
                        gt_frames=gt_frame_count,
                        generated_frames=current_frame_count,
                        source="cached_or_trimmed",
                        traj_conditioning=applied_traj_conditioning,
                    )
                    count += 1
                    continue
            pred_dir = ensure_dir(pred_dir)

            # Pad if needed
            needed = n_previous + n_frames_to_generate
            if n < needed:
                pad_len = needed - n
                action = torch.cat([action, action[-1:].repeat(pad_len, 1)], dim=0)
                delta_action = torch.cat([delta_action, delta_action[-1:].repeat(pad_len, 1)], dim=0)
                c2w = torch.cat([c2w, c2w[-1:].repeat(pad_len, 1, 1)], dim=0)
                w2c = torch.cat([w2c, w2c[-1:].repeat(pad_len, 1, 1)], dim=0)
                n = int(action.shape[0])

            n_chunk_to_pred = int(math.ceil(float(n_frames_to_generate) / chunk))

            # Run inference
            original_size = tuple(int(x) for x in img.shape[-2:])
            with (
                open(os.devnull, "w") as quiet,
                contextlib.redirect_stdout(quiet),
                contextlib.redirect_stderr(quiet),
            ):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    infer_out = model.inference(
                        model_cfg,
                        img.to(device),
                        action,
                        delta_action,
                        c2w,
                        w2c,
                        intrinsic,
                        str(pred_dir),
                        n_chunk_to_pred,
                        chunk=chunk,
                        n_previous=n_previous,
                        n_valid=n - n_previous,
                        unconditional_guidance_scale=cfg_val,
                        guidance_rescale=gr_val,
                        ddim_steps=ddim_steps,
                        saving_tag="",
                        saving_video=False,
                        video_dir=None,
                        lambda_guide=0.0,
                        return_latents=bool(save_latents),
                    )
            gen_frame_count = _count_frame_files(pred_dir)
            if gen_frame_count < n_frames_to_generate:
                raise RuntimeError(
                    f"Generated only {gen_frame_count}/{n_frames_to_generate} frames for {ep_id}"
                )
            if gen_frame_count > n_frames_to_generate:
                _trim_generated_frame_dir(pred_dir, n_frames_to_generate)
            if save_traj_videos and isinstance(infer_out, tuple) and len(infer_out) >= 2:
                save_video(_to_video_frames(infer_out[1])[:n_frames_to_generate], ep_out / "traj_condition.mp4", fps=8)
                debug = _projection_debug_stats(
                    action=action,
                    w2c=w2c,
                    intrinsic=intrinsic,
                    sample_size=tuple(model_cfg.data.params.train.params.sample_size),
                    original_size=original_size,
                    traj_gripper_z_offset=float(getattr(model, "traj_gripper_z_offset", 0.23)),
                    traj_keypoint_scale=float(getattr(model, "traj_keypoint_scale", 0.1)),
                )
                with (ep_out / "traj_projection_debug.json").open("w", encoding="utf-8") as f:
                    json.dump(debug, f, indent=2)
            if save_latents:
                internal_pred_latents_raw = None
                if isinstance(infer_out, tuple) and len(infer_out) >= 3:
                    internal_pred_latents_raw = infer_out[2]
                try:
                    if _save_episode_latents(
                        model=model,
                        model_cfg=model_cfg,
                        device=device,
                        ep_record=ep_record,
                        ep_out=ep_out,
                        pred_dir=pred_dir,
                        n_previous=n_previous,
                        n_frames_to_generate=n_frames_to_generate,
                        internal_pred_latents_raw=internal_pred_latents_raw,
                    ):
                        latents_backfilled += 1
                except Exception as exc:
                    latent_failures += 1
                    print(f"[infer WARNING] {ep_id}: latent save failed: {exc}")
            _write_generation_meta(
                ep_out=ep_out,
                ep_id=ep_id,
                n_previous=n_previous,
                n_frames_setting=n_frames_setting,
                n_frames_to_generate=n_frames_to_generate,
                n_frames_mode=n_frames_mode,
                action_frames=n,
                gt_frames=gt_frame_count,
                generated_frames=int(min(_count_frame_files(pred_dir), n_frames_to_generate)),
                source="generated",
                traj_conditioning=applied_traj_conditioning,
            )
            count += 1
            # Free GPU memory cache between episodes to prevent OOM fragmentation
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"[infer] {ep_id}: {e}")
            traceback.print_exc()
            skipped += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    summary = {
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "output_dir": str(out_root),
        "worker_id": worker_id,
        "total_workers": total_workers,
        "episodes_requested": len(episodes),
        "episodes_completed": count,
        "episodes_skipped": skipped,
        "n_frames_to_generate": n_frames_setting,
        "save_latents": bool(save_latents),
        "latents_saved_or_backfilled": latents_backfilled,
        "latent_failures": latent_failures,
        "traj_conditioning": applied_traj_conditioning,
    }
    if total_workers > 1:
        save_json_path = out_root / f"inference_summary_worker_{worker_id:03d}_of_{total_workers:03d}.json"
    else:
        save_json_path = out_root / "inference_summary.json"
    with open(save_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[infer] done: {count} completed, {skipped} skipped → {save_json_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EVAC val inference for AL eval")
    parser.add_argument("--checkpoint", required=True, help="Retrained EVAC checkpoint (.ckpt)")
    parser.add_argument("--config", required=True, help="Pipeline config YAML (al_robotwin.yaml / al_agibot.yaml)")
    parser.add_argument("--manifest", required=True, help="Val manifest JSON (al_val.json)")
    parser.add_argument("--output", default=None, help="Output directory (default: derived from checkpoint)")
    parser.add_argument("--max_episodes", "--max-episodes", dest="max_episodes", type=int, default=None, help="Only run inference for the first N manifest episodes")
    parser.add_argument(
        "--n_frames_to_generate",
        "--n-frames-to-generate",
        "--n_frames",
        "--n-frames",
        "--num_frames",
        "--num-frames",
        "--num_inference_frames",
        "--num-inference-frames",
        dest="n_frames_to_generate",
        type=_n_frames_cli_value,
        default="auto",
        help=(
            "Frames to generate per episode: positive integer or 'auto'. "
            "Default: auto (episode num_frames - n_previous; ignores legacy config 64/200)."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_shards", "--num-shards", dest="num_shards", type=int, default=1, help="Number of GPU shards")
    parser.add_argument("--workers_per_gpu", "--workers-per-gpu", dest="workers_per_gpu", type=int, default=1, help="Workers per GPU")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs")
    parser.add_argument(
        "--export_compare",
        "--export-compare",
        "--export_videos",
        "--export-videos",
        nargs="?",
        const=3,
        default=None,
        type=int,
        metavar="N",
        dest="export_compare",
        help=(
            "Export compare outputs for the first N manifest episodes: pred mp4 files "
            "plus GT-vs-pred contact sheets grouped by 3 episodes. "
            "Default when flag is present: 3. Legacy alias: --export_videos."
        ),
    )
    parser.add_argument("--video_output", "--video-output", default=None, help="Compare output directory (default: {output}_compare)")
    parser.add_argument("--video_fps", "--video-fps", type=int, default=8, help="Preview video FPS")
    parser.add_argument("--save_traj_videos", "--save-traj-videos", action="store_true", help="Save per-episode traj_condition.mp4 for action/camera debugging")
    parser.add_argument(
        "--save_latents",
        "--save-latents",
        dest="save_latents",
        action="store_true",
        default=True,
        help="Save/backfill latent_pred.npy and latent_gt.npy for latent_loss eval (default: enabled)",
    )
    parser.add_argument(
        "--no_save_latents",
        "--no-save-latents",
        dest="save_latents",
        action="store_false",
        help="Disable latent export/backfill during val inference",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate episodes even when pred_frames already exists")
    parser.add_argument("--_worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_gpu", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_total-workers", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    output = args.output
    if output is None:
        ckpt = Path(args.checkpoint)
        retrain_dir = ckpt.parents[2]
        output = str(retrain_dir / "val_infer")

    total_workers = args.num_shards * args.workers_per_gpu
    if total_workers > 1:
        gpu_list = [s.strip() for s in args.gpus.split(",")] if args.gpus else [str(i) for i in range(args.num_shards)]
        # Count total episodes
        payload = load_json(args.manifest)
        all_eps = flatten_manifest_items(payload)
        manifest_count = len(all_eps)
        if args.max_episodes is not None and manifest_count < args.max_episodes:
            print(
                f"[infer] WARNING: --max_episodes={args.max_episodes} requested, "
                f"but manifest only has {manifest_count} episodes"
            )
        if args.max_episodes is not None:
            all_eps = all_eps[:args.max_episodes]
        per_worker = math.ceil(len(all_eps) / float(total_workers)) if total_workers else 0
        print(f"[infer] {len(all_eps)} val episodes → up to {per_worker} per worker × {total_workers} workers")
        print(f"[infer] GPUs: {gpu_list} | {args.workers_per_gpu} workers/GPU")
        procs = []
        for sid in range(args.num_shards):
            for wid in range(args.workers_per_gpu):
                worker_id = sid * args.workers_per_gpu + wid
                cmd = [
                    sys.executable, __file__,
                    "--checkpoint", args.checkpoint,
                    "--config", args.config,
                    "--manifest", args.manifest,
                    "--output", output,
                    "--num-shards", "1",
                    "--workers-per-gpu", "1",
                    "--_worker-id", str(worker_id),
                    "--_gpu", gpu_list[sid],
                    "--_total-workers", str(total_workers),
                ]
                if args.max_episodes is not None:
                    cmd += ["--max-episodes", str(args.max_episodes)]
                if args.overwrite:
                    cmd += ["--overwrite"]
                if args.save_traj_videos:
                    cmd += ["--save-traj-videos"]
                if args.n_frames_to_generate is not None:
                    cmd += ["--n-frames-to-generate", str(args.n_frames_to_generate)]
                if not args.save_latents:
                    cmd += ["--no-save-latents"]
                p = subprocess.Popen(cmd)
                procs.append((p, sid, wid, worker_id))
        for p, sid, wid, wid_full in procs:
            p.wait()
            if p.returncode != 0:
                print(f"[infer] WARNING: worker {wid_full} (shard {sid}/{wid}) exited with code {p.returncode}")
        out_root = ensure_dir(output)
        worker_summaries = []
        missing_summaries = []
        for worker_id in range(total_workers):
            summary_path = out_root / f"inference_summary_worker_{worker_id:03d}_of_{total_workers:03d}.json"
            if summary_path.exists():
                summary = load_json(summary_path)
                summary["summary_path"] = str(summary_path)
                worker_summaries.append(summary)
            else:
                missing_summaries.append(str(summary_path))

        failed_workers = [
            {
                "worker_id": wid_full,
                "shard_id": sid,
                "worker_index_on_gpu": wid,
                "returncode": p.returncode,
            }
            for p, sid, wid, wid_full in procs
            if p.returncode != 0
        ]
        merged = {
            "checkpoint": args.checkpoint,
            "manifest": args.manifest,
            "output_dir": str(out_root),
            "total_workers": total_workers,
            "episodes_requested": sum(int(s.get("episodes_requested", 0)) for s in worker_summaries),
            "episodes_completed": sum(int(s.get("episodes_completed", 0)) for s in worker_summaries),
            "episodes_skipped": sum(int(s.get("episodes_skipped", 0)) for s in worker_summaries),
            "workers_completed": len(worker_summaries),
            "workers_failed": failed_workers,
            "missing_worker_summaries": missing_summaries,
            "worker_summaries": worker_summaries,
        }
        merged_path = out_root / "inference_summary.json"
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(
            f"[infer] all workers finished: {merged['episodes_completed']} completed, "
            f"{merged['episodes_skipped']} skipped → {merged_path}"
        )
        if failed_workers or missing_summaries:
            sys.exit(1)
        if args.export_compare is not None:
            export_compare_outputs(
                output_dir=output,
                manifest_path=args.manifest,
                config_path=args.config,
                compare_output_dir=args.video_output,
                compare_count=args.export_compare,
                fps=args.video_fps,
            )
        return

    # Single-worker mode
    total_workers = args._total_workers
    if total_workers <= 1:
        total_workers = max(1, args.num_shards * args.workers_per_gpu)
    if args._gpu:
        device_str = f"cuda:{args._gpu}"
    elif args.gpus:
        device_str = f"cuda:{args.gpus.split(',')[0].strip()}"
    else:
        device_str = args.device
    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=output,
        max_episodes=args.max_episodes,
        device_str=device_str,
        worker_id=args._worker_id,
        total_workers=total_workers,
        overwrite=args.overwrite,
        save_traj_videos=args.save_traj_videos,
        save_latents=args.save_latents,
        n_frames_override=args.n_frames_to_generate,
    )
    if args.export_compare is not None and args._total_workers <= 1:
        export_compare_outputs(
            output_dir=output,
            manifest_path=args.manifest,
            config_path=args.config,
            compare_output_dir=args.video_output,
            compare_count=args.export_compare,
            fps=args.video_fps,
        )


if __name__ == "__main__":
    main()
