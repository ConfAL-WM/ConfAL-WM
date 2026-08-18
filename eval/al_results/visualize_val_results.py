#!/usr/bin/env python3
"""Visualize validation predictions, trajectory overlays, and C3 confidence maps.

This script is intentionally lightweight: it reads existing outputs from
run_val_inference.py or score_pool_with_c3.py and never launches EVAC/C3.

Default mode writes a GT-vs-prediction contact sheet. Add --vis-traj to include
trajectory-condition rows. Add --save_conf_map/--save_conf_video to export C3
confidence and risk visualizations from saved conf_map.npy files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import ensure_dir, flatten_manifest_items, load_json, load_yaml

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _episode_id(item: dict[str, Any]) -> str:
    return str(item.get("episode_id") or item.get("ep_id") or item.get("folder") or "")


def _apply_rewrites(path: Path, rewrites: list[tuple[str, str]] | None = None) -> Path:
    text = str(path)
    for src, dst in rewrites or []:
        if text.startswith(src):
            return Path(dst + text[len(src) :])
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _to_uint8_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        out = arr
    else:
        out = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if out.ndim != 4 or out.shape[-1] < 3:
        raise ValueError(f"frames must have shape [T,H,W,3], got {out.shape}")
    return out[..., :3]


def _read_frame_dir(frame_dir: Path, limit: int | None = None) -> np.ndarray:
    paths = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]
    return np.stack(frames, axis=0)


def _read_video(video_path: Path, limit: int | None = None) -> np.ndarray:
    frames: list[np.ndarray] = []
    last_error: Exception | None = None
    if imageio is not None:
        try:
            reader = imageio.get_reader(str(video_path))
            try:
                for idx, frame in enumerate(reader):
                    if limit is not None and idx >= int(limit):
                        break
                    frames.append(np.asarray(frame, dtype=np.uint8)[..., :3])
            finally:
                reader.close()
            if frames:
                return np.stack(frames, axis=0)
        except Exception as exc:
            last_error = exc
            frames = []
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(str(video_path))
            try:
                while True:
                    if limit is not None and len(frames) >= int(limit):
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
            if frames:
                return np.stack(frames, axis=0)
        except Exception as exc:
            last_error = exc
            frames = []
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        try:
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    str(video_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            size_text = probe.stdout.strip().splitlines()[0]
            width, height = [int(x) for x in size_text.split("x")[:2]]
            cmd = [ffmpeg, "-v", "error", "-i", str(video_path)]
            if limit is not None:
                cmd.extend(["-frames:v", str(int(limit))])
            cmd.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            raw = np.frombuffer(proc.stdout, dtype=np.uint8)
            frame_size = width * height * 3
            usable = raw.size // frame_size
            if usable > 0:
                return raw[: usable * frame_size].reshape(usable, height, width, 3)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"No video backend could read {video_path}: {last_error}") from last_error
    raise RuntimeError("Reading mp4 files needs imageio, cv2, or ffmpeg")


def _load_frame_source(path: Path, limit: int | None = None) -> np.ndarray:
    if path.is_dir():
        return _read_frame_dir(path, limit=limit)
    if path.is_file():
        if path.suffix.lower() == ".mp4":
            return _read_video(path, limit=limit)
        return np.expand_dims(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8), axis=0)
    raise FileNotFoundError(str(path))


def _load_pred_frames(ep_out: Path) -> np.ndarray:
    pred_dir = ep_out / "pred_frames"
    if pred_dir.is_dir():
        return _read_frame_dir(pred_dir)
    for name in ("pred_video.mp4", "pred.mp4", "outputs.mp4"):
        path = ep_out / name
        if path.exists():
            return _read_video(path)
    raise FileNotFoundError(f"No pred_frames or pred video found under {ep_out}")


def _candidate_gt_sources(item: dict[str, Any], rewrites: list[tuple[str, str]] | None = None) -> list[tuple[Path, bool]]:
    sources: list[tuple[Path, bool]] = []

    for key in ("gt_frames_dir", "frames_dir"):
        if item.get(key):
            sources.append((_apply_rewrites(Path(str(item[key])), rewrites), True))
    for key in ("gt_video",):
        if item.get(key):
            sources.append((_apply_rewrites(Path(str(item[key])), rewrites), True))

    if item.get("path_gt"):
        path_gt = _apply_rewrites(Path(str(item["path_gt"])), rewrites)
        frames_rel = str((item.get("files") or {}).get("frames_dir") or "video")
        sources.append((path_gt / frames_rel, True))

    if item.get("episode_dir"):
        ep_dir = _apply_rewrites(Path(str(item["episode_dir"])), rewrites)
        sources.append((ep_dir / "frames", True))
        sources.append((ep_dir / "head_color.mp4", True))

    if item.get("source_path"):
        source_path = _apply_rewrites(Path(str(item["source_path"])), rewrites)
        sources.append((source_path / "frames", True))
        sources.append((source_path / "head_color.mp4", True))

    if item.get("path"):
        path = _apply_rewrites(Path(str(item["path"])), rewrites)
        files = item.get("files") or {}
        if files.get("frames_dir"):
            sources.append((path / str(files["frames_dir"]), True))
        if files.get("video"):
            sources.append((path / str(files["video"]), True))
        sources.append((path / "frames", True))
        sources.append((path / "head_color.mp4", True))

    return sources


def _load_gt_frames(
    item: dict[str, Any],
    *,
    n_previous: int,
    n_frames: int,
    rewrites: list[tuple[str, str]] | None = None,
) -> np.ndarray | None:
    for source, slice_future in _candidate_gt_sources(item, rewrites):
        try:
            raw = _load_frame_source(source)
            if slice_future and raw.shape[0] > n_previous:
                raw = raw[n_previous : n_previous + n_frames]
            else:
                raw = raw[:n_frames]
            if raw.shape[0] > 0:
                return _to_uint8_frames(raw)
        except Exception:
            continue
    return None


def _load_traj_frames(ep_out: Path, n_frames: int) -> np.ndarray | None:
    path = ep_out / "traj_condition.mp4"
    if not path.exists():
        return None
    try:
        return _to_uint8_frames(_read_video(path, limit=n_frames))
    except Exception:
        return None


def _n_previous(ep_out: Path) -> int:
    for name in ("generation_meta.json", "inference_meta.json"):
        meta = _read_json(ep_out / name)
        if meta.get("n_condition_frames") is not None:
            return int(meta["n_condition_frames"])
    return 4


def _frame_indices(n_frames: int, count: int) -> list[int]:
    if n_frames <= 0:
        return []
    if n_frames == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, min(count, n_frames))})


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=34) or [""]:
            while draw.textbbox((0, 0), line, font=font)[2] > max_width and len(line) > 4:
                line = line[:-1]
            lines.append(line)
    return "\n".join(lines)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int = 210,
) -> None:
    x, y = xy
    text = _fit_text(draw, text, max_width, font)
    bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=3)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.multiline_text((x, y), text, font=font, fill=(255, 255, 255), spacing=3)


def _resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.asarray(Image.fromarray(frame.astype(np.uint8), mode="RGB").resize((width, height), Image.Resampling.BILINEAR))


def _resize_scalar(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    img = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    return np.asarray(img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def _colorize(arr: np.ndarray, *, cmap_name: str, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    if plt is not None:
        rgb = plt.get_cmap(cmap_name)(x)[..., :3]
        return (rgb * 255.0).round().astype(np.uint8)
    if cmap_name in {"inferno", "magma"}:
        return np.stack(
            [
                (255 * x).astype(np.uint8),
                (120 * np.sqrt(x)).astype(np.uint8),
                (50 * (1.0 - x)).astype(np.uint8),
            ],
            axis=-1,
        )
    return np.stack(
        [
            (255 * (1.0 - x)).astype(np.uint8),
            (80 * (1.0 - np.abs(x - 0.5) * 2.0)).astype(np.uint8),
            (255 * x).astype(np.uint8),
        ],
        axis=-1,
    )


def _error_vmax(value_map: np.ndarray | None, percentile: float = 98.0) -> float:
    if value_map is None:
        return 1.0
    finite = np.asarray(value_map, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(finite, percentile))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def _blank_tile(size: tuple[int, int], text: str) -> np.ndarray:
    width, height = size
    img = Image.new("RGB", (width, height), (32, 32, 32))
    draw = ImageDraw.Draw(img)
    font = _font(14)
    label = _fit_text(draw, text, width - 24, font)
    bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=3)
    draw.multiline_text(
        ((width - (bbox[2] - bbox[0])) // 2, (height - (bbox[3] - bbox[1])) // 2),
        label,
        font=font,
        fill=(220, 220, 220),
        spacing=3,
        align="center",
    )
    return np.asarray(img)


def _traj_mask(traj: np.ndarray, bg: int = 50, threshold: int = 35) -> np.ndarray:
    diff = np.max(np.abs(traj.astype(np.int16) - int(bg)), axis=-1)
    return diff > int(threshold)


def _overlay_traj(base: np.ndarray, traj: np.ndarray, *, alpha: float = 0.55) -> np.ndarray:
    if traj.shape[:2] != base.shape[:2]:
        traj = _resize_rgb(traj, (base.shape[1], base.shape[0]))
    mask = _traj_mask(traj)
    out = base.astype(np.float32)
    out[mask] = alpha * traj[mask].astype(np.float32) + (1.0 - alpha) * out[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def _mat_from_xyz_quat(row: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = float(row[3]), float(row[4]), float(row[5]), float(row[6])
    rot = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rot
    out[:3, 3] = row[:3]
    return out


def _pad_extrinsic(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-2:] == (3, 4):
        out = np.zeros(arr.shape[:-2] + (4, 4), dtype=np.float32)
        out[..., :3, :4] = arr
        out[..., 3, 3] = 1.0
        return out
    return arr


def _as_time_intrinsic(arr: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (3, 3):
        return np.repeat(arr[None], n_frames, axis=0)
    return arr


def _render_ee_projection_on_canvas(
    *,
    canvas: np.ndarray,
    action_row: np.ndarray,
    intrinsic: np.ndarray,
    w2c: np.ndarray,
    min_depth: float,
    traj_gripper_z_offset: float,
    traj_keypoint_scale: float,
    radius: int,
) -> np.ndarray:
    pil_img = Image.fromarray(canvas.astype(np.uint8), "RGB").convert("RGBA")
    draw = ImageDraw.Draw(pil_img, "RGBA")
    width, height = pil_img.size

    gripper_to_eef = np.eye(4, dtype=np.float32)
    gripper_to_eef[2, 3] = float(traj_gripper_z_offset)
    ee_points = np.asarray(
        [
            [0, 0, 0, 1],
            [traj_keypoint_scale, 0, 0, 1],
            [0, traj_keypoint_scale, 0, 1],
            [0, 0, traj_keypoint_scale, 1],
        ],
        dtype=np.float32,
    ).T
    arm_specs = [
        (0, (255, 64, 64, 170), [(255, 0, 0, 220), (0, 255, 0, 220), (0, 0, 255, 220)]),
        (8, (64, 180, 255, 170), [(255, 255, 0, 220), (0, 255, 255, 220), (255, 0, 255, 220)]),
    ]

    for arm_start, circle_color, line_colors in arm_specs:
        if action_row.shape[0] < arm_start + 7:
            continue
        pose = _mat_from_xyz_quat(action_row[arm_start : arm_start + 7])
        pts = w2c @ pose @ gripper_to_eef @ ee_points
        with np.errstate(divide="ignore", invalid="ignore"):
            uvw = intrinsic @ pts[:3]
            uv = np.stack([uvw[0] / uvw[2], uvw[1] / uvw[2]], axis=-1)
        valid = (pts[2] > float(min_depth)) & np.isfinite(uv).all(axis=-1)
        base = uv[0]
        if not bool(valid[0] and 0 <= base[0] < width and 0 <= base[1] < height):
            continue
        bx, by = float(base[0]), float(base[1])
        draw.ellipse((bx - radius, by - radius, bx + radius, by + radius), fill=circle_color)
        for point, color in zip(uv[1:], line_colors):
            px, py = float(point[0]), float(point[1])
            if np.isfinite(px) and np.isfinite(py):
                draw.line((bx, by, px, py), fill=color, width=max(2, radius // 3))

    return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)


def _resolve_record_file(
    item: dict[str, Any],
    *,
    direct_keys: tuple[str, ...],
    fallback_names: tuple[str, ...],
    rewrites: list[tuple[str, str]] | None,
) -> Path | None:
    for key in direct_keys:
        if item.get(key):
            path = _apply_rewrites(Path(str(item[key])), rewrites)
            if path.exists():
                return path
    for root_key in ("path", "source_path", "episode_dir"):
        if not item.get(root_key):
            continue
        root = _apply_rewrites(Path(str(item[root_key])), rewrites)
        for name in fallback_names:
            path = root / name
            if path.exists():
                return path
    return None


def _compute_manual_traj_projection(
    *,
    item: dict[str, Any],
    n_frames: int,
    n_previous: int,
    ref_hw: tuple[int, int],
    traj_gripper_z_offset: float,
    traj_keypoint_scale: float,
    min_depth: float,
    radius: int,
    rewrites: list[tuple[str, str]] | None,
) -> np.ndarray:
    actions_path = _resolve_record_file(
        item,
        direct_keys=("actions_path",),
        fallback_names=("actions_evac.npy", "actions.npy"),
        rewrites=rewrites,
    )
    camera_path = _resolve_record_file(
        item,
        direct_keys=("camera_path",),
        fallback_names=("camera.npz",),
        rewrites=rewrites,
    )
    if actions_path is None or camera_path is None:
        raise FileNotFoundError("manual trajectory projection needs actions_path/actions_evac.npy and camera_path/camera.npz")

    actions = np.load(actions_path).astype(np.float32)
    camera = np.load(camera_path)
    intrinsic_all = _as_time_intrinsic(camera["intrinsic_cv"], len(actions))
    extrinsic = np.asarray(camera["extrinsic_cv"], dtype=np.float32)
    if extrinsic.ndim == 2:
        extrinsic = np.repeat(extrinsic[None], len(actions), axis=0)
    w2c_all = _pad_extrinsic(extrinsic)
    ref_h, ref_w = ref_hw

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        src_idx = n_previous + i
        if src_idx >= len(actions) or src_idx >= len(intrinsic_all) or src_idx >= len(w2c_all):
            break
        canvas = np.full((ref_h, ref_w, 3), 50, dtype=np.uint8)
        frames.append(
            _render_ee_projection_on_canvas(
                canvas=canvas,
                action_row=actions[src_idx],
                intrinsic=intrinsic_all[src_idx],
                w2c=w2c_all[src_idx],
                min_depth=min_depth,
                traj_gripper_z_offset=traj_gripper_z_offset,
                traj_keypoint_scale=traj_keypoint_scale,
                radius=radius,
            )
        )
    if not frames:
        raise FileNotFoundError("manual trajectory projection produced no frames")
    return np.stack(frames, axis=0)


def _traj_mask_stats(traj: np.ndarray, *, source: str) -> dict[str, Any]:
    frames: list[int] = []
    centers: list[tuple[float, float]] = []
    pixels: list[int] = []
    for idx, frame in enumerate(traj):
        mask = _traj_mask(frame)
        count = int(mask.sum())
        if count <= 50:
            continue
        y, x = np.where(mask)
        frames.append(idx)
        centers.append((float(x.mean()), float(y.mean())))
        pixels.append(count)
    return {
        "traj_source": source,
        "frames": int(len(traj)),
        "nonblank_frames": len(frames),
        "first_nonblank": frames[0] if frames else None,
        "last_nonblank": frames[-1] if frames else None,
        "sample_nonblank_frames": frames[:20],
        "mean_center_xy": [
            float(np.mean([c[0] for c in centers])) if centers else None,
            float(np.mean([c[1] for c in centers])) if centers else None,
        ],
        "max_mask_pixels": max(pixels) if pixels else 0,
    }


def _overlay_risk(base: np.ndarray, risk_hw: np.ndarray, *, alpha_scale: float = 0.70) -> np.ndarray:
    risk_up = np.clip(_resize_scalar(risk_hw, base.shape[:2]), 0.0, 1.0)
    heat = _colorize(risk_up, cmap_name="inferno", vmin=0.0, vmax=1.0).astype(np.float32)
    alpha = (np.power(risk_up, 0.6) * float(alpha_scale))[..., None]
    out = alpha * heat + (1.0 - alpha) * base.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _write_video(frames: list[np.ndarray] | np.ndarray, output_path: Path, *, fps: int) -> None:
    arr = _to_uint8_frames(np.asarray(frames))
    ensure_dir(output_path.parent)
    if imageio is not None:
        try:
            imageio.mimwrite(
                str(output_path),
                list(arr),
                fps=int(fps),
                format="FFMPEG",
                quality=8,
                macro_block_size=1,
            )
            return
        except Exception:
            pass
    h, w = arr.shape[1:3]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, input=arr.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return
    if cv2 is None:
        raise RuntimeError("Writing mp4 files needs imageio, cv2, or ffmpeg")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))
    try:
        for frame in arr:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _draw_colorbar(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    cmap_name: str,
    vmin: float,
    vmax: float,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    grad = np.linspace(vmax, vmin, max(2, height), dtype=np.float32)[:, None]
    bar = _colorize(grad, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
    bar = np.repeat(bar, width, axis=1)
    canvas.paste(Image.fromarray(bar), (x, y))
    draw.rectangle((x, y, x + width, y + height), outline=(230, 230, 230), width=1)
    draw.text((x + width + 5, y), f"{vmax:.3g}", font=font, fill=(255, 255, 255))
    draw.text((x + width + 5, y + height - 14), f"{vmin:.3g}", font=font, fill=(255, 255, 255))
    draw.text((x, y + height + 3), label, font=font, fill=(220, 220, 220))


def _write_sheet(
    *,
    rows: list[tuple[str, list[np.ndarray]]],
    frame_labels: list[str],
    out_path: Path,
    title: str,
    tile_w: int,
    tile_h: int,
    row_bars: list[dict[str, Any] | None] | None = None,
) -> None:
    label_w = 260
    gap = 8
    header_h = 44
    row_h = tile_h + gap
    n_cols = len(frame_labels)
    row_bars = row_bars or [None for _ in rows]
    bar_region_w = 98 if any(row_bars) else 0
    canvas_w = label_w + n_cols * tile_w + (n_cols + 1) * gap + bar_region_w
    canvas_h = header_h + len(rows) * row_h + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(15)
    label_font = _font(14)
    small_font = _font(12)
    draw.text((gap, 10), title, font=title_font, fill=(255, 255, 255))

    y = header_h
    for row_idx, (row_label, tiles) in enumerate(rows):
        _draw_label(draw, (gap, y + 10), row_label, font=label_font, max_width=label_w - 2 * gap)
        for col, tile in enumerate(tiles):
            x = label_w + gap + col * (tile_w + gap)
            tile_rgb = _resize_rgb(tile, (tile_w, tile_h))
            canvas.paste(Image.fromarray(tile_rgb), (x, y))
            _draw_label(draw, (x + 6, y + 6), frame_labels[col], font=small_font, max_width=tile_w - 12)
        if row_idx < len(row_bars) and row_bars[row_idx]:
            spec = row_bars[row_idx] or {}
            bar_x = label_w + gap + n_cols * (tile_w + gap)
            _draw_colorbar(
                canvas,
                draw,
                x=bar_x,
                y=y,
                width=18,
                height=tile_h - 22,
                cmap_name=str(spec.get("cmap", "inferno")),
                vmin=float(spec.get("vmin", 0.0)),
                vmax=float(spec.get("vmax", 1.0)),
                label=str(spec.get("label", "")),
                font=small_font,
            )
        y += row_h

    ensure_dir(out_path.parent)
    canvas.save(out_path)


def _load_confidence(ep_out: Path) -> np.ndarray:
    path = ep_out / "conf_map.npy"
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"conf_map.npy must have shape [T,H,W], got {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def _load_latent_oracle_error(ep_out: Path, conf_shape: tuple[int, int, int]) -> tuple[np.ndarray | None, str | None]:
    pred_path = ep_out / "latent_pred.npy"
    gt_path = ep_out / "latent_gt.npy"
    if not pred_path.exists() or not gt_path.exists():
        return None, "missing latent_pred.npy or latent_gt.npy"
    try:
        from eval.confidence_eval.core.io_utils import compute_latent_oracle_error_map

        err = compute_latent_oracle_error_map(np.load(pred_path), np.load(gt_path), conf_shape)
        return np.asarray(err, dtype=np.float32), None
    except Exception as exc:
        return None, str(exc)


def _load_pixel_oracle_error(
    pred_frames: np.ndarray,
    gt_frames: np.ndarray | None,
    conf_shape: tuple[int, int, int],
) -> tuple[np.ndarray | None, str | None]:
    if gt_frames is None:
        return None, "missing GT frames"
    try:
        from eval.confidence_eval.core.io_utils import compute_pixel_oracle_error_map

        pred_f = np.asarray(pred_frames, dtype=np.float32) / 255.0
        gt_f = np.asarray(gt_frames, dtype=np.float32) / 255.0
        err = compute_pixel_oracle_error_map(pred_f, gt_f, conf_shape)
        return np.asarray(err, dtype=np.float32), None
    except Exception as exc:
        return None, str(exc)


def _canonicalize_latent(latent: np.ndarray) -> np.ndarray:
    arr = np.asarray(latent, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"latent tensor must have 4 dims, got {arr.shape}")
    if arr.shape[0] <= 8 and arr.shape[1] > 8:
        arr = np.transpose(arr, (1, 0, 2, 3))
    return arr


def _load_latent_maps(ep_out: Path) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    pred_path = ep_out / "latent_pred.npy"
    gt_path = ep_out / "latent_gt.npy"
    if not pred_path.exists() or not gt_path.exists():
        return None, None, "missing latent_pred.npy or latent_gt.npy"
    try:
        pred = _canonicalize_latent(np.load(pred_path))
        gt = _canonicalize_latent(np.load(gt_path))
        t = min(int(pred.shape[0]), int(gt.shape[0]))
        pred = pred[:t]
        gt = gt[:t]
        pred_map = np.mean(np.abs(pred), axis=1)
        gt_map = np.mean(np.abs(gt), axis=1)
        return pred_map.astype(np.float32), gt_map.astype(np.float32), None
    except Exception as exc:
        return None, None, str(exc)


def _comparison_rows(gt: np.ndarray | None, pred: np.ndarray, traj: np.ndarray | None, indices: list[int]) -> list[tuple[str, list[np.ndarray]]]:
    size = (pred.shape[2], pred.shape[1])
    rows: list[tuple[str, list[np.ndarray]]] = []
    if gt is not None:
        rows.append(("GT future frames", [gt[min(i, len(gt) - 1)] for i in indices]))
    else:
        rows.append(("GT future frames", [_blank_tile(size, "GT missing") for _ in indices]))
    rows.append(("EVAC prediction", [pred[i] for i in indices]))
    if traj is not None and len(traj) > 0:
        rows.append(("Trajectory condition", [traj[min(i, len(traj) - 1)] for i in indices]))
        rows.append(("GT + traj_condition", [_overlay_traj(gt[min(i, len(gt) - 1)], traj[min(i, len(traj) - 1)]) if gt is not None else _blank_tile(size, "GT missing") for i in indices])
        )
        rows.append(("Pred + traj_condition", [_overlay_traj(pred[i], traj[min(i, len(traj) - 1)]) for i in indices]))
    return rows


def _pixel_confidence_rows(
    *,
    gt: np.ndarray | None,
    pred: np.ndarray,
    conf: np.ndarray,
    pixel_error: np.ndarray | None,
    pixel_note: str | None,
    indices: list[int],
) -> list[tuple[str, list[np.ndarray]]]:
    size = (pred.shape[2], pred.shape[1])
    risk = 1.0 - conf
    error_vmax = _error_vmax(pixel_error)

    rows: list[tuple[str, list[np.ndarray]]] = []
    if gt is not None:
        rows.append(("GT future frames", [gt[min(i, len(gt) - 1)] for i in indices]))
    else:
        rows.append(("GT future frames", [_blank_tile(size, "GT missing") for _ in indices]))
    rows.append(("EVAC prediction", [pred[i] for i in indices]))
    if pixel_error is not None:
        rows.append(
            (
                "Oracle Pixel Error",
                [_colorize(pixel_error[min(i, pixel_error.shape[0] - 1)], cmap_name="magma", vmin=0.0, vmax=error_vmax) for i in indices],
            )
        )
    else:
        rows.append(("Oracle Pixel Error", [_blank_tile(size, pixel_note or "pixel error missing") for _ in indices]))
    rows.append(("Confidence heatmap", [_colorize(conf[i], cmap_name="RdBu", vmin=0.0, vmax=1.0) for i in indices]))
    rows.append(("Risk Map = 1 - confidence", [_colorize(risk[i], cmap_name="inferno", vmin=0.0, vmax=1.0) for i in indices]))
    rows.append(("Risk Map overlay on prediction", [_overlay_risk(pred[i], risk[i]) for i in indices]))
    return rows


def _pixel_confidence_bars(pixel_error: np.ndarray | None) -> list[dict[str, Any] | None]:
    pixel_error_bar = (
        {"cmap": "magma", "vmin": 0.0, "vmax": _error_vmax(pixel_error), "label": "pixel err"}
        if pixel_error is not None
        else None
    )
    return [
        None,
        None,
        pixel_error_bar,
        {"cmap": "RdBu", "vmin": 0.0, "vmax": 1.0, "label": "conf"},
        {"cmap": "inferno", "vmin": 0.0, "vmax": 1.0, "label": "risk"},
        {"cmap": "inferno", "vmin": 0.0, "vmax": 1.0, "label": "risk"},
    ]


def _latent_confidence_rows(
    *,
    pred_latent_map: np.ndarray | None,
    gt_latent_map: np.ndarray | None,
    latent_error: np.ndarray | None,
    latent_note: str | None,
    pred: np.ndarray,
    conf: np.ndarray,
    indices: list[int],
) -> list[tuple[str, list[np.ndarray]]]:
    size = (pred.shape[2], pred.shape[1])
    risk = 1.0 - conf
    latent_vmax = _error_vmax(
        np.concatenate(
            [
                x.reshape(-1)
                for x in (pred_latent_map, gt_latent_map)
                if x is not None and x.size > 0
            ],
            axis=0,
        )
        if pred_latent_map is not None or gt_latent_map is not None
        else None
    )
    error_vmax = _error_vmax(latent_error)

    if gt_latent_map is not None:
        gt_tiles = [_colorize(gt_latent_map[min(i, gt_latent_map.shape[0] - 1)], cmap_name="viridis", vmin=0.0, vmax=latent_vmax) for i in indices]
    else:
        gt_tiles = [_blank_tile(size, latent_note or "latent GT missing") for _ in indices]
    if pred_latent_map is not None:
        pred_tiles = [_colorize(pred_latent_map[min(i, pred_latent_map.shape[0] - 1)], cmap_name="viridis", vmin=0.0, vmax=latent_vmax) for i in indices]
    else:
        pred_tiles = [_blank_tile(size, latent_note or "latent pred missing") for _ in indices]
    if latent_error is not None:
        err_tiles = [_colorize(latent_error[min(i, latent_error.shape[0] - 1)], cmap_name="magma", vmin=0.0, vmax=error_vmax) for i in indices]
    else:
        err_tiles = [_blank_tile(size, latent_note or "rollout latent error missing") for _ in indices]
    if pred_latent_map is not None:
        overlay_tiles = []
        for i in indices:
            idx = min(i, pred_latent_map.shape[0] - 1)
            base = _colorize(pred_latent_map[idx], cmap_name="viridis", vmin=0.0, vmax=latent_vmax)
            overlay_tiles.append(_overlay_risk(base, risk[min(i, risk.shape[0] - 1)]))
    else:
        overlay_tiles = [_blank_tile(size, latent_note or "latent pred missing") for _ in indices]

    return [
        ("GT latent magnitude", gt_tiles),
        ("Pred latent magnitude", pred_tiles),
        ("Rollout Latent L1 Error", err_tiles),
        ("Confidence heatmap", [_colorize(conf[i], cmap_name="RdBu", vmin=0.0, vmax=1.0) for i in indices]),
        ("Risk Map = 1 - confidence", [_colorize(risk[i], cmap_name="inferno", vmin=0.0, vmax=1.0) for i in indices]),
        ("Risk Map overlay on pred latent", overlay_tiles),
    ]


def _latent_confidence_bars(
    *,
    pred_latent_map: np.ndarray | None,
    gt_latent_map: np.ndarray | None,
    latent_error: np.ndarray | None,
) -> list[dict[str, Any] | None]:
    if pred_latent_map is not None or gt_latent_map is not None:
        latent_values = np.concatenate(
            [
                x.reshape(-1)
                for x in (pred_latent_map, gt_latent_map)
                if x is not None and x.size > 0
            ],
            axis=0,
        )
    else:
        latent_values = None
    latent_vmax = _error_vmax(latent_values)
    latent_bar = {"cmap": "viridis", "vmin": 0.0, "vmax": latent_vmax, "label": "|z|"} if latent_values is not None else None
    latent_error_bar = (
        {"cmap": "magma", "vmin": 0.0, "vmax": _error_vmax(latent_error), "label": "rollout L1"}
        if latent_error is not None
        else None
    )
    return [
        latent_bar if gt_latent_map is not None else None,
        latent_bar if pred_latent_map is not None else None,
        latent_error_bar,
        {"cmap": "RdBu", "vmin": 0.0, "vmax": 1.0, "label": "conf"},
        {"cmap": "inferno", "vmin": 0.0, "vmax": 1.0, "label": "risk"},
        {"cmap": "inferno", "vmin": 0.0, "vmax": 1.0, "label": "risk"},
    ]


def _nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key)
    return cur if isinstance(cur, dict) else {}


def _config_traj_conditioning(cfg: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in (
        ("model", "params", "traj_conditioning"),
        ("retraining", "traj_conditioning"),
        ("scoring", "traj_conditioning"),
    ):
        merged.update(_nested_dict(cfg, *path))
    return merged


def _resolve_float(cli_value: float | None, cfg: dict[str, Any], key: str, default: float) -> float:
    if cli_value is not None:
        return float(cli_value)
    if cfg.get(key) is not None:
        return float(cfg[key])
    return float(default)


def _resolve_int(cli_value: int | None, cfg: dict[str, Any], key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    if cfg.get(key) is not None:
        return int(cfg[key])
    return int(default)


def visualize_episode(
    *,
    item: dict[str, Any],
    pred_dir: Path,
    output_root: Path,
    vis_traj: bool,
    rewrites: list[tuple[str, str]],
    traj_gripper_z_offset: float,
    traj_keypoint_scale: float,
    min_depth: float,
    sheet_radius: int,
    save_conf_map: bool,
    save_conf_video: bool,
    conf_video_fps: int,
    samples_per_row: int,
    tile_w: int,
    tile_h: int,
) -> dict[str, Any]:
    ep_id = _episode_id(item)
    if not ep_id:
        return {"episode_id": "", "status": "skipped", "reason": "missing episode_id"}
    ep_out = pred_dir / ep_id
    ep_viz = ensure_dir(output_root / ep_id)
    pred = _to_uint8_frames(_load_pred_frames(ep_out))
    n_previous = _n_previous(ep_out)
    gt = _load_gt_frames(item, n_previous=n_previous, n_frames=int(pred.shape[0]), rewrites=rewrites)
    traj = None
    traj_source = None
    traj_reason = None
    if vis_traj:
        traj = _load_traj_frames(ep_out, n_frames=int(pred.shape[0]))
        if traj is not None:
            traj_source = "mp4"
        else:
            try:
                traj = _compute_manual_traj_projection(
                    item=item,
                    n_frames=int(pred.shape[0]),
                    n_previous=n_previous,
                    ref_hw=(int(pred.shape[1]), int(pred.shape[2])),
                    traj_gripper_z_offset=traj_gripper_z_offset,
                    traj_keypoint_scale=traj_keypoint_scale,
                    min_depth=min_depth,
                    radius=sheet_radius,
                    rewrites=rewrites,
                )
                traj_source = "manual_ee_projection"
            except Exception as exc:
                traj_reason = str(exc)
    n = int(pred.shape[0])
    if gt is not None:
        n = min(n, int(gt.shape[0]))
    if traj is not None:
        n = min(n, int(traj.shape[0]))
    if n <= 0:
        return {"episode_id": ep_id, "status": "skipped", "reason": "no aligned frames"}
    pred = pred[:n]
    gt = gt[:n] if gt is not None else None
    traj = traj[:n] if traj is not None else None
    indices = _frame_indices(n, samples_per_row)
    frame_labels = [f"t={i}" for i in indices]

    comparison_path = ep_viz / "comparison_traj.png"
    _write_sheet(
        rows=_comparison_rows(gt, pred, traj, indices),
        frame_labels=frame_labels,
        out_path=comparison_path,
        title=f"{ep_id} | GT / prediction" + (" / trajectory" if traj is not None else ""),
        tile_w=tile_w,
        tile_h=tile_h,
    )
    if traj is not None:
        with (ep_viz / "traj_mask_stats.json").open("w", encoding="utf-8") as f:
            json.dump(_traj_mask_stats(traj, source=str(traj_source or "unknown")), f, indent=2)

    result = {
        "episode_id": ep_id,
        "status": "ok",
        "comparison_traj": str(comparison_path),
        "n_frames": n,
        "n_condition_frames": n_previous,
        "gt_loaded": gt is not None,
        "traj_loaded": traj is not None,
        "vis_traj": bool(vis_traj),
        "traj_source": traj_source,
        "traj_gripper_z_offset": float(traj_gripper_z_offset) if vis_traj else None,
    }
    if traj_reason:
        result["traj_reason"] = traj_reason

    if save_conf_map or save_conf_video:
        conf_path = ep_out / "conf_map.npy"
        if not conf_path.exists():
            result["confidence_status"] = "skipped"
            result["confidence_reason"] = "missing conf_map.npy"
        else:
            conf = _load_confidence(ep_out)
            conf_n = min(n, int(conf.shape[0]))
            conf = conf[:conf_n]
            pred_conf = pred[:conf_n]
            gt_conf = gt[:conf_n] if gt is not None else None
            conf_indices = _frame_indices(conf_n, samples_per_row)
            pixel_error = None
            pixel_note = None
            latent_error = None
            latent_note = None
            if save_conf_map:
                pixel_error, pixel_note = _load_pixel_oracle_error(pred_conf, gt_conf, tuple(conf.shape))
                latent_error, latent_note = _load_latent_oracle_error(ep_out, tuple(conf.shape))
                if latent_error is not None:
                    latent_error = latent_error[:conf_n]

            pixel_path = None
            latent_path = None
            if save_conf_map:
                pixel_path = ep_viz / "confidence_pixel_comparison.png"
                _write_sheet(
                    rows=_pixel_confidence_rows(
                        gt=gt_conf,
                        pred=pred_conf,
                        conf=conf,
                        pixel_error=pixel_error[:conf_n] if pixel_error is not None else None,
                        pixel_note=pixel_note,
                        indices=conf_indices,
                    ),
                    frame_labels=[f"t={i}" for i in conf_indices],
                    out_path=pixel_path,
                    title=f"{ep_id} | C3 confidence, pixel error, and risk",
                    tile_w=tile_w,
                    tile_h=tile_h,
                    row_bars=_pixel_confidence_bars(pixel_error),
                )

            pred_latent_map = gt_latent_map = None
            latent_map_note = None
            if save_conf_map:
                pred_latent_map, gt_latent_map, latent_map_note = _load_latent_maps(ep_out)
            if save_conf_map and (latent_error is not None or pred_latent_map is not None or gt_latent_map is not None):
                latent_path = ep_viz / "confidence_latent_comparison.png"
                latent_note_final = latent_note or latent_map_note
                _write_sheet(
                    rows=_latent_confidence_rows(
                        pred_latent_map=pred_latent_map[:conf_n] if pred_latent_map is not None else None,
                        gt_latent_map=gt_latent_map[:conf_n] if gt_latent_map is not None else None,
                        latent_error=latent_error,
                        latent_note=latent_note_final,
                        pred=pred_conf,
                        conf=conf,
                        indices=conf_indices,
                    ),
                    frame_labels=[f"t={i}" for i in conf_indices],
                    out_path=latent_path,
                    title=f"{ep_id} | C3 confidence, rollout latent L1, and risk",
                    tile_w=tile_w,
                    tile_h=tile_h,
                    row_bars=_latent_confidence_bars(
                        pred_latent_map=pred_latent_map,
                        gt_latent_map=gt_latent_map,
                        latent_error=latent_error,
                    ),
                )

            if save_conf_video:
                conf_video_frames = []
                risk_overlay_frames = []
                risk = 1.0 - conf
                for frame_idx in range(conf_n):
                    conf_up = _resize_scalar(conf[frame_idx], pred_conf.shape[1:3])
                    conf_video_frames.append(_colorize(conf_up, cmap_name="RdBu", vmin=0.0, vmax=1.0))
                    risk_overlay_frames.append(_overlay_risk(pred_conf[frame_idx], risk[frame_idx]))
                heatmap_video_path = ep_viz / "confidence_heatmap.mp4"
                overlay_video_path = ep_viz / "confidence_overlay.mp4"
                _write_video(conf_video_frames, heatmap_video_path, fps=conf_video_fps)
                _write_video(risk_overlay_frames, overlay_video_path, fps=conf_video_fps)
                result["confidence_heatmap_video"] = str(heatmap_video_path)
                result["confidence_overlay_video"] = str(overlay_video_path)

            result["confidence_status"] = "ok"
            if pixel_path is not None:
                result["confidence_pixel_comparison"] = str(pixel_path)
            if latent_path is not None:
                result["confidence_latent_comparison"] = str(latent_path)
            result["pixel_oracle_loaded"] = pixel_error is not None
            result["latent_oracle_loaded"] = latent_error is not None
            if pixel_note:
                result["pixel_oracle_note"] = pixel_note
            if latent_note:
                result["latent_oracle_note"] = latent_note
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize validation prediction outputs")
    parser.add_argument("--pred-dir", "--pred_dir", required=True, help="Validation prediction directory containing per-episode outputs")
    parser.add_argument("--manifest", required=True, help="Validation manifest JSON")
    parser.add_argument("--config", default=None, help="Optional AL config for summary context and trajectory-conditioning defaults")
    parser.add_argument("--output-dir", "--output_dir", default=None, help="Output directory, default: {pred_dir}_visualize")
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=3)
    parser.add_argument("--episode-id", "--episode_id", action="append", default=None, help="Specific episode id to visualize; can be repeated")
    parser.add_argument("--path-rewrite", "--path_rewrite", action="append", default=[], help="Runtime path rewrite FROM=TO for manifest paths; can be repeated")
    parser.add_argument("--samples-per-row", "--samples_per_row", "--samples", dest="samples_per_row", type=int, default=6)
    parser.add_argument("--tile-width", "--tile_width", type=int, default=256)
    parser.add_argument("--tile-height", "--tile_height", type=int, default=160)
    parser.add_argument("--vis-traj", "--vis_traj", action="store_true", help="Add trajectory-condition, GT+traj, and Pred+traj rows")
    parser.add_argument("--traj-gripper-z-offset", "--traj_gripper_z_offset", type=float, default=None, help="Manual trajectory fallback gripper Z offset; CLI overrides config")
    parser.add_argument("--traj-keypoint-scale", "--traj_keypoint_scale", type=float, default=None, help="Manual trajectory fallback keypoint axis scale; CLI overrides config")
    parser.add_argument("--min-depth", "--min_depth", type=float, default=0.03, help="Manual trajectory fallback minimum projection depth")
    parser.add_argument("--sheet-radius", "--sheet_radius", type=int, default=None, help="Manual trajectory fallback marker radius; CLI overrides config traj_radius")
    parser.add_argument("--save_conf_map", "--save-conf-map", action="store_true", help="Also save confidence pixel/latent comparison sheets from conf_map.npy")
    parser.add_argument("--save_conf_video", "--save-conf-video", action="store_true", help="Also save confidence_heatmap.mp4 and confidence_overlay.mp4 next to the images")
    parser.add_argument("--conf-video-fps", "--conf_video_fps", type=int, default=10)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = pred_dir.parent / f"{pred_dir.name}_visualize"
    output_root = ensure_dir(output_root)

    items = flatten_manifest_items(load_json(args.manifest))
    if args.episode_id:
        wanted = {str(x) for x in args.episode_id}
        items = [item for item in items if _episode_id(item) in wanted]
    elif args.max_episodes is not None:
        items = items[: max(0, int(args.max_episodes))]

    rewrites: list[tuple[str, str]] = []
    for rewrite in args.path_rewrite or []:
        if "=" not in rewrite:
            raise ValueError(f"--path-rewrite expects FROM=TO, got {rewrite!r}")
        src, dst = rewrite.split("=", 1)
        rewrites.append((src.rstrip("/"), dst.rstrip("/")))

    cfg_summary: dict[str, Any] = {}
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)
        cfg_summary = {
            "run": cfg.get("run", {}),
            "phase": cfg.get("phase", {}),
        }
    traj_cfg = _config_traj_conditioning(cfg)
    traj_gripper_z_offset = _resolve_float(args.traj_gripper_z_offset, traj_cfg, "traj_gripper_z_offset", 0.23)
    traj_keypoint_scale = _resolve_float(args.traj_keypoint_scale, traj_cfg, "traj_keypoint_scale", 0.1)
    sheet_radius = _resolve_int(args.sheet_radius, traj_cfg, "traj_radius", 14)
    cfg_summary["traj_conditioning_resolved"] = {
        "traj_gripper_z_offset": traj_gripper_z_offset,
        "traj_keypoint_scale": traj_keypoint_scale,
        "traj_radius": sheet_radius,
        "min_depth": float(args.min_depth),
    }

    results = []
    for item in items:
        try:
            results.append(
                visualize_episode(
                    item=item,
                    pred_dir=pred_dir,
                    output_root=output_root,
                    vis_traj=bool(args.vis_traj),
                    rewrites=rewrites,
                    traj_gripper_z_offset=traj_gripper_z_offset,
                    traj_keypoint_scale=traj_keypoint_scale,
                    min_depth=float(args.min_depth),
                    sheet_radius=sheet_radius,
                    save_conf_map=bool(args.save_conf_map),
                    save_conf_video=bool(args.save_conf_video),
                    conf_video_fps=max(1, int(args.conf_video_fps)),
                    samples_per_row=max(1, int(args.samples_per_row)),
                    tile_w=max(32, int(args.tile_width)),
                    tile_h=max(32, int(args.tile_height)),
                )
            )
        except Exception as exc:
            results.append({"episode_id": _episode_id(item), "status": "skipped", "reason": str(exc)})

    summary = {
        "pred_dir": str(pred_dir),
        "manifest": str(args.manifest),
        "config": str(args.config) if args.config else None,
        "config_summary": cfg_summary,
        "output_dir": str(output_root),
        "vis_traj": bool(args.vis_traj),
        "path_rewrites": rewrites,
        "save_conf_map": bool(args.save_conf_map),
        "save_conf_video": bool(args.save_conf_video),
        "episodes_requested": len(items),
        "episodes_ok": sum(1 for row in results if row.get("status") == "ok"),
        "results": results,
    }
    with (output_root / "visualize_val_results_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[visualize] wrote {summary['episodes_ok']} episode visualizations -> {output_root}")
    skipped = [row for row in results if row.get("status") != "ok"]
    if skipped:
        print(f"[visualize] skipped {len(skipped)} episodes; see visualize_val_results_summary.json")


if __name__ == "__main__":
    main()
