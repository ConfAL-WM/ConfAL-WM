from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class EpisodeRecord:
    episode_id: str
    pred_frames_dir: str | None = None
    pred_video: str | None = None
    gt_frames_dir: str | None = None
    gt_video: str | None = None
    conf_map_path: str | None = None
    actions_path: str | None = None
    split: str = "unknown"
    ood_type: str | None = None
    task_name: str | None = None
    camera_name: str | None = None
    scene_id: str | None = None
    latent_pred_path: str | None = None
    latent_gt_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: str | None, base_dir: str | os.PathLike[str]) -> str | None:
    if path is None:
        return None
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(str(base_dir), expanded))


def _infer_episode_id(entry: dict) -> str:
    for key in ("episode_id", "ep_id", "id"):
        if entry.get(key) is not None:
            return str(entry[key])
    for key in ("conf_map_path", "conf_map_npy", "pred_frames_dir", "pred_video"):
        value = entry.get(key)
        if value:
            return Path(str(value)).parent.name
    raise ValueError(f"Cannot infer episode_id for manifest entry: {entry}")


def _infer_task_and_episode(episode_id: str) -> tuple[str | None, str | None]:
    if "_" in episode_id:
        parts = episode_id.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[0], parts[1]
    match = re.match(r"^(\d+)[-_](\d+)", episode_id)
    if match:
        return match.group(1), match.group(2)
    return None, None


def _infer_gt_dir(gt_root: str | None, episode_id: str) -> str | None:
    if gt_root is None:
        return None
    task_id, ep_id = _infer_task_and_episode(episode_id)
    if task_id is None or ep_id is None:
        return None
    candidate = Path(gt_root) / task_id / ep_id / "video"
    return str(candidate) if candidate.exists() else None


def normalize_manifest_entry(entry: dict, base_dir: str | os.PathLike[str], gt_root: str | None = None) -> EpisodeRecord:
    episode_id = _infer_episode_id(entry)
    conf_map_path = entry.get("conf_map_path", entry.get("conf_map_npy"))
    pred_frames_dir = entry.get("pred_frames_dir")
    pred_video = entry.get("pred_video")
    gt_frames_dir = entry.get("gt_frames_dir")
    gt_video = entry.get("gt_video")
    latent_pred_path = entry.get("latent_pred_path")
    latent_gt_path = entry.get("latent_gt_path")

    if gt_frames_dir is None and gt_video is None:
        inferred_gt_dir = _infer_gt_dir(gt_root, episode_id)
        gt_frames_dir = gt_frames_dir or inferred_gt_dir

    record = EpisodeRecord(
        episode_id=str(episode_id),
        pred_frames_dir=resolve_path(pred_frames_dir, base_dir),
        pred_video=resolve_path(pred_video, base_dir),
        gt_frames_dir=resolve_path(gt_frames_dir, base_dir),
        gt_video=resolve_path(gt_video, base_dir),
        conf_map_path=resolve_path(conf_map_path, base_dir),
        actions_path=resolve_path(entry.get("actions_path", entry.get("actions_npy")), base_dir),
        split=str(entry.get("split", "unknown")),
        ood_type=entry.get("ood_type"),
        task_name=entry.get("task_name"),
        camera_name=entry.get("camera_name"),
        scene_id=entry.get("scene_id"),
        latent_pred_path=resolve_path(latent_pred_path, base_dir),
        latent_gt_path=resolve_path(latent_gt_path, base_dir),
    )
    return record


def load_manifest(manifest_path: str, gt_root: str | None = None) -> list[EpisodeRecord]:
    path = Path(manifest_path)
    base_dir = str(path.parent)
    records: list[EpisodeRecord] = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                records.append(normalize_manifest_entry(entry, base_dir=base_dir, gt_root=gt_root))
        return records

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        for entry in payload:
            records.append(normalize_manifest_entry(entry, base_dir=base_dir, gt_root=gt_root))
        return records

    if isinstance(payload, dict) and "episodes" in payload:
        for entry in payload["episodes"]:
            records.append(normalize_manifest_entry(entry, base_dir=base_dir, gt_root=gt_root))
        return records

    if isinstance(payload, dict) and "splits" in payload:
        for split_name, entries in payload["splits"].items():
            for entry in entries:
                item = dict(entry)
                item.setdefault("split", split_name)
                records.append(normalize_manifest_entry(item, base_dir=base_dir, gt_root=gt_root))
        return records

    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def write_manifest(records: Sequence[EpisodeRecord], output_path: str) -> str:
    path = Path(output_path)
    ensure_dir(path.parent)
    if path.suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([record.to_dict() for record in records], f, ensure_ascii=False, indent=2)
    return str(path)


def build_manifest_from_standardized_root(
    standardized_root: str,
    gt_root: str | None = None,
    split: str = "val",
    output_path: str | None = None,
) -> str:
    root = Path(standardized_root)
    records: list[EpisodeRecord] = []
    for episode_dir in sorted(root.iterdir()):
        if not episode_dir.is_dir():
            continue
        conf_map_path = episode_dir / "conf_map.npy"
        pred_frames_dir = episode_dir / "pred_frames"
        latent_pred_path = episode_dir / "latent_pred.npy"
        latent_gt_path = episode_dir / "latent_gt.npy"
        pred_video = None
        for name in ("pred_video.mp4", "pred.mp4", "outputs.mp4"):
            candidate = episode_dir / name
            if candidate.exists():
                pred_video = candidate
                break
        if not conf_map_path.exists():
            continue
        if not pred_frames_dir.exists() and pred_video is None:
            continue
        records.append(
            EpisodeRecord(
                episode_id=episode_dir.name,
                pred_frames_dir=str(pred_frames_dir) if pred_frames_dir.exists() else None,
                pred_video=str(pred_video) if pred_video is not None else None,
                gt_frames_dir=_infer_gt_dir(gt_root, episode_dir.name),
                conf_map_path=str(conf_map_path),
                actions_path=str(episode_dir / "actions.npy") if (episode_dir / "actions.npy").exists() else None,
                latent_pred_path=str(latent_pred_path) if latent_pred_path.exists() else None,
                latent_gt_path=str(latent_gt_path) if latent_gt_path.exists() else None,
                split=split,
            )
        )

    if output_path is None:
        output_path = str(root / "confidence_eval_manifest.json")
    return write_manifest(records, output_path)


def load_conf_map(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"conf_map expected shape [T,H,W], got {arr.shape} @ {path}")
    return np.clip(arr, 0.0, 1.0)


def _load_frames_from_dir(directory: str, n_frames: int | None = None) -> np.ndarray:
    paths = sorted(
        list(Path(directory).glob("frame_*.png"))
        + list(Path(directory).glob("frame_*.jpg"))
        + list(Path(directory).glob("frame_*.jpeg"))
    )
    if not paths:
        raise FileNotFoundError(f"No frame_*.png/jpg found in directory: {directory}")
    if n_frames is not None:
        paths = paths[:n_frames]
    frames = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)


def _load_frames_from_video(video_path: str, n_frames: int | None = None) -> np.ndarray:
    frames = []
    if iio is not None:
        video = iio.imiter(video_path)
        for idx, frame in enumerate(video):
            if n_frames is not None and idx >= n_frames:
                break
            frames.append(np.asarray(frame, dtype=np.float32)[..., :3] / 255.0)
        if frames:
            return np.stack(frames, axis=0)
    if cv2 is None:
        raise RuntimeError("Neither imageio.v3 nor cv2 available for video reading")
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n_frames is not None and idx >= n_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise FileNotFoundError(f"Cannot read frames from video: {video_path}")
    return np.stack(frames, axis=0)


def load_frames(pred_frames_dir: str | None = None, pred_video: str | None = None, n_frames: int | None = None) -> np.ndarray:
    if pred_frames_dir is not None and Path(pred_frames_dir).exists():
        return _load_frames_from_dir(pred_frames_dir, n_frames=n_frames)
    if pred_video is not None and Path(pred_video).exists():
        return _load_frames_from_video(pred_video, n_frames=n_frames)
    raise FileNotFoundError("Neither usable frames_dir nor video path available")


def _resize_frames(frames: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    resized = []
    for frame in frames:
        pil = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
        pil = pil.resize((w, h), Image.BILINEAR)
        resized.append(np.asarray(pil, dtype=np.float32) / 255.0)
    return np.stack(resized, axis=0)


def _adaptive_avg_pool2d_np(array_2d: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    in_h, in_w = array_2d.shape
    out_h, out_w = output_hw
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        h0 = int(np.floor(i * in_h / out_h))
        h1 = int(np.ceil((i + 1) * in_h / out_h))
        h1 = max(h1, h0 + 1)
        for j in range(out_w):
            w0 = int(np.floor(j * in_w / out_w))
            w1 = int(np.ceil((j + 1) * in_w / out_w))
            w1 = max(w1, w0 + 1)
            out[i, j] = float(np.mean(array_2d[h0:h1, w0:w1]))
    return out


def _pool_to_conf_shape(error_map: np.ndarray, conf_hw: tuple[int, int]) -> np.ndarray:
    pooled = []
    for frame_err in np.asarray(error_map, dtype=np.float32):
        pooled.append(_adaptive_avg_pool2d_np(frame_err, output_hw=conf_hw))
    return np.stack(pooled, axis=0)


def compute_pixel_oracle_error_map(pred_frames: np.ndarray, gt_frames: np.ndarray, conf_shape: tuple[int, int, int]) -> np.ndarray:
    t = min(pred_frames.shape[0], gt_frames.shape[0], conf_shape[0])
    pred = np.asarray(pred_frames[:t], dtype=np.float32)
    gt = np.asarray(gt_frames[:t], dtype=np.float32)
    if pred.shape[1:3] != gt.shape[1:3]:
        gt = _resize_frames(gt, pred.shape[1:3])
    mae = np.mean(np.abs(pred - gt), axis=-1)
    return _pool_to_conf_shape(mae, conf_hw=(conf_shape[1], conf_shape[2]))


def _canonicalize_latent(latent: np.ndarray) -> np.ndarray:
    arr = np.asarray(latent, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"latent tensor expected 4 dimensions, got {arr.shape}")
    if arr.shape[0] <= 8 and arr.shape[1] > 8:
        arr = np.transpose(arr, (1, 0, 2, 3))
    return arr


def compute_latent_oracle_error_map(pred_latent: np.ndarray, gt_latent: np.ndarray, conf_shape: tuple[int, int, int]) -> np.ndarray:
    pred = _canonicalize_latent(pred_latent)
    gt = _canonicalize_latent(gt_latent)
    t = min(pred.shape[0], gt.shape[0], conf_shape[0])
    pred = pred[:t]
    gt = gt[:t]
    if pred.shape[-2:] != gt.shape[-2:]:
        zoom_factors = (
            1.0,
            1.0,
            pred.shape[-2] / gt.shape[-2],
            pred.shape[-1] / gt.shape[-1],
        )
        gt = ndimage.zoom(gt, zoom=zoom_factors, order=1)
    mae = np.mean(np.abs(pred - gt), axis=1)
    return _pool_to_conf_shape(mae, conf_hw=(conf_shape[1], conf_shape[2]))


def save_json(data: Any, path: str) -> None:
    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: _convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_convert(item) for item in obj]
        if isinstance(obj, tuple):
            return [_convert(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(data), f, ensure_ascii=False, indent=2)


def save_csv(rows: Sequence[dict[str, Any]], path: str) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
