#!/usr/bin/env python3
"""Build RoboTwin gripper pseudo-labels for YOLO fine-tuning.

The converted RoboTwin episodes already contain camera calibration plus 3D
end-effector positions.  This script projects those 3D points into the head RGB
image and writes a compact YOLO dataset with two classes:

  0: left gripper
  1: right gripper

The labels are pseudo-labels around projected EE centers, so a small manual
inspection pass is still useful before treating the detector as final.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import ensure_dir, flatten_manifest_items, load_json, load_yaml, save_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "left gripper", 1: "right gripper"}


def _resolve_path(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else (_REPO / p)


def _load_records(config_path: str | None, manifest_path: str | None, converted_root: str | None) -> list[dict[str, Any]]:
    cfg = load_yaml(config_path) if config_path else {}
    manifest = (
        manifest_path
        or cfg.get("external_split", {}).get("converted_manifest")
        or cfg.get("datasets", {}).get("robotwin", {}).get("converted_manifest")
    )
    root = converted_root or cfg.get("datasets", {}).get("robotwin", {}).get("converted_root")

    if manifest:
        manifest_p = _resolve_path(manifest)
        if manifest_p is None or not manifest_p.exists():
            raise FileNotFoundError(f"Converted manifest not found: {manifest}")
        return flatten_manifest_items(load_json(manifest_p))

    if not root:
        raise ValueError("Provide --manifest, --converted_root, or a config with datasets.robotwin.converted_manifest")
    root_p = _resolve_path(root)
    if root_p is None or not root_p.is_dir():
        raise FileNotFoundError(f"Converted root not found: {root}")

    records: list[dict[str, Any]] = []
    for ep_dir in sorted(root_p.iterdir()):
        if not ep_dir.is_dir():
            continue
        if (ep_dir / "frames").is_dir() and (ep_dir / "camera.npz").exists():
            records.append(
                {
                    "episode_id": ep_dir.name,
                    "episode_dir": str(ep_dir),
                    "frames_dir": str(ep_dir / "frames"),
                    "actions_path": str(ep_dir / "actions_evac.npy"),
                    "camera_path": str(ep_dir / "camera.npz"),
                    "proprio_path": str(ep_dir / "proprio_stats.h5"),
                }
            )
    return records


def _episode_dir(record: dict[str, Any]) -> Path:
    for key in ("episode_dir", "path"):
        if record.get(key):
            return Path(record[key])
    if record.get("frames_dir"):
        return Path(record["frames_dir"]).parent
    raise KeyError(f"Cannot infer episode_dir from record keys: {sorted(record)}")


def _episode_path(ep_dir: Path, value: str | os.PathLike[str] | None, default_name: str) -> Path:
    p = Path(value) if value else ep_dir / default_name
    return p if p.is_absolute() else ep_dir / p


def _frame_files(ep_dir: Path, record: dict[str, Any]) -> list[Path]:
    frames_dir = _episode_path(ep_dir, record.get("frames_dir"), "frames")
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    return files


def _load_ee_positions(ep_dir: Path, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    actions_path = _episode_path(ep_dir, record.get("actions_path"), "actions_evac.npy")
    if actions_path.exists():
        actions = np.load(actions_path).astype(np.float32)
        if actions.ndim == 2 and actions.shape[1] >= 11:
            return actions[:, 0:3], actions[:, 8:11], str(actions_path)
        raise ValueError(f"Unexpected actions_evac shape in {actions_path}: {actions.shape}")

    proprio_path = _episode_path(ep_dir, record.get("proprio_path"), "proprio_stats.h5")
    if not proprio_path.exists():
        raise FileNotFoundError(f"Missing actions_evac.npy and proprio_stats.h5 for {ep_dir}")
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for proprio_stats.h5 fallback") from exc
    with h5py.File(proprio_path, "r") as f:
        pos = np.asarray(f["state/end/position"], dtype=np.float32)
    if pos.ndim == 3 and pos.shape[1] >= 2 and pos.shape[2] >= 3:
        return pos[:, 0, :3], pos[:, 1, :3], str(proprio_path)
    if pos.ndim == 2 and pos.shape[1] >= 6:
        return pos[:, :3], pos[:, 3:6], str(proprio_path)
    raise ValueError(f"Unexpected state/end/position shape in {proprio_path}: {pos.shape}")


def _as_time_array(arr: np.ndarray, n: int, tail_shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == tail_shape:
        return np.repeat(arr[None, ...], n, axis=0)
    if arr.ndim == len(tail_shape) + 1 and arr.shape[1:] == tail_shape:
        return arr
    raise ValueError(f"{name} has unsupported shape {arr.shape}, expected {tail_shape} or [T,{tail_shape}]")


def _load_camera(ep_dir: Path, record: dict[str, Any], n: int) -> tuple[np.ndarray, np.ndarray, str]:
    camera_path = _episode_path(ep_dir, record.get("camera_path"), "camera.npz")
    if not camera_path.exists():
        raise FileNotFoundError(f"camera.npz not found for {ep_dir}")
    cam = np.load(camera_path)
    if "intrinsic_cv" not in cam or "extrinsic_cv" not in cam:
        raise KeyError(f"{camera_path} must contain intrinsic_cv and extrinsic_cv")
    intrinsic = _as_time_array(cam["intrinsic_cv"], n, (3, 3), "intrinsic_cv")
    extrinsic = np.asarray(cam["extrinsic_cv"], dtype=np.float32)
    if extrinsic.shape == (3, 4):
        padded = np.eye(4, dtype=np.float32)
        padded[:3, :4] = extrinsic
        extrinsic = padded
    elif extrinsic.ndim == 3 and extrinsic.shape[1:] == (3, 4):
        padded = np.repeat(np.eye(4, dtype=np.float32)[None, ...], extrinsic.shape[0], axis=0)
        padded[:, :3, :4] = extrinsic
        extrinsic = padded
    extrinsic = _as_time_array(extrinsic, n, (4, 4), "extrinsic_cv")
    return intrinsic, extrinsic, str(camera_path)


def _project(point_xyz: np.ndarray, intrinsic: np.ndarray, w2c: np.ndarray, min_depth: float) -> tuple[float, float, float] | None:
    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = point_xyz.astype(np.float32)
    cam = w2c @ point_h
    z = float(cam[2])
    if not np.isfinite(z) or z <= min_depth:
        return None
    uvw = intrinsic @ cam[:3]
    if abs(float(uvw[2])) < 1e-8:
        return None
    u = float(uvw[0] / uvw[2])
    v = float(uvw[1] / uvw[2])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return u, v, z


def _yolo_box(
    u: float,
    v: float,
    width: int,
    height: int,
    *,
    box_size: float,
    min_box_size: float,
    margin: float,
) -> tuple[float, float, float, float] | None:
    if u < -margin or u >= width + margin or v < -margin or v >= height + margin:
        return None
    half = float(box_size) / 2.0
    x1 = max(0.0, u - half)
    y1 = max(0.0, v - half)
    x2 = min(float(width), u + half)
    y2 = min(float(height), v + half)
    bw = x2 - x1
    bh = y2 - y1
    if bw < min_box_size or bh < min_box_size:
        return None
    return (
        (x1 + x2) / 2.0 / float(width),
        (y1 + y2) / 2.0 / float(height),
        bw / float(width),
        bh / float(height),
    )


def _link_or_copy(src: Path, dst: Path, *, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _render_visualization(src_img: Path, labels: list[str]) -> Image.Image:
    colors = {0: (238, 70, 70), 1: (70, 150, 255)}
    with Image.open(src_img).convert("RGB") as im:
        im = im.copy()
        draw = ImageDraw.Draw(im)
        width, height = im.size
        for line in labels:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = [float(x) for x in parts[1:]]
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            color = colors.get(cls_id, (255, 230, 80))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), CLASS_NAMES.get(cls_id, str(cls_id)), fill=color)
        return im


def _write_contact_sheet(rows: list[dict[str, Any]], dst_path: Path) -> None:
    if not rows:
        return
    rows = sorted(
        rows,
        key=lambda r: (
            int(r.get("record_index", 0)),
            int(r.get("visualization_frame_index_in_episode", 0)),
            int(r.get("frame_idx", 0)),
        ),
    )
    thumb_w, thumb_h = 240, 180
    caption_h = 28
    cols = min(3, max(1, len(rows)))
    rendered: list[Image.Image] = []
    for row in rows:
        im = _render_visualization(Path(row["source_image"]), list(row["label_lines"]))
        im.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h + caption_h), "white")
        canvas.paste(im, ((thumb_w - im.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        caption = f"ep{int(row['record_index']):05d} f{int(row['frame_idx']):06d} {row['split']}"
        draw.text((4, thumb_h + 6), caption, fill=(0, 0, 0))
        rendered.append(canvas)

    sheet_rows = int(math.ceil(len(rendered) / float(cols)))
    sheet = Image.new("RGB", (cols * thumb_w, sheet_rows * (thumb_h + caption_h)), "white")
    for idx, im in enumerate(rendered):
        sheet.paste(im, ((idx % cols) * thumb_w, (idx // cols) * (thumb_h + caption_h)))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst_path, quality=95)


def _process_episode(record: dict[str, Any], rec_idx: int, split: str, params: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "items": [],
        "rows": [],
        "visualizations": [],
        "frames_seen": 0,
        "frames_newly_written": 0,
        "frames_existing_skipped": 0,
        "empty_frames_kept": 0,
        "labels_written": 0,
        "failed": None,
    }
    try:
        ep_dir = _episode_dir(record)
        episode_id = str(record.get("episode_id") or ep_dir.name)
        frames = _frame_files(ep_dir, record)
        left_xyz, right_xyz, action_source = _load_ee_positions(ep_dir, record)
        n = min(len(frames), len(left_xyz), len(right_xyz))
        intrinsic, w2c, camera_source = _load_camera(ep_dir, record, n)
        n = min(n, len(intrinsic), len(w2c))

        frame_stride = max(1, int(params["frame_stride"]))
        for frame_idx in range(0, n, frame_stride):
            src_img = frames[frame_idx]
            with Image.open(src_img) as im:
                width, height = im.size

            labels: list[str] = []
            for cls_id, xyz in [(0, left_xyz[frame_idx]), (1, right_xyz[frame_idx])]:
                projected = _project(xyz, intrinsic[frame_idx], w2c[frame_idx], float(params["min_depth"]))
                if projected is None:
                    continue
                u, v, _z = projected
                box = _yolo_box(
                    u,
                    v,
                    width,
                    height,
                    box_size=float(params["box_size"]),
                    min_box_size=float(params["min_box_size"]),
                    margin=float(params["margin"]),
                )
                if box is None:
                    continue
                labels.append(f"{cls_id} " + " ".join(f"{x:.8f}" for x in box))

            result["frames_seen"] += 1
            if not labels and not bool(params["keep_empty"]):
                continue

            stem = f"{episode_id}_{frame_idx:06d}"
            item = {
                "episode_id": episode_id,
                "frame_idx": int(frame_idx),
                "split": split,
                "stem": stem,
                "src_img": str(src_img),
                "src_suffix": src_img.suffix.lower(),
                "labels": labels,
                "action_source": action_source,
                "camera_source": camera_source,
                "record_index": int(rec_idx),
            }

            if not bool(params.get("write_files", False)):
                result["items"].append(item)
                continue

            out_dir = Path(params["out_dir"])
            dst_img = out_dir / "images" / split / f"{stem}{src_img.suffix.lower()}"
            dst_label = out_dir / "labels" / split / f"{stem}.txt"
            already_complete = dst_img.exists() and dst_label.exists()
            if already_complete:
                result["frames_existing_skipped"] += 1
            else:
                _link_or_copy(src_img, dst_img, copy_images=bool(params["copy_images"]))
                dst_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                result["frames_newly_written"] += 1
            if not labels:
                result["empty_frames_kept"] += 1

            row = {
                "episode_id": episode_id,
                "frame_idx": int(frame_idx),
                "split": split,
                "image": str(dst_img),
                "label": str(dst_label),
                "num_labels": len(labels),
                "action_source": action_source,
                "camera_source": camera_source,
                "record_index": int(rec_idx),
            }
            result["rows"].append(row)
            result["labels_written"] += len(labels)

            vis_episode_indices = params.get("vis_episode_indices", set())
            vis_frames_per_episode = max(0, int(params.get("vis_frames_per_episode", 0)))
            if rec_idx in vis_episode_indices and len(result["visualizations"]) < vis_frames_per_episode:
                vis_row = dict(row)
                vis_row["source_image"] = str(src_img)
                vis_row["label_lines"] = labels
                vis_row["visualization_frame_index_in_episode"] = len(result["visualizations"])
                result["visualizations"].append(vis_row)
    except Exception as exc:
        result["failed"] = {
            "episode_id": str(record.get("episode_id") or record.get("episode_dir") or record.get("path")),
            "error": str(exc),
            "record_index": int(rec_idx),
        }
    return result


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    records = _load_records(args.config, args.manifest, args.converted_root)
    if args.max_episodes is not None:
        records = records[: args.max_episodes]
    if not records:
        raise ValueError("No RoboTwin records found")

    out_dir = _resolve_path(args.output_dir)
    if out_dir is None:
        raise ValueError("--output_dir is required")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        ensure_dir(out_dir / "images" / split)
        ensure_dir(out_dir / "labels" / split)

    rng = random.Random(args.seed)
    episode_indices = list(range(len(records)))
    rng.shuffle(episode_indices)
    n_val = int(round(len(records) * args.val_ratio))
    if len(records) > 1:
        n_val = max(1, min(len(records) - 1, n_val))
    val_indices = set(episode_indices[:n_val])

    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "num_records": len(records),
        "output_dir": str(out_dir),
        "box_size": float(args.box_size),
        "frame_stride": int(args.frame_stride),
        "max_frames": args.max_frames,
        "workers": int(args.workers),
        "file_writes_in_workers": args.max_frames is None,
        "vis_examples_requested": int(args.vis_examples),
        "vis_episodes_requested": int(args.vis_examples),
        "vis_frames_per_episode": int(args.vis_frames_per_episode),
        "camera_convention": "intrinsic_cv plus extrinsic_cv treated as world-to-camera",
        "class_names": CLASS_NAMES,
        "episodes_failed": [],
        "frames_seen": 0,
        "frames_written": 0,
        "frames_newly_written": 0,
        "frames_existing_skipped": 0,
        "empty_frames_kept": 0,
        "labels_written": 0,
        "visualizations_written": 0,
    }
    visualizations: list[dict[str, Any]] = []
    vis_episode_indices = set(range(max(0, int(args.vis_examples))))
    vis_counts_by_record: dict[int, int] = {}

    params = {
        "frame_stride": int(args.frame_stride),
        "box_size": float(args.box_size),
        "min_box_size": float(args.min_box_size),
        "margin": float(args.margin),
        "min_depth": float(args.min_depth),
        "keep_empty": bool(args.keep_empty),
        "copy_images": bool(args.copy_images),
        "out_dir": str(out_dir),
        "write_files": args.max_frames is None,
        "vis_episode_indices": vis_episode_indices,
        "vis_frames_per_episode": int(args.vis_frames_per_episode),
    }

    def _consume_episode_result(ep_result: dict[str, Any]) -> None:
        report["frames_seen"] += int(ep_result.get("frames_seen", 0))
        failed = ep_result.get("failed")
        if failed and len(report["episodes_failed"]) < 100:
            report["episodes_failed"].append(failed)
        if ep_result.get("rows"):
            episode_rows = list(ep_result.get("rows", []))
            rows.extend(episode_rows)
            episode_visualizations = list(ep_result.get("visualizations", []))
            visualizations.extend(episode_visualizations)
            report["frames_written"] += len(episode_rows)
            report["frames_newly_written"] += int(ep_result.get("frames_newly_written", 0))
            report["frames_existing_skipped"] += int(ep_result.get("frames_existing_skipped", 0))
            report["empty_frames_kept"] += int(ep_result.get("empty_frames_kept", 0))
            report["labels_written"] += int(ep_result.get("labels_written", 0))
            report["visualizations_written"] += len(episode_visualizations)
            return
        for item in ep_result.get("items", []):
            if args.max_frames is not None and report["frames_written"] >= args.max_frames:
                return
            split = item["split"]
            src_img = Path(item["src_img"])
            labels = list(item["labels"])
            stem = item["stem"]
            dst_img = out_dir / "images" / split / f"{stem}{item['src_suffix']}"
            dst_label = out_dir / "labels" / split / f"{stem}.txt"
            already_complete = dst_img.exists() and dst_label.exists()
            if already_complete:
                report["frames_existing_skipped"] += 1
            else:
                _link_or_copy(src_img, dst_img, copy_images=bool(args.copy_images))
                dst_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                report["frames_newly_written"] += 1
            if not labels:
                report["empty_frames_kept"] += 1

            row = {
                "episode_id": item["episode_id"],
                "frame_idx": int(item["frame_idx"]),
                "split": split,
                "image": str(dst_img),
                "label": str(dst_label),
                "num_labels": len(labels),
                "action_source": item["action_source"],
                "camera_source": item["camera_source"],
                "record_index": int(item["record_index"]),
            }
            rows.append(row)
            report["frames_written"] += 1
            report["labels_written"] += len(labels)

            record_index = int(item["record_index"])
            vis_count = vis_counts_by_record.get(record_index, 0)
            if (
                record_index in vis_episode_indices
                and vis_count < max(0, int(args.vis_frames_per_episode))
            ):
                vis_row = dict(row)
                vis_row["source_image"] = str(src_img)
                vis_row["label_lines"] = labels
                vis_row["visualization_frame_index_in_episode"] = vis_count
                visualizations.append(vis_row)
                vis_counts_by_record[record_index] = vis_count + 1
                report["visualizations_written"] += 1

    jobs = [
        (rec_idx, record, "val" if rec_idx in val_indices else "train")
        for rec_idx, record in enumerate(records)
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = (
            _process_episode(record, rec_idx, split, params)
            for rec_idx, record, split in jobs
        )
        for ep_result in tqdm(iterator, total=len(jobs), desc="[yolo labels]", unit="ep"):
            if args.max_frames is not None and report["frames_written"] >= args.max_frames:
                break
            _consume_episode_result(ep_result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_episode, record, rec_idx, split, params)
                for rec_idx, record, split in jobs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="[yolo labels]", unit="ep"):
                _consume_episode_result(future.result())
                if args.max_frames is not None and report["frames_written"] >= args.max_frames:
                    for pending in futures:
                        pending.cancel()
                    break

    rows = sorted(
        rows,
        key=lambda r: (int(r.get("record_index", 0)), int(r.get("frame_idx", 0)), str(r.get("split", ""))),
    )
    visualizations = sorted(
        visualizations,
        key=lambda r: (
            int(r.get("record_index", 0)),
            int(r.get("visualization_frame_index_in_episode", 0)),
            int(r.get("frame_idx", 0)),
        ),
    )

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    visualization_path = out_dir / "visualization.jpg"
    _write_contact_sheet(visualizations, visualization_path)
    if visualizations:
        report["visualization_path"] = str(visualization_path)
    save_json(rows, out_dir / "pseudo_label_manifest.json")
    save_json(visualizations, out_dir / "visualization_manifest.json")
    save_json(report, out_dir / "projection_report.json")
    print(
        f"[yolo labels] dataset rows={report['frames_written']} "
        f"(new={report['frames_newly_written']}, existing={report['frames_existing_skipped']}) "
        f"boxes={report['labels_written']} -> {out_dir}"
    )
    if report["episodes_failed"]:
        print(f"[yolo labels] WARNING: {len(report['episodes_failed'])} episodes failed; see projection_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RoboTwin gripper pseudo-labels for YOLO")
    parser.add_argument("--config", default="configs/agibotworld/al_robotwin.yaml", help="Pipeline config with RoboTwin converted_manifest")
    parser.add_argument("--manifest", default=None, help="Converted RoboTwin manifest JSON")
    parser.add_argument("--converted_root", "--converted-root", dest="converted_root", default=None, help="Converted RoboTwin episode root")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="eval/retrain_yolo/robotwin_gripper_yolo")
    parser.add_argument("--max_episodes", "--max-episodes", dest="max_episodes", type=int, default=None)
    parser.add_argument("--max_frames", "--max-frames", dest="max_frames", type=int, default=None, help="Maximum written YOLO frames; default is full data after stride/filtering")
    parser.add_argument("--frame_stride", "--frame-stride", dest="frame_stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="Episode-level parallel workers")
    parser.add_argument("--vis_examples", "--vis-examples", dest="vis_examples", type=int, default=5, help="Include this many episodes in output_dir/visualization.jpg")
    parser.add_argument("--vis_frames_per_episode", "--vis-frames-per-episode", dest="vis_frames_per_episode", type=int, default=3, help="Include this many labeled frames per selected episode in visualization.jpg")
    parser.add_argument("--val_ratio", "--val-ratio", dest="val_ratio", type=float, default=0.1)
    parser.add_argument("--box_size", "--box-size", dest="box_size", type=float, default=36.0, help="Pseudo bbox side length in pixels")
    parser.add_argument("--min_box_size", "--min-box-size", dest="min_box_size", type=float, default=4.0)
    parser.add_argument("--margin", type=float, default=8.0, help="Allow projected centers this many pixels outside the image before clipping")
    parser.add_argument("--min_depth", "--min-depth", dest="min_depth", type=float, default=0.03)
    parser.add_argument("--keep_empty", "--keep-empty", dest="keep_empty", action="store_true", help="Keep frames with no projected gripper label")
    parser.add_argument("--copy_images", "--copy-images", dest="copy_images", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--overwrite", action="store_true", help="Remove and rebuild the output dataset directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
