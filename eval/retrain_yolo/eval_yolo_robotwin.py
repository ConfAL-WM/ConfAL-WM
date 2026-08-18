#!/usr/bin/env python3
"""Evaluate a fine-tuned YOLO gripper detector on RoboTwin validation data.

Produces:
  - Per-class and overall mAP / precision / recall
  - Per-episode classification report
  - Optional: visual spot-check of predictions on sample frames
  - PR curve exported as PNG

Usage::

  python eval/retrain_yolo/eval_yolo_robotwin.py \\
    --weights eval/retrain_yolo/runs/robotwin_gripper_yolo26s-3/weights/best.pt \\
    --dataset_dir eval/retrain_yolo/robotwin_gripper_yolo \\
    --split val
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import save_json  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_REPO / p)


def _shorten_label(name: str, max_len: int = 50) -> str:
    """Compress RoboTwin filenames for display.

    ``adjust_bottle_aloha-agilex_randomized_500_ep013-0_frame00010``
    → ``adj_bottle_ep013-0_f010``
    """
    for sep in ("_aloha-agilex_", "_aloha_agilex_"):
        if sep in name:
            task_part = name.split(sep)[0]
            rest = name.split(sep, 1)[1]
            for prefix in ("randomized_500_", "rand_500_"):
                if rest.startswith(prefix):
                    rest = rest[len(prefix):]
                    break
            name = f"{task_part}_{rest}"
            break
    # compact frame number:  _000090 → _f90,  _frame000090 → _f90
    name = name.replace("frame", "")
    import re
    name = re.sub(r"_0+(\d+)", r"_f\1", name)
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def _make_shortname_val_dataset(orig_data_yaml: Path, temp_root: Path) -> Path:
    """Create a temp val dataset where image symlinks have compact names.

    Returns the path to the temporary ``data.yaml``.
    """
    import shutil

    with orig_data_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    orig_path = Path(cfg.get("path", orig_data_yaml.parent)).resolve()
    val_rel = cfg.get("val", "images/val")
    src_img_dir = orig_path / val_rel
    src_lbl_dir = orig_path / "labels" / Path(val_rel).name

    dst_img_dir = temp_root / "images" / "val"
    dst_lbl_dir = temp_root / "labels" / "val"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        short = _shorten_label(img_path.stem) + img_path.suffix.lower()
        os.symlink(img_path.resolve(), dst_img_dir / short)
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            os.symlink(lbl_path.resolve(), dst_lbl_dir / (short.rsplit(".", 1)[0] + ".txt"))
        count += 1

    if count == 0:
        raise FileNotFoundError(f"No images found in {src_img_dir}")

    temp_yaml = temp_root / "data.yaml"
    temp_cfg = {
        "path": str(temp_root.resolve()),
        "train": str(cfg.get("train", "images/train")),
        "val": "images/val",
        "names": cfg.get("names", {0: "left gripper", 1: "right gripper"}),
    }
    with temp_yaml.open("w") as f:
        yaml.safe_dump(temp_cfg, f, sort_keys=False)
    print(f"[yolo eval] short-name val dataset: {count} images -> {temp_root}")
    return temp_yaml


def _merge_validation_plots(out_dir: Path) -> Path | None:
    """Merge ultralytics per‑metric PNGs into a single 2×3 contact sheet.

    Looks for the six files produced by ``model.val(plots=True)``::

      BoxPR_curve.png   BoxP_curve.png   BoxR_curve.png
      BoxF1_curve.png   confusion_matrix.png   confusion_matrix_normalized.png

    Returns the path to the merged image, or *None* if fewer than two sources exist.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_files = [
        "BoxPR_curve.png", "PR_curve.png",
        "BoxP_curve.png", "P_curve.png",
        "BoxR_curve.png", "R_curve.png",
        "BoxF1_curve.png", "F1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ]
    existing = [(name, out_dir / name) for name in plot_files if (out_dir / name).exists()]
    if len(existing) < 2:
        return None

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (name, path) in enumerate(existing):
        img = plt.imread(str(path))
        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(name.replace("_", " ").replace(".png", ""), fontsize=10)
        ax.axis("off")

    # hide unused subplots
    for idx in range(len(existing), nrows * ncols):
        axes[idx].axis("off")

    fig.suptitle("YOLO Gripper Detector — Validation Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()

    merged_path = out_dir / "validation_curves_combined.png"
    fig.savefig(merged_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[yolo eval] merged validation plots -> {merged_path}")
    return merged_path


def _device_from_gpus(gpus: str | None) -> str:
    if gpus is None or str(gpus).strip() == "":
        return "0"
    value = str(gpus).strip()
    if value.lower() in {"cpu", "mps"}:
        return value.lower()
    return ",".join(part.strip() for part in value.split(",") if part.strip())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required. Install it or run inside the EWMBench conda env."
        ) from exc

    weights = _resolve_path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    dataset_dir = _resolve_path(args.dataset_dir)
    data_yaml = _resolve_path(args.data_yaml) if args.data_yaml else dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")

    device = _device_from_gpus(args.gpus)

    # Load model
    model = YOLO(str(weights))

    # ---- build temp dataset with shortened filenames → clean batch titles ----
    import tempfile as _tempfile

    _tmpdir = Path(_tempfile.mkdtemp(prefix="yolo_eval_short_"))
    _eval_yaml = _make_shortname_val_dataset(data_yaml, _tmpdir)

    # ---- validation metrics (standard ultralytics val) ----
    print(f"[yolo eval] Running validation on {_eval_yaml} (split={args.split}) ...")
    val_results = model.val(
        data=str(_eval_yaml),
        split=args.split,
        device=device,
        batch=int(args.batch),
        imgsz=int(args.imgsz),
        workers=int(args.workers),
        save_json=False,
        plots=True,
        project=str(_resolve_path(args.project)),
        name=args.name,
        exist_ok=True,
    )

    # ---- collect metrics ----
    metrics: dict[str, Any] = {
        "weights": str(weights),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
    }

    # val_results is a Results or dict-like object
    if hasattr(val_results, "results_dict"):
        for k, v in val_results.results_dict.items():
            metrics[k] = float(v) if isinstance(v, (int, float)) else v
    elif isinstance(val_results, dict):
        metrics.update(val_results)

    # Top-level summary keys
    keys_of_interest = [
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/precision(B)_per_class", "metrics/recall(B)_per_class",
        "metrics/mAP50(B)_per_class", "metrics/mAP50-95(B)_per_class",
        "fitness",
    ]
    summary: dict[str, Any] = {"weights": str(weights), "split": args.split}
    for key in keys_of_interest:
        clean = key.replace("metrics/", "").replace("(B)", "").replace(" ", "_")
        val = metrics.get(key)
        if val is not None:
            if isinstance(val, (list, tuple)):
                summary[clean] = [float(v) for v in val]
            elif isinstance(val, (int, float)):
                summary[clean] = float(val)
            else:
                summary[clean] = val

    # Per-class breakdown
    class_names: dict[int, str] = {0: "left gripper", 1: "right gripper"}
    if data_yaml.exists():
        with data_yaml.open() as f:
            ds_cfg = yaml.safe_load(f) or {}
        if isinstance(ds_cfg.get("names"), dict):
            class_names = {int(k): str(v) for k, v in ds_cfg["names"].items()}

    per_class: dict[str, Any] = {}
    for metric_key, label in [
        ("metrics/mAP50(B)_per_class", "mAP50"),
        ("metrics/mAP50-95(B)_per_class", "mAP50-95"),
        ("metrics/precision(B)_per_class", "precision"),
        ("metrics/recall(B)_per_class", "recall"),
    ]:
        vals = metrics.get(metric_key, [])
        if isinstance(vals, (list, tuple)):
            per_class[label] = {class_names.get(i, f"cls_{i}"): float(v) for i, v in enumerate(vals)}

    summary["per_class"] = per_class

    # ---- save ----
    out_dir = _resolve_path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eval_summary.json"
    save_json(summary, out_json)

    # merge per-metric PNGs into one 2×3 contact sheet, then delete originals
    merged = _merge_validation_plots(out_dir)
    if merged is not None:
        for pattern in ("*_curve.png", "*confusion_matrix*.png"):
            for p in sorted(out_dir.glob(pattern)):
                if p != merged:
                    p.unlink()

    # remove redundant labels-only batch images (pred already overlays labels + scores)
    for lb in sorted(out_dir.glob("val_batch*_labels.jpg")):
        lb.unlink()

    # ---- clean up temp short-name dataset ----
    import shutil as _shutil
    _shutil.rmtree(str(_tmpdir), ignore_errors=True)
    del _eval_yaml, _tmpdir

    # ---- speed benchmark ----
    if not args.skip_speed:
        print("[yolo eval] Running speed benchmark ...")
        speed = model.val(
            data=str(data_yaml),
            split=args.split,
            device=device,
            batch=1,
            imgsz=int(args.imgsz),
            plots=False,
            save_json=False,
            project=str(out_dir),
            name="speed_benchmark",
            exist_ok=True,
        )
        if hasattr(speed, "speed"):
            summary["speed_ms_per_image"] = {
                "preprocess": float(speed.speed.get("preprocess", 0)),
                "inference": float(speed.speed.get("inference", 0)),
                "loss": float(speed.speed.get("loss", 0)),
                "postprocess": float(speed.speed.get("postprocess", 0)),
            }

    # ---- print summary ----
    print(f"\n[yolo eval] ===== {weights.name} =====")
    print(f"  mAP50:      {summary.get('mAP50'):.4f}" if summary.get('mAP50') is not None else "  mAP50: N/A")
    print(f"  mAP50-95:   {summary.get('mAP50-95'):.4f}" if summary.get('mAP50-95') is not None else "  mAP50-95: N/A")
    print(f"  Precision:  {summary.get('precision'):.4f}" if summary.get('precision') is not None else "  Precision: N/A")
    print(f"  Recall:     {summary.get('recall'):.4f}" if summary.get('recall') is not None else "  Recall: N/A")
    if per_class:
        print("  Per-class mAP50:", {k: f"{v:.4f}" for k, v in per_class.get("mAP50", {}).items()})
    print(f"  Saved: {out_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoboTwin gripper YOLO detector")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO checkpoint (.pt)")
    parser.add_argument("--dataset_dir", "--dataset-dir", dest="dataset_dir",
                        default="eval/retrain_yolo/robotwin_gripper_yolo")
    parser.add_argument("--data_yaml", "--data-yaml", dest="data_yaml", default=None,
                        help="Override data.yaml path (default: <dataset_dir>/data.yaml)")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (val / train / test)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gpus", default="0", help="GPU id(s)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="eval/retrain_yolo/eval_runs")
    parser.add_argument("--name", default=None, help="Run name (default: auto-derived from weights filename)")
    parser.add_argument("--skip_speed", "--skip-speed", dest="skip_speed", action="store_true",
                        help="Skip speed benchmarking")
    args = parser.parse_args()

    if args.name is None:
        args.name = Path(args.weights).stem

    evaluate(args)


if __name__ == "__main__":
    main()
