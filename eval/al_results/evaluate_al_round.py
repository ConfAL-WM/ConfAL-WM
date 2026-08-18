#!/usr/bin/env python3
"""Evaluate a checkpoint against a validation manifest.

Expected workflow (this script only computes metrics; it does NOT run EVAC):

  1. Generate predictions for the eval set via run_val_inference.py or score_pool.py:
     python eval/al_results/run_val_inference.py \\
       --checkpoint .../checkpoints/epoch=*-step=*.ckpt \\
       --config configs/agibotworld/al_robotwin.yaml \\
       --manifest al_runs/robotwin_al/manifests/al_val.json \\
       --num_shards 2 --workers_per_gpu 4 --gpus 0,1

  2. Compute metrics:
     python eval/al_results/evaluate_al_round.py \\
       --checkpoint .../checkpoints/epoch=*-step=*.ckpt \\
       --score_method c3 --select_method c3_tail_risk --weighting frame_patch \\
       --val-manifest al_runs/robotwin_al/manifests/al_val.json \\
       --pred-dir al_runs/robotwin_al/retrain/c3_persistent_risk_frame_patch/val_infer \\
       --output al_runs/robotwin_al/eval/c3_persistent_risk_frame_patch.json \\
       --metrics pixel_mae

  For the warmup v1 baseline:
       --select_method warmup --weighting none
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import load_json, load_yaml, save_json
from eval.al_results.metrics import (
    compute_cvar,
    compute_ewmbench,
    compute_episode_error,
    compute_latent_loss,
    compute_pixel_mae_from_paths,
    compute_risk_reduction,
    flatten_metrics,
    load_metrics_json,
    summarize_losses,
)
from eval.al_results.metrics.ewmbench_metrics import _compute_mean_from_results as _ewmbench_means_from_results
from eval.al_results.utils import ensure_dir, load_frames, normalize_frames

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
SCORE_METHOD_ALIASES = {
    "confidence": "c3",
    "c3_confidence": "c3",
    "robo_reward": "roboreward",
    "robometer-prog": "robometer_prog",
    "robometer-progress": "robometer_prog",
    "robometer-pref": "robometer_pref",
    "robometer-preference": "robometer_pref",
    "prm-as-judge": "prm_judge",
}
WEIGHTING_LABEL_ALIASES = {
    "none": "none",
    "oversampling": "oversampling",
    "confidence_guided_oversampling": "oversampling",
    "frame": "frame",
    "frame_weight": "frame",
    "frame_weighting": "frame",
    "frame_patch": "frame_patch",
    "frame_patch_weight": "frame_patch",
    "frame_patch_weighting": "frame_patch",
    "hybrid": "frame_patch",
    "patch": "frame_patch",
    "patch_weight": "frame_patch",
    "patch_weighting": "frame_patch",
    "loss_map": "frame_patch",
    "patch_only": "patch_only",
    "patch_weight_only": "patch_only",
    "pure_patch": "patch_only",
}
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260716
PSNR_NORMALIZATION_MAX = 30.0
CLIP_SCORE_NORMALIZATION_MAX = 100.0
EWMBENCH_AGGREGATES = {
    "Reconstruction": ("psnr", "ssim"),
    "Motion": ("traj_hsd", "traj_dyn", "traj_ndtw"),
    "Semantics": ("logics", "semantics_CLIPScore", "semantics_BLEUScore"),
}
EWMBENCH_NORMALIZED_ORDER = (
    "Reconstruction",
    "Motion",
    "Semantics",
    "psnr",
    "ssim",
    "scene_consistency",
    "logics",
    "semantics_CLIPScore",
    "semantics_BLEUScore",
    "traj_hsd",
    "traj_dyn",
    "traj_ndtw",
    "diversity",
)


def _safe_name(value: str | None, default: str) -> str:
    text = str(value or default).strip().lower()
    return text.replace("/", "_").replace(" ", "_")


def _canonical_score_method(value: str | None) -> str | None:
    if value is None:
        return None
    name = _safe_name(value, "")
    return SCORE_METHOD_ALIASES.get(name, name)


def _canonical_weighting_label(value: str | None) -> str:
    name = _safe_name(value, "none")
    return WEIGHTING_LABEL_ALIASES.get(name, name)


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bootstrap_ci(
    values: list[float],
    *,
    statistic: str = "mean",
    confidence: float = BOOTSTRAP_CI_LEVEL,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """Bootstrap a confidence interval for one episode-level metric."""
    import numpy as np

    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return None

    def _stat(sample: np.ndarray) -> float:
        if statistic == "median":
            return float(np.median(sample))
        if statistic == "cvar90":
            q = float(np.quantile(sample, 0.9))
            tail = sample[sample >= q]
            return float(tail.mean()) if tail.size else float(sample.max())
        return float(sample.mean())

    point = _stat(arr)
    if n == 1 or n_resamples <= 0:
        return {
            "mean": point,
            "lower": point,
            "upper": point,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    samples = arr[idx]
    if statistic == "median":
        stats = np.median(samples, axis=1)
    elif statistic == "cvar90":
        qs = np.quantile(samples, 0.9, axis=1)
        stats = np.asarray([
            row[row >= q].mean() if np.any(row >= q) else row.max()
            for row, q in zip(samples, qs)
        ])
    else:
        stats = samples.mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": point,
        "lower": float(np.quantile(stats, alpha)),
        "upper": float(np.quantile(stats, 1.0 - alpha)),
    }


def _paired_bootstrap_delta_ci(
    current: dict[str, float],
    reference: dict[str, float],
    *,
    higher_is_better: bool,
    confidence: float = BOOTSTRAP_CI_LEVEL,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float] | None:
    """Paired bootstrap CI for signed method-reference deltas."""
    import numpy as np

    keys = sorted(set(current) & set(reference))
    if not keys:
        return None
    sign = 1.0 if higher_is_better else -1.0
    diffs = np.asarray(
        [
            sign * (float(current[k]) - float(reference[k]))
            for k in keys
            if math.isfinite(float(current[k])) and math.isfinite(float(reference[k]))
        ],
        dtype=np.float64,
    )
    n = int(diffs.size)
    if n == 0:
        return None
    point = float(diffs.mean())
    if n == 1 or n_resamples <= 0:
        return {"mean": point, "lower": point, "upper": point}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    stats = diffs[idx].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": point,
        "lower": float(np.quantile(stats, alpha)),
        "upper": float(np.quantile(stats, 1.0 - alpha)),
    }


def _paired_bootstrap_delta_ci_from_lists(
    current: list[float],
    reference: list[float],
    *,
    higher_is_better: bool,
    confidence: float = BOOTSTRAP_CI_LEVEL,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float] | None:
    if not current or not reference or len(current) != len(reference):
        return None
    current_map = {str(i): float(v) for i, v in enumerate(current)}
    reference_map = {str(i): float(v) for i, v in enumerate(reference)}
    return _paired_bootstrap_delta_ci(
        current_map,
        reference_map,
        higher_is_better=higher_is_better,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def _normalize_ewmbench_metric(metric_name: str, value: Any) -> float | None:
    num = _as_finite_float(value)
    if num is None:
        return None
    if metric_name == "psnr":
        return max(0.0, min(1.0, num / PSNR_NORMALIZATION_MAX))
    if metric_name == "semantics_CLIPScore":
        return max(0.0, min(1.0, num / CLIP_SCORE_NORMALIZATION_MAX))
    return max(0.0, min(1.0, num))


def _ordered_metric_block(values: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in EWMBENCH_NORMALIZED_ORDER:
        if key in values:
            ordered[key] = values[key]
    for key in sorted(values):
        if key not in ordered:
            ordered[key] = values[key]
    return ordered


def _ordered_ci_block(values: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for prefix in ("ewmbench.", ""):
        for key in EWMBENCH_NORMALIZED_ORDER:
            full_key = f"{prefix}{key}"
            if full_key in values:
                ordered[full_key] = values[full_key]
    for key in sorted(values):
        if key not in ordered:
            ordered[key] = values[key]
    return ordered


def _aggregate_metric_lists(metric_values: dict[str, list[float]]) -> dict[str, list[float]]:
    aggregates: dict[str, list[float]] = {}
    for aggregate_name, component_names in EWMBENCH_AGGREGATES.items():
        components = [metric_values.get(name, []) for name in component_names]
        if not all(components):
            continue
        n = min(len(vals) for vals in components)
        if n <= 0:
            continue
        aggregates[aggregate_name] = [
            float(sum(vals[i] for vals in components))
            for i in range(n)
        ]
    return aggregates


def _aggregate_metric_maps(metric_maps: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for aggregate_name, component_names in EWMBENCH_AGGREGATES.items():
        components = [metric_maps.get(name, {}) for name in component_names]
        if not all(components):
            continue
        keys = set(components[0])
        for comp in components[1:]:
            keys &= set(comp)
        if not keys:
            continue
        aggregates[aggregate_name] = {
            key: float(sum(comp[key] for comp in components))
            for key in sorted(keys)
        }
    return aggregates


def _collect_ewmbench_metric_maps(results_json: str | Path | None) -> dict[str, dict[str, float]]:
    """Extract episode-level EWMBench values from evac_c3_results.json.

    Mirrors ewmbench_metrics._compute_mean_from_results(), but keeps the raw
    episode values so evaluate_al_round.py can bootstrap confidence intervals.
    """
    if not results_json:
        return {}
    path = Path(results_json)
    if not path.exists():
        text = str(results_json)
        marker = "ConfAL-WM/"
        if marker in text:
            fallback = _REPO / text.split(marker, 1)[1]
            if fallback.exists():
                path = fallback
    if not path.exists():
        return {}
    import json

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    def _append_number(value: Any, out: list[float]) -> None:
        num = _as_finite_float(value)
        if num is not None:
            out.append(num)

    def _collect_generic(obj: Any, out: list[float]) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                _collect_generic(value, out)
        elif isinstance(obj, list):
            # Some EWMBench metrics are [overall_mean, per_episode_records].
            if (
                len(obj) == 2
                and isinstance(obj[0], (int, float, str))
                and isinstance(obj[1], list)
            ):
                _collect_generic(obj[1], out)
                return
            for value in obj:
                _collect_generic(value, out)
        else:
            _append_number(obj, out)

    values: dict[str, dict[str, float]] = {}
    compound = {"trajectory_consistency", "semantics"}

    def _store(metric: str, key: str, value: Any) -> None:
        num = _as_finite_float(value)
        if num is not None:
            values.setdefault(metric, {})[key] = num

    def _nested_trial_values(metric: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for task_id, episodes in payload.items():
            if not isinstance(episodes, dict):
                continue
            for episode_id, trials in episodes.items():
                if isinstance(trials, dict):
                    for trial_id, value in trials.items():
                        if isinstance(value, dict):
                            continue
                        _store(metric, f"{task_id}/{episode_id}/{trial_id}", value)

    for metric, payload in data.items():
        if metric in compound:
            continue
        if metric == "scene_consistency" and isinstance(payload, list) and len(payload) == 2:
            # Prefer explicit per-video records over the leading aggregate.
            records = payload[1]
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict) and "video_results" in record:
                        video_path = str(record.get("video_path", ""))
                        parts = Path(video_path).parts
                        key = "/".join(parts[-4:-1]) if len(parts) >= 4 else video_path
                        _store(metric, key, record.get("video_results"))
                if values.get(metric):
                    continue
        elif metric in {"psnr", "ssim"}:
            _nested_trial_values(metric, payload)
            if values.get(metric):
                continue
        elif metric == "logics" and isinstance(payload, dict):
            for gid, value in payload.items():
                try:
                    gid_contents = str(gid).split("_dataset_")[-1].split("_")
                    task_id = gid_contents[0]
                    trial_id = gid_contents[-1]
                    episode_id = gid_contents[1] if len(gid_contents) == 3 else "_".join(gid_contents[1:-1])
                    _store(metric, f"{task_id}/{episode_id}/{trial_id}", value)
                except Exception:
                    _store(metric, str(gid), value)
            if values.get(metric):
                continue
        metric_values: list[float] = []
        _collect_generic(payload, metric_values)
        for idx, value in enumerate(metric_values):
            _store(metric, str(idx), value)

    traj_values: dict[str, list[tuple[str, float]]] = {"hsd": [], "dyn": [], "ndtw": []}
    def _collect_traj(obj: Any, prefix: tuple[str, ...] = ()) -> None:
        if isinstance(obj, dict):
            for key in ("hsd", "dyn", "ndtw"):
                if key in obj:
                    num = _as_finite_float(obj[key])
                    if num is not None:
                        traj_values[key].append(("/".join(prefix), num))
            for key, value in obj.items():
                _collect_traj(value, prefix + (str(key),))
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                _collect_traj(value, prefix + (str(idx),))
    _collect_traj(data.get("trajectory_consistency", {}))
    for key, vals in traj_values.items():
        if vals:
            values[f"traj_{key}"] = {k: v for k, v in vals}

    # Semantics is stored flat as {"500": {epXXX: {trial: {BLEUScore,CLIPScore}}}},
    # WITHOUT task / full-episode-name context, so its raw keys ("500/epXXX/trial")
    # cannot intersect with every other metric's "task/fullname/trial" keys. Recover
    # the canonical full-episode key by ORDER-MERGING the per-episode semantics
    # records (in val-processing order, a subsequence of the other metrics) against
    # the canonical episode order taken from logics/psnr (task-keyed, full names).
    canonical: list[tuple[str, str]] = []  # (full_key "task/fullname/trial", ep_token)
    for _src in ("logics", "psnr", "ssim"):
        for full_key in values.get(_src, {}):
            parts = full_key.split("/")
            if len(parts) >= 3:
                canonical.append((full_key, parts[1].split("_ep")[-1]))
        if canonical:
            break

    sem_records: list[tuple[str, float | None, float | None]] = []  # (ep_token, BLEU, CLIP)
    def _collect_sem(obj: Any, prefix: tuple[str, ...] = ()) -> None:
        if isinstance(obj, dict):
            if "BLEUScore" in obj or "CLIPScore" in obj:
                ep_key = prefix[-2] if len(prefix) >= 2 else (prefix[-1] if prefix else "")
                ep_token = ep_key[2:] if ep_key.startswith("ep") else ep_key
                sem_records.append(
                    (ep_token, _as_finite_float(obj.get("BLEUScore")), _as_finite_float(obj.get("CLIPScore")))
                )
                return
            for key, value in obj.items():
                _collect_sem(value, prefix + (str(key),))
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                _collect_sem(value, prefix + (str(idx),))
    _collect_sem(data.get("semantics", {}))

    if canonical and sem_records:
        ptr = 0
        bleu_map: dict[str, float] = {}
        clip_map: dict[str, float] = {}
        for full_key, ep_token in canonical:
            if ptr >= len(sem_records):
                break
            if sem_records[ptr][0] == ep_token:
                _, bleu, clip = sem_records[ptr]
                if bleu is not None:
                    bleu_map[full_key] = bleu
                if clip is not None:
                    clip_map[full_key] = clip
                ptr += 1
        if bleu_map:
            values["semantics_BLEUScore"] = bleu_map
        if clip_map:
            values["semantics_CLIPScore"] = clip_map
    else:
        # Fallback: legacy flat keys when canonical alignment is impossible.
        for _name, _idx in (("BLEUScore", 1), ("CLIPScore", 2)):
            legacy = {tok: rec[_idx] for rec in sem_records if rec[_idx] is not None for tok in [rec[0]]}
            if legacy:
                values[f"semantics_{_name}"] = legacy

    return values


def _collect_ewmbench_metric_values(results_json: str | Path | None) -> dict[str, list[float]]:
    maps = _collect_ewmbench_metric_maps(results_json)
    return {key: list(value_map.values()) for key, value_map in maps.items()}


def _normalize_ewmbench_maps(raw_maps: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for metric_name, value_map in raw_maps.items():
        out: dict[str, float] = {}
        for key, value in value_map.items():
            norm = _normalize_ewmbench_metric(metric_name, value)
            if norm is not None:
                out[key] = norm
        if out:
            normalized[metric_name] = out
    normalized.update(_aggregate_metric_maps(normalized))
    return normalized


def _normalize_ewmbench_values(raw_values: dict[str, list[float]]) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for metric_name, values in raw_values.items():
        out = [
            norm
            for value in values
            if (norm := _normalize_ewmbench_metric(metric_name, value)) is not None
        ]
        if out:
            normalized[metric_name] = out
    normalized.update(_aggregate_metric_lists(normalized))
    return normalized


def _mean_from_metric_values(metric_values: dict[str, list[float]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if finite:
            means[metric_name] = round(sum(finite) / len(finite), 6)
    return _ordered_metric_block(means)


def _with_eval_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Merge eval-only defaults into pipeline configs without changing data paths.

    For dict-valued keys (``ewmbench``, ``risk``) the defaults fill in
    individual missing or empty entries rather than replacing the whole block,
    so pipeline-config overrides (e.g. ``ewmbench.repo``) are preserved while
    eval-config settings (e.g. ``ewmbench.python``) still flow through when the
    pipeline config leaves them blank.
    """
    defaults_path = _REPO / "eval/al_results/configs/eval_config.yaml"
    if not defaults_path.exists():
        return config
    defaults = load_yaml(defaults_path)
    merged = dict(config)
    for key in ["risk", "ewmbench"]:
        if key not in merged:
            if key in defaults:
                merged[key] = defaults[key]
        elif isinstance(defaults.get(key), dict) and isinstance(merged.get(key), dict):
            deep = dict(defaults[key])
            for k, v in merged[key].items():
                # empty strings / None are treated as "not set" → keep default
                if v or (not isinstance(v, str) and v is not None):
                    deep[k] = v
            merged[key] = deep
    return merged


def _apply_ewmbench_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    ewm_cfg = dict(cfg.get("ewmbench", {}) or {})
    if args.use_ewmbench_evaluate_py:
        ewm_cfg["use_official_evaluate_py"] = True
    if getattr(args, "ewmbench_gpus", None):
        ewm_cfg["gpus"] = str(args.ewmbench_gpus).strip()
    if getattr(args, "save_yolo_videos", None) is not None:
        ewm_cfg["save_yolo_videos"] = str(args.save_yolo_videos).strip()
    cfg["ewmbench"] = ewm_cfg
    return cfg


# ---------------------------------------------------------------------------
# EWMBench trajectory preprocessing (YOLO gripper detection → traj.npy)
# ---------------------------------------------------------------------------

def _resolve_yolo_ckpt(ewm_cfg: dict[str, Any], ewmbench_repo: str) -> str:
    """Resolve the YOLO checkpoint path.

    Checks (in order):
    1. ``ewm_cfg["ckpt"]["yolo_world_ckpt"]`` — EVAC eval config
    2. ``{ewmbench_repo}/config.yaml`` → ``ckpt.yolo_world_ckpt`` — EWMBench repo config
    """
    ckpt = ewm_cfg.get("ckpt", {}).get("yolo_world_ckpt", "") if isinstance(ewm_cfg.get("ckpt"), dict) else ""
    if ckpt and Path(ckpt).exists():
        return str(Path(ckpt))

    repo_cfg_path = Path(ewmbench_repo) / "config.yaml"
    if repo_cfg_path.exists():
        repo_cfg = load_yaml(repo_cfg_path)
        if isinstance(repo_cfg, dict):
            ckpt = repo_cfg.get("ckpt", {}).get("yolo_world_ckpt", "")
            if isinstance(ckpt, str) and ckpt:
                p = Path(ckpt)
                if not p.is_absolute():
                    p = Path(ewmbench_repo) / p
                if p.exists():
                    return str(p)
    return ""


def _run_ewmbench_traj_preprocessing(
    episodes: list[dict[str, Any]],
    pred_dir: Path,
    gt_dir: Path,
    ewmbench_repo: str,
    ewmbench_python: str,
    yolo_ckpt: str,
    find_pred_source: Any,
    find_gt_source: Any,
    ewmbench_gpus: str = "",
) -> None:
    """Generate ``traj/traj.npy`` for pred and GT via EWMBench YOLO detection.

    Only processes episodes where the file is missing.  Pred outputs land in
    ``{val_infer}/{episode}/traj/``; GT outputs land alongside the GT frames.
    A single temporary script is written and executed with the EWMBench conda
    python so that ``ultralytics`` (YOLO) is available.
    """
    import subprocess
    import tempfile

    import numpy as np

    def _traj_needs_regenerate(traj_path: Path) -> bool:
        """Return True if *traj_path* is missing or contains no valid detections."""
        if not traj_path.exists():
            return True
        try:
            arr = np.load(traj_path)
            # all (-1, -1) means YOLO detected nothing — stale cache
            return not np.any(arr >= 0)
        except Exception:
            return True

    work_items: list[tuple[str, str, str]] = []  # (input_dir, output_parent, label)

    for ep in episodes:
        ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
        if not ep_id:
            continue

        # --- pred side ---
        pred_source = find_pred_source(pred_dir, ep)
        if pred_source is not None and pred_source.is_dir():
            pred_ep_root = (
                pred_source.parent
                if pred_source.name in {"pred_frames", "frames", "video"}
                else pred_source
            )
            if _traj_needs_regenerate(pred_ep_root / "traj" / "traj.npy"):
                work_items.append((str(pred_source), str(pred_ep_root), f"pred:{ep_id}"))

        # --- GT side ---
        gt_source = find_gt_source(gt_dir, ep)
        if gt_source is not None and gt_source.is_dir():
            gt_ep_root = (
                gt_source.parent
                if gt_source.name in {"frames", "video"}
                else gt_source
            )
            if _traj_needs_regenerate(gt_ep_root / "traj" / "traj.npy"):
                if os.access(str(gt_ep_root), os.W_OK):
                    work_items.append((str(gt_source), str(gt_ep_root), f"gt:{ep_id}"))

    if not work_items:
        print("[ewmbench preproc] All traj/traj.npy files already present — nothing to do")
        return

    print(f"[ewmbench preproc] {len(work_items)} traj sources need generation "
          f"(pred={sum(1 for _, _, l in work_items if l.startswith('pred:'))}, "
          f"gt={sum(1 for _, _, l in work_items if l.startswith('gt:'))})")

    # Build a single batch script.  We use our own detection loop (lower conf,
    # resize to training resolution) instead of EWMBench's detection_tracking.py
    # which uses conf=0.8 and resize to 640×480.
    script_lines = [
        "import logging, os, sys, cv2",
        "import numpy as np",
        "from collections import defaultdict",
        "from tqdm import tqdm",
        "from ultralytics import YOLO",
        "",
        "os.environ.setdefault('YOLO_VERBOSE', 'false')",
        "logging.getLogger('ultralytics').setLevel(logging.ERROR)",
        "logging.getLogger('ultralytics').propagate = False",
        "",
        "_CONF = 0.05",
        f"_MODEL_PATH = {yolo_ckpt!r}",
        f"_WORK = {work_items!r}",
        "",
        "_model = YOLO(_MODEL_PATH).to('cuda:0')",
        "",
        "_failed = 0",
        "for _input, _output_dir, _label in tqdm(_WORK, desc='[ewmbench preproc]', unit='src', ncols=80):",
        "    try:",
        "        _traj_dir = os.path.join(_output_dir, 'traj')",
        "        os.makedirs(_traj_dir, exist_ok=True)",
        "        _image_files = sorted([f for f in os.listdir(_input) if f.lower().endswith(('.jpg','.jpeg','.png'))])",
        "        _traj = []",
        "        for _fname in _image_files:",
        "            _img = cv2.imread(os.path.join(_input, _fname))",
        "            if _img is None:",
        "                continue",
        "            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)",
        "            _h, _w = _img.shape[:2]",
        "            # let ultralytics handle preprocessing (letterbox, normalize)",
        "            _results = _model.track(_img, persist=True, conf=_CONF, imgsz=640, verbose=False)",
        "            _boxes = _results[0].boxes",
        "            _clses = _boxes.cls.cpu().tolist() if _boxes.cls is not None else []",
        "            _confs = _boxes.conf.cpu().tolist() if _boxes.conf is not None else []",
        "            # pick best detection per class, normalize to [0,1] by original image size",
        "            _best = {}",
        "            for _i, (_c, _f) in enumerate(zip(_clses, _confs)):",
        "                if int(_c) not in _best or _f > _best[int(_c)][1]:",
        "                    _best[int(_c)] = (_i, _f)",
        "            _lx = _ly = _rx = _ry = -1.0",
        "            if 0 in _best and _best[0][1] > 0:",
        "                _xywh = _boxes[_best[0][0]].xywh.cpu().numpy()[0]",
        "                _lx, _ly = float(_xywh[0]) / _w, float(_xywh[1]) / _h",
        "            if 1 in _best and _best[1][1] > 0:",
        "                _xywh = _boxes[_best[1][0]].xywh.cpu().numpy()[0]",
        "                _rx, _ry = float(_xywh[0]) / _w, float(_xywh[1]) / _h",
        "            _traj.append([(_lx, _ly), (_rx, _ry)])",
        "        np.save(os.path.join(_traj_dir, 'traj.npy'), np.array(_traj, dtype=np.float32).reshape(-1, 2, 2))",
        "    except Exception as _e:",
        "        _failed += 1",
        "        tqdm.write(f'[ewmbench preproc]  FAILED {_label}: {_e}')",
        f"print(f'[ewmbench preproc] done: {{len(_WORK) - _failed}}/{{len(_WORK)}} ok, {{_failed}} failed')",
        "",
    ]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(script_lines))
        tmp.close()

        _env = os.environ.copy()
        if ewmbench_gpus:
            _env["CUDA_VISIBLE_DEVICES"] = str(ewmbench_gpus).strip()
        proc = subprocess.run(
            [ewmbench_python, tmp.name],
            env=_env, text=True, timeout=14400,
        )
        if proc.returncode != 0:
            print(f"\n[ewmbench preproc] WARNING: exit code {proc.returncode}")
    finally:
        os.unlink(tmp.name)


def _load_episode_list(manifest_path: str) -> list[dict[str, Any]]:
    payload = load_json(manifest_path)
    if isinstance(payload, dict):
        items = payload.get("items", payload.get("episodes", []))
        if not items and "records" in payload:
            items = payload["records"]
        return items
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected manifest format: {type(payload)}")


def _dir_has_images(path: Path) -> bool:
    return path.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir())


def _first_video_file(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    videos = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    return videos[0] if videos else None


def _frame_source_from_path(
    path: Path,
    preferred_files: list[str] | None = None,
    *,
    allow_direct_images: bool = True,
) -> Path | None:
    """Return an image directory or video file that load_frames/EWMBench can address."""
    preferred_files = preferred_files or []
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return path
    if not path.is_dir():
        return None

    for name in preferred_files:
        candidate = path / name
        source = _frame_source_from_path(candidate)
        if source is not None:
            return source

    for sub in ["pred_frames", "frames", "video"]:
        candidate = path / sub
        if _dir_has_images(candidate):
            return candidate
        video = _first_video_file(candidate)
        if video is not None:
            return video

    if allow_direct_images and _dir_has_images(path):
        return path

    for sub in ["pred_video", "videos"]:
        video = _first_video_file(path / sub)
        if video is not None:
            return video

    video = _first_video_file(path)
    if video is not None:
        return video
    return None


def _find_pred_source(pred_dir: Path, ep: dict[str, Any]) -> Path | None:
    """Look for prediction frames/video under pred_dir/{episode_id}."""
    ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
    if not ep_id:
        return None
    for candidate in [pred_dir / ep_id, pred_dir / ep_id.replace("/", "_")]:
        source = _frame_source_from_path(candidate, preferred_files=["outputs.mp4"])
        if source is not None:
            return source
    return None


def _gt_root_candidates(gt_dir: Path) -> list[Path]:
    roots = [gt_dir]
    for candidate in [
        gt_dir / "gt_dataset",
        gt_dir / "validation" / "gt_dataset",
        gt_dir / "WorldModel" / "validation" / "gt_dataset",
    ]:
        if candidate not in roots:
            roots.append(candidate)
    return roots


def _agibot_gt_candidates(ep: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
    task_id = ep.get("task_id")
    raw_ep = ep.get("raw_episode_id")
    seg = ep.get("segment_id")
    if task_id is not None and raw_ep is not None:
        raw = str(raw_ep)
        if seg is not None:
            seg_s = str(seg)
            out.extend([(str(task_id), raw + seg_s), (str(task_id), f"{raw}-{seg_s}")])
        out.append((str(task_id), raw))

    if ep_id:
        if "-" in ep_id:
            parts = ep_id.split("-")
        elif "_" in ep_id:
            parts = ep_id.split("_")
        else:
            parts = []
        if len(parts) >= 2:
            task = parts[0]
            rest = parts[1:]
            out.append((task, "".join(rest)))
            out.append((task, rest[0]))
            if len(rest) >= 2:
                out.append((task, f"{rest[0]}-{rest[1]}"))

    # Preserve order while dropping duplicates.
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _find_gt_video(gt_dir: Path, ep: dict) -> Path | None:
    """Look for GT data under gt_dir or via episode metadata.

    AgiBot gt_dataset layout:  gt_dir/{task_id}/{ep_num}/video/
    External (RoboTwin etc.):   {episode_dir}/frames/  (absolute path from manifest)
    """
    # --- Strategy 1: episode_dir from external_worldmodel manifest ---
    ep_dir = ep.get("episode_dir", "")
    if ep_dir:
        source = _frame_source_from_path(Path(ep_dir))
        if source is not None:
            return source

    for key in ["frames_dir", "video_dir", "video_mp4"]:
        value = ep.get(key, "")
        if value:
            source = _frame_source_from_path(Path(value))
            if source is not None:
                return source

    # --- Strategy 1b: path field from AgiBot manifest (points to episode dir) ---
    ep_path = ep.get("path", "")
    if ep_path:
        p = Path(ep_path)
        files = ep.get("files", {})
        if isinstance(files, dict):
            preferred = [str(files.get("video", ""))] if files.get("video") else []
        else:
            preferred = []
        source = _frame_source_from_path(p, preferred_files=preferred, allow_direct_images=False)
        if source is not None:
            return source

    # --- Strategy 2: AgiBot gt_dataset ---
    for root in _gt_root_candidates(gt_dir):
        for task_id, ep_folder in _agibot_gt_candidates(ep):
            for candidate in [root / task_id / ep_folder, root / f"{task_id}-{ep_folder}"]:
                source = _frame_source_from_path(candidate)
                if source is not None:
                    return source

    # --- Strategy 4: direct ep_id ---
    ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
    for root in _gt_root_candidates(gt_dir):
        source = _frame_source_from_path(root / ep_id)
        if source is not None:
            return source
    return None


def _find_gt_image_dir(gt_dir: Path, ep: dict) -> Path | None:
    """Find an image-frame GT directory for tools that cannot read videos.

    EWMBench's upstream loader only accepts directories containing image files.
    For AgiBot, prefer validation-style ``gt_dataset`` roots over manifest
    ``path`` fields, because phase-1 manifests often point to train mp4 dirs.
    """
    ep_dir = ep.get("episode_dir", "")
    if ep_dir:
        source = _frame_source_from_path(Path(ep_dir))
        if source is not None and source.is_dir():
            return source

    for key in ["frames_dir", "video_dir"]:
        value = ep.get(key, "")
        if value:
            source = _frame_source_from_path(Path(value))
            if source is not None and source.is_dir():
                return source

    for root in _gt_root_candidates(gt_dir):
        for task_id, ep_folder in _agibot_gt_candidates(ep):
            for candidate in [root / task_id / ep_folder, root / f"{task_id}-{ep_folder}"]:
                source = _frame_source_from_path(candidate)
                if source is not None and source.is_dir():
                    return source

    ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
    for root in _gt_root_candidates(gt_dir):
        source = _frame_source_from_path(root / ep_id)
        if source is not None and source.is_dir():
            return source

    ep_path = ep.get("path", "")
    if ep_path:
        source = _frame_source_from_path(Path(ep_path), allow_direct_images=False)
        if source is not None and source.is_dir():
            return source
    return None


def _find_gt_ewmbench_source(gt_dir: Path, ep: dict) -> Path | None:
    """Find GT for EWMBench: prefer image dirs, then allow mp4 fallback."""
    source = _find_gt_image_dir(gt_dir, ep)
    if source is not None:
        return source
    return _find_gt_video(gt_dir, ep)


def _resolve_gt_dir(gt_dir: str | None, config: dict[str, Any], val_manifest: str) -> str:
    if gt_dir and gt_dir != ".":
        return gt_dir
    candidates: list[Any] = [
        config.get("gt_video_dir"),
        config.get("gt_dir"),
        config.get("data", {}).get("gt_path") if isinstance(config.get("data"), dict) else None,
    ]
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    for key in ["val_root", "validation_root"]:
        if data_cfg.get(key):
            candidates.append(Path(data_cfg[key]) / "gt_dataset")
            candidates.append(data_cfg[key])
    if data_cfg.get("worldmodel_root"):
        candidates.append(Path(data_cfg["worldmodel_root"]) / "validation" / "gt_dataset")
    try:
        payload = load_json(val_manifest)
        if isinstance(payload, dict) and payload.get("root"):
            candidates.append(Path(payload["root"]) / "gt_dataset")
            candidates.append(payload["root"])
    except Exception:
        pass
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return gt_dir or "."


def _compute_internal_episode_values(
    *,
    episodes: list[dict[str, Any]],
    pred_dir: str | Path,
    gt_dir: str | Path,
    do_latent: bool,
    do_pixel: bool,
    do_risk: bool,
    alpha: float,
    beta: float,
    pred_start_frame: int,
    gt_start_frame: int,
) -> dict[str, dict[str, float]]:
    """Compute per-episode internal metric values for paired bootstrap refs."""
    import numpy as np

    pred_p = Path(pred_dir)
    gt_p = Path(gt_dir)
    values: dict[str, dict[str, float]] = {
        "latent_loss.mean": {},
        "pixel_mae.mean": {},
        "risk.risk_cvar90": {},
    }
    for ep in episodes:
        ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
        if not ep_id:
            continue
        latent_mean: float | None = None
        pixel_mean: float | None = None
        try:
            if do_pixel or do_risk:
                pred_source = _find_pred_source(pred_p, ep)
                gt_source = _find_gt_video(gt_p, ep)
                if pred_source is not None and gt_source is not None:
                    pixel_summary = compute_pixel_mae_from_paths(
                        pred_source,
                        gt_source,
                        pred_start_frame=pred_start_frame,
                        gt_start_frame=gt_start_frame,
                    )
                    if pixel_summary.get("mean") is not None:
                        pixel_mean = float(pixel_summary["mean"])
                        values["pixel_mae.mean"][ep_id] = pixel_mean
            if do_latent or do_risk:
                latent_pred_path = pred_p / ep_id / "latent_pred.npy"
                latent_gt_path = pred_p / ep_id / "latent_gt.npy"
                if latent_pred_path.exists() and latent_gt_path.exists():
                    pred_latent = np.load(latent_pred_path).astype(np.float32)
                    gt_latent = np.load(latent_gt_path).astype(np.float32)
                    if pred_latent.shape != gt_latent.shape:
                        if (
                            pred_latent.ndim == gt_latent.ndim
                            and pred_latent.ndim >= 1
                            and pred_latent.shape[1:] == gt_latent.shape[1:]
                        ):
                            common_t = min(int(pred_latent.shape[0]), int(gt_latent.shape[0]))
                            pred_latent = pred_latent[:common_t]
                            gt_latent = gt_latent[:common_t]
                        else:
                            continue
                    latent_summary = compute_latent_loss(pred_latent, gt_latent)
                    if latent_summary.get("mean") is not None:
                        latent_mean = float(latent_summary["mean"])
                        values["latent_loss.mean"][ep_id] = latent_mean
            if do_risk and (latent_mean is not None or pixel_mean is not None):
                l_err = latent_mean if latent_mean is not None else 0.0
                p_err = pixel_mean if pixel_mean is not None else 0.0
                values["risk.risk_cvar90"][ep_id] = float(alpha * l_err + beta * p_err)
        except Exception:
            continue
    return {key: val for key, val in values.items() if val}


def _auto_config_from_paths(val_manifest: str, pred_dir: str) -> dict[str, Any]:
    """Best-effort pipeline config discovery for simple README commands."""
    text = f"{val_manifest} {pred_dir}".lower()
    candidates: list[Path] = []
    if "robotwin" in text:
        candidates.append(_REPO / "configs/agibotworld/al_robotwin.yaml")
    if "agibot" in text:
        candidates.append(_REPO / "configs/agibotworld/al_agibot.yaml")
    candidates.extend(
        [
            _REPO / "eval/al_results/configs/eval_config.yaml",
        ]
    )
    for path in candidates:
        if path.exists():
            cfg = _with_eval_defaults(load_yaml(path))
            cfg["_auto_loaded_config"] = str(path)
            print(f"[eval] auto-loaded config: {path}")
            return cfg
    return {}


def evaluate_round(
    checkpoint_path: str,
    round_id: int,
    method: str,
    val_manifest: str,
    pred_dir: str,
    gt_dir: str | None,
    output: str,
    metrics: list[str],
    config: dict[str, Any] | None = None,
    weighting: str = "",
    score_method: str | None = None,
    select_method: str | None = None,
    base_metrics_path: str | None = None,
    prev_metrics_path: str | None = None,
    paired_bootstrap_ref: str | None = None,
    ewmbench_output_dir: str | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    risk_cfg = cfg.get("risk", {})
    alpha = float(risk_cfg.get("alpha", 1.0))
    beta = float(risk_cfg.get("beta", 1.0))
    cvar_q = float(risk_cfg.get("cvar_q", 0.9))
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    gt_start_frame = int(eval_cfg.get("gt_start_frame", cfg.get("gt_start_frame", cfg.get("n_previous", 4))))
    pred_start_frame = int(eval_cfg.get("pred_start_frame", cfg.get("pred_start_frame", 0)))

    episodes = _load_episode_list(val_manifest)
    if not episodes:
        raise ValueError(f"No episodes in manifest: {val_manifest}")

    metrics = [m for m in metrics if m]
    known_metrics = {"latent_loss", "pixel_mae", "risk_reduction", "ewmbench"}
    unknown = sorted(set(metrics) - known_metrics)
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {sorted(known_metrics)}")

    pred_p = Path(pred_dir)
    gt_dir = _resolve_gt_dir(gt_dir, cfg, val_manifest)
    gt_p = Path(gt_dir)

    all_latent_errors: list[float] = []
    all_pixel_errors: list[float] = []
    latent_summary: dict[str, float | None] = {}
    pixel_summary: dict[str, float | None] = {}
    ep_errors: list[float] = []
    num_eval = 0
    missing_pred = 0
    missing_gt = 0
    missing_latent = 0
    failed = 0
    invalid_latent = 0
    truncated_latent = 0
    skipped_examples: list[dict[str, str]] = []
    missing_latent_examples: list[dict[str, Any]] = []
    invalid_latent_examples: list[dict[str, Any]] = []
    failed_examples: list[dict[str, str]] = []
    internal_episode_values: dict[str, dict[str, float]] = {
        "latent_loss.mean": {},
        "pixel_mae.mean": {},
        "risk.risk_cvar90": {},
    }

    do_latent = "latent_loss" in metrics
    do_pixel = "pixel_mae" in metrics
    do_risk = "risk_reduction" in metrics
    do_ewmbench = "ewmbench" in metrics
    do_internal = do_latent or do_pixel or do_risk

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    if do_internal:
        for ep in tqdm(episodes, desc=f"[eval] {method}", unit="ep"):
            ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
            if not ep_id:
                continue

            pred_source = _find_pred_source(pred_p, ep)
            gt_source = _find_gt_video(gt_p, ep)

            if pred_source is None or gt_source is None:
                if pred_source is None:
                    missing_pred += 1
                if gt_source is None:
                    missing_gt += 1
                if len(skipped_examples) < 20:
                    skipped_examples.append(
                        {
                            "episode_id": ep_id,
                            "reason": "missing_pred" if pred_source is None else "missing_gt",
                        }
                    )
                continue

            latent_mean: float | None = None
            pixel_mean: float | None = None

            try:
                # --- pixel MAE ---
                if do_pixel or do_risk:
                    pixel_summary = compute_pixel_mae_from_paths(
                        pred_source,
                        gt_source,
                        pred_start_frame=pred_start_frame,
                        gt_start_frame=gt_start_frame,
                    )
                    if pixel_summary.get("mean") is not None:
                        pixel_mean = float(pixel_summary["mean"])
                        all_pixel_errors.append(pixel_mean)
                        internal_episode_values["pixel_mae.mean"][ep_id] = pixel_mean

                # --- latent loss ---
                if do_latent or do_risk:
                    latent_pred_path = pred_p / ep_id / "latent_pred.npy"
                    latent_gt_path = pred_p / ep_id / "latent_gt.npy"
                    if latent_pred_path.exists() and latent_gt_path.exists():
                        import numpy as np

                        try:
                            pred_latent = np.load(latent_pred_path).astype(np.float32)
                            gt_latent = np.load(latent_gt_path).astype(np.float32)
                            if pred_latent.shape != gt_latent.shape:
                                if (
                                    pred_latent.ndim == gt_latent.ndim
                                    and pred_latent.ndim >= 1
                                    and pred_latent.shape[1:] == gt_latent.shape[1:]
                                ):
                                    common_t = min(int(pred_latent.shape[0]), int(gt_latent.shape[0]))
                                    if common_t <= 0:
                                        raise ValueError(
                                            f"Shape mismatch with empty common prefix: "
                                            f"pred {pred_latent.shape} vs gt {gt_latent.shape}"
                                        )
                                    truncated_latent += 1
                                    if len(invalid_latent_examples) < 20:
                                        invalid_latent_examples.append(
                                            {
                                                "episode_id": ep_id,
                                                "reason": "latent_time_truncated",
                                                "pred_shape": list(pred_latent.shape),
                                                "gt_shape": list(gt_latent.shape),
                                                "used_frames": common_t,
                                            }
                                        )
                                    pred_latent = pred_latent[:common_t]
                                    gt_latent = gt_latent[:common_t]
                                else:
                                    raise ValueError(
                                        f"Shape mismatch: pred {pred_latent.shape} vs gt {gt_latent.shape}"
                                    )
                            latent_summary = compute_latent_loss(pred_latent, gt_latent)
                            if latent_summary.get("mean") is not None:
                                latent_mean = float(latent_summary["mean"])
                                all_latent_errors.append(latent_mean)
                                internal_episode_values["latent_loss.mean"][ep_id] = latent_mean
                        except Exception as latent_exc:
                            invalid_latent += 1
                            if len(invalid_latent_examples) < 20:
                                invalid_latent_examples.append(
                                    {
                                        "episode_id": ep_id,
                                        "reason": "latent_invalid",
                                        "error": str(latent_exc),
                                        "latent_pred_path": str(latent_pred_path),
                                        "latent_gt_path": str(latent_gt_path),
                                    }
                                )
                    else:
                        missing_latent += 1
                        if len(missing_latent_examples) < 20:
                            missing_latent_examples.append(
                                {
                                    "episode_id": ep_id,
                                    "latent_pred_exists": latent_pred_path.exists(),
                                    "latent_gt_exists": latent_gt_path.exists(),
                                }
                            )

                # --- episode-level error ---
                if do_risk:
                    if latent_mean is not None or pixel_mean is not None:
                        l_err = latent_mean if latent_mean is not None else 0.0
                        p_err = pixel_mean if pixel_mean is not None else 0.0
                        ep_error = float(alpha * l_err + beta * p_err)
                        ep_errors.append(ep_error)
                        internal_episode_values["risk.risk_cvar90"][ep_id] = ep_error
                num_eval += 1

            except Exception as e:
                import traceback
                print(f"[eval] {ep_id}: {e}")
                traceback.print_exc()
                failed += 1
                if len(failed_examples) < 20:
                    failed_examples.append({"episode_id": ep_id, "error": str(e)})
                continue

    # --- aggregate ---
    num_eps = num_eval

    latent_result: dict[str, float | None] = {
        "mean": None, "median": None, "cvar90": None,
    }
    if all_latent_errors:
        import numpy as np

        latent_result = summarize_losses(np.asarray(all_latent_errors, dtype=np.float32))

    pixel_result: dict[str, float | None] = {
        "mean": None, "median": None, "cvar90": None,
    }
    if all_pixel_errors:
        import numpy as np

        pixel_result = summarize_losses(np.asarray(all_pixel_errors, dtype=np.float32))

    risk_result: dict[str, float | None] = {
        "risk_cvar90": None,
    }
    if ep_errors:
        risk_result["risk_cvar90"] = compute_cvar(ep_errors, cvar_q)

    result: dict[str, Any] = {
        "method": method,
        "score_method": score_method,
        "select_method": select_method,
        "weighting": weighting or "none",
        "round_id": round_id,
        "checkpoint": checkpoint_path,
        "val_manifest": val_manifest,
        "pred_dir": pred_dir,
        "gt_dir": gt_dir,
        "pred_start_frame": pred_start_frame,
        "gt_start_frame": gt_start_frame,
        "num_manifest_episodes": len(episodes),
        "num_eval_episodes": num_eps,
        "num_missing_pred": missing_pred,
        "num_missing_gt": missing_gt,
        "num_missing_latent": missing_latent,
        "num_invalid_latent": invalid_latent,
        "num_truncated_latent": truncated_latent,
        "num_failed_episodes": failed,
    }
    if skipped_examples:
        result["skipped_examples"] = skipped_examples
    if missing_latent_examples:
        result["missing_latent_examples"] = missing_latent_examples
    if invalid_latent_examples:
        result["invalid_latent_examples"] = invalid_latent_examples
    if failed_examples:
        result["failed_examples"] = failed_examples
    if do_latent:
        result["latent_loss"] = latent_result
    if do_pixel:
        result["pixel_mae"] = pixel_result
    if do_risk:
        result["risk"] = risk_result
    if do_ewmbench:
        # Explicit --ewmbench-output-dir wins; otherwise derive from --output so
        # the ewmbench artifacts sit next to the metrics json. Distinct output
        # dirs per checkpoint are required to avoid cross-step races when
        # evaluating multiple steps in parallel.
        if ewmbench_output_dir:
            ewmbench_output_dir = Path(ewmbench_output_dir)
        else:
            ewmbench_output_dir = Path(output).parent / f"{Path(output).stem}_ewmbench"
        try:
            module_file = Path(sys.modules[compute_ewmbench.__module__].__file__).resolve()
            ewmbench_cfg = cfg.get("ewmbench", {}) if isinstance(cfg.get("ewmbench"), dict) else {}
            use_official = bool(ewmbench_cfg.get("use_official_evaluate_py", False))
            print(
                "[eval] EWMBench adapter: "
                f"module={module_file}, "
                f"repo={ewmbench_cfg.get('repo', 'third_party/EWMBench')}, "
                f"decode_video_files={ewmbench_cfg.get('decode_video_files', True)}, "
                f"use_official_evaluate_py={use_official}"
            )

            # traj generation is handled inside compute_ewmbench (after adapter
            # tree build, so mp4 GT sources are already decoded to frames).

            ewmbench_result = compute_ewmbench(
                episodes=episodes,
                pred_dir=pred_p,
                gt_dir=gt_p,
                output_dir=ewmbench_output_dir,
                config=cfg,
                project_root=_REPO,
                find_pred_source=_find_pred_source,
                find_gt_source=_find_gt_ewmbench_source,
            )
        except Exception as e:
            ewmbench_cfg = cfg.get("ewmbench", {}) if isinstance(cfg.get("ewmbench"), dict) else {}
            if bool(ewmbench_cfg.get("strict", False)):
                raise
            print(f"[eval WARNING] EWMBench failed: {e}")
            ewmbench_result = {
                "status": "failed",
                "error": str(e),
                "output_dir": str(ewmbench_output_dir),
            }
        result["ewmbench"] = ewmbench_result
        if not do_internal:
            result["num_eval_episodes"] = ewmbench_result.get("num_eval_episodes", 0)
    if do_internal and num_eps == 0:
        print(
            "[eval WARNING] no episodes were successfully evaluated "
            f"(missing_pred={missing_pred}, missing_gt={missing_gt}, failed={failed})."
        )

    mean_result: dict[str, Any] = {}
    self_ci_result: dict[str, Any] = {}
    if do_latent and all_latent_errors:
        for stat in ("mean", "median", "cvar90"):
            ci = _bootstrap_ci(all_latent_errors, statistic=stat)
            if ci is not None:
                self_ci_result[f"latent_loss.{stat}"] = ci
    if do_pixel and all_pixel_errors:
        for stat in ("mean", "median", "cvar90"):
            ci = _bootstrap_ci(all_pixel_errors, statistic=stat)
            if ci is not None:
                self_ci_result[f"pixel_mae.{stat}"] = ci
    if do_risk and ep_errors:
        ci = _bootstrap_ci(ep_errors, statistic="cvar90")
        if ci is not None:
            self_ci_result["risk.risk_cvar90"] = ci
    current_ewmbench_maps: dict[str, dict[str, float]] = {}
    current_ewmbench_values: dict[str, list[float]] = {}
    if do_ewmbench:
        ewmbench_results_json = result.get("ewmbench", {}).get("results_json")
        current_ewmbench_maps = _normalize_ewmbench_maps(
            _collect_ewmbench_metric_maps(ewmbench_results_json)
        )
        # Derive aligned per-episode lists from the (key-aligned) maps so that
        # aggregates use the map intersection instead of a fragile index-wise sum
        # across metrics with different episode coverage / ordering.
        current_ewmbench_values = {
            metric_name: list(value_map.values())
            for metric_name, value_map in current_ewmbench_maps.items()
        }
        mean_result.update(_mean_from_metric_values(current_ewmbench_values))
        for metric_name in _ordered_metric_block(current_ewmbench_values):
            ci = _bootstrap_ci(current_ewmbench_values[metric_name], statistic="mean")
            if ci is not None:
                self_ci_result[f"ewmbench.{metric_name}"] = ci
        # Trajectory detector-failure-aware breakdown. EWMBench stores
        # hsd=dyn=ndtw=0.000 when the YOLO gripper detector fails on a prediction
        # (a worst-score, since traj metrics are higher=better). traj_hsd/dyn/ndtw
        # + Motion above already include those 0s (canonical EWMBench). Add the
        # detected-only "clean" means + EEF Detection Rate so both views are
        # available. Inject ONLY the new keys (don't overwrite normalized
        # psnr/ssim/clip already in mean_result).
        _TRAJ_CLEAN_KEYS = (
            "traj_hsd_clean", "traj_dyn_clean", "traj_ndtw_clean", "Motion_clean",
            "eef_detection_rate", "n_trajectory_episodes", "n_detected_episodes",
        )
        try:
            _traj_means = _ewmbench_means_from_results(Path(ewmbench_results_json))
            for _k in _TRAJ_CLEAN_KEYS:
                if _k in _traj_means:
                    mean_result[_k] = _traj_means[_k]
        except Exception:
            pass
    if mean_result:
        result["mean"] = mean_result
    if self_ci_result:
        result["self_bootstrap_0.95CI"] = _ordered_ci_block(self_ci_result)

    paired_ci_result: dict[str, Any] = {}
    ref_path = Path(paired_bootstrap_ref) if paired_bootstrap_ref else None
    if ref_path is not None and ref_path.exists() and ref_path.resolve() != Path(output).resolve():
        try:
            ref_result = load_json(ref_path)
            if do_internal:
                ref_pred_dir = ref_result.get("pred_dir")
                ref_gt_dir = ref_result.get("gt_dir", gt_dir)
                if ref_pred_dir:
                    ref_internal = _compute_internal_episode_values(
                        episodes=episodes,
                        pred_dir=ref_pred_dir,
                        gt_dir=ref_gt_dir,
                        do_latent=do_latent,
                        do_pixel=do_pixel,
                        do_risk=do_risk,
                        alpha=alpha,
                        beta=beta,
                        pred_start_frame=pred_start_frame,
                        gt_start_frame=gt_start_frame,
                    )
                    for metric_name in ("latent_loss.mean", "pixel_mae.mean", "risk.risk_cvar90"):
                        if metric_name in internal_episode_values and metric_name in ref_internal:
                            ci = _paired_bootstrap_delta_ci(
                                internal_episode_values[metric_name],
                                ref_internal[metric_name],
                                higher_is_better=False,
                            )
                            if ci is not None:
                                paired_ci_result[metric_name] = ci
            if do_ewmbench:
                ref_ewmbench_maps = _normalize_ewmbench_maps(
                    _collect_ewmbench_metric_maps(
                        (ref_result.get("ewmbench", {}) or {}).get("results_json")
                    )
                )
                ref_ewmbench_values = {
                    metric_name: list(value_map.values())
                    for metric_name, value_map in ref_ewmbench_maps.items()
                }
                for metric_name in _ordered_metric_block(current_ewmbench_maps):
                    if metric_name in ref_ewmbench_maps:
                        ci = _paired_bootstrap_delta_ci(
                            current_ewmbench_maps[metric_name],
                            ref_ewmbench_maps[metric_name],
                            higher_is_better=True,
                        )
                        if ci is not None:
                            paired_ci_result[f"ewmbench.{metric_name}"] = ci
                for metric_name in _ordered_metric_block(current_ewmbench_values):
                    key = f"ewmbench.{metric_name}"
                    if key in paired_ci_result or metric_name not in ref_ewmbench_values:
                        continue
                    ci = _paired_bootstrap_delta_ci_from_lists(
                        current_ewmbench_values[metric_name],
                        ref_ewmbench_values[metric_name],
                        higher_is_better=True,
                    )
                    if ci is not None:
                        paired_ci_result[key] = ci
        except Exception as exc:
            print(f"[eval WARNING] paired bootstrap skipped: {exc}")
    if paired_ci_result:
        result["paired_bootstrap_delta_0.95CI"] = _ordered_ci_block(paired_ci_result)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)
    print(f"[eval] wrote metrics to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one AL-round checkpoint against a validation manifest"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to AL-trained checkpoint (.pt/.ckpt)")
    parser.add_argument("--round-id", type=int, default=0, help="Round identifier for multi-round AL (default: 0)")
    parser.add_argument("--method", default=None, help="Deprecated/legacy method name; use --select_method for new AL runs")
    parser.add_argument("--score_method", "--score-method", dest="score_method", default=None, help="Score method name, e.g. c3, random, robometer_prog")
    parser.add_argument("--select_method", "--select-method", dest="select_method", default=None, help="Selection method name, e.g. c3_tail_risk, random")
    parser.add_argument("--weighting", default="none", help="Weighting mode (none, frame_patch, frame, patch_only, oversampling)")
    parser.add_argument("--val-manifest", required=True, help="JSON manifest of eval episodes")
    parser.add_argument("--pred-dir", required=True, help="Directory with pre-computed predictions")
    parser.add_argument("--gt-dir", default=None, help="Root of GT video/episode data (optional if manifest/config items contain GT paths)")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output metrics.json path. Default: {run_root}/eval/{score_method}_{select_method}_{weighting}.json "
            "when --score_method is set, otherwise {run_root}/eval/{method}_{weighting}.json."
        ),
    )
    parser.add_argument(
        "--ewmbench-output-dir",
        "--ewmbench_output_dir",
        dest="ewmbench_output_dir",
        default=None,
        help=(
            "Directory for EWMBench artifacts (evac_c3_results.json, traj_cache, etc.). "
            "Default: <output_stem>_ewmbench/ next to --output. "
            "Pass a distinct dir per checkpoint so parallel / multi-step evals don't "
            "overwrite each other's EWMBench results."
        ),
    )
    parser.add_argument("--metrics", default="latent_loss,pixel_mae,risk_reduction",
                        help="Comma-separated metric names to compute")
    parser.add_argument("--config", default=None, help="Optional eval config YAML")
    parser.add_argument(
        "--use_ewmbench_evaluate_py",
        "--use-ewmbench-evaluate-py",
        dest="use_ewmbench_evaluate_py",
        action="store_true",
        help="Use EWMBench's official evaluate.py and the configured EWMBench Python environment",
    )
    parser.add_argument(
        "--ewmbench-gpus",
        dest="ewmbench_gpus",
        default=None,
        help="GPU id(s) for EWMBench subprocess (sets CUDA_VISIBLE_DEVICES). "
             "Useful for pinning to a specific GPU to avoid OOM when running multiple evals.",
    )
    parser.add_argument(
        "--save_yolo_videos",
        "--save-yolo-videos",
        "--save_yolo_video",
        "--save-yolo-video",
        dest="save_yolo_videos",
        nargs="?",
        const="3",
        default=None,
        help=(
            "Save YOLO gripper detection overlay videos for trajectory debugging. "
            "Pass N to save the first N episodes, pass 'all' for every episode, "
            "or omit the value to use N=3. Disabled when the flag is not present."
        ),
    )
    parser.add_argument(
        "--paired-bootstrap-ref",
        "--paired_bootstrap_ref",
        dest="paired_bootstrap_ref",
        default=None,
        help=(
            "Reference eval JSON for paired bootstrap deltas. Defaults to "
            "{output_dir}/warmup_none.json when it exists."
        ),
    )
    parser.add_argument(
        "--no-paired-bootstrap",
        dest="no_paired_bootstrap",
        action="store_true",
        help="Disable paired bootstrap delta computation.",
    )
    args = parser.parse_args()
    effective_select_method = _safe_name(args.select_method or args.method, "")
    if not effective_select_method:
        parser.error("Provide --select_method (new pipeline) or --method (legacy pipeline)")
    effective_method = effective_select_method
    effective_score_method = _canonical_score_method(args.score_method)
    effective_weighting = _canonical_weighting_label(args.weighting)

    config = None
    if args.config:
        config = load_yaml(args.config)
    else:
        config = _auto_config_from_paths(args.val_manifest, args.pred_dir)
    config = _apply_ewmbench_cli_overrides(config or {}, args)

    # Default output: derive from val-manifest path
    output = args.output
    if output is None:
        vm = Path(args.val_manifest)
        # val_manifest = al_runs/{run}/manifests/al_val.json → run_root = al_runs/{run}
        run_root = vm.parent.parent  # manifests/.. → run dir
        eval_dir = run_root / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        if effective_score_method:
            base_stem = f"{effective_score_method}_{effective_select_method}_{effective_weighting}"
            stem = base_stem
            # Mirror the retrain run-dir suffix (e.g. a seed '_3407') so that
            # different-seed runs write separate eval files instead of
            # overwriting each other's <base_stem>.json. The run dir is the
            # path component right after 'retrain/' in --checkpoint/--pred-dir.
            for _p in (args.checkpoint, args.pred_dir):
                if not _p:
                    continue
                _parts = Path(_p).parts
                if "retrain" in _parts:
                    _i = _parts.index("retrain")
                    if _i + 1 < len(_parts) and _parts[_i + 1]:
                        _rundir = _parts[_i + 1]
                        if _rundir != base_stem and _rundir.startswith(base_stem + "_"):
                            stem = _rundir
                    break
            output = str(eval_dir / f"{stem}.json")
        else:
            output = str(eval_dir / f"{effective_method}_{effective_weighting}.json")

    paired_ref = None
    if not args.no_paired_bootstrap:
        paired_ref = args.paired_bootstrap_ref
        if paired_ref is None:
            candidate = Path(output).parent / "warmup_none.json"
            if candidate.exists() and candidate.resolve() != Path(output).resolve():
                paired_ref = str(candidate)

    evaluate_round(
        checkpoint_path=args.checkpoint,
        round_id=args.round_id,
        method=effective_method,
        weighting=effective_weighting,
        score_method=effective_score_method,
        select_method=effective_select_method,
        val_manifest=args.val_manifest,
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output=output,
        metrics=[m.strip() for m in args.metrics.split(",")],
        config=config,
        paired_bootstrap_ref=paired_ref,
        ewmbench_output_dir=args.ewmbench_output_dir,
    )


if __name__ == "__main__":
    main()
