from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


TRAIN_REQUIRED_FILES = [
    "head_color.mp4",
    "head_extrinsic_params_aligned.json",
    "head_intrinsic_params.json",
    "proprio_stats.h5",
]

EXTERNAL_WORLDMODEL_REQUIRED_FILES = [
    "head_color.mp4",
    "head_extrinsic_params_aligned.json",
    "head_intrinsic_params.json",
    "proprio_stats.h5",
    "actions_evac.npy",
    "camera.npz",
    "meta.json",
]


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def save_json(obj: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_train_folder_name(folder_name: str) -> tuple[str, str, str] | None:
    parts = folder_name.split("-")
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def episode_key(record: dict[str, Any]) -> str:
    return f"{record['task_id']}-{record['raw_episode_id']}"


def record_uid(record: dict[str, Any]) -> str:
    return record.get("episode_id") or record.get("folder") or (
        f"{record['task_id']}-{record['raw_episode_id']}-{record['segment_id']}"
    )


def scan_agibot_train_segments(
    train_root: str | os.PathLike[str],
    *,
    min_frames: int = 16,
    count_frames: bool = True,
    max_episodes: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan flat AgiBot train folders and return valid segment records.

    The active-learning split is clean at the episode key level:
    ``(task_id, raw_episode_id)``. Each segment directory is still kept as a
    candidate item because the raw train layout stores segments separately.
    """
    from prepare_data import get_n_frames_from_extrinsic, get_n_frames_from_proprio

    train_root = Path(train_root)
    if not train_root.is_dir():
        raise FileNotFoundError(f"train_root not found: {train_root}")

    records: list[dict[str, Any]] = []
    missing_counter: Counter[str] = Counter()
    skipped_bad_name = 0
    skipped_short = 0

    for folder in sorted(train_root.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_train_folder_name(folder.name)
        if parsed is None:
            skipped_bad_name += 1
            continue
        task_id, raw_ep_id, segment_id = parsed
        missing = [name for name in TRAIN_REQUIRED_FILES if not (folder / name).exists()]
        if missing:
            missing_counter.update(missing)
            continue

        if count_frames:
            n_frames = get_n_frames_from_extrinsic(str(folder), "head_extrinsic_params_aligned.json")
            if n_frames <= 0:
                n_frames = get_n_frames_from_proprio(str(folder), "proprio_stats.h5")
            if n_frames < min_frames:
                skipped_short += 1
                continue
        else:
            n_frames = -1

        uid = folder.name
        records.append(
            {
                "episode_id": uid,
                "folder": folder.name,
                "path": str(folder.resolve()),
                "task_id": task_id,
                "raw_episode_id": raw_ep_id,
                "segment_id": segment_id,
                "episode_key": f"{task_id}-{raw_ep_id}",
                "split": None,
                "files": {
                    "video": "head_color.mp4",
                    "extrinsic": "head_extrinsic_params_aligned.json",
                    "intrinsic": "head_intrinsic_params.json",
                    "proprio": "proprio_stats.h5",
                },
                "n_frames": n_frames,
                "source": "agibot_train_flat",
            }
        )
        if max_episodes is not None and len(records) >= max_episodes:
            break

    stats = {
        "train_root": str(train_root.resolve()),
        "valid_segments": len(records),
        "valid_episode_keys": len({r["episode_key"] for r in records}),
        "valid_tasks": len({r["task_id"] for r in records}),
        "skipped_bad_name": skipped_bad_name,
        "skipped_short": skipped_short,
        "missing_files_total": int(sum(missing_counter.values())),
        "missing_files": dict(missing_counter),
    }
    return records, stats


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_task = Counter(r["task_id"] for r in rows)
    return {
        "segments": len(rows),
        "episode_keys": len({r["episode_key"] for r in rows}),
        "tasks": len(by_task),
        "by_task": dict(sorted(by_task.items())),
    }


def _assign_episode_level(
    records: list[dict[str, Any]],
    *,
    c3_train_ratio: float,
    candidate_pool_ratio: float,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    keys = sorted({r["episode_key"] for r in records})
    rng.shuffle(keys)
    n = len(keys)
    n_seed = int(round(n * c3_train_ratio))
    n_seed = max(0, min(n, n_seed))
    remaining = n - n_seed
    n_pool = int(round(n * candidate_pool_ratio)) if candidate_pool_ratio <= 1.0 else int(candidate_pool_ratio)
    n_pool = max(0, min(remaining, n_pool))
    split_by_key = {key: "c3_train_split" for key in keys[:n_seed]}
    for key in keys[n_seed : n_seed + n_pool]:
        split_by_key[key] = "candidate_pool"
    for key in keys[n_seed + n_pool :]:
        split_by_key[key] = "unused"
    return split_by_key


def _assign_task_level(
    records: list[dict[str, Any]],
    *,
    c3_train_ratio: float,
    candidate_pool_ratio: float,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    tasks = sorted({r["task_id"] for r in records})
    rng.shuffle(tasks)
    n = len(tasks)
    n_seed = int(round(n * c3_train_ratio))
    n_seed = max(0, min(n, n_seed))
    remaining = n - n_seed
    n_pool = int(round(n * candidate_pool_ratio)) if candidate_pool_ratio <= 1.0 else int(candidate_pool_ratio)
    n_pool = max(0, min(remaining, n_pool))
    task_split = {task: "c3_train_split" for task in tasks[:n_seed]}
    for task in tasks[n_seed : n_seed + n_pool]:
        task_split[task] = "candidate_pool"
    for task in tasks[n_seed + n_pool :]:
        task_split[task] = "unused"
    return {r["episode_key"]: task_split[r["task_id"]] for r in records}


def build_clean_al_split(
    *,
    train_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    split_mode: str = "episode",
    c3_train_ratio: float = 0.5,
    candidate_pool_ratio: float = 0.5,
    random_seed: int = 42,
    min_frames: int = 16,
    max_episodes: int | None = None,
    count_frames: bool = True,
) -> dict[str, Any]:
    """Build AgiBot phase-1 split manifests.

    AgiBot here is an in-domain hard-case replay source because the EVAC base
    model has already seen AgiBot-style data. The only protocol-facing split
    files written by new runs are ``c3_train_split.json`` and
    ``candidate_pool.json``. For phase-1 retraining, ``c3_train_split`` can also
    serve as the replay buffer; it is not an EVAC seed-training set.
    """
    records, scan_stats = scan_agibot_train_segments(
        train_root,
        min_frames=min_frames,
        count_frames=count_frames,
        max_episodes=max_episodes,
    )
    if not records:
        raise ValueError(f"No valid AgiBot train records found under {train_root}")
    if split_mode not in {"episode", "task"}:
        raise ValueError(f"Unsupported split_mode={split_mode!r}; expected 'episode' or 'task'")

    if split_mode == "episode":
        split_by_key = _assign_episode_level(
            records,
            c3_train_ratio=c3_train_ratio,
            candidate_pool_ratio=candidate_pool_ratio,
            seed=random_seed,
        )
    else:
        split_by_key = _assign_task_level(
            records,
            c3_train_ratio=c3_train_ratio,
            candidate_pool_ratio=candidate_pool_ratio,
            seed=random_seed,
        )

    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_membership: dict[str, str] = {}
    for rec in records:
        split = split_by_key[rec["episode_key"]]
        item = dict(rec)
        item["split"] = split
        splits[split].append(item)
        prev = seen_membership.setdefault(rec["episode_key"], split)
        if prev != split:
            raise AssertionError(f"Episode leakage detected for {rec['episode_key']}: {prev} vs {split}")

    output_dir = ensure_dir(output_dir)
    for stale_name in [
        "seed_train.json",
        "al_pool.json",
        "unused_train.json",
        "replay_split.json",
        "al_val.json",
        "al_test.json",
    ]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    # Split unused into val / test
    unused = splits.get("unused", [])
    n_unused = len(unused)
    n_val = n_unused // 2
    splits.setdefault("al_val", []).extend(unused[:n_val])
    splits.setdefault("al_test", []).extend(unused[n_val:])

    manifest_paths: dict[str, str] = {}
    for name in ["c3_train_split", "candidate_pool", "al_val", "al_test"]:
        path = output_dir / f"{name}.json"
        items_for_split = splits.get(name, [])
        save_json({
            "split": name,
            "items": items_for_split,
            "stats": summarize_records(items_for_split),
        }, path)
        manifest_paths[name] = str(path)

    summary = {
        "split_mode": split_mode,
        "random_seed": random_seed,
        "ratios": {
            "c3_train_ratio": c3_train_ratio,
            "candidate_pool_ratio": candidate_pool_ratio,
        },
        "scan": scan_stats,
        "splits": {
            name: summarize_records(splits.get(name, []))
            for name in ["c3_train_split", "candidate_pool"]
        },
        "unused": summarize_records(splits.get("unused", [])),
        "manifest_paths": manifest_paths,
        "leakage_check": {
            "episode_keys_are_disjoint": True,
            "note": (
                "episode_key is task_id-raw_episode_id; all segment dirs for that key stay in one split. "
                "For AgiBot this supports in-domain hard-case replay, not clean external AL."
            ),
        },
    }
    save_json(summary, output_dir / "split_summary.json")
    return summary


def flatten_manifest_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "items" in payload and isinstance(payload["items"], list):
            return payload["items"]
        if "episodes" in payload and isinstance(payload["episodes"], list):
            return payload["episodes"]
    raise ValueError("Manifest must be a list or a dict containing an 'items' list")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def external_source_path_value(item: dict[str, Any]) -> str:
    return str(item.get("source_path") or item.get("path") or item.get("episode_dir") or "")


def external_source_path(item: dict[str, Any]) -> Path:
    value = external_source_path_value(item)
    return Path(value) if value else Path("")


def hydrate_external_worldmodel_item(item: dict[str, Any]) -> dict[str, Any]:
    """Fill stale external_worldmodel rows from their converted meta.json.

    Older RoboTwin manifests could mark already-converted episodes as
    ``_skipped=true`` with ``num_frames=0``.  When the converted episode exists,
    the on-disk meta.json is the source of truth and lets downstream steps
    repair those rows instead of silently propagating bad metadata.
    """
    row = dict(item)
    source_value = external_source_path_value(row)
    ep_dir = Path(source_value) if source_value else Path("")
    meta_value = row.get("meta_path") or (ep_dir / "meta.json" if source_value else None)
    meta_path = Path(str(meta_value)) if meta_value else None
    meta: dict[str, Any] = {}
    if meta_path is not None and meta_path.exists():
        try:
            loaded = load_json(meta_path)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = {}

    if meta:
        for key in (
            "dataset",
            "task_name",
            "robot",
            "variant",
            "clean_or_randomized",
            "camera",
            "action_dim",
            "evac_action_compatible",
            "action_representation",
            "is_dual_arm",
            "active_arm",
            "left_arm_moving",
            "right_arm_moving",
            "left_move_sum",
            "right_move_sum",
        ):
            if row.get(key) in (None, "") and meta.get(key) is not None:
                row[key] = meta[key]
        if not row.get("num_frames"):
            row["num_frames"] = meta.get("num_frames_used") or meta.get("num_frames_raw")
        if not row.get("episode_id") and meta.get("episode_id"):
            row["episode_id"] = meta["episode_id"]

        # Repair the specific stale-manifest case produced by the converter
        # skip path: files and meta exist, but the row says skipped/zero.
        meta_frames = _safe_int(meta.get("num_frames_used") or meta.get("num_frames_raw"))
        if row.get("_skipped") and meta_frames and meta.get("evac_action_compatible", True):
            row["_skipped"] = False
            row["_repaired_from_meta"] = True

    source_value = external_source_path_value(row)
    if source_value:
        row.setdefault("episode_dir", str(ep_dir))
        row.setdefault("source_path", str(ep_dir))
        row.setdefault("frames_dir", str(ep_dir / "frames"))
        row.setdefault("actions_path", str(ep_dir / "actions_evac.npy"))
        row.setdefault("actions_delta_path", str(ep_dir / "actions_delta_evac.npy"))
        row.setdefault("proprio_path", str(ep_dir / "actions_evac.npy"))
        row.setdefault("camera_path", str(ep_dir / "camera.npz"))
        row.setdefault("meta_path", str(ep_dir / "meta.json"))
    row.setdefault("format", "external_worldmodel")
    return row


def validate_external_worldmodel_item(
    item: dict[str, Any],
    *,
    min_frames: int = 1,
    require_paths: bool = False,
    require_training_files: bool = False,
) -> list[str]:
    reasons: list[str] = []
    row = item
    if not row.get("episode_id"):
        reasons.append("missing episode_id")
    if not external_source_path_value(row):
        reasons.append("missing episode_dir/source_path")
    if row.get("_skipped"):
        reasons.append("_skipped is true")

    num_frames = _safe_int(row.get("num_frames"))
    if num_frames is None:
        reasons.append("missing num_frames")
    elif num_frames < int(min_frames):
        reasons.append(f"num_frames {num_frames} < min_frames {int(min_frames)}")

    if row.get("evac_action_compatible") is False:
        reasons.append("evac_action_compatible is false")
    action_dim = _safe_int(row.get("action_dim"))
    if action_dim is not None and action_dim not in {14, 16}:
        reasons.append(f"unsupported action_dim {action_dim}")

    if require_paths:
        source_value = external_source_path_value(row)
        ep_dir = external_source_path(row)
        if source_value and not ep_dir.is_dir():
            reasons.append(f"episode_dir does not exist: {ep_dir}")
        required = [
            row.get("frames_dir") or (ep_dir / "frames"),
            row.get("actions_path") or (ep_dir / "actions_evac.npy"),
            row.get("camera_path") or (ep_dir / "camera.npz"),
            row.get("meta_path") or (ep_dir / "meta.json"),
        ]
        for value in required:
            path = Path(str(value))
            if not path.exists():
                reasons.append(f"missing path {path}")
        if require_training_files and str(ep_dir):
            for name in EXTERNAL_WORLDMODEL_REQUIRED_FILES:
                if not (ep_dir / name).exists():
                    reasons.append(f"missing training/scoring file {name}")
    return reasons


def filter_valid_external_worldmodel_items(
    items: list[dict[str, Any]],
    *,
    min_frames: int = 1,
    require_paths: bool | str = "auto",
    require_training_files: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from tqdm import tqdm
    hydrated = [hydrate_external_worldmodel_item(item) for item in tqdm(items, desc="hydrating manifest items", unit="item")]
    if require_paths == "auto":
        require_paths_bool = any(
            external_source_path_value(item) and external_source_path(item).exists()
            for item in hydrated
        )
    else:
        require_paths_bool = bool(require_paths)

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for item in tqdm(hydrated, desc="validating items", unit="item"):
        reasons = validate_external_worldmodel_item(
            item,
            min_frames=min_frames,
            require_paths=require_paths_bool,
            require_training_files=require_training_files,
        )
        if reasons:
            invalid.append(
                {
                    "episode_id": item.get("episode_id"),
                    "episode_dir": item.get("episode_dir") or item.get("source_path"),
                    "reasons": reasons,
                }
            )
        else:
            valid.append(item)

    reason_counts: Counter[str] = Counter()
    for item in invalid:
        reason_counts.update(str(reason) for reason in item.get("reasons", []))
    stats = {
        "input_items": len(items),
        "valid_items": len(valid),
        "invalid_items": len(invalid),
        "min_frames": int(min_frames),
        "require_paths": require_paths_bool,
        "require_training_files": bool(require_training_files),
        "repaired_from_meta": sum(1 for item in valid if item.get("_repaired_from_meta")),
        "invalid_reason_counts": dict(reason_counts.most_common()),
    }
    return valid, invalid, stats


def manifest_episode_keys(payload_or_items: Any) -> set[str]:
    """Return episode-level keys for leakage checks.

    Preferred key is ``episode_key`` (task_id-raw_episode_id). Fallbacks keep
    compatibility with scored/selected manifests that may only have episode_id.
    """
    items = flatten_manifest_items(payload_or_items)
    keys: set[str] = set()
    for item in items:
        key = item.get("episode_key")
        if not key and item.get("task_id") and item.get("raw_episode_id"):
            key = f"{item['task_id']}-{item['raw_episode_id']}"
        if not key:
            key = item.get("episode_id") or item.get("ep_id") or item.get("folder")
        if key:
            keys.add(str(key))
    return keys


def check_manifest_overlap(
    left_payload: Any,
    right_payload: Any,
    *,
    left_name: str,
    right_name: str,
    allow_overlap: bool = False,
) -> dict[str, Any]:
    left_keys = manifest_episode_keys(left_payload)
    right_keys = manifest_episode_keys(right_payload)
    overlap = sorted(left_keys & right_keys)
    result = {
        "left_name": left_name,
        "right_name": right_name,
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "overlap_count": len(overlap),
        "overlap_episode_keys": overlap[:100],
        "allow_overlap": allow_overlap,
    }
    if overlap and not allow_overlap:
        preview = ", ".join(overlap[:10])
        raise ValueError(
            f"Episode-level leakage detected between {left_name} and {right_name}: "
            f"{len(overlap)} overlaps. Examples: {preview}. "
            "Set c3_probe.allow_seen_pool=true only for an intentional ablation."
        )
    return result


def validate_confidence_map(
    conf_map: np.ndarray,
    *,
    confidence_format: str = "probability",
    out_of_range: str = "error",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate/convert confidence maps before risk computation.

    ``probability`` expects C3 confidence in [0, 1]. ``logits`` applies sigmoid
    and stores probabilities downstream so all AL selectors share one meaning:
    higher confidence means lower risk.
    """
    arr = np.asarray(conf_map, dtype=np.float32)
    if confidence_format not in {"probability", "logits"}:
        raise ValueError(f"confidence_format must be probability or logits, got {confidence_format!r}")
    info = {
        "input_format": confidence_format,
        "input_min": float(np.nanmin(arr)),
        "input_max": float(np.nanmax(arr)),
        "converted_to": "probability",
    }
    if confidence_format == "logits":
        arr = 1.0 / (1.0 + np.exp(-arr))
        info["output_min"] = float(np.nanmin(arr))
        info["output_max"] = float(np.nanmax(arr))
        return arr.astype(np.float32), info

    bad = bool(np.nanmin(arr) < -1e-4 or np.nanmax(arr) > 1.0 + 1e-4)
    if bad:
        msg = (
            f"confidence_format=probability but conf_map range is "
            f"[{info['input_min']:.6f}, {info['input_max']:.6f}]"
        )
        if out_of_range == "error":
            raise ValueError(msg)
        if out_of_range == "warning":
            print(f"[confidence_format WARNING] {msg}; clipping to [0,1].")
        else:
            raise ValueError("out_of_range must be 'error' or 'warning'")
    arr = np.clip(arr, 0.0, 1.0)
    info["output_min"] = float(np.nanmin(arr))
    info["output_max"] = float(np.nanmax(arr))
    return arr.astype(np.float32), info


def compute_risk_stats(conf_map: np.ndarray, tail_percents: tuple[float, ...] = (5.0, 10.0)) -> dict[str, float]:
    """Compute selection-only risk statistics from C3 confidence.

    No GT or oracle error is used here. ``conf_map`` is confidence in [0, 1],
    therefore risk is defined as ``1 - confidence``.
    """
    if conf_map.ndim != 3:
        raise ValueError(f"conf_map must have shape [T,H,W], got {conf_map.shape}")
    conf = np.clip(conf_map.astype(np.float32), 0.0, 1.0)
    risk = 1.0 - conf
    flat = risk.reshape(-1)
    stats: dict[str, float] = {
        "mean_conf": float(conf.mean()),
        "mean_risk": float(risk.mean()),
        "max_risk": float(risk.max()),
        "std_risk": float(risk.std()),
        "risk_area": float((risk > 0.5).mean()),
    }
    for pct in tail_percents:
        k = max(1, int(math.ceil(flat.size * (pct / 100.0))))
        top = np.partition(flat, flat.size - k)[-k:]
        stats[f"tail_risk_top{int(pct)}"] = float(top.mean())

    frame_tail = []
    for frame in risk:
        vals = frame.reshape(-1)
        k = max(1, int(math.ceil(vals.size * 0.10)))
        frame_tail.append(float(np.partition(vals, vals.size - k)[-k:].mean()))
    frame_tail_arr = np.asarray(frame_tail, dtype=np.float32)
    m = max(1, int(math.ceil(len(frame_tail_arr) * 0.20)))
    stats["persistent_risk"] = float(np.partition(frame_tail_arr, len(frame_tail_arr) - m)[-m:].mean())
    stats["high_risk_frame_fraction"] = float((frame_tail_arr > 0.5).mean())
    return stats


def write_csv(rows: list[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def score_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
