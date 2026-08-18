#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import ensure_dir, flatten_manifest_items, load_json, load_yaml, save_json


DEFAULT_SCORE_METHOD = "c3"
EXTERNAL_SCORE_METHODS = {
    "gvl",
    "roboreward",
    "robometer_prog",
    "robometer_pref",
    "prm_judge",
    "lrm",
    "lrms",
    "lrm_progress",
    "lrm_completion",
    "lrm_contrastive",
}
SCORE_METHOD_ALIASES = {
    "confidence": "c3",
    "c3_confidence": "c3",
    "random": "random",
    "gvl": "gvl",
    "robo-reward": "roboreward",
    "robo_reward": "roboreward",
    "roboreward": "roboreward",
    "robometer-prog": "robometer_prog",
    "robometer_prog": "robometer_prog",
    "robometer-progress": "robometer_prog",
    "robometer-pref": "robometer_pref",
    "robometer_pref": "robometer_pref",
    "robometer-preference": "robometer_pref",
    "prm-as-judge": "prm_judge",
    "prm-judge": "prm_judge",
    "prm_judge": "prm_judge",
    "lrm": "lrm",
    "lrms": "lrms",
    "lrm-progress": "lrm_progress",
    "lrm_progress": "lrm_progress",
    "lrm-completion": "lrm_completion",
    "lrm_completion": "lrm_completion",
    "lrm-contrastive": "lrm_contrastive",
    "lrm_contrastive": "lrm_contrastive",
}


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _run_root(cfg: dict[str, Any]) -> Path:
    root_dir = _get(cfg, "project.root_dir", _get(cfg, "run.root", "al_runs"))
    run_name = _get(cfg, "project.run_name", _get(cfg, "run.name", "debug_al"))
    return Path(root_dir) / str(run_name)


def _safe_name(value: str | None, default: str) -> str:
    text = str(value or default).strip().lower()
    text = SCORE_METHOD_ALIASES.get(text, text)
    return text.replace("/", "_").replace(" ", "_")


def _selection_root(cfg: dict[str, Any], run_root: Path) -> Path:
    prescreen_cfg = dict(_get(cfg, "task_prescreen", {}) or {})
    return Path(prescreen_cfg.get("output_dir") or _get(cfg, "selection.output_root") or run_root / "selection")


def _task_prescreen_pred_dir(pool_root: Path, pred_dir_override: str | None) -> Path:
    if pred_dir_override:
        return Path(pred_dir_override)
    legacy_dir = pool_root / "task_prescreen"
    namespaced_dir = pool_root / "pred" / "task_prescreen"
    if legacy_dir.exists():
        return legacy_dir
    return namespaced_dir


def _derive_paths(
    cfg: dict[str, Any],
    *,
    score_method: str,
    select_method: str | None,
    manifest_override: str | None,
    output_dir_override: str | None,
    pred_dir_override: str | None,
) -> tuple[Path, Path, Path, str]:
    run_root = _run_root(cfg)
    prescreen_cfg = dict(_get(cfg, "task_prescreen", {}) or {})
    selection_root = _selection_root(cfg, run_root)
    pool_root = run_root / "pool_scores"
    score_method = _safe_name(score_method, DEFAULT_SCORE_METHOD)
    if select_method:
        select_method = _safe_name(select_method, "tail_risk")
        combo = f"{score_method}_{select_method}"
        manifest = Path(manifest_override or selection_root / combo / "selected_candidates.json")
        score_dir = Path(output_dir_override or pool_root / f"{combo}_selected_scores")
        pred_dir = Path(pred_dir_override or pool_root / "pred" / "selected_pool")
        stage = "selected"
    else:
        manifest = Path(
            manifest_override
            or prescreen_cfg.get("prescreen_manifest")
            or selection_root / "task_prescreen_pool.json"
        )
        score_dir = Path(output_dir_override or pool_root / f"{score_method}_task_prescreen_scores")
        pred_dir = _task_prescreen_pred_dir(pool_root, pred_dir_override)
        stage = "task_prescreen"
    return manifest, score_dir, pred_dir, stage


def _episode_id(item: dict[str, Any], fallback: int) -> str:
    return str(item.get("episode_id") or item.get("ep_id") or item.get("folder") or f"item_{fallback:06d}")


def _clip01(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out))


def _task_text(item: dict[str, Any]) -> str:
    for key in ("task", "task_description", "language_instruction", "instruction", "prompt"):
        value = item.get(key)
        if value:
            return str(value)
    task_name = str(item.get("task_name") or item.get("task_id") or "robot task")
    return task_name.replace("_", " ")


def _video_path(item: dict[str, Any]) -> str:
    for key in (
        "video_path",
        "head_video_path",
        "rgb_video_path",
        "mp4_path",
        "head_color_path",
    ):
        value = item.get(key)
        if value:
            return str(value)
    episode_dir = item.get("episode_dir") or item.get("source_path") or item.get("path")
    if episode_dir:
        return str(Path(str(episode_dir)) / "head_color.mp4")
    frames_dir = item.get("frames_dir")
    if frames_dir:
        return str(frames_dir)
    return ""


def _write_baseline_manifest(
    *,
    items: list[dict[str, Any]],
    manifest_path: Path,
    pred_dir: Path | None = None,
    video_kind: str = "gt",
) -> dict[str, dict[str, Any]]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    by_episode: dict[str, dict[str, Any]] = {}
    with manifest_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            ep_id = _episode_id(item, idx)
            gt_video_path = _video_path(item)
            pred_frames_dir = _pred_frames_dir(pred_dir, ep_id) if video_kind == "pred" and pred_dir is not None else None
            if pred_frames_dir is not None and not pred_frames_dir.is_absolute():
                pred_frames_dir = pred_frames_dir.resolve()
            score_video_path = str(pred_frames_dir) if video_kind == "pred" and pred_frames_dir is not None else gt_video_path
            if video_kind == "pred" and pred_frames_dir is not None and not _has_pred_frames(pred_frames_dir.parent):
                raise FileNotFoundError(
                    f"Missing EVAC prediction frames for external baseline scoring: {pred_frames_dir}"
                )
            row = dict(item)
            if video_kind != "pred":
                row.pop("pred_frames_dir", None)
            row.update(
                {
                    "episode_id": ep_id,
                    "video_path": score_video_path,
                    "score_video_path": score_video_path,
                    "score_video_kind": video_kind,
                    "gt_video_path": gt_video_path,
                    "source_video_path": gt_video_path,
                    "pred_frames_dir": str(pred_frames_dir) if pred_frames_dir is not None else None,
                    "task": _task_text(item),
                }
            )
            by_episode[ep_id] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return by_episode


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _has_pred_frames(ep_pred_dir: Path) -> bool:
    frames_dir = ep_pred_dir / "pred_frames"
    if not frames_dir.is_dir():
        return False
    return any(p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir())


def _pred_frames_dir(pred_dir: Path, ep_id: str) -> Path:
    return pred_dir / ep_id / "pred_frames"


def _write_subset_manifest(source_manifest: Path, items: list[dict[str, Any]], out_path: Path) -> Path:
    manifest_data = load_json(str(source_manifest))
    if isinstance(manifest_data, dict):
        reduced = dict(manifest_data)
        reduced["items"] = items
    else:
        reduced = items
    save_json(reduced, out_path)
    return out_path


def _external_baseline_method(score_method: str) -> str:
    return {
        "lrm": "lrm",
        "lrms": "lrms",
    }.get(score_method, score_method)


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


GT_SCORE_FILE = "baseline_scores_gt.jsonl"
PRED_SCORE_FILE = "baseline_scores_pred.jsonl"
FINAL_SCORE_FILE = "baseline_scores.jsonl"


def _load_score_rows(score_dir: Path, name: str) -> tuple[list[dict[str, Any]], Path | None]:
    path = score_dir / name
    if not path.exists():
        return [], None
    return _read_jsonl(path), path


def _load_legacy_gt_rows_from_scored_pool(score_dir: Path) -> tuple[list[dict[str, Any]], Path | None]:
    """Recover old external GT-only rows from scored_pool.json.

    Older multi-shard external scoring runs wrote scored_pool.json but did not
    materialize baseline_scores.jsonl.  Those rows predate GT-vs-pred delta
    scoring, so they can be treated as cached GT scores for --skip_gt_scores.
    """
    path = score_dir / "scored_pool.json"
    if not path.exists():
        return [], None
    try:
        payload = load_json(str(path))
    except Exception:
        return [], None
    items = payload.get("items", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("score_ready", True) is not True:
            continue
        if item.get("score_delta_mode"):
            return [], None
        row = dict(item)
        row["score_video_kind"] = "gt"
        row["score_ready"] = True
        row["episode_score"] = row.get(
            "baseline_episode_score",
            row.get("episode_score", row.get("acquisition_score", row.get("tail_risk_top5", 0.0))),
        )
        row["acquisition_score"] = row.get(
            "baseline_acquisition_score",
            row.get("acquisition_score", row.get("tail_risk_top5", row.get("episode_score", 0.0))),
        )
        row["frame_scores"] = row.get("frame_scores", row.get("baseline_frame_scores", []))
        row["method"] = row.get("baseline_method", row.get("method"))
        row["extra"] = row.get("extra", {})
        row.pop("pred_frames_dir", None)
        rows.append(row)
    return rows, path


def _merge_rows_by_episode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        ep_id = str(row.get("episode_id") or "")
        if not ep_id:
            continue
        if ep_id not in merged or row.get("score_ready", True):
            merged[ep_id] = dict(row)
    return list(merged.values())


def _rows_by_episode(rows: list[dict[str, Any]], needed_eps: set[str] | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ep_id = str(row.get("episode_id") or "")
        if not ep_id:
            continue
        if needed_eps is not None and ep_id not in needed_eps:
            continue
        if row.get("score_ready", True) is not True:
            continue
        out.setdefault(ep_id, dict(row))
    return out



def _prediction_cache_dirs(pool_root: Path) -> list[Path]:
    dirs = [p for p in sorted((pool_root / "pred").glob("*")) if p.is_dir()]
    legacy_task_prescreen = pool_root / "task_prescreen"
    if legacy_task_prescreen.is_dir() and all(p.resolve() != legacy_task_prescreen.resolve() for p in dirs):
        dirs.append(legacy_task_prescreen)
    return dirs


def _missing_prediction_items(
    items: list[dict[str, Any]],
    *,
    pred_dir: Path,
    force: bool,
) -> list[dict[str, Any]]:
    if force:
        return list(items)
    return [
        item for idx, item in enumerate(items)
        if not _has_pred_frames(pred_dir / _episode_id(item, idx))
    ]


def _successful_score_rows(
    rows: list[dict[str, Any]],
    needed_eps: set[str],
    *,
    require_pred_video: bool = False,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ep_id = str(row.get("episode_id") or "")
        if not ep_id or ep_id not in needed_eps:
            continue
        if row.get("score_ready", True) is not True:
            continue
        if require_pred_video and not row.get("pred_frames_dir"):
            continue
        out.setdefault(ep_id, dict(row))
    return out


def _read_scored_pool_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = load_json(str(path))
    except Exception:
        return []
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return [dict(row) for row in items if isinstance(row, dict)]


def _find_raw_score_cache_rows(
    *,
    score_method: str,
    score_dir: Path,
    pool_root: Path,
    needed_eps: set[str],
    names: str,
    video_kind: str,
) -> list[dict[str, Any]]:
    by_ep: dict[str, dict[str, Any]] = {}
    for score_candidate in sorted(pool_root.glob(f"{score_method}_*_scores")):
        if not score_candidate.is_dir() or score_candidate.resolve() == score_dir.resolve():
            continue
        rows, path = _load_score_rows(score_candidate, names)
        if path is None:
            continue
        for row in rows:
            ep_id = str(row.get("episode_id") or "")
            if not ep_id or ep_id not in needed_eps:
                continue
            if row.get("score_ready", True) is not True:
                continue
            if video_kind == "pred" and not row.get("pred_frames_dir"):
                continue
            cached = dict(row)
            cached.setdefault("_cached_from_score", str(path))
            by_ep.setdefault(ep_id, cached)
    return list(by_ep.values())


def _run_c3_module(
    args: argparse.Namespace,
    *,
    manifest: Path,
    score_dir: Path,
    pred_dir: Path,
    prediction_only: bool,
    max_episodes: int | None,
    workers_per_gpu: int,
) -> None:
    gpu_ids: list[str] = []
    if getattr(args, "gpus", None):
        gpu_ids = [s.strip() for s in str(args.gpus).split(",") if s.strip()]
    num_shards = max(1, int(getattr(args, "num_shards", 1)))
    if gpu_ids and num_shards > len(gpu_ids):
        num_shards = len(gpu_ids)
    workers_per_gpu = max(1, int(workers_per_gpu))
    total_workers = num_shards * workers_per_gpu

    score_dir = ensure_dir(score_dir)
    if total_workers > 1:
        for stale in sorted(score_dir.glob("scored_pool_*.json")):
            stale.unlink()
        if getattr(args, "overwrite", False) and (score_dir / "scored_pool.json").exists():
            (score_dir / "scored_pool.json").unlink()

    if total_workers <= 1:
        # Single worker: call run_from_config inline (original behaviour)
        from al_pipeline.score_pool_with_c3 import run_from_config

        device_override: str | None = None
        if gpu_ids:
            device_override = f"cuda:{gpu_ids[0]}"
        elif getattr(args, "device", None) is not None:
            device_override = args.device
        print(
            f"[score_pool] C3 module call shard=0/1 worker=0/1 "
            f"prediction_only={prediction_only}"
        )
        run_from_config(
            args.config,
            max_episodes=max_episodes,
            manifest_override=str(manifest),
            output_dir_override=str(pred_dir),
            score_dir_override=str(score_dir),
            evac_checkpoint_override=args.evac_checkpoint,
            device_override=device_override,
            shard_id=0,
            num_shards=1,
            worker_id=0,
            workers_per_gpu=1,
            overwrite=bool(args.overwrite),
            n_frames_to_generate_override=args.n_frames_to_generate,
            save_traj_videos=bool(args.save_traj_videos),
            prediction_only=prediction_only,
        )
        return

    # Multi-worker: spawn subprocesses (same pattern as _run_baseline_scorer
    # and score_pool_with_c3.py:main).
    print(
        f"[score_pool] spawning {workers_per_gpu} workers x {num_shards} shards "
        f"= {total_workers} total (prediction_only={prediction_only})"
    )
    c3_script = str(_REPO / "al_pipeline" / "score_pool_with_c3.py")
    procs: list[subprocess.Popen] = []
    last_cmd: list[str] = []
    for sid in range(num_shards):
        for wid in range(workers_per_gpu):
            device_str: str | None = None
            if gpu_ids:
                device_str = f"cuda:{gpu_ids[sid % len(gpu_ids)]}"
            elif getattr(args, "device", None) is not None:
                device_str = args.device

            cmd = [
                sys.executable, c3_script,
                "--config", args.config,
                "--manifest", str(manifest),
                "--output-dir", str(pred_dir),
                "--score-dir", str(score_dir),
                "--shard_id", str(sid),
                "--num_shards", str(num_shards),
                "--workers_per_gpu", str(workers_per_gpu),
                "--_worker_id", str(wid),
            ]
            if args.evac_checkpoint:
                cmd += ["--evac_checkpoint", args.evac_checkpoint]
            if max_episodes is not None:
                cmd += ["--max_episodes", str(max_episodes)]
            if args.n_frames_to_generate is not None:
                cmd += ["--n_frames_to_generate", str(args.n_frames_to_generate)]
            if device_str:
                cmd += ["--device", device_str]
            if bool(args.overwrite):
                cmd += ["--overwrite"]
            if bool(args.save_traj_videos):
                cmd += ["--save_traj_videos"]
            if prediction_only:
                cmd += ["--prediction-only"]

            last_cmd = cmd
            p = subprocess.Popen(cmd)
            procs.append(p)
            print(
                f"[score_pool]  started shard {sid}/{num_shards} worker {wid}/{workers_per_gpu} "
                f"(pid={p.pid}, gpu={gpu_ids[sid % len(gpu_ids)] if gpu_ids else 'auto'})"
            )
    for p in procs:
        p.wait()
    for p in procs:
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, last_cmd)
    print(f"[score_pool] all {total_workers} C3 workers finished")


def _run_prediction_backend(
    args: argparse.Namespace,
    *,
    manifest: Path,
    pred_dir: Path,
    score_dir: Path,
    max_episodes: int | None = None,
) -> None:
    print(f"[score_pool] EVAC prediction/cache dir: {pred_dir}")
    print(f"[score_pool] prediction-only manifest: {manifest}")
    _run_c3_module(
        args,
        manifest=manifest,
        pred_dir=pred_dir,
        score_dir=score_dir,
        prediction_only=True,
        max_episodes=max_episodes,
        workers_per_gpu=max(1, int(getattr(args, "workers_per_gpu", 1))),
    )


def _ensure_external_predictions(
    args: argparse.Namespace,
    *,
    cfg: dict[str, Any],
    manifest: Path,
    items: list[dict[str, Any]],
    pred_dir: Path,
    score_dir: Path,
) -> tuple[int, int]:
    """Ensure EVAC predictions exist for all *items* under *pred_dir*.

    With unified pred directories (task_prescreen / selected_pool), pred
    frames are already shared across score methods — no copying needed.
    Missing episodes are generated on demand; ``--overwrite`` forces
    regeneration of all episodes.
    """
    force = bool(getattr(args, "overwrite", False))
    missing = _missing_prediction_items(items, pred_dir=pred_dir, force=force)
    cached_count = 0 if force else len(items) - len(missing)
    if not missing:
        print(f"[score_pool] EVAC predictions ready for {len(items)} episode(s)")
        return cached_count, 0

    ensure_dir(score_dir)
    reduced_manifest = _write_subset_manifest(
        manifest,
        missing,
        score_dir / "manifest_missing_predictions.json",
    )
    aux_score_dir = ensure_dir(score_dir / "_evac_prediction_only")
    print(
        f"[score_pool] generating EVAC predictions for {len(missing)}/{len(items)} "
        "episode(s) before external baseline scoring"
    )
    _run_prediction_backend(
        args,
        manifest=reduced_manifest,
        pred_dir=pred_dir,
        score_dir=aux_score_dir,
        max_episodes=len(missing),
    )

    still_missing = _missing_prediction_items(items, pred_dir=pred_dir, force=False)
    if still_missing:
        missing_ids = [_episode_id(item, idx) for idx, item in enumerate(still_missing[:10])]
        raise RuntimeError(
            "EVAC prediction generation finished but some pred_frames are still missing: "
            + ", ".join(missing_ids)
        )
    return cached_count, len(missing)


def _recover_orphaned_shards(score_dir: Path) -> None:
    """Merge orphaned multi-worker shard files left by an interrupted run.

    A previous run killed mid-way leaves ``{name}_s{sid}_w{wid}.jsonl`` files.
    Score shards (gt/pred/final) have their successful rows merged into the
    canonical ``{name}.jsonl``; manifest shards are pure inputs and are just
    deleted.  Run this before loading existing rows so the missing-episode
    computation is correct.
    """
    import re

    shard_re = re.compile(r"^(.+?)_s\d+_w\d+\.jsonl$")
    by_base: dict[str, list[Path]] = {}
    for f in sorted(score_dir.glob("*_s*_w*.jsonl")):
        m = shard_re.match(f.name)
        if not m:
            continue
        base_name = m.group(1) + ".jsonl"
        by_base.setdefault(base_name, []).append(f)

    for base_name, shards in by_base.items():
        base_path = score_dir / base_name
        if base_name.startswith("baseline_manifest"):
            for s in shards:
                s.unlink(missing_ok=True)
            continue
        existing: dict[str, dict[str, Any]] = {}
        if base_path.exists():
            for row in _read_jsonl(base_path):
                ep = str(row.get("episode_id") or "")
                if ep:
                    existing[ep] = row
        recovered = 0
        for s in shards:
            for row in _read_jsonl(s):
                if row.get("score_ready", True) is not True:
                    continue
                ep = str(row.get("episode_id") or "")
                if ep and ep not in existing:
                    existing[ep] = row
                    recovered += 1
        if existing:
            _write_jsonl_rows(base_path, list(existing.values()))
        for s in shards:
            s.unlink(missing_ok=True)
        if shards:
            print(
                f"[score_pool] recovered {recovered} row(s) from {len(shards)} "
                f"orphaned shard(s) into {base_name}"
            )


def _run_baseline_scorer(
    args: argparse.Namespace,
    *,
    items: list[dict[str, Any]],
    score_dir: Path,
    manifest_path: Path,
    output_path: Path,
    pred_dir: Path | None,
    video_kind: str,
    baseline_method: str,
    baseline_config: Path,
) -> list[dict[str, Any]]:
    if not items:
        return []

    cwd = str(_REPO / "baselines" / "evac_al_baselines")
    gpu_ids: list[str] = []
    if getattr(args, "gpus", None):
        gpu_ids = [s.strip() for s in str(args.gpus).split(",") if s.strip()]
    num_shards = max(1, int(getattr(args, "num_shards", 1)))
    workers_per_gpu = max(1, int(getattr(args, "workers_per_gpu", 1)))
    if gpu_ids and num_shards > len(gpu_ids):
        num_shards = len(gpu_ids)
    total_procs = num_shards * workers_per_gpu

    if total_procs <= 1:
        _write_baseline_manifest(
            items=items,
            manifest_path=manifest_path,
            pred_dir=pred_dir,
            video_kind=video_kind,
        )
        tmp_output = output_path.with_name(f"{output_path.stem}_new.jsonl")
        if tmp_output.exists():
            tmp_output.unlink()
        cmd = [
            sys.executable,
            str(_REPO / "baselines" / "evac_al_baselines" / "run_score.py"),
            "--manifest", str(manifest_path),
            "--method", baseline_method,
            "--config", str(baseline_config),
            "--output", str(tmp_output),
        ]
        print(f"[score_pool] {video_kind} baseline manifest: {manifest_path}")
        print(f"[score_pool] {video_kind} baseline output  : {tmp_output}")
        subprocess.run(cmd, cwd=cwd, check=True)
        if not tmp_output.exists():
            raise FileNotFoundError(f"External baseline did not produce {tmp_output}")
        rows = _read_jsonl(tmp_output)
        tmp_output.unlink()
        return rows

    print(f"[score_pool] spawning {workers_per_gpu} workers x {num_shards} shards = {total_procs} total ({video_kind})")
    procs: list[subprocess.Popen] = []
    outputs: list[Path] = []
    last_cmd: list[str] = []
    for sid in range(num_shards):
        for wid in range(workers_per_gpu):
            bucket = sid * workers_per_gpu + wid
            shard_items = items[bucket::total_procs]
            if not shard_items:
                continue
            shard_manifest = (score_dir / f"{manifest_path.stem}_s{sid}_w{wid}.jsonl").resolve()
            shard_output = (score_dir / f"{output_path.stem}_s{sid}_w{wid}.jsonl").resolve()
            _write_baseline_manifest(
                items=shard_items,
                manifest_path=shard_manifest,
                pred_dir=pred_dir,
                video_kind=video_kind,
            )
            outputs.append(shard_output)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[sid % len(gpu_ids)]
            cmd = [
                sys.executable,
                str(_REPO / "baselines" / "evac_al_baselines" / "run_score.py"),
                "--manifest", str(shard_manifest),
                "--method", baseline_method,
                "--config", str(baseline_config),
                "--output", str(shard_output),
            ]
            last_cmd = cmd
            p = subprocess.Popen(cmd, cwd=cwd, env=env)
            procs.append(p)
            print(
                f"[score_pool]  started {video_kind} shard {sid} worker {wid} "
                f"(pid={p.pid}, gpu={gpu_ids[sid % len(gpu_ids)] if gpu_ids else 'auto'})"
            )
    for p in procs:
        p.wait()
    for p in procs:
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, last_cmd)

    rows: list[dict[str, Any]] = []
    for out in outputs:
        if out.exists():
            rows.extend(_read_jsonl(out))
            out.unlink()
    # Clean up shard manifests now that all workers succeeded.
    for sid in range(num_shards):
        for wid in range(workers_per_gpu):
            shard_manifest = score_dir / f"{manifest_path.stem}_s{sid}_w{wid}.jsonl"
            shard_manifest.unlink(missing_ok=True)
    return rows


def _score_missing_external_rows(
    args: argparse.Namespace,
    *,
    all_items: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    score_dir: Path,
    manifest_path: Path,
    output_path: Path,
    pred_dir: Path | None,
    video_kind: str,
    baseline_method: str,
    baseline_config: Path,
    skip_scoring: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    needed_eps = {_episode_id(item, idx) for idx, item in enumerate(all_items)}
    existing_by_ep = _rows_by_episode(existing_rows, needed_eps)
    missing_items = [
        item for idx, item in enumerate(all_items)
        if _episode_id(item, idx) not in existing_by_ep
    ]
    if not missing_items:
        print(f"[score_pool] {video_kind}: all {len(all_items)} episode(s) already scored")
        return list(existing_by_ep.values()), len(existing_by_ep), 0

    if skip_scoring:
        print(
            f"[score_pool] {video_kind}: --skip_gt_scores active; "
            f"scoring only {len(missing_items)} episode(s) with no cached score"
        )
    else:
        print(
            f"[score_pool] {video_kind}: scoring {len(missing_items)}/{len(all_items)} missing episode(s); "
            "existing rows are kept"
        )
    new_rows = _run_baseline_scorer(
        args,
        items=missing_items,
        score_dir=score_dir,
        manifest_path=manifest_path,
        output_path=output_path,
        pred_dir=pred_dir,
        video_kind=video_kind,
        baseline_method=baseline_method,
        baseline_config=baseline_config,
    )
    merged = _merge_rows_by_episode(list(existing_by_ep.values()) + new_rows)
    _write_jsonl_rows(output_path, merged)
    return merged, len(existing_by_ep), len(new_rows)


def _as_float_list(value: Any) -> list[float]:
    if isinstance(value, list):
        vals = value
    elif isinstance(value, tuple):
        vals = list(value)
    elif value is None:
        vals = []
    else:
        vals = [value]
    out = []
    for item in vals:
        out.append(_clip01(item))
    return out


def _resample_scores(scores: list[float], target_len: int) -> list[float]:
    if target_len <= 0:
        return []
    if not scores:
        return [0.0] * target_len
    if len(scores) == target_len:
        return [_clip01(x) for x in scores]
    if len(scores) == 1:
        return [_clip01(scores[0])] * target_len
    import numpy as np

    src_x = np.linspace(0.0, 1.0, num=len(scores), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    arr = np.interp(dst_x, src_x, np.asarray(scores, dtype=np.float32))
    return [_clip01(x) for x in arr.tolist()]


def _risk_stats_from_frame_risk(frame_risk: list[float], fallback: float) -> dict[str, float]:
    import numpy as np

    vals = np.asarray(frame_risk if frame_risk else [fallback], dtype=np.float32)
    vals = np.clip(vals, 0.0, 1.0)
    stats = {
        "mean_risk": float(vals.mean()),
        "max_risk": float(vals.max()),
        "std_risk": float(vals.std()),
        "risk_area": float((vals > 0.5).mean()),
    }
    for pct in (5.0, 10.0):
        k = max(1, int(math.ceil(vals.size * (pct / 100.0))))
        top = np.partition(vals, vals.size - k)[-k:]
        stats[f"tail_risk_top{int(pct)}"] = float(top.mean())
    m = max(1, int(math.ceil(vals.size * 0.20)))
    stats["persistent_risk"] = float(np.partition(vals, vals.size - m)[-m:].mean())
    stats["high_risk_frame_fraction"] = float((vals > 0.5).mean())
    return stats


def _build_delta_rows(
    *,
    all_items: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    score_method: str,
    baseline_method: str,
    select_method: str | None,
    pred_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_by_ep = _rows_by_episode(gt_rows)
    pred_by_ep = _rows_by_episode(pred_rows)
    final_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, item in enumerate(all_items):
        ep_id = _episode_id(item, idx)
        gt = gt_by_ep.get(ep_id)
        pred = pred_by_ep.get(ep_id)
        if gt is None or pred is None:
            failures.append(
                {
                    "episode_id": ep_id,
                    "method": baseline_method,
                    "error": "missing gt score" if gt is None else "missing pred score",
                }
            )
            continue

        gt_episode = _clip01(gt.get("episode_score", gt.get("baseline_episode_score", 0.0)))
        pred_episode = _clip01(pred.get("episode_score", pred.get("baseline_episode_score", 0.0)))
        episode_delta = pred_episode - gt_episode  # signed: positive = pred better than gt
        episode_score = (episode_delta + 1.0) / 2.0  # map [-1,1] → [0,1]
        episode_risk = 1.0 - episode_score  # [0,1], high = bad

        gt_frames = _as_float_list(gt.get("frame_scores", gt.get("baseline_frame_scores", [])))
        pred_frames = _as_float_list(pred.get("frame_scores", pred.get("baseline_frame_scores", [])))
        target_len = max(len(gt_frames), len(pred_frames), 1)
        gt_frames = _resample_scores(gt_frames or [gt_episode], target_len)
        pred_frames = _resample_scores(pred_frames or [pred_episode], target_len)
        frame_delta = [p - g for g, p in zip(gt_frames, pred_frames)]  # signed
        frame_conf = [(d + 1.0) / 2.0 for d in frame_delta]  # map [-1,1] → [0,1]
        frame_risk = [1.0 - c for c in frame_conf]  # [0,1], high = bad
        risk_stats = _risk_stats_from_frame_risk(frame_risk, episode_risk)
        acquisition = float(risk_stats.get("tail_risk_top5", episode_risk))
        pred_frames_dir = _pred_frames_dir(pred_dir, ep_id)
        gt_video_path = _video_path(item)

        row = dict(pred)
        row.update(
            {
                "episode_id": ep_id,
                "score_method": score_method,
                "baseline_method": baseline_method,
                "method": baseline_method,
                "select_method": select_method,
                "score_ready": True,
                "score_delta_mode": "signed_delta_higher_is_better",
                "score_video_kind": "delta",
                "episode_score": episode_score,
                "pred_episode_score": pred_episode,
                "gt_episode_score": gt_episode,
                "episode_score_gap": episode_delta,
                "acquisition_score": acquisition,
                "baseline_acquisition_score": acquisition,
                "frame_scores": frame_conf,
                "frame_conf_scores": frame_conf,
                "baseline_frame_scores": frame_conf,
                "frame_risk_scores": frame_risk,
                "gt_frame_scores": gt_frames,
                "pred_frame_scores": pred_frames,
                "video_path": str(pred_frames_dir),
                "pred_frames_dir": str(pred_frames_dir),
                "score_video_path": str(pred_frames_dir),
                "gt_video_path": gt_video_path,
                "source_video_path": gt_video_path,
                "gt_score_raw": gt,
                "pred_score_raw": pred,
                "extra": {
                    "delta_mode": "signed_delta_mapped_to_01",
                    "formula": "frame_score = (pred - gt + 1) / 2",
                    "higher_is_better": True,
                    "gt_extra": gt.get("extra", {}),
                    "pred_extra": pred.get("extra", {}),
                },
                **risk_stats,
            }
        )
        final_rows.append(row)
    return final_rows, failures


def _build_external_scored_pool(
    *,
    score_dir: Path,
    pred_dir: Path,
    all_items: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    precomputed_failures: list[dict[str, Any]] | None = None,
    manifest: Path,
    baseline_manifest: Path,
    baseline_output: Path,
    baseline_scores_gt: Path | None = None,
    baseline_scores_pred: Path | None = None,
    score_method: str,
    baseline_method: str,
    select_method: str | None,
    cached_count: int,
    newly_scored_count: int,
    prediction_cached_count: int,
    prediction_new_count: int,
) -> dict[str, Any]:
    """Shared post-processing: build scored_pool.json from scored rows + original items."""
    by_episode: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(all_items):
        ep_id = _episode_id(item, idx)
        by_episode[ep_id] = item

    scored: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = list(precomputed_failures or [])
    seen_episode_ids: set[str] = set()
    for raw in scored_rows:
        ep_id = str(raw.get("episode_id") or "")
        if ep_id:
            seen_episode_ids.add(ep_id)
        original = by_episode.get(ep_id, {})
        if raw.get("score_ready", True) is not True:
            failures.append(
                {
                    "episode_id": ep_id,
                    "method": baseline_method,
                    "error": raw.get("error", "score_ready is false"),
                    "raw": raw,
                }
            )
            continue
        acquisition = _clip01(raw.get("acquisition_score", raw.get("tail_risk_top5", raw.get("episode_score", 0.0))))
        episode_score = _clip01(raw.get("episode_score", 0.0))
        frame_scores = raw.get("frame_conf_scores", raw.get("frame_scores", raw.get("baseline_frame_scores", [])))
        if not isinstance(frame_scores, list):
            frame_scores = []
        frame_scores = [_clip01(x) for x in frame_scores]
        pred_frames_dir = _pred_frames_dir(pred_dir, ep_id)
        gt_video_path = _video_path(original)
        row = dict(original)
        row.update(
            {
                "episode_id": ep_id or _episode_id(original, len(scored)),
                "score_method": score_method,
                "baseline_method": baseline_method,
                "method": baseline_method,
                "select_method": select_method,
                "score_ready": True,
                "episode_score": episode_score,
                "acquisition_score": acquisition,
                "baseline_episode_score": episode_score,
                "baseline_acquisition_score": acquisition,
                "frame_scores": frame_scores,
                "baseline_frame_scores": frame_scores,
                "frame_conf_scores": frame_scores,
                "frame_risk_scores": raw.get("frame_risk_scores", []),
                "gt_frame_scores": raw.get("gt_frame_scores", []),
                "pred_frame_scores": raw.get("pred_frame_scores", []),
                "gt_episode_score": raw.get("gt_episode_score"),
                "pred_episode_score": raw.get("pred_episode_score"),
                "episode_score_gap": raw.get("episode_score_gap"),
                "score_delta_mode": raw.get("score_delta_mode"),
                "video_path": str(pred_frames_dir),
                "pred_frames_dir": str(pred_frames_dir),
                "score_video_path": str(pred_frames_dir),
                "gt_video_path": gt_video_path,
                "source_video_path": gt_video_path,
                "mean_risk": _clip01(raw.get("mean_risk", acquisition)),
                "tail_risk_top5": _clip01(raw.get("tail_risk_top5", acquisition)),
                "tail_risk_top10": _clip01(raw.get("tail_risk_top10", acquisition)),
                "persistent_risk": _clip01(raw.get("persistent_risk", acquisition)),
                "risk_area": _clip01(raw.get("risk_area", acquisition)),
                "max_risk": _clip01(raw.get("max_risk", acquisition)),
                "std_risk": _clip01(raw.get("std_risk", 0.0)),
                "high_risk_frame_fraction": _clip01(raw.get("high_risk_frame_fraction", 0.0)),
                "extra": raw.get("extra", {}),
            }
        )
        scored.append(row)

    failure_episode_ids = {str(item.get("episode_id") or "") for item in failures}
    for ep_id in sorted(set(by_episode) - seen_episode_ids - failure_episode_ids):
        failures.append(
            {
                "episode_id": ep_id,
                "method": baseline_method,
                "error": "baseline did not return a row for this episode",
            }
        )

    out = {
        "source_manifest": str(manifest),
        "baseline_manifest": str(baseline_manifest),
        "baseline_scores": str(baseline_output),
        "baseline_scores_gt": str(baseline_scores_gt) if baseline_scores_gt else None,
        "baseline_scores_pred": str(baseline_scores_pred) if baseline_scores_pred else None,
        "items": scored,
        "failures": failures,
        "stats": {
            "requested": len(all_items),
            "scored": len(scored),
            "failed": len(failures),
            "cached": cached_count,
            "newly_scored_this_run": newly_scored_count,
            "prediction_cached": prediction_cached_count,
            "prediction_new_this_run": prediction_new_count,
        },
        "protocol": {
            "score_method": score_method,
            "baseline_method": baseline_method,
            "select_method": select_method,
            "no_evac_inference": False,
            "prediction_cache_dir": str(pred_dir),
            "scored_video": "external baselines score both GT and EVAC warmup/v1 prediction frames",
            "risk_definition": "risk = 1 - (pred - gt + 1) / 2, signed delta mapped to [0,1]; higher risk = worse prediction",
            "supports_patch_weight": False,
            "supports_frame_weight": True,
        },
    }
    save_json(out, score_dir / "scored_pool.json")
    print(
        f"[score_pool] {score_method} scored {len(scored)} items "
        f"({len(failures)} failures) -> {score_dir / 'scored_pool.json'}"
    )
    if all_items and not scored:
        raise RuntimeError(
            f"score_method={score_method} produced no valid scores. "
            f"Inspect failures in {score_dir / 'scored_pool.json'}; the baseline server/env may not be ready."
        )
    return out


def _run_external_backend(
    args: argparse.Namespace,
    *,
    cfg: dict[str, Any],
    manifest: Path,
    score_dir: Path,
    pred_dir: Path,
    score_method: str,
    select_method: str | None,
) -> dict[str, Any]:
    payload = load_json(str(manifest))
    all_items = flatten_manifest_items(payload)
    if args.max_episodes is not None:
        all_items = all_items[: int(args.max_episodes)]

    overwrite = bool(getattr(args, "overwrite", False))
    skip_gt_scores = bool(getattr(args, "skip_gt_scores", False))
    score_dir = ensure_dir(score_dir)
    pred_dir = ensure_dir(pred_dir)
    baseline_config = Path(args.baseline_config or _REPO / "baselines" / "evac_al_baselines" / "configs" / "baselines.yaml")
    baseline_method = _external_baseline_method(score_method)

    baseline_manifest = (score_dir / "baseline_manifest.jsonl").resolve()
    baseline_output = (score_dir / FINAL_SCORE_FILE).resolve()
    baseline_output_gt = (score_dir / GT_SCORE_FILE).resolve()
    baseline_output_pred = (score_dir / PRED_SCORE_FILE).resolve()

    # ---- Recover orphaned shard files from an interrupted previous run ----
    _recover_orphaned_shards(score_dir)

    # ---- Handle --overwrite: clear previous outputs ----
    if overwrite:
        remove_names = {FINAL_SCORE_FILE, PRED_SCORE_FILE}
        if not skip_gt_scores:
            remove_names.add(GT_SCORE_FILE)
        for f in [
            baseline_manifest,
            score_dir / "scored_pool.json",
            *[score_dir / name for name in sorted(remove_names)],
        ]:
            if f.exists():
                f.unlink()
                print(f"[score_pool] overwrite: removed {f}")

    prediction_cached_count, prediction_new_count = _ensure_external_predictions(
        args,
        cfg=cfg,
        manifest=manifest,
        items=all_items,
        pred_dir=pred_dir,
        score_dir=score_dir,
    )

    gt_existing_rows: list[dict[str, Any]] = []
    pred_existing_rows: list[dict[str, Any]] = []
    gt_existing_path: Path | None = None
    pred_existing_path: Path | None = None
    if not overwrite or skip_gt_scores:
        gt_existing_rows, gt_existing_path = _load_score_rows(score_dir, GT_SCORE_FILE)
    if not overwrite:
        pred_existing_rows, pred_existing_path = _load_score_rows(score_dir, PRED_SCORE_FILE)
    if skip_gt_scores and not gt_existing_rows and not overwrite:
        legacy_rows, legacy_path = _load_score_rows(score_dir, FINAL_SCORE_FILE)
        if legacy_rows and not any(row.get("score_delta_mode") for row in legacy_rows):
            gt_existing_rows, gt_existing_path = legacy_rows, legacy_path
            print(
                "[score_pool] --skip_gt_scores: using legacy baseline_scores.jsonl "
                "as GT scores because no *_gt file was found"
            )
    if skip_gt_scores and not gt_existing_rows and not overwrite:
        legacy_rows, legacy_path = _load_legacy_gt_rows_from_scored_pool(score_dir)
        if legacy_rows:
            gt_existing_rows, gt_existing_path = legacy_rows, legacy_path
            print(
                "[score_pool] --skip_gt_scores: recovered legacy GT scores from "
                f"{legacy_path}"
            )
    # Always scan other same-method score dirs for existing GT/pred scores.
    needed_eps = {_episode_id(item, idx) for idx, item in enumerate(all_items)}
    pool_root = _run_root(cfg) / "pool_scores"
    gt_cross = _find_raw_score_cache_rows(
        score_method=score_method,
        score_dir=score_dir,
        pool_root=pool_root,
        needed_eps=needed_eps,
        names=GT_SCORE_FILE,
        video_kind="gt",
    )
    pred_cross = _find_raw_score_cache_rows(
        score_method=score_method,
        score_dir=score_dir,
        pool_root=pool_root,
        needed_eps=needed_eps,
        names=PRED_SCORE_FILE,
        video_kind="pred",
    )
    if gt_cross:
        before = len(_rows_by_episode(gt_existing_rows, needed_eps))
        gt_existing_rows = _merge_rows_by_episode(gt_existing_rows + gt_cross)
        after = len(_rows_by_episode(gt_existing_rows, needed_eps))
        if after > before:
            print(f"[score_pool] reused {after - before} GT score row(s) from other {score_method}_* dirs")
    if pred_cross:
        before = len(_rows_by_episode(pred_existing_rows, needed_eps))
        pred_existing_rows = _merge_rows_by_episode(pred_existing_rows + pred_cross)
        after = len(_rows_by_episode(pred_existing_rows, needed_eps))
        if after > before:
            print(f"[score_pool] reused {after - before} pred score row(s) from other {score_method}_* dirs")

    if gt_existing_path:
        print(f"[score_pool] gt score cache: {gt_existing_path} ({len(gt_existing_rows)} rows)")
    if pred_existing_path:
        print(f"[score_pool] pred score cache: {pred_existing_path} ({len(pred_existing_rows)} rows)")

    gt_rows, gt_cached_count, gt_new_count = _score_missing_external_rows(
        args,
        all_items=all_items,
        existing_rows=gt_existing_rows,
        score_dir=score_dir,
        manifest_path=baseline_manifest,
        output_path=baseline_output_gt,
        pred_dir=None,
        video_kind="gt",
        baseline_method=baseline_method,
        baseline_config=baseline_config,
        skip_scoring=skip_gt_scores,
    )
    if not skip_gt_scores:
        _write_jsonl_rows(baseline_output_gt, gt_rows)
    elif gt_rows and not baseline_output_gt.exists():
        _write_jsonl_rows(baseline_output_gt, gt_rows)
        print(f"[score_pool] repaired cached GT scores -> {baseline_output_gt}")

    pred_rows, pred_cached_count, pred_new_count = _score_missing_external_rows(
        args,
        all_items=all_items,
        existing_rows=pred_existing_rows,
        score_dir=score_dir,
        manifest_path=baseline_manifest,
        output_path=baseline_output_pred,
        pred_dir=pred_dir,
        video_kind="pred",
        baseline_method=baseline_method,
        baseline_config=baseline_config,
    )
    _write_jsonl_rows(baseline_output_pred, pred_rows)
    _write_baseline_manifest(
        items=all_items,
        manifest_path=baseline_manifest,
        pred_dir=pred_dir,
        video_kind="pred",
    )

    final_rows, delta_failures = _build_delta_rows(
        all_items=all_items,
        gt_rows=gt_rows,
        pred_rows=pred_rows,
        score_method=score_method,
        baseline_method=baseline_method,
        select_method=select_method,
        pred_dir=pred_dir,
    )
    _write_jsonl_rows(baseline_output, final_rows)
    print(
        f"[score_pool] external delta scores: {len(final_rows)} ready, "
        f"{len(delta_failures)} incomplete -> {baseline_output}"
    )

    return _build_external_scored_pool(
        score_dir=score_dir,
        pred_dir=pred_dir,
        all_items=all_items,
        scored_rows=final_rows,
        precomputed_failures=delta_failures,
        manifest=manifest,
        baseline_manifest=baseline_manifest,
        baseline_output=baseline_output,
        baseline_scores_gt=baseline_output_gt if baseline_output_gt.exists() else gt_existing_path,
        baseline_scores_pred=baseline_output_pred,
        score_method=score_method,
        baseline_method=baseline_method,
        select_method=select_method,
        cached_count=gt_cached_count + pred_cached_count,
        newly_scored_count=gt_new_count + pred_new_count,
        prediction_cached_count=prediction_cached_count,
        prediction_new_count=prediction_new_count,
    )


def _write_random_scores(
    *,
    cfg: dict[str, Any],
    manifest: Path,
    output_dir: Path,
    score_method: str,
    select_method: str | None,
    seed: int | None,
    max_episodes: int | None,
) -> dict[str, Any]:
    payload = load_json(str(manifest))
    items = flatten_manifest_items(payload)
    if max_episodes is not None:
        items = items[: int(max_episodes)]
    rng = random.Random(int(seed if seed is not None else cfg.get("seed", 42)))
    scored: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        score = float(rng.random())
        row = dict(item)
        row.update(
            {
                "episode_id": _episode_id(item, idx),
                "score_method": score_method,
                "select_method": select_method,
                "score_ready": True,
                "random_score": score,
                "mean_risk": score,
                "tail_risk_top5": score,
                "tail_risk_top10": score,
                "persistent_risk": score,
                "risk_area": score,
            }
        )
        scored.append(row)
    output_dir = ensure_dir(output_dir)
    out = {
        "source_manifest": str(manifest),
        "items": scored,
        "failures": [],
        "stats": {
            "requested": len(items),
            "scored": len(scored),
            "failed": 0,
            "cached": 0,
            "newly_scored_this_run": len(scored),
        },
        "protocol": {
            "score_method": score_method,
            "select_method": select_method,
            "no_inference": True,
            "risk_definition": "random uniform score in [0, 1)",
        },
    }
    save_json(out, output_dir / "scored_pool.json")
    print(f"[score_pool] random scored {len(scored)} items -> {output_dir / 'scored_pool.json'}")
    return out


def _find_cached_results(
    manifest: Path,
    score_dir: Path,
    pred_dir: Path,
    score_method: str,
    pool_root: Path,
) -> tuple[list[dict[str, Any]], Path, int]:
    """Scan existing pred/score dirs for episodes that already have both.

    Returns (cached_score_items, reduced_manifest_path, cache_hit_count).
    The reduced manifest excludes cached episodes so the backend only
    processes what is genuinely missing.
    """
    manifest_data = load_json(str(manifest))
    all_items = flatten_manifest_items(manifest_data)
    needed_eps: set[str] = set()
    ep_to_item: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(all_items):
        ep_id = _episode_id(item, idx)
        needed_eps.add(ep_id)
        ep_to_item[ep_id] = item

    # ---- scan existing pred dirs ----
    ep_to_pred_src: dict[str, Path] = {}
    for pred_candidate in _prediction_cache_dirs(pool_root):
        if pred_candidate.resolve() == pred_dir.resolve():
            continue  # don't scan our own target
        for ep_dir in pred_candidate.iterdir():
            if ep_dir.is_dir() and ep_dir.name in needed_eps:
                if ep_dir.name not in ep_to_pred_src:
                    ep_to_pred_src[ep_dir.name] = ep_dir

    # ---- scan existing score dirs with matching score_method prefix ----
    ep_to_score_entry: dict[str, dict[str, Any]] = {}
    for score_candidate in sorted(pool_root.glob(f"{score_method}_*_scores")):
        if score_candidate.resolve() == score_dir.resolve():
            continue
        scored_pool_path = score_candidate / "scored_pool.json"
        if not scored_pool_path.exists():
            continue
        try:
            pool_data = load_json(str(scored_pool_path))
        except Exception:
            continue
        for entry in pool_data.get("items", []):
            ep_id = str(entry.get("episode_id") or "")
            if ep_id in needed_eps and ep_id not in ep_to_score_entry:
                ep_to_score_entry[ep_id] = dict(entry)

    # ---- episodes with BOTH pred and score are cache hits ----
    cached_ep_ids: set[str] = set()
    cached_scores: list[dict[str, Any]] = []
    for ep_id in needed_eps:
        if ep_id in ep_to_pred_src and ep_id in ep_to_score_entry:
            cached_ep_ids.add(ep_id)
            entry = dict(ep_to_score_entry[ep_id])
            entry.setdefault("_cached_from_pred", str(ep_to_pred_src[ep_id].parent))
            entry.setdefault("_cached_from_score", str(ep_to_score_entry[ep_id].get("_source") or ""))
            cached_scores.append(entry)

    # ---- copy pred caches ----
    pred_dir = ensure_dir(pred_dir)
    for ep_id in sorted(cached_ep_ids):
        src = ep_to_pred_src[ep_id]
        dst = pred_dir / ep_id
        if not dst.exists():
            shutil.copytree(str(src), str(dst))

    # ---- build reduced manifest ----
    remaining = [it for idx, it in enumerate(all_items) if _episode_id(it, idx) not in cached_ep_ids]
    reduced_manifest_path = score_dir / "manifest_reduced.json"
    if isinstance(manifest_data, dict):
        reduced = dict(manifest_data)
        reduced["items"] = remaining
    else:
        reduced = remaining
    save_json(reduced, reduced_manifest_path)

    return cached_scores, reduced_manifest_path, len(cached_ep_ids)


def _merge_cached_scores(
    score_dir: Path,
    cached_scores: list[dict[str, Any]],
    score_method: str,
    select_method: str | None,
    manifest: Path,
    pred_dir: Path,
) -> None:
    """Merge cached score entries into an existing scored_pool.json."""
    src = score_dir / "scored_pool.json"
    if not src.exists():
        return
    payload = load_json(str(src))
    existing_ids = {str(it.get("episode_id", "")) for it in payload.get("items", [])}
    new_entries = []
    for entry in cached_scores:
        ep_id = str(entry.get("episode_id", ""))
        if ep_id not in existing_ids:
            entry.setdefault("score_method", score_method)
            entry.setdefault("select_method", select_method)
            entry.setdefault("score_ready", True)
            entry.setdefault("_cached", True)
            new_entries.append(entry)
    if new_entries:
        payload["items"] = list(payload.get("items", [])) + new_entries
        payload.setdefault("stats", {})
        payload["stats"]["cached"] = payload["stats"].get("cached", 0) + len(new_entries)
        save_json(payload, src)

    # copy any extra files from pred dir
    for name in ("invalid_candidate_pool_items.json",):
        if (pred_dir / name).exists() and not (score_dir / name).exists():
            shutil.copy2(pred_dir / name, score_dir / name)


def _write_cached_only_scored_pool(
    score_dir: Path,
    cached_scores: list[dict[str, Any]],
    manifest: Path,
    score_method: str,
    select_method: str | None,
    pred_dir: Path,
) -> None:
    """Write scored_pool.json when every episode was a cache hit (no backend run)."""
    score_dir = ensure_dir(score_dir)
    out = {
        "source_manifest": str(manifest),
        "items": cached_scores,
        "failures": [],
        "stats": {
            "requested": len(cached_scores),
            "scored": len(cached_scores),
            "failed": 0,
            "cached": len(cached_scores),
            "newly_scored_this_run": 0,
        },
        "protocol": {
            "score_method": score_method,
            "select_method": select_method,
            "prediction_cache_dir": str(pred_dir),
            "score_output_dir": str(score_dir),
            "all_cached": True,
        },
    }
    save_json(out, score_dir / "scored_pool.json")
    print(f"[score_pool] all {len(cached_scores)} episodes cached -> {score_dir / 'scored_pool.json'}")


def _run_c3_backend(args: argparse.Namespace, *, manifest: Path, score_dir: Path, pred_dir: Path) -> None:
    print(f"[score_pool] c3 prediction/cache dir: {pred_dir}")
    print(f"[score_pool] c3 score dir          : {score_dir}")
    _run_c3_module(
        args,
        manifest=manifest,
        pred_dir=pred_dir,
        score_dir=score_dir,
        prediction_only=False,
        max_episodes=args.max_episodes,
        workers_per_gpu=max(1, int(args.workers_per_gpu)),
    )
    ensure_dir(score_dir)
    src = score_dir / "scored_pool.json"
    if not src.exists():
        raise FileNotFoundError(f"C3 backend did not produce {src}")
    payload = load_json(str(src))
    payload.setdefault("protocol", {})
    payload["protocol"].update(
        {
            "score_method": "c3",
            "select_method": args.select_method,
            "prediction_cache_dir": str(pred_dir),
            "score_output_dir": str(score_dir),
        }
    )
    save_json(payload, score_dir / "scored_pool.json")
    for name in ("invalid_candidate_pool_items.json",):
        if (pred_dir / name).exists():
            shutil.copy2(pred_dir / name, score_dir / name)
    print(f"[score_pool] wrote structured score file -> {score_dir / 'scored_pool.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AL pools with pluggable score methods")
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_method", "--score-method", dest="score_method", default=DEFAULT_SCORE_METHOD)
    parser.add_argument("--select_method", "--select-method", dest="select_method", default=None)
    parser.add_argument("--manifest", default=None, help="Override derived manifest path")
    parser.add_argument("--output-dir", default=None, help="Override derived score output dir")
    parser.add_argument("--pred-dir", default=None, help="Override derived EVAC prediction/cache dir")
    parser.add_argument("--evac_checkpoint", "--evac-checkpoint", dest="evac_checkpoint", default=None)
    parser.add_argument("--max_episodes", "--max-episodes", dest="max_episodes", type=int, default=None)
    parser.add_argument("--n_frames_to_generate", "--n-frames-to-generate", "--n-frames", dest="n_frames_to_generate", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard_id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-score all episodes from scratch (delete existing scores and re-run). "
                             "Without this flag, already-scored episodes are automatically skipped.")
    parser.add_argument("--save_traj_videos", "--save-traj-videos", action="store_true", default=False)
    parser.add_argument("--skip_gt_scores", "--skip-gt-scores", action="store_true", default=False,
                        help=(
                            "External score methods only: do not run GT scoring. "
                            "Use an existing baseline_scores_gt.jsonl and only fill pred scores."
                        ))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline_config", "--baseline-config", dest="baseline_config", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    score_method = _safe_name(args.score_method, DEFAULT_SCORE_METHOD)
    args.score_method = score_method
    manifest, score_dir, pred_dir, stage = _derive_paths(
        cfg,
        score_method=score_method,
        select_method=args.select_method,
        manifest_override=args.manifest,
        output_dir_override=args.output_dir,
        pred_dir_override=args.pred_dir,
    )
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found for {stage} scoring: {manifest}")

    if score_method == "random":
        _write_random_scores(
            cfg=cfg,
            manifest=manifest,
            output_dir=score_dir,
            score_method=score_method,
            select_method=args.select_method,
            seed=args.seed,
            max_episodes=args.max_episodes,
        )
        return

    if score_method == "c3":
        pool_root = _run_root(cfg) / "pool_scores"
        cached_scores, reduced_manifest, cache_hits = _find_cached_results(
            manifest=manifest,
            score_dir=score_dir,
            pred_dir=pred_dir,
            score_method=score_method,
            pool_root=pool_root,
        )
        total = len(flatten_manifest_items(load_json(str(manifest))))
        remain = total - cache_hits
        if remain > 0:
            print(f"[score_pool] {cache_hits}/{total} episodes cached "
                  f"({100 * cache_hits / max(1, total):.1f}%), {remain} remaining")
            _run_c3_backend(args, manifest=reduced_manifest, score_dir=score_dir, pred_dir=pred_dir)
        if cached_scores:
            _merge_cached_scores(
                score_dir=score_dir,
                cached_scores=cached_scores,
                score_method=score_method,
                select_method=args.select_method,
                manifest=manifest,
                pred_dir=pred_dir,
            )
        if not remain and cached_scores:
            _write_cached_only_scored_pool(
                score_dir=score_dir,
                cached_scores=cached_scores,
                manifest=manifest,
                score_method=score_method,
                select_method=args.select_method,
                pred_dir=pred_dir,
            )
        return

    if score_method in EXTERNAL_SCORE_METHODS:
        _run_external_backend(
            args,
            cfg=cfg,
            manifest=manifest,
            score_dir=score_dir,
            pred_dir=pred_dir,
            score_method=score_method,
            select_method=args.select_method,
        )
        return
    raise ValueError(f"Unknown score_method={score_method!r}; expected c3, random, or one of {sorted(EXTERNAL_SCORE_METHODS)}")


if __name__ == "__main__":
    main()
