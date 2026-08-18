#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from al_pipeline.sample_selectors import normalize_values
from al_pipeline.select_active_samples import (
    _filter_valid_scored_items,
    _load_available_scored_items,
    _invalid_reason_counts,
)
from al_pipeline.utils import (
    ensure_dir,
    filter_valid_external_worldmodel_items,
    flatten_manifest_items,
    load_json,
    load_yaml,
    save_json,
    score_distribution,
    write_csv,
)


DEFAULT_METHOD = "tail_risk"
DEFAULT_SCORE_METHOD = "c3"
DEFAULT_SCORE_KEY = "tail_risk_top5"
DEFAULT_TASK_KEY_FIELDS = ("task_name", "task_id")
C3_TRAIN_SPLIT_SUFFIX = "with_c3_train_split"
WEIGHTING_ALIASES = {
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
    # Legacy patch names now mean the new frame+residual-patch method.
    "patch": "frame_patch",
    "patch_weight": "frame_patch",
    "patch_weighting": "frame_patch",
    "loss_map": "frame_patch",
    # Pure patch remains available as an explicit ablation.
    "patch_only": "patch_only",
    "patch_weight_only": "patch_only",
    "pure_patch": "patch_only",
}
# Selection methods operate on a shared scored-pool contract.  For C3,
# tail_risk_top5 is the real top-5% patch risk; for scalar external baselines
# it is filled from their acquisition score so tail_risk remains method-neutral.
METHOD_SCORE_KEYS = {
    "mean_risk": "mean_risk",
    "tail_risk": "tail_risk_top5",
    "persistent_risk": "persistent_risk",
    "c3_persistent_risk": "persistent_risk",  # backward-compatible alias
    "random": "random_score",
}
SCORE_METHOD_ALIASES = {
    "confidence": "c3",
    "c3_confidence": "c3",
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


def _task_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(_get(cfg, "task_prescreen", {}) or {})


def _safe_name(value: str | None, default: str) -> str:
    text = str(value or default).strip().lower()
    return text.replace("/", "_").replace(" ", "_")


def _canonical_score_method(value: str | None, default: str = DEFAULT_SCORE_METHOD) -> str:
    name = _safe_name(value, default)
    return SCORE_METHOD_ALIASES.get(name, name)


def _canonical_weighting_name(value: str | None) -> str | None:
    if value is None:
        return None
    name = _safe_name(value, "")
    if not name:
        return None
    return WEIGHTING_ALIASES.get(name, name)


def _resolve_score_method(
    cfg: dict[str, Any],
    prescreen_cfg: dict[str, Any],
    score_method_override: str | None = None,
) -> str:
    return _canonical_score_method(
        score_method_override
        or prescreen_cfg.get("score_method")
        or _get(cfg, "selection.score_method")
        or DEFAULT_SCORE_METHOD,
        DEFAULT_SCORE_METHOD,
    )


def _selection_root(cfg: dict[str, Any], prescreen_cfg: dict[str, Any], run_root: Path) -> Path:
    return Path(
        prescreen_cfg.get("output_dir")
        or _get(cfg, "selection.output_root")
        or run_root / "selection"
    )


def _selection_run_dir(
    cfg: dict[str, Any],
    prescreen_cfg: dict[str, Any],
    run_root: Path,
    *,
    score_method: str,
    select_method: str,
) -> Path:
    return _selection_root(cfg, prescreen_cfg, run_root) / f"{_safe_name(score_method, DEFAULT_SCORE_METHOD)}_{_safe_name(select_method, DEFAULT_METHOD)}"


def _default_prescreen_scored_pool(
    cfg: dict[str, Any],
    prescreen_cfg: dict[str, Any],
    run_root: Path,
    *,
    score_method: str,
) -> Path:
    return run_root / "pool_scores" / f"{_safe_name(score_method, DEFAULT_SCORE_METHOD)}_task_prescreen_scores" / "scored_pool.json"


def _default_detailed_scored_pool(
    cfg: dict[str, Any],
    prescreen_cfg: dict[str, Any],
    run_root: Path,
    *,
    score_method: str,
    select_method: str,
) -> Path:
    return run_root / "pool_scores" / f"{_safe_name(score_method, DEFAULT_SCORE_METHOD)}_{_safe_name(select_method, DEFAULT_METHOD)}_selected_scores" / "scored_pool.json"


def _task_key(item: dict[str, Any], fields: list[str] | tuple[str, ...] | str | None = None) -> str:
    fields = fields or DEFAULT_TASK_KEY_FIELDS
    if isinstance(fields, str):
        fields = [fields]
    values = []
    for field in fields:
        value = item.get(str(field))
        if value is not None and str(value) != "":
            values.append(str(value))
    if values:
        return "::".join(values)
    return str(item.get("task_name") or item.get("task_id") or "unknown")


def _task_key_fields(prescreen_cfg: dict[str, Any]) -> list[str] | tuple[str, ...]:
    fields = prescreen_cfg.get("task_key_fields")
    if isinstance(fields, str):
        return [fields]
    if isinstance(fields, (list, tuple)) and fields:
        return [str(x) for x in fields]
    return DEFAULT_TASK_KEY_FIELDS


def _episode_id(item: dict[str, Any]) -> str:
    return str(item.get("episode_id") or item.get("ep_id") or item.get("folder") or "")


def _source_path_value(item: dict[str, Any]) -> str:
    return str(item.get("source_path") or item.get("path") or item.get("episode_dir") or "")


def _merge_key(item: dict[str, Any]) -> str:
    source = _source_path_value(item)
    if source:
        return "path:" + str(Path(source).expanduser())
    return "episode:" + _episode_id(item)


def _resolve_task_selection_method(
    cfg: dict[str, Any],
    prescreen_cfg: dict[str, Any],
    method_override: str | None = None,
) -> str:
    return _safe_name(
        method_override
        or prescreen_cfg.get("selection_method")
        or _get(cfg, "selection.method")
        or DEFAULT_METHOD,
        DEFAULT_METHOD,
    )


def _normalize_task_scores(values: list[float], method: str) -> list[float]:
    if method != "rank":
        return normalize_values(values, method)
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.size == 1:
        return [1.0]
    out = np.zeros_like(arr, dtype=np.float32)
    unique_vals = sorted(float(x) for x in set(arr.tolist()))
    denom = max(1, len(unique_vals) - 1)
    for rank, value in enumerate(unique_vals):
        out[arr == value] = float(rank) / float(denom)
    return out.tolist()


def _load_candidate_items(
    cfg: dict[str, Any],
    *,
    candidate_manifest: str | None = None,
    require_training_files: bool = True,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None, list[dict[str, Any]]]:
    run_root = _run_root(cfg)
    prescreen_cfg = _task_cfg(cfg)
    manifests_cfg = _get(cfg, "manifests", {})
    path = (
        candidate_manifest
        or prescreen_cfg.get("candidate_pool_manifest")
        or manifests_cfg.get("candidate_pool")
        or _get(cfg, "scoring.candidate_pool_manifest")
        or str(run_root / "manifests" / "candidate_pool.json")
    )
    payload = load_json(path)
    items = flatten_manifest_items(payload)
    data_format = _get(cfg, "scoring.data_format", "external_worldmodel")
    external_filter_stats = None
    invalid_external_items: list[dict[str, Any]] = []
    if data_format == "external_worldmodel":
        items, invalid_external_items, external_filter_stats = filter_valid_external_worldmodel_items(
            items,
            min_frames=int(prescreen_cfg.get("min_valid_frames", _get(cfg, "scoring.min_valid_frames", 1))),
            require_paths=prescreen_cfg.get("require_paths", _get(cfg, "scoring.require_paths", "auto")),
            require_training_files=require_training_files,
        )
    if not items:
        raise ValueError(f"No valid candidate items found in {path}")
    return items, str(path), external_filter_stats, invalid_external_items


def _group_by_task(
    items: list[dict[str, Any]],
    task_key_fields: list[str] | tuple[str, ...] | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[_task_key(item, task_key_fields)].append(item)
    for rows in groups.values():
        rows.sort(key=lambda x: (_episode_id(x), str(x.get("episode_dir") or x.get("path") or "")))
    return dict(sorted(groups.items()))


def _pick_representatives(
    rows: list[dict[str, Any]],
    *,
    k: int,
    strategy: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    k = min(k, len(rows))
    if strategy == "first":
        return rows[:k]
    if strategy == "middle":
        if k == 1:
            return [rows[len(rows) // 2]]
        idxs = np.linspace(0, len(rows) - 1, num=k, dtype=int).tolist()
        return [rows[i] for i in idxs]
    if strategy == "random":
        return rng.sample(rows, k)
    raise ValueError("representative_strategy must be one of: random, first, middle")


def build_prescreen_manifest(
    config_path: str,
    *,
    candidate_manifest: str | None = None,
    output_dir: str | None = None,
    representatives_per_task: int | None = None,
    representative_strategy: str | None = None,
    seed: int | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    run_root = _run_root(cfg)
    prescreen_cfg = _task_cfg(cfg)
    out_dir = ensure_dir(output_dir or _selection_root(cfg, prescreen_cfg, run_root))
    reps_per_task = int(representatives_per_task or prescreen_cfg.get("representatives_per_task", 1))
    strategy = representative_strategy or prescreen_cfg.get("representative_strategy", "random")
    rng = random.Random(int(seed if seed is not None else prescreen_cfg.get("random_seed", cfg.get("seed", 42))))

    items, source_manifest, filter_stats, invalid_items = _load_candidate_items(
        cfg,
        candidate_manifest=candidate_manifest,
        require_training_files=True,
    )
    task_key_fields = _task_key_fields(prescreen_cfg)
    groups = _group_by_task(items, task_key_fields)
    task_names = list(groups)
    if max_tasks is not None:
        task_names = task_names[: int(max_tasks)]

    reps: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task in task_names:
        rows = groups[task]
        picked = _pick_representatives(rows, k=reps_per_task, strategy=strategy, rng=rng)
        picked_ids = {_episode_id(row) for row in picked}
        for row in picked:
            out = dict(row)
            out["split"] = "task_prescreen"
            out["task_prescreen_representative"] = True
            out["task_pool_size"] = len(rows)
            out["representative_strategy"] = strategy
            reps.append(out)
        task_rows.append(
            {
                "task_name": task,
                "available_episodes": len(rows),
                "representatives": len(picked),
                "representative_episode_ids": sorted(picked_ids),
            }
        )

    manifest_path = out_dir / "task_prescreen_pool.json"
    summary_path = out_dir / "task_prescreen_summary.json"
    invalid_path = out_dir / "invalid_candidate_items.json"
    save_json(
        {
            "dataset": _get(cfg, "phase.dataset", "robotwin"),
            "split": "task_prescreen",
            "items": reps,
            "stats": {
                "candidate_items": len(items),
                "tasks": len(task_names),
                "representatives": len(reps),
                "representatives_per_task": reps_per_task,
                "representative_strategy": strategy,
                "task_key_fields": list(task_key_fields),
            },
        },
        manifest_path,
    )
    save_json(
        {
            "source_manifest": source_manifest,
            "output_manifest": str(manifest_path),
            "tasks": task_rows,
            "external_filter": filter_stats,
            "invalid_candidate_items": str(invalid_path) if invalid_items else None,
        },
        summary_path,
    )
    save_json({"items": invalid_items, "count": len(invalid_items), "filter_stats": filter_stats}, invalid_path)
    print(f"[task prescreen] representatives: {len(reps)} episodes from {len(task_names)} tasks")
    print(f"  manifest: {manifest_path}")
    print(f"  summary : {summary_path}")
    return {"manifest": str(manifest_path), "summary": str(summary_path), "items": len(reps)}


def _score_for_task(rows: list[dict[str, Any]], score_key: str) -> float:
    values = []
    for row in rows:
        try:
            val = float(row.get(score_key))
        except Exception:
            continue
        if math.isfinite(val):
            values.append(val)
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _resolve_budget(
    cfg: dict[str, Any],
    *,
    n_items: int,
    budget_override: int | None,
) -> tuple[int, str]:
    prescreen_cfg = _task_cfg(cfg)
    selection_cfg = dict(_get(cfg, "selection", {}) or {})
    if budget_override is not None:
        budget = int(budget_override)
        source = "cli"
    elif prescreen_cfg.get("selection_budget") is not None:
        budget = int(prescreen_cfg["selection_budget"])
        source = "task_prescreen.selection_budget"
    elif selection_cfg.get("budget") is not None:
        budget = int(selection_cfg["budget"])
        source = "selection.budget"
    else:
        ratio = float(prescreen_cfg.get("selection_budget_ratio", selection_cfg.get("budget_ratio", 0.4)))
        budget = int(math.ceil(n_items * ratio))
        source = "task_prescreen.selection_budget_ratio"
    return max(0, min(n_items, budget)), source


def _allocate_quotas(
    task_rows: list[dict[str, Any]],
    *,
    total_budget: int,
    min_per_task: int,
    max_per_task: int | None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in task_rows]
    if total_budget <= 0:
        for row in rows:
            row["task_budget"] = 0
        return rows

    for row in rows:
        cap = int(row["available_episodes"])
        if max_per_task is not None:
            cap = min(cap, int(max_per_task))
        row["_cap"] = max(0, cap)
        row["task_budget"] = 0

    rows.sort(key=lambda r: (float(r.get("score_norm", 0.0)), float(r.get("task_score_for_allocation", 0.0))), reverse=True)
    min_total = sum(min(int(row["_cap"]), int(min_per_task)) for row in rows)
    if min_per_task > 0 and total_budget < min_total:
        remaining = total_budget
        for row in rows:
            if remaining <= 0:
                break
            if int(row["_cap"]) > 0:
                row["task_budget"] = 1
                remaining -= 1
    else:
        for row in rows:
            row["task_budget"] = min(int(row["_cap"]), int(min_per_task))

    remaining = total_budget - sum(int(row["task_budget"]) for row in rows)
    while remaining > 0:
        open_rows = [row for row in rows if int(row["task_budget"]) < int(row["_cap"])]
        if not open_rows:
            break
        weights = np.asarray(
            [max(0.0, float(row.get("allocation_weight", 0.0))) for row in open_rows],
            dtype=np.float64,
        )
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        desired = weights / float(weights.sum()) * remaining
        added = 0
        fractions: list[tuple[float, int]] = []
        for idx, (row, want) in enumerate(zip(open_rows, desired)):
            room = int(row["_cap"]) - int(row["task_budget"])
            add = min(room, int(math.floor(float(want))))
            if add > 0:
                row["task_budget"] = int(row["task_budget"]) + add
                added += add
            fractions.append((float(want) - math.floor(float(want)), idx))
        remaining -= added
        if remaining <= 0:
            break
        fractions.sort(reverse=True)
        progressed = False
        for _, idx in fractions:
            if remaining <= 0:
                break
            row = open_rows[idx]
            if int(row["task_budget"]) < int(row["_cap"]):
                row["task_budget"] = int(row["task_budget"]) + 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    for row in rows:
        row.pop("_cap", None)
    rows.sort(key=lambda r: str(r["task_name"]))
    return rows


def select_candidates_from_prescreen(
    config_path: str,
    *,
    candidate_manifest: str | None = None,
    prescreen_scores: str | None = None,
    output_dir: str | None = None,
    method: str | None = None,
    score_method: str | None = None,
    select_method: str | None = None,
    budget: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    run_root = _run_root(cfg)
    prescreen_cfg = _task_cfg(cfg)
    score_method_name = _resolve_score_method(cfg, prescreen_cfg, score_method)
    method_name = _resolve_task_selection_method(cfg, prescreen_cfg, select_method or method)
    out_dir = ensure_dir(
        output_dir
        or _selection_run_dir(
            cfg,
            prescreen_cfg,
            run_root,
            score_method=score_method_name,
            select_method=method_name,
        )
    )
    rng = random.Random(int(seed if seed is not None else prescreen_cfg.get("random_seed", cfg.get("seed", 42))))

    items, source_manifest, filter_stats, invalid_items = _load_candidate_items(
        cfg,
        candidate_manifest=candidate_manifest,
        require_training_files=True,
    )
    task_key_fields = _task_key_fields(prescreen_cfg)
    groups = _group_by_task(items, task_key_fields)
    scored_pool = prescreen_scores or str(
        _default_prescreen_scored_pool(
            cfg,
            prescreen_cfg,
            run_root,
            score_method=score_method_name,
        )
    )
    legacy_scored_pool = (
        prescreen_cfg.get("prescreen_scored_pool")
        or str(
            Path(prescreen_cfg.get("prescreen_score_output_dir") or run_root / "pool_scores" / "task_prescreen_scores")
            / "scored_pool.json"
        )
    )
    if not prescreen_scores and not Path(scored_pool).exists() and Path(str(legacy_scored_pool)).exists():
        scored_pool = str(legacy_scored_pool)
    if Path(str(scored_pool)).exists():
        scored_items, scored_source, scored_count = _load_available_scored_items(str(scored_pool))
    elif score_method_name == "random" or method_name == "random":
        scored_items, scored_source, scored_count = [], f"random/no-prescreen-score-file:{scored_pool}", 0
    else:
        scored_items, scored_source, scored_count = _load_available_scored_items(str(scored_pool))
    score_key = str(METHOD_SCORE_KEYS.get(method_name) or prescreen_cfg.get("score_key") or DEFAULT_SCORE_KEY)
    score_direction = str(prescreen_cfg.get("score_direction", "high")).lower()
    if score_direction not in {"high", "low"}:
        raise ValueError("task_prescreen.score_direction must be 'high' or 'low'")

    scored_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_items:
        scored_by_task[_task_key(row, task_key_fields)].append(row)

    raw_scores = []
    for task in groups:
        raw_scores.append(_score_for_task(scored_by_task.get(task, []), score_key))
    if method_name == "random":
        raw_scores = [1.0 for _ in raw_scores]
    finite_scores = [x for x in raw_scores if math.isfinite(x)]
    missing_policy = str(prescreen_cfg.get("missing_task_score", "median"))
    if finite_scores:
        if missing_policy == "zero":
            fallback_score = 0.0
        elif missing_policy == "min":
            fallback_score = float(np.min(finite_scores))
        else:
            fallback_score = float(np.median(finite_scores))
    else:
        fallback_score = 0.0
    raw_scores = [fallback_score if not math.isfinite(x) else x for x in raw_scores]
    alloc_scores = raw_scores if score_direction == "high" else [-x for x in raw_scores]
    score_norm = _normalize_task_scores(
        [float(x) for x in alloc_scores],
        str(prescreen_cfg.get("score_normalization", "rank")),
    )

    min_multiplier = float(prescreen_cfg.get("min_multiplier", 0.25))
    max_multiplier = float(prescreen_cfg.get("max_multiplier", 2.0))
    allocation_base = str(prescreen_cfg.get("allocation_base", "uniform"))
    task_rows = []
    for idx, task in enumerate(groups):
        base = len(groups[task]) if allocation_base == "available" else 1.0
        norm = float(score_norm[idx]) if score_norm else 0.0
        weight = float(base) * (min_multiplier + (max_multiplier - min_multiplier) * norm)
        task_rows.append(
            {
                "task_name": task,
                "available_episodes": len(groups[task]),
                "scored_representatives": len(scored_by_task.get(task, [])),
                "score_key": score_key,
                "raw_task_score": float(raw_scores[idx]),
                "task_score_for_allocation": float(alloc_scores[idx]),
                "score_norm": norm,
                "allocation_weight": weight,
            }
        )

    total_budget, budget_source = _resolve_budget(cfg, n_items=len(items), budget_override=budget)
    task_rows = _allocate_quotas(
        task_rows,
        total_budget=total_budget,
        min_per_task=int(prescreen_cfg.get("min_episodes_per_task", 1)),
        max_per_task=(
            int(prescreen_cfg["max_episodes_per_task"])
            if prescreen_cfg.get("max_episodes_per_task") is not None
            else None
        ),
    )
    task_meta = {row["task_name"]: row for row in task_rows}
    selected: list[dict[str, Any]] = []
    for task, rows in groups.items():
        quota = int(task_meta[task].get("task_budget", 0))
        if quota <= 0:
            continue
        shuffled = list(rows)
        rng.shuffle(shuffled)
        for row in shuffled[:quota]:
            out = dict(row)
            out.update(
                {
                    "split": "task_prescreen_selected_candidates",
                    "score_method": score_method_name,
                    "selection_method": method_name,
                    "select_method": method_name,
                    "selection_stage": "task_budget_random_candidate",
                    "task_prescreen_score_key": score_key,
                    "task_prescreen_score": float(task_meta[task]["raw_task_score"]),
                    "task_prescreen_score_norm": float(task_meta[task]["score_norm"]),
                    "task_budget": quota,
                    "task_available_episodes": int(task_meta[task]["available_episodes"]),
                }
            )
            selected.append(out)

    selected_path = out_dir / "selected_candidates.json"
    plan_path = out_dir / "task_selection_plan.json"
    csv_path = out_dir / "task_selection_plan.csv"
    invalid_path = out_dir / "invalid_candidate_items.json"
    save_json(
        {
            "dataset": _get(cfg, "phase.dataset", "robotwin"),
            "split": "task_prescreen_selected_candidates",
            "items": selected,
            "stats": {
                "candidate_items": len(items),
                "selected_candidates": len(selected),
                "tasks": len(groups),
                "tasks_with_budget": sum(1 for row in task_rows if int(row.get("task_budget", 0)) > 0),
                "budget": total_budget,
                "budget_source": budget_source,
                "score_method": score_method_name,
                "selection_method": method_name,
                "select_method": method_name,
                "score_key": score_key,
                "score_direction": score_direction,
                "task_key_fields": list(task_key_fields),
            },
        },
        selected_path,
    )
    save_json(
        {
            "source_candidate_manifest": source_manifest,
            "source_prescreen_scores": scored_source,
            "source_prescreen_score_ready_items": scored_count,
            "selected_candidate_manifest": str(selected_path),
            "task_rows": task_rows,
            "external_filter": filter_stats,
            "invalid_candidate_items": str(invalid_path) if invalid_items else None,
        },
        plan_path,
    )
    write_csv(task_rows, csv_path)
    save_json({"items": invalid_items, "count": len(invalid_items), "filter_stats": filter_stats}, invalid_path)
    print(
        f"[task prescreen] selected {len(selected)} raw candidates "
        f"from {len(groups)} tasks using score_method={score_method_name} "
        f"select_method={method_name} score_key={score_key}"
    )
    print(f"  selected candidates: {selected_path}")
    print(f"  plan               : {plan_path}")
    return {"selected_candidates": str(selected_path), "plan": str(plan_path), "items": len(selected)}


def _load_plan_metadata(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = load_json(str(p))
    if isinstance(payload, dict) and "items" in payload:
        rows = payload.get("items", [])
    else:
        rows = flatten_manifest_items(payload)
    out = {}
    for row in rows:
        ep_id = _episode_id(row)
        if ep_id:
            out[ep_id] = row
    return out


def finalize_scored_selection(
    config_path: str,
    *,
    scored_pool: str | None = None,
    selected_candidates: str | None = None,
    output_dir: str | None = None,
    method: str | None = None,
    score_method: str | None = None,
    select_method: str | None = None,
    weighting: str | None = None,
    include_c3_train_split: bool = False,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    run_root = _run_root(cfg)
    prescreen_cfg = _task_cfg(cfg)
    task_key_fields = _task_key_fields(prescreen_cfg)
    selection_cfg = dict(_get(cfg, "selection", {}) or {})
    score_method_name = _resolve_score_method(cfg, prescreen_cfg, score_method)
    method_name = _safe_name(
        select_method
        or method
        or prescreen_cfg.get("final_selection_method")
        or prescreen_cfg.get("selection_method")
        or selection_cfg.get("select_method")
        or selection_cfg.get("method")
        or DEFAULT_METHOD,
        DEFAULT_METHOD,
    )
    method_dir = _selection_run_dir(
        cfg,
        prescreen_cfg,
        run_root,
        score_method=score_method_name,
        select_method=method_name,
    )
    out_dir = ensure_dir(
        output_dir
        or prescreen_cfg.get("final_selection_output_dir")
        or selection_cfg.get("output_dir")
        or method_dir
    )
    scored_pool_path = scored_pool or str(
        _default_detailed_scored_pool(
            cfg,
            prescreen_cfg,
            run_root,
            score_method=score_method_name,
            select_method=method_name,
        )
    )
    legacy_scored_pool = (
        prescreen_cfg.get("detailed_scored_pool")
        or str(
            Path(prescreen_cfg.get("detailed_score_output_dir") or run_root / "pool_scores" / "selected_scores")
            / "scored_pool.json"
        )
    )
    if not scored_pool and not Path(scored_pool_path).exists() and Path(str(legacy_scored_pool)).exists():
        scored_pool_path = str(legacy_scored_pool)
    selected_candidates_path = selected_candidates or str(method_dir / "selected_candidates.json")
    legacy_selected_candidates = (
        prescreen_cfg.get("selected_candidate_manifest")
        or str(Path(prescreen_cfg.get("output_dir") or run_root / "task_prescreen") / "selected_candidates.json")
    )
    if not selected_candidates and not Path(selected_candidates_path).exists() and Path(str(legacy_selected_candidates)).exists():
        selected_candidates_path = str(legacy_selected_candidates)

    weighting_name = _canonical_weighting_name(weighting)
    _skip_detailed_score = weighting_name == "none"
    if weighting_name in {"frame_patch", "patch_only"} and score_method_name != "c3":
        raise ValueError(
            f"{weighting_name} requires dense C3 confidence maps. "
            f"score_method={score_method_name!r} only provides trajectory-level baseline scores; "
            "use --weighting none, --weighting oversampling, or --weighting frame."
        )
    if not _skip_detailed_score and Path(str(scored_pool_path)).exists():
        raw_items, scored_source, raw_count = _load_available_scored_items(str(scored_pool_path))
        if score_method_name == "c3":
            valid_items, invalid_items = _filter_valid_scored_items(raw_items, Path(scored_pool_path).parent)
        else:
            valid_items, invalid_items = raw_items, []
        detailed_score_available = True
    else:
        raw_items, scored_source, raw_count = [], f"not-found:{scored_pool_path}", 0
        valid_items, invalid_items = [], []
        detailed_score_available = False
        if _skip_detailed_score:
            scored_source = f"skipped-detailed-score:weighting={weighting_name}"
    invalid_reason_counts = _invalid_reason_counts(invalid_items)
    if weighting_name in {"frame_patch", "frame", "patch_only"} and not detailed_score_available:
        raise FileNotFoundError(
            f"--weighting {weighting_name} requires selected-subset detailed scores, "
            f"but scored_pool was not found: {scored_pool_path}"
        )
    if weighting_name in {"frame_patch", "frame", "patch_only"} and detailed_score_available and not valid_items:
        raise ValueError(
            f"--weighting {weighting_name} found scored_pool but no valid scored items: "
            f"{scored_pool_path}; invalid_reason_counts={invalid_reason_counts}"
        )
    plan_meta = _load_plan_metadata(str(selected_candidates_path))
    score_key = str(prescreen_cfg.get("final_score_key", _get(cfg, "retraining.sample_weight_score_key", DEFAULT_SCORE_KEY)))
    if not detailed_score_available:
        score_key = str(METHOD_SCORE_KEYS.get(method_name) or METHOD_SCORE_KEYS.get(score_method_name) or "task_prescreen_score")

    selected: list[dict[str, Any]] = []
    source_items = valid_items if valid_items else list(plan_meta.values())
    for item in source_items:
        ep_id = _episode_id(item)
        merged = dict(plan_meta.get(ep_id, {}))
        merged.update(item)
        merged["score_method"] = str(score_method_name)
        merged["selection_method"] = str(method_name)
        merged["select_method"] = str(method_name)
        merged["selection_stage"] = "selected_only_detailed_scored"
        if detailed_score_available:
            merged["selection_stage"] = "selected_only_detailed_scored"
        else:
            merged["selection_stage"] = "selected_candidates_without_detailed_score"
            merged.setdefault("score_ready", False)
        try:
            merged["selection_score"] = float(
                merged.get(
                    score_key,
                    merged.get(
                        "task_prescreen_score",
                        merged.get(DEFAULT_SCORE_KEY, merged.get("random_score", 0.0)),
                    ),
                )
            )
        except Exception:
            merged["selection_score"] = 0.0
        merged["task_prescreen_candidate"] = ep_id in plan_meta
        selected.append(merged)

    c3_train_split_path = (
        _get(cfg, "manifests.score_method_train_split")
        or _get(cfg, "manifests.c3_train_split")
        or _get(cfg, "c3_probe.trained_split_manifest")
    )
    added_c3_train_items: list[dict[str, Any]] = []
    skipped_duplicate_c3_train = 0
    if include_c3_train_split:
        if not c3_train_split_path:
            raise ValueError(
                "--include_c3_train_split requires manifests.score_method_train_split "
                "or manifests.c3_train_split in the config"
            )
        c3_path = Path(str(c3_train_split_path))
        if not c3_path.exists():
            raise FileNotFoundError(f"c3_train_split manifest not found: {c3_path}")
        seen = {_merge_key(item) for item in selected}
        for item in flatten_manifest_items(load_json(c3_path)):
            row = dict(item)
            key = _merge_key(row)
            if key in seen:
                skipped_duplicate_c3_train += 1
                continue
            source_path = _source_path_value(row)
            if source_path:
                row.setdefault("source_path", source_path)
                row.setdefault("episode_dir", source_path)
            row.setdefault("episode_id", _episode_id(row) or Path(source_path).name)
            row.setdefault("dataset", _get(cfg, "phase.dataset", "robotwin"))
            row.setdefault("format", _get(cfg, "scoring.data_format", "external_worldmodel"))
            row["score_method"] = str(score_method_name)
            row["selection_method"] = str(method_name)
            row["select_method"] = str(method_name)
            row["selection_stage"] = "c3_train_split_replay"
            row["source"] = "c3_train_split"
            row["anti_forgetting_replay"] = True
            row["task_prescreen_candidate"] = False
            row["selection_score"] = 0.0
            row.setdefault("mean_risk", 0.0)
            row.setdefault(DEFAULT_SCORE_KEY, 0.0)
            row.setdefault("persistent_risk", 0.0)
            row.setdefault("risk_area", 0.0)
            row.setdefault("score_ready", False)
            row["risk_stats_path"] = None
            row["conf_map_path"] = None
            selected.append(row)
            added_c3_train_items.append(row)
            seen.add(key)

    _prefix = f"{weighting_name}_" if weighting_name else ""
    _suffix = f"_{C3_TRAIN_SPLIT_SUFFIX}" if include_c3_train_split else ""
    selected_path = out_dir / f"{_prefix}selected{_suffix}.json"
    summary_path = out_dir / f"{_prefix}selection_summary{_suffix}.json"
    csv_path = out_dir / f"{_prefix}selection_scores{_suffix}.csv"
    invalid_path = out_dir / f"{_prefix}invalid_scored_items{_suffix}.json"
    task_counts = dict(sorted(Counter(_task_key(row, task_key_fields) for row in selected).items()))
    summary = {
        "method": str(method_name),
        "score_method": str(score_method_name),
        "selection_method": str(method_name),
        "select_method": str(method_name),
        "weighting": str(weighting_name) if weighting_name else None,
        "phase": _get(cfg, "phase.name", ""),
        "dataset": _get(cfg, "phase.dataset", "robotwin"),
        "data_format": _get(cfg, "scoring.data_format", "external_worldmodel"),
        "budget": len(selected),
        "source_scored_pool": scored_source,
        "source_selected_candidates": str(selected_candidates_path),
        "detailed_score_available": detailed_score_available,
        "raw_score_ready_items": raw_count,
        "valid_score_ready_items": len(valid_items),
        "invalid_score_ready_items": len(invalid_items),
        "invalid_reason_counts": invalid_reason_counts,
        "invalid_items_path": str(invalid_path) if invalid_items else None,
        "include_c3_train_split": bool(include_c3_train_split),
        "c3_train_split_manifest": str(c3_train_split_path) if include_c3_train_split else None,
        "c3_train_split_added_items": len(added_c3_train_items),
        "c3_train_split_duplicate_items": skipped_duplicate_c3_train,
        "selected_episode_ids": [_episode_id(row) for row in selected],
        "selected_task_counts": task_counts,
        "score_key": score_key,
        "score_distribution": score_distribution([float(row.get("selection_score", 0.0)) for row in selected]),
        "protocol": {
            "task_level_prescreen": True,
            "full_pool_scoring": False,
            "detailed_c3_maps_only_for_selected_subset": True,
            "risk_definition": "risk = 1 - confidence",
            "task_key_fields": list(task_key_fields),
        },
    }
    selected_payload = {
        "phase": _get(cfg, "phase.name", ""),
        "dataset": _get(cfg, "phase.dataset", "robotwin"),
        "data_format": _get(cfg, "scoring.data_format", "external_worldmodel"),
        "score_method": str(score_method_name),
        "method": str(method_name),
        "selection_method": str(method_name),
        "select_method": str(method_name),
        "weighting": str(weighting_name) if weighting_name else None,
        "budget": len(selected),
        "base_selected_items": len(selected) - len(added_c3_train_items),
        "include_c3_train_split": bool(include_c3_train_split),
        "c3_train_split_manifest": str(c3_train_split_path) if include_c3_train_split else None,
        "c3_train_split_added_items": len(added_c3_train_items),
        "items": selected,
        "summary_path": str(summary_path),
    }
    save_json(selected_payload, selected_path)
    save_json(summary, summary_path)
    write_csv(selected, csv_path)
    save_json(
        {
            "items": invalid_items,
            "count": len(invalid_items),
            "reason_counts": invalid_reason_counts,
            "source_scored_pool": scored_source,
        },
        invalid_path,
    )
    detail_label = "detailed-scored" if detailed_score_available else "candidate-only"
    print(f"[task prescreen] finalized {len(selected)} {detail_label} samples")
    if include_c3_train_split:
        print(
            f"  c3_train_split replay: +{len(added_c3_train_items)} "
            f"(duplicates skipped: {skipped_duplicate_c3_train})"
        )
    print(f"  selected: {selected_path}")
    print(f"  summary : {summary_path}")
    if invalid_items:
        print(f"  invalid : {invalid_path}")
    return {"selected": str(selected_path), "summary": str(summary_path), "items": len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-level prescreen selection utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build one/few representative episodes per task for quick scoring")
    p_build.add_argument("--config", required=True)
    p_build.add_argument("--candidate-manifest", default=None)
    p_build.add_argument("--output-dir", default=None)
    p_build.add_argument("--representatives-per-task", type=int, default=None)
    p_build.add_argument("--representative-strategy", choices=["random", "first", "middle"], default=None)
    p_build.add_argument("--seed", type=int, default=None)
    p_build.add_argument("--max-tasks", type=int, default=None)

    p_select = sub.add_parser("select", help="Allocate task budgets from prescreen scores and sample raw candidates")
    p_select.add_argument("--config", required=True)
    p_select.add_argument("--candidate-manifest", default=None)
    p_select.add_argument("--prescreen-scores", default=None)
    p_select.add_argument("--output-dir", default=None)
    p_select.add_argument("--score_method", "--score-method", dest="score_method", default=None, help="Score source used for task prescreen, e.g. c3, random, robometer_prog")
    p_select.add_argument("--select_method", "--select-method", dest="select_method", default=None, help="Task-budget selection method, e.g. tail_risk, mean_risk, random")
    p_select.add_argument("--method", default=None, help="Backward-compatible alias for --select_method")
    p_select.add_argument("--budget", type=int, default=None)
    p_select.add_argument("--seed", type=int, default=None)

    p_finalize = sub.add_parser("finalize", help="Convert detailed selected-subset scores into retraining selected.json")
    p_finalize.add_argument("--config", required=True)
    p_finalize.add_argument("--scored-pool", default=None)
    p_finalize.add_argument("--selected-candidates", default=None)
    p_finalize.add_argument("--output-dir", default=None)
    p_finalize.add_argument("--score_method", "--score-method", dest="score_method", default=None)
    p_finalize.add_argument("--select_method", "--select-method", dest="select_method", default=None)
    p_finalize.add_argument("--method", default=None, help="Backward-compatible alias for --select_method")
    p_finalize.add_argument(
        "--weighting",
        default=None,
        help=(
            "Weighting mode for the finalized manifest. 'none' skips detailed-score lookup; "
            "frame_patch/frame/patch_only require selected-subset detailed scores."
        ),
    )
    p_finalize.add_argument(
        "--include_c3_train_split",
        "--include-c3-train-split",
        dest="include_c3_train_split",
        action="store_true",
        help=(
            "Append manifests.c3_train_split / score_method_train_split to the finalized "
            "selected manifest for anti-forgetting replay. Output files get a "
            f"_{C3_TRAIN_SPLIT_SUFFIX} suffix."
        ),
    )

    args = parser.parse_args()
    if args.cmd == "build":
        build_prescreen_manifest(
            args.config,
            candidate_manifest=args.candidate_manifest,
            output_dir=args.output_dir,
            representatives_per_task=args.representatives_per_task,
            representative_strategy=args.representative_strategy,
            seed=args.seed,
            max_tasks=args.max_tasks,
        )
    elif args.cmd == "select":
        if args.budget is not None and args.budget <= 0:
            parser.error("--budget must be positive")
        select_candidates_from_prescreen(
            args.config,
            candidate_manifest=args.candidate_manifest,
            prescreen_scores=args.prescreen_scores,
            output_dir=args.output_dir,
            method=args.method,
            score_method=args.score_method,
            select_method=args.select_method,
            budget=args.budget,
            seed=args.seed,
        )
    elif args.cmd == "finalize":
        finalize_scored_selection(
            args.config,
            scored_pool=args.scored_pool,
            selected_candidates=args.selected_candidates,
            output_dir=args.output_dir,
            method=args.method,
            score_method=args.score_method,
            select_method=args.select_method,
            weighting=args.weighting,
            include_c3_train_split=args.include_c3_train_split,
        )


if __name__ == "__main__":
    main()
