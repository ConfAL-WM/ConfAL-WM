from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from al_pipeline.sample_selectors import (
    COMPOSITE_TERMS,
    SELECTOR_REGISTRY,
    selected_cluster_counts,
    selected_task_counts,
)
from al_pipeline.utils import (
    flatten_manifest_items,
    load_json,
    load_yaml,
    save_json,
    score_distribution,
    write_csv,
)


RISK_KEYS = ("tail_risk_top5", "tail_risk_top10", "mean_risk", "persistent_risk", "risk_area")
REQUIRED_PATH_FIELDS = (
    "cache_dir",
    "pred_frames_dir",
    "conf_map_path",
    "actions_path",
    "hdec_embedding_path",
    "risk_stats_path",
    "meta_path",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _path_exists(value: Any) -> bool:
    return bool(value) and Path(str(value)).exists()


def _has_prediction_frames(path: Any) -> bool:
    if not path:
        return False
    pred_dir = Path(str(path))
    if not pred_dir.is_dir():
        return False
    return any(p.suffix.lower() in IMAGE_EXTS for p in pred_dir.iterdir())


def _npy_shape(path: Any) -> tuple[int, ...] | None:
    if not path:
        return None
    try:
        return tuple(np.load(str(path), mmap_mode="r").shape)
    except Exception:
        return None


def _fill_path_if_missing(item: dict[str, Any], key: str, path: Path) -> None:
    if not item.get(key) and path.exists():
        item[key] = str(path)


def _hydrate_scored_item(item: dict[str, Any], pool_dir: Path) -> dict[str, Any]:
    """Recover standard scored-pool fields from an episode cache directory.

    This keeps selection robust when scored_pool.json was manually rebuilt with
    only partial rows while the per-episode pool_scores/{episode_id}/ cache is
    still intact.
    """
    row = dict(item)
    ep_id = str(row.get("episode_id") or row.get("ep_id") or "")
    cache_dir = Path(str(row.get("cache_dir"))) if row.get("cache_dir") else None
    if (cache_dir is None or not cache_dir.exists()) and ep_id:
        cache_dir = pool_dir / ep_id
    if cache_dir is None:
        return row
    row.setdefault("cache_dir", str(cache_dir))

    _fill_path_if_missing(row, "pred_frames_dir", cache_dir / "pred_frames")
    _fill_path_if_missing(row, "conf_map_path", cache_dir / "conf_map.npy")
    _fill_path_if_missing(row, "actions_path", cache_dir / "actions.npy")
    _fill_path_if_missing(row, "hdec_embedding_path", cache_dir / "hdec_embedding.npy")
    _fill_path_if_missing(row, "risk_stats_path", cache_dir / "risk_stats.json")
    _fill_path_if_missing(row, "meta_path", cache_dir / "meta.json")
    _fill_path_if_missing(row, "latent_pred_path", cache_dir / "latent_pred.npy")
    _fill_path_if_missing(row, "latent_gt_path", cache_dir / "latent_gt.npy")

    risk_path = row.get("risk_stats_path")
    if risk_path and Path(str(risk_path)).exists():
        try:
            risk_stats = load_json(str(risk_path))
            for key in RISK_KEYS:
                if key not in row and key in risk_stats:
                    row[key] = risk_stats[key]
        except Exception:
            pass

    meta_path = row.get("meta_path")
    if meta_path and Path(str(meta_path)).exists():
        try:
            meta = load_json(str(meta_path))
            for key in ("episode_id", "task_id", "task_name", "dataset", "data_format", "raw_episode_id", "segment_id"):
                if not row.get(key) and meta.get(key) is not None:
                    row[key] = meta[key]
        except Exception:
            pass

    return row


def _validate_scored_item(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    ep_id = item.get("episode_id")
    if not ep_id:
        reasons.append("missing episode_id")
    if item.get("score_ready", True) is not True:
        reasons.append("score_ready is false")

    for key in RISK_KEYS:
        if not _is_finite_number(item.get(key)):
            reasons.append(f"missing/non-finite {key}")

    for key in REQUIRED_PATH_FIELDS:
        if not _path_exists(item.get(key)):
            reasons.append(f"missing path {key}")

    if item.get("pred_frames_dir") and not _has_prediction_frames(item.get("pred_frames_dir")):
        reasons.append("pred_frames_dir has no image frames")

    conf_shape = _npy_shape(item.get("conf_map_path"))
    if conf_shape is not None and len(conf_shape) != 3:
        reasons.append(f"conf_map shape is {conf_shape}, expected [T,H,W]")

    action_shape = _npy_shape(item.get("actions_path"))
    if action_shape is not None and (len(action_shape) != 2 or action_shape[0] <= 0):
        reasons.append(f"actions shape is {action_shape}, expected [T,D]")

    emb_path = item.get("hdec_embedding_path")
    if emb_path and Path(str(emb_path)).exists():
        try:
            emb = np.load(str(emb_path)).astype(np.float32).reshape(-1)
            if emb.size == 0:
                reasons.append("hdec_embedding is empty")
            elif not bool(np.isfinite(emb).all()):
                reasons.append("hdec_embedding contains non-finite values")
        except Exception as exc:
            reasons.append(f"hdec_embedding load failed: {exc!r}")

    for key in ("latent_pred_path", "latent_gt_path"):
        path = item.get(key)
        if path and not Path(str(path)).exists():
            reasons.append(f"missing path {key}")

    return reasons


def _filter_valid_scored_items(items: list[dict[str, Any]], pool_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for raw_item in tqdm(items, desc="[al selection] validating scored items", unit="ep", dynamic_ncols=True):
        item = _hydrate_scored_item(raw_item, pool_dir)
        reasons = _validate_scored_item(item)
        if reasons:
            invalid.append(
                {
                    "episode_id": item.get("episode_id"),
                    "task_id": item.get("task_id"),
                    "cache_dir": item.get("cache_dir"),
                    "reasons": reasons,
                }
            )
        else:
            valid.append(item)
    return valid, invalid


def _invalid_reason_counts(invalid_items: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in invalid_items:
        counts.update(str(reason) for reason in item.get("reasons", []))
    return dict(counts.most_common())


def _load_available_scored_items(scored_pool_path: str) -> tuple[list[dict[str, Any]], str, int]:
    """Load scored items, supporting partially-scored pools via shard files.

    Tries the merged ``scored_pool.json`` first.  When that doesn't exist
    (scoring is still in progress or was interrupted), assembles the pool
    from individual ``scored_pool_*.json`` shard files on the fly.

    Returns (items, source_description, raw_item_count) where *source_description* is a
    human-readable string suitable for the summary.
    """
    pool_path = Path(scored_pool_path)
    pool_dir = pool_path.parent
    if pool_path.exists():
        payload = load_json(str(pool_path))
        items = [row for row in flatten_manifest_items(payload) if row.get("score_ready", True)]
        return items, str(pool_path), len(items)

    # Merged file not found — assemble from individual shard/worker files
    output_dir = pool_dir
    shard_files = sorted(output_dir.glob("scored_pool_*.json"))
    if not shard_files:
        raise FileNotFoundError(
            f"No scored pool found at {pool_path} or as shard files in {output_dir}"
        )

    items: list[dict[str, Any]] = []
    for sf in shard_files:
        payload = load_json(str(sf))
        items.extend(
            row for row in flatten_manifest_items(payload) if row.get("score_ready", True)
        )
    source_desc = f"{len(shard_files)} shard files in {output_dir}"
    print(
        f"[al selection] merged scored_pool.json not found — "
        f"assembled {len(items)} scored items from {len(shard_files)} shard file(s)"
    )
    return items, source_desc, len(items)


def _resolve_budget(selection_cfg: dict[str, Any], n_items: int, budget_override: int | None = None) -> tuple[int, str]:
    if budget_override is not None:
        budget = int(budget_override)
        source = "cli"
    elif selection_cfg.get("budget") is not None:
        budget = int(selection_cfg["budget"])
        source = "config_budget"
    else:
        ratio = float(selection_cfg.get("budget_ratio", 0.1))
        budget = int(np.ceil(n_items * ratio))
        source = "config_budget_ratio"
    return max(0, min(n_items, budget)), source


def run_selection(
    config_path: str,
    *,
    method_override: str | None = None,
    budget_override: int | None = None,
    scored_pool_override: str | None = None,
    output_dir_override: str | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    run_name = _get(cfg, "project.run_name", _get(cfg, "run.name", "debug_al"))
    root_dir = _get(cfg, "project.root_dir", _get(cfg, "run.root", "al_runs"))
    dataset_name = _get(cfg, "phase.dataset", "agibot")
    data_format = _get(cfg, "evac.data_format", _get(cfg, "scoring.data_format", "agibot"))
    run_root = Path(root_dir) / run_name
    selection_cfg = _get(cfg, "selection", {})
    method = method_override or selection_cfg.get("method", "diverse")
    if method not in SELECTOR_REGISTRY:
        raise ValueError(f"Unknown selector {method!r}. Available: {sorted(SELECTOR_REGISTRY)}")

    scored_pool = scored_pool_override or selection_cfg.get("scored_pool") or str(run_root / "pool_scores" / "scored_pool.json")
    out_dir = Path(output_dir_override or selection_cfg.get("output_dir") or str(run_root / "selections" / method))
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / "selected.json"
    summary_path = out_dir / "selection_summary.json"
    csv_path = out_dir / "selection_scores.csv"
    invalid_path = out_dir / "invalid_scored_items.json"

    items_raw, scored_source, raw_item_count = _load_available_scored_items(scored_pool)
    scored_pool_dir = Path(scored_pool).parent
    items, invalid_items = _filter_valid_scored_items(items_raw, scored_pool_dir)
    invalid_reason_counts = _invalid_reason_counts(invalid_items)
    save_json(
        {
            "items": invalid_items,
            "count": len(invalid_items),
            "reason_counts": invalid_reason_counts,
            "source_scored_pool": scored_source,
        },
        invalid_path,
    )
    if invalid_items:
        print(
            f"[al selection WARNING] filtered {len(invalid_items)} invalid/problem scored items "
            f"from {raw_item_count} score-ready rows"
        )
        preview = ", ".join(f"{k}={v}" for k, v in list(invalid_reason_counts.items())[:5])
        if preview:
            print(f"  invalid reasons: {preview}")
        print(f"  invalid : {invalid_path}")
    if not items:
        raise ValueError(f"No valid score-ready items found in {scored_pool}")
    budget, budget_source = _resolve_budget(selection_cfg, raw_item_count, budget_override=budget_override)
    if budget <= 0:
        raise ValueError(f"Selection budget resolved to {budget}; check budget/budget_ratio")
    if budget > len(items):
        print(
            f"[al selection WARNING] budget={budget} exceeds valid items={len(items)} after filtering; "
            f"capping budget to {len(items)}"
        )
        budget = len(items)

    selector_cfg = dict(selection_cfg.get("params", {}))
    selector_cfg.update(selection_cfg.get(method, {}))
    selector_cfg.setdefault("score_normalization", selection_cfg.get("score_normalization", "rank"))
    selector_cfg.setdefault("show_progress", True)
    seed = int(selection_cfg.get("random_seed", cfg.get("seed", 42)))
    selector = SELECTOR_REGISTRY[method](selector_cfg, seed=seed)
    selected, score_rows = selector.select(items, budget)

    selected_ids = [row["episode_id"] for row in selected]
    summary = {
        "method": method,
        "selection_method": method,
        "phase": _get(cfg, "phase.name", ""),
        "dataset": dataset_name,
        "data_format": data_format,
        "budget": budget,
        "budget_source": budget_source,
        "budget_override": budget_override,
        "budget_base_count": raw_item_count,
        "source_scored_pool": scored_source,
        "valid_score_ready_items": len(items),
        "invalid_score_ready_items": len(invalid_items),
        "invalid_reason_counts": invalid_reason_counts,
        "invalid_items_path": str(invalid_path) if invalid_items else None,
        "selected_episode_ids": selected_ids,
        "score_normalization": selector_cfg.get("score_normalization", "rank"),
        "score_distribution": score_distribution([float(r.get("selection_score", 0.0)) for r in score_rows]),
        "score_component_distributions": {
            "raw": {
                term: score_distribution([float(r.get(f"raw_{term}", r.get(term, 0.0))) for r in score_rows])
                for term in COMPOSITE_TERMS
            },
            "normalized": {
                term: score_distribution([float(r.get(f"norm_{term}", 0.0)) for r in score_rows])
                for term in COMPOSITE_TERMS
            },
        },
        "selected_task_counts": selected_task_counts(selected),
        "selected_cluster_counts": selected_cluster_counts(selected),
        "selected_metric_means": {
            key: float(np.mean([float(r.get(key, 0.0)) for r in selected])) if selected else 0.0
            for key in ["tail_risk_top5", "tail_risk_top10", "mean_risk", "persistent_risk", "risk_area"]
        },
        "config": selector_cfg,
        "protocol": {
            "no_future_gt_for_selection": True,
            "risk_definition": "risk = 1 - confidence",
        },
    }
    selected_payload = {
        "phase": _get(cfg, "phase.name", ""),
        "dataset": dataset_name,
        "data_format": data_format,
        "method": method,
        "selection_method": method,
        "budget": budget,
        "items": selected,
        "summary_path": str(summary_path),
    }
    save_json(selected_payload, selected_path)
    save_json(summary, summary_path)
    write_csv(score_rows, csv_path)
    print(f"[al selection] method={method} selected={len(selected)}/{len(items)} valid ({raw_item_count} score-ready)")
    print(f"  selected: {selected_path}")
    print(f"  summary : {summary_path}")
    print(f"  scores  : {csv_path}")
    if invalid_items:
        print(f"  invalid : {invalid_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Select active-learning samples from C3-scored pool")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default=None, help="Override selection.method from config")
    parser.add_argument("--budget", type=int, default=None, help="Override selection budget with an explicit number of valid scored samples")
    parser.add_argument("--scored-pool", default=None, help="Override selection.scored_pool")
    parser.add_argument("--output-dir", default=None, help="Override selection.output_dir")
    args = parser.parse_args()
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be a positive integer")
    run_selection(
        args.config,
        method_override=args.method,
        budget_override=args.budget,
        scored_pool_override=args.scored_pool,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
