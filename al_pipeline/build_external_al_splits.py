from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import (
    ensure_dir,
    filter_valid_external_worldmodel_items,
    flatten_manifest_items,
    load_json,
    load_yaml,
    save_json,
)
from tqdm import tqdm


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(items),
        "tasks": len({x.get("task_name", "unknown") for x in items}),
        "robots": dict(sorted(Counter(str(x.get("robot", "unknown")) for x in items).items())),
        "by_task": dict(sorted(Counter(str(x.get("task_name", "unknown")) for x in items).items())),
    }


def build_external_splits(
    *,
    dataset_name: str,
    converted_manifest: str | Path,
    output_dir: str | Path,
    split_mode: str = "episode",
    c3_train_ratio: float | None = None,
    candidate_pool_ratio: float | None = None,
    pool_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    max_episodes: int | None = None,
    min_frames: int = 1,
    require_paths: bool | str = "auto",
) -> dict[str, Any]:
    payload = load_json(converted_manifest)
    raw_items = flatten_manifest_items(payload)
    print(f"[external split] loaded {len(raw_items)} items from {converted_manifest}")
    items = [dict(item, dataset=dataset_name, format=item.get("format", "external_worldmodel")) for item in raw_items]
    items, invalid_items, filter_stats = filter_valid_external_worldmodel_items(
        items,
        min_frames=int(min_frames),
        require_paths=require_paths,
        require_training_files=True,
    )
    if max_episodes is not None:
        items = items[:max_episodes]
    if not items:
        raise ValueError(
            f"No valid converted external episodes found in {converted_manifest}; "
            f"filter_stats={filter_stats}"
        )
    if split_mode not in {"episode", "task"}:
        raise ValueError("split_mode must be episode or task")

    rng = random.Random(random_seed)
    if c3_train_ratio is None and candidate_pool_ratio is None:
        c3_train_ratio = 0.0
        candidate_pool_ratio = pool_ratio
    elif c3_train_ratio is None:
        c3_train_ratio = max(0.0, 1.0 - float(candidate_pool_ratio) - val_ratio - test_ratio)
    elif candidate_pool_ratio is None:
        candidate_pool_ratio = max(0.0, 1.0 - float(c3_train_ratio) - val_ratio - test_ratio)
    c3_train_ratio = float(c3_train_ratio)
    candidate_pool_ratio = float(candidate_pool_ratio)

    split_by_ep: dict[str, str] = {}
    if split_mode == "episode":
        eps = sorted(str(item["episode_id"]) for item in items)
        rng.shuffle(eps)
        n = len(eps)
        n_c3 = max(0, min(n, int(round(n * c3_train_ratio))))
        n_pool = max(0, min(n - n_c3, int(round(n * candidate_pool_ratio))))
        n_val = max(0, min(n - n_c3 - n_pool, int(round(n * val_ratio))))
        for ep in eps[:n_c3]:
            split_by_ep[ep] = "c3_train_split"
        for ep in eps[n_c3 : n_c3 + n_pool]:
            split_by_ep[ep] = "candidate_pool"
        for ep in eps[n_c3 + n_pool : n_c3 + n_pool + n_val]:
            split_by_ep[ep] = "heldout"
        for ep in eps[n_c3 + n_pool + n_val :]:
            split_by_ep[ep] = "heldout"
    else:
        tasks = sorted({str(item.get("task_name", "unknown")) for item in items})
        rng.shuffle(tasks)
        n = len(tasks)
        n_c3 = max(0, min(n, int(round(n * c3_train_ratio))))
        n_pool = max(0, min(n - n_c3, int(round(n * candidate_pool_ratio))))
        n_val = max(0, min(n - n_c3 - n_pool, int(round(n * val_ratio))))
        task_split = {task: "c3_train_split" for task in tasks[:n_c3]}
        task_split.update({task: "candidate_pool" for task in tasks[n_c3 : n_c3 + n_pool]})
        task_split.update({task: "heldout" for task in tasks[n_c3 + n_pool :]})
        for item in items:
            split_by_ep[str(item["episode_id"])] = task_split[str(item.get("task_name", "unknown"))]

    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in tqdm(items, desc="assigning splits", unit="item"):
        split = split_by_ep[str(item["episode_id"])]
        row = dict(item)
        row["split"] = split
        splits[split].append(row)

    output_dir = ensure_dir(output_dir)
    invalid_path = output_dir / "invalid_external_items.json"
    save_json(
        {
            "items": invalid_items,
            "count": len(invalid_items),
            "filter_stats": filter_stats,
            "source_manifest": str(converted_manifest),
        },
        invalid_path,
    )
    for stale_name in [
        "external_c3_train_split.json",
        "external_candidate_pool.json",
        "external_pool.json",
        "external_val.json",
        "external_test.json",
    ]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    # Split heldout into val (first n_val) and test (remainder)
    heldout = splits.get("heldout", [])
    n_val_heldout = max(0, int(round(n * val_ratio)))
    val_items = heldout[:n_val_heldout]
    test_items = heldout[n_val_heldout:]

    paths = {}
    for split, items_for_split in tqdm([
        ("c3_train_split", splits.get("c3_train_split", [])),
        ("candidate_pool", splits.get("candidate_pool", [])),
        ("al_val", val_items),
        ("al_test", test_items),
    ], desc="saving splits", unit="split"):
        path = output_dir / f"{split}.json"
        save_json({"dataset": dataset_name, "split": split, "items": items_for_split, "stats": _summarize(items_for_split)}, path)
        paths[split] = str(path)
        print(f"[external split]   {split}: {len(items_for_split)} episodes")
    summary = {
        "dataset": dataset_name,
        "format": "external_worldmodel",
        "split_mode": split_mode,
        "random_seed": random_seed,
        "ratios": {
            "c3_train_ratio": c3_train_ratio,
            "candidate_pool_ratio": candidate_pool_ratio,
            "fallback_candidate_pool_ratio": pool_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "manifest_paths": paths,
        "source_manifest": str(converted_manifest),
        "source_items": len(raw_items),
        "filter": {
            **filter_stats,
            "invalid_items_path": str(invalid_path) if invalid_items else None,
        },
        "splits": {
            name: _summarize(items_for_split)
            for name, items_for_split in [
                ("c3_train_split", splits.get("c3_train_split", [])),
                ("candidate_pool", splits.get("candidate_pool", [])),
                ("al_val", val_items),
                ("al_test", test_items),
            ]
        },
    }
    save_json(summary, output_dir / "split_summary.json")
    return summary


def build_from_config(config_path: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    project_root = Path(_get(cfg, "project.root_dir", "al_runs"))
    run_name = _get(cfg, "project.run_name", _get(cfg, "run.name", "multiphase_al"))
    run_root = project_root / run_name
    dataset_name = _get(cfg, "external_split.dataset_name", _get(cfg, "datasets.robotwin.name", "robotwin"))
    dataset_cfg = _get(cfg, f"datasets.{dataset_name}", {})
    converted_manifest = _get(cfg, "external_split.converted_manifest") or dataset_cfg.get("converted_manifest")
    if not converted_manifest:
        converted_root = dataset_cfg.get("converted_root")
        if not converted_root:
            raise ValueError("Need external_split.converted_manifest or datasets.<name>.converted_root")
        converted_manifest = str(Path(converted_root) / dataset_name / "manifests" / "all.json")
    split_cfg = _get(cfg, "external_split", {})
    return build_external_splits(
        dataset_name=dataset_name,
        converted_manifest=converted_manifest,
        output_dir=split_cfg.get("output_dir") or str(run_root / "manifests"),
        split_mode=split_cfg.get("mode", "episode"),
        c3_train_ratio=split_cfg.get("c3_train_ratio"),
        candidate_pool_ratio=split_cfg.get("candidate_pool_ratio"),
        pool_ratio=float(split_cfg.get("pool_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        random_seed=int(split_cfg.get("random_seed", cfg.get("seed", 42))),
        max_episodes=split_cfg.get("max_episodes"),
        min_frames=int(split_cfg.get("min_frames", 1)),
        require_paths=split_cfg.get("require_paths", "auto"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RoboTwin/LIBERO external active-learning splits")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = build_from_config(args.config)
    print(f"[external split] {summary['dataset']} paths: {summary['manifest_paths']}")


if __name__ == "__main__":
    main()
