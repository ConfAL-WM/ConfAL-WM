#!/usr/bin/env python3
"""Method-neutral EVAC active-learning retraining launcher."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tqdm import tqdm

from al_pipeline.utils import (
    ensure_dir,
    filter_valid_external_worldmodel_items,
    flatten_manifest_items,
    load_json,
    load_yaml,
    save_json,
)
from al_pipeline.weighting import sample_weight_from_stats

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
C3_TRAIN_SPLIT_SUFFIX = "with_c3_train_split"
WEIGHTING_ALIASES = {
    "none": "none",
    "oversampling": "confidence_guided_oversampling",
    "confidence_guided_oversampling": "confidence_guided_oversampling",
    "sample": "sample_weight",
    "sample_weight": "sample_weight",
    "sample_weighting": "sample_weight",
    # The old patch names now point to the frame+residual-patch default.
    "patch": "frame_patch",
    "patch_weight": "frame_patch",
    "patch_weighting": "frame_patch",
    "loss_map": "frame_patch",
    "frame_patch": "frame_patch",
    "frame_patch_weight": "frame_patch",
    "frame_patch_weighting": "frame_patch",
    "hybrid": "frame_patch",
    # Keep pure patch available only as an explicit ablation.
    "patch_only": "patch_weight",
    "patch_weight_only": "patch_weight",
    "pure_patch": "patch_weight",
    "frame": "frame_weight",
    "frame_weight": "frame_weight",
    "frame_weighting": "frame_weight",
}
LOSS_MAP_WEIGHTINGS = {"patch_weight", "frame_weight", "frame_patch"}
C3_DENSE_WEIGHTINGS = {"patch_weight", "frame_patch"}


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set_nested(cfg: dict[str, Any], path: list[str], value: Any) -> None:
    cur = cfg
    for part in path[:-1]:
        cur = cur.setdefault(part, {})
    cur[path[-1]] = value


def _pop_nested_key(cfg: dict[str, Any], key: str) -> None:
    """Safely pop *key* from *cfg* if it exists."""
    cfg.pop(key, None)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _safe_name(value: Any, default: str) -> str:
    text = str(value or default).strip().lower()
    return text.replace("/", "_").replace(" ", "_")


def _canonical_score_method(value: Any, default: str = "c3") -> str:
    name = _safe_name(value, default)
    return SCORE_METHOD_ALIASES.get(name, name)


def _canonical_weighting(value: Any, default: str = "none") -> str:
    name = _safe_name(value, default)
    resolved = WEIGHTING_ALIASES.get(name)
    if resolved is None:
        valid = ", ".join(sorted(WEIGHTING_ALIASES))
        raise ValueError(f"weighting_mode must be one of: {valid}; got {value!r}")
    return resolved


def _weighting_manifest_name(weighting_mode: str) -> str:
    if weighting_mode == "confidence_guided_oversampling":
        return "oversampling"
    if weighting_mode == "patch_weight":
        return "patch_only"
    if weighting_mode == "frame_weight":
        return "frame"
    return weighting_mode


def _is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
            return True
    except PermissionError:
        print("[al retrain WARNING] Cannot probe master_port availability in this environment; using requested port")
        return True
    except OSError:
        return False


def _resolve_master_port(base_port: int, *, max_tries: int = 100) -> int:
    for offset in range(max(1, int(max_tries))):
        port = int(base_port) + offset
        if _is_port_available(port):
            if offset:
                print(f"[al retrain] master_port {base_port} is busy; using free port {port}")
            return port
    raise RuntimeError(f"No free master_port found in range [{base_port}, {base_port + max_tries - 1}]")


def _load_selected(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"selected manifest must contain an items list: {path}")
    items = payload["items"]
    if not isinstance(items, list) or not items:
        raise ValueError(f"selected manifest has no items: {path}")
    return items



def _source_path_from_selected(item: dict[str, Any]) -> Path:
    meta_path = item.get("meta_path")
    meta = load_json(meta_path) if meta_path and Path(meta_path).exists() else {}
    return Path(meta.get("source_path") or item.get("source_path", ""))


def _source_path_from_replay(item: dict[str, Any]) -> Path:
    return Path(item.get("path") or item.get("source_path") or item.get("episode_dir", ""))


def _build_frame_conf_maps(
    oversampling_rows: list[dict[str, Any]],
    score_method: str,
    scored_pool_dir: Path,
    out_dir: Path,
    default_h: int = 20,
    default_w: int = 32,
) -> list[dict[str, Any]]:
    """Generate frame-uniform conf_map.npy for frame_weight mode.

    C3: spatially-averages existing conf_map.npy → uniform per-frame weights.
    Non-C3 (lrms etc.): reads frame_conf_scores from scored_pool.json. These
    are computed as 1 - max(0, gt_score - pred_score), then tiled into a
    spatially uniform [T, H, W] conf_map.
    """
    import numpy as np

    ensure_dir(out_dir)

    # Load scored pool for non-C3 frame_scores lookup
    scored_by_ep: dict[str, dict[str, Any]] = {}
    if score_method != "c3":
        scored_pool_path = scored_pool_dir / "scored_pool.json"
        if scored_pool_path.exists():
            pool = load_json(str(scored_pool_path))
            for entry in pool.get("items", []):
                ep_id = str(entry.get("episode_id", ""))
                if ep_id:
                    scored_by_ep[ep_id] = entry

    conf_rows: list[dict[str, Any]] = []
    for row in oversampling_rows:
        ep_id = str(row.get("episode_id", ""))
        ep_out_dir = ensure_dir(out_dir / ep_id)
        dst = ep_out_dir / "conf_map.npy"

        if score_method == "c3":
            src_path = row.get("conf_map_path")
            if src_path and Path(src_path).exists():
                conf = np.load(str(src_path))  # [T, H, W]
                frame_w = conf.mean(axis=(1, 2), keepdims=True)  # [T, 1, 1]
                frame_conf = np.broadcast_to(frame_w, conf.shape).copy()
                np.save(str(dst), frame_conf.astype(np.float32))
            else:
                continue
        else:
            entry = scored_by_ep.get(ep_id, {})
            frame_scores = entry.get("frame_conf_scores") or entry.get("frame_scores") or entry.get("baseline_frame_scores") or []
            if not frame_scores:
                # fallback: use episode_score as single frame weight
                ep_score = float(entry.get("episode_score") or entry.get("baseline_episode_score") or 0.5)
                frame_scores = [ep_score]
            frame_w = np.clip(np.array(frame_scores, dtype=np.float32), 0.0, 1.0)
            T = len(frame_w)
            frame_conf = np.broadcast_to(
                frame_w[:, None, None], (T, default_h, default_w)
            ).copy()
            np.save(str(dst), frame_conf.astype(np.float32))

        row_copy = dict(row)
        row_copy["conf_map_path"] = str(dst)
        conf_rows.append(row_copy)

    return conf_rows


def _build_frame_patch_conf_maps(
    oversampling_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate frame+residual-patch C3 maps for frame_patch mode.

    The map is built in confidence space:

        conf_hybrid = frame_mean(conf) + alpha * (conf - frame_mean(conf))

    alpha=0 is pure frame weighting; alpha=1 recovers the old pure patch map.
    """
    import numpy as np

    ensure_dir(out_dir)
    alpha = float(alpha)
    conf_rows: list[dict[str, Any]] = []
    for row in oversampling_rows:
        src_path = row.get("conf_map_path")
        if not src_path or not Path(src_path).exists():
            continue
        ep_id = str(row.get("episode_id", ""))
        ep_out_dir = ensure_dir(out_dir / ep_id)
        dst = ep_out_dir / "conf_map.npy"
        conf = np.load(str(src_path)).astype(np.float32)  # [T, H, W]
        frame_conf = conf.mean(axis=(1, 2), keepdims=True)
        hybrid_conf = frame_conf + alpha * (conf - frame_conf)
        hybrid_conf = np.clip(hybrid_conf, 0.0, 1.0)
        np.save(str(dst), hybrid_conf.astype(np.float32))

        row_copy = dict(row)
        row_copy["conf_map_path"] = str(dst)
        row_copy["frame_patch_alpha"] = alpha
        conf_rows.append(row_copy)

    return conf_rows


def _build_training_metadata(
    replay_items: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    retrain_cfg: dict[str, Any],
    *,
    dataset_name: str,
    selected_source_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build training manifest rows without creating any symlinks.

    Both replay and selected data are read directly from the original
    data root by the DataLoader.  When patch weighting is enabled, conf_map
    matching uses source_path as the lookup key (see _load_conf_map_manifest /
    _lookup_conf_map_path in agibotworld_challenge_dataset.py).
    """
    lambda_conf = float(retrain_cfg.get("lambda_conf", 1.0))
    w_min = float(retrain_cfg.get("weight_clip_min", 1.0))
    w_max = float(retrain_cfg.get("weight_clip_max", 3.0))
    score_key = str(retrain_cfg.get("sample_weight_score_key", "tail_risk_top5"))
    oversample_by_weight = bool(retrain_cfg.get("oversample_by_weight", False))
    max_repeats = max(1, int(retrain_cfg.get("max_oversample_repeats", 3)))

    rows = []
    merged_items = []

    for item in replay_items:
        source_path = _source_path_from_replay(item)
        row = {
            "episode_id": item.get("episode_id", source_path.name),
            "episode_key": item.get("episode_key"),
            "task_id": item.get("task_id"),
            "task_name": item.get("task_name", item.get("task_id")),
            "dataset": item.get("dataset", dataset_name),
            "source": "replay",
            "source_path": str(source_path),
            "sample_weight": 1.0,
            "oversample_repeats": 1,
            "active_learning_guided": False,
            "confidence_guided": False,
        }
        rows.append(row)
        merged_items.append(dict(row))

    for item in tqdm(selected, desc="[al retrain] indexing selected samples", unit="ep", dynamic_ncols=True):
        source_path = _source_path_from_selected(item)
        risk_stats_path = item.get("risk_stats_path")
        risk_stats = load_json(risk_stats_path) if risk_stats_path and Path(risk_stats_path).exists() else item
        weight = sample_weight_from_stats(
            risk_stats,
            score_key=score_key,
            lambda_conf=lambda_conf,
            weight_clip_min=w_min,
            weight_clip_max=w_max,
        )
        repeats = 1
        if oversample_by_weight:
            repeats = max(1, min(max_repeats, int(round(float(weight)))))
        row = {
            "episode_id": item.get("episode_id"),
            "episode_key": item.get("episode_key"),
            "task_id": item.get("task_id"),
            "task_name": item.get("task_name", item.get("task_id")),
            "dataset": item.get("dataset", dataset_name),
            "source": selected_source_label,
            "source_path": str(source_path),
            "active_learning_oversampling_weight": weight,
            "confidence_oversampling_weight": weight,
            "sample_weight": weight,
            "oversample_repeats": repeats,
            "risk_stats_path": item.get("risk_stats_path"),
            "conf_map_path": item.get("conf_map_path"),
            "active_learning_guided": True,
            "confidence_guided": True,
        }
        rows.append(row)
        merged_items.append({
            **row,
            "selected_by": item.get("selection_method"),
            "risk_score": item.get("selection_score", item.get("tail_risk_top5")),
        })
    return rows, merged_items


def prepare_retrain(
    config_path: str,
    *,
    stage: str = "selected",
    launch: bool = False,
    gpus: str | None = None,
    nproc: int | None = None,
    method: str | None = None,
    score_method: str | None = None,
    select_method: str | None = None,
    weighting: str | None = None,
    include_c3_train_split: bool = False,
    patch_frame_hyp: float | None = None,
    save_best_loss: str | None = None,
    save_every: bool | None = None,
    master_port: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Prepare and optionally launch EVAC retraining with active-learning samples.

    Parameters
    ----------
    method:
        Deprecated alias for ``select_method``.
    weighting:
        Override ``retraining.weighting_mode`` from config.  One of
        ``none``, ``oversampling``, ``frame_patch``, ``frame``, ``patch_only``.
        Also used in the default output directory name
        (e.g. ``c3_tail_risk_frame_patch``).
    """
    print(f"[al retrain] loading config: {config_path}")
    cfg = load_yaml(config_path)
    run_name = _get(cfg, "project.run_name", _get(cfg, "run.name", "debug_al"))
    root_dir = _get(cfg, "project.root_dir", _get(cfg, "run.root", "al_runs"))
    dataset_name = _get(cfg, "phase.dataset", "agibot")
    data_format = _get(cfg, "scoring.data_format", "agibot")
    run_root = Path(root_dir) / run_name
    base_retrain_cfg = dict(_get(cfg, "retraining", {}) or {})
    selection_cfg = _get(cfg, "selection", {})
    task_prescreen_cfg = dict(_get(cfg, "task_prescreen", {}) or {})
    if stage not in {"selected", "warmup"}:
        raise ValueError("stage must be 'selected' or 'warmup'")
    _score_method = (
        score_method
        or selection_cfg.get("score_method")
        or task_prescreen_cfg.get("score_method")
        or "c3"
    )
    _select_method = (
        select_method
        or method
        or selection_cfg.get("select_method")
        or selection_cfg.get("method")
        or task_prescreen_cfg.get("selection_method")
        or "tail_risk"
    )
    _score_method = _canonical_score_method(_score_method, "c3")
    _select_method = _safe_name(_select_method, "tail_risk")
    # --- Resolve seed ---
    # When --seed is set, it is forwarded to trainer/trainer.py (overriding its
    # default 123) AND appended to the default retrain output directory name
    # (..._{seed}). When not set, behavior is unchanged: trainer.py keeps its
    # default seed and no suffix is added to the folder name.
    _seed_suffix = f"_{seed}" if seed is not None else ""
    if seed is not None:
        print(f"[al retrain] seed={seed} (overrides trainer default 123; folder gets {_seed_suffix!r} suffix)")
    if stage == "warmup":
        warmup_cfg = dict(_get(cfg, "warmup_retraining", _get(cfg, "warmup", {})) or {})
        retrain_cfg = _deep_update(base_retrain_cfg, warmup_cfg)
        retrain_cfg["train_mode"] = warmup_cfg.get(
            "train_mode",
            "external_replay_only" if data_format == "external_worldmodel" else "replay_only",
        )
        if not warmup_cfg.get("replay_manifest"):
            retrain_cfg["replay_manifest"] = (
                _get(cfg, "manifests.score_method_train_split")
                or _get(cfg, "manifests.c3_train_split")
            )
        if not warmup_cfg.get("output_dir"):
            retrain_cfg["output_dir"] = str(run_root / "retrain" / "evac_v1_warmup")
        if weighting is None:
            retrain_cfg["weighting_mode"] = "none"
        retrain_cfg["selected_manifest"] = None
        print("[al retrain] stage=warmup: training EVAC v1 on score_method_train_split/replay data")
    else:
        retrain_cfg = base_retrain_cfg
        print("[al retrain] stage=selected: training EVAC v2 on the finalized active subset")

    # --- Resolve method ---
    _method = "warmup" if stage == "warmup" and method is None and select_method is None else _select_method
    if stage != "warmup":
        print(f"[al retrain] score_method={_score_method} select_method={_select_method}")
    if method and not select_method:
        print("[al retrain] --method is deprecated; treating it as --select_method")

    # --- Resolve weighting ---
    _weighting = weighting or str(retrain_cfg.get("weighting_mode", "none"))
    if weighting and _weighting != retrain_cfg.get("weighting_mode"):
        print(f"[al retrain] --weighting={_weighting} (overrides config retraining.weighting_mode={retrain_cfg.get('weighting_mode')})")
    elif retrain_cfg.get("weighting_mode"):
        print(f"[al retrain] weighting_mode={_weighting} (from config retraining.weighting_mode)")
    else:
        print(f"[al retrain] weighting_mode={_weighting} (default)")

    requested_weighting_mode = _safe_name(_weighting, "none")
    weighting_mode = _canonical_weighting(requested_weighting_mode)
    weighting_manifest_name = _weighting_manifest_name(weighting_mode)
    if requested_weighting_mode != weighting_manifest_name:
        print(f"[al retrain] weighting alias: {requested_weighting_mode} -> {weighting_manifest_name}")

    train_mode = retrain_cfg.get("train_mode", "replay_plus_selected")
    print(f"[al retrain] method={_method} weighting={weighting_manifest_name} train_mode={train_mode}")
    valid_modes = {
        "selected_only",
        "replay_only",
        "replay_plus_selected",
        "external_selected",
        "external_replay_only",
        "external_replay_plus_selected",
        "simulated_seed_plus_selected",
    }
    if train_mode not in valid_modes:
        raise ValueError(f"retraining.train_mode must be one of {sorted(valid_modes)}")
    needs_selected = train_mode in {
        "selected_only",
        "replay_plus_selected",
        "external_selected",
        "external_replay_plus_selected",
        "simulated_seed_plus_selected",
    }

    # --- Resolve selected manifest ---
    selected_manifest = retrain_cfg.get("selected_manifest")
    if needs_selected:
        if selected_manifest:
            print(f"[al retrain] selected_manifest={selected_manifest} (from config retraining.selected_manifest)")
        else:
            manifest_suffix = f"_{C3_TRAIN_SPLIT_SUFFIX}" if include_c3_train_split else ""
            combo_manifest = run_root / "selection" / f"{_score_method}_{_select_method}" / f"selected{manifest_suffix}.json"
            weighted_manifest = (
                run_root / "selection" / f"{_score_method}_{_select_method}" / f"{weighting_manifest_name}_selected{manifest_suffix}.json"
                if weighting_manifest_name
                else None
            )
            legacy_weighted_manifests = []
            if weighting_manifest_name == "frame_patch":
                legacy_weighted_manifests.extend(
                    [
                        run_root / "selection" / f"{_score_method}_{_select_method}" / f"patch_weight_selected{manifest_suffix}.json",
                        run_root / "selection" / f"{_score_method}_{_select_method}" / f"patch_selected{manifest_suffix}.json",
                    ]
                )
            elif weighting_manifest_name == "patch_only":
                legacy_weighted_manifests.append(
                    run_root / "selection" / f"{_score_method}_{_select_method}" / f"patch_weight_selected{manifest_suffix}.json"
                )
            elif weighting_manifest_name == "frame":
                legacy_weighted_manifests.append(
                    run_root / "selection" / f"{_score_method}_{_select_method}" / f"frame_weight_selected{manifest_suffix}.json"
                )
            legacy_manifest = run_root / "selections" / _method / "selected.json"
            # Prefer weighted manifest (--weighting none → none_selected.json etc.),
            # then combo manifest, then legacy.
            resolved = None
            legacy_candidate = None if include_c3_train_split else legacy_manifest
            for candidate in (weighted_manifest, *legacy_weighted_manifests, combo_manifest, legacy_candidate):
                if candidate is not None and candidate.exists():
                    resolved = candidate
                    break
            selected_manifest = str(resolved) if resolved else str(combo_manifest)
            print(
                f"[al retrain] selected_manifest={selected_manifest} "
                f"(default: selection/{_score_method}_{_select_method}/{'selected' + manifest_suffix + '.json' if not weighted_manifest else weighted_manifest.name})"
            )
    else:
        selected_manifest = None
        print(f"[al retrain] selected_manifest skipped for train_mode={train_mode}")

    if weighting_mode in C3_DENSE_WEIGHTINGS and _score_method != "c3":
        raise ValueError(
            f"weighting={weighting_manifest_name} requires score_method=c3 because dense conf_map.npy files "
            f"are only produced by C3 scoring; got score_method={_score_method!r}. "
            "Use weighting=none, weighting=oversampling, or weighting=frame for external baseline score methods."
        )
    if weighting_mode == "sample_weight":
        print(
            "[al retrain] NOTE: weighting=sample_weight generates per-episode weights "
            "but the current EVAC trainer does not natively consume them. "
            "Falling back to oversampling-based approximation. "
            "Use weighting=oversampling for equivalent behavior."
        )
    base_train_config = retrain_cfg.get("evac_train_config") or _get(cfg, "model.evac_train_config")
    if not base_train_config:
        raise ValueError("Config must define retraining.evac_train_config or model.evac_train_config")
    with open(base_train_config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f) or {}
    train_target = str(_get(train_cfg, "data.params.train.target", ""))
    if train_mode.startswith("external") and "agibot" in train_target.lower() and not bool(
        retrain_cfg.get("allow_external_with_agibot_loader", False)
    ):
        raise NotImplementedError(
            f"{train_mode} prepared external_worldmodel samples, but base train config uses {train_target}. "
            "Add an EVAC-compatible external Dataset adapter or explicitly set "
            "retraining.allow_external_with_agibot_loader=true for a local experiment you know is compatible."
        )

    # Default output dir: retrain/{method}_{weighting_short}/
    _weighting_short = weighting_manifest_name
    _retrain_suffix = f"_{C3_TRAIN_SPLIT_SUFFIX}" if include_c3_train_split and stage != "warmup" else ""
    out_dir = ensure_dir(
        retrain_cfg.get("output_dir")
        or str(
            run_root / "retrain" / (
                f"{_score_method}_{_select_method}_{_weighting_short}{_retrain_suffix}"
                if stage != "warmup"
                else f"{_method}_{_weighting_short}"
            )
        )
    )
    # When --seed is set, append _{seed} to the leaf folder so distinct seeds
    # never collide. This applies uniformly to the auto-derived name AND any
    # config-pinned output_dir (e.g. warmup's evac_v1_warmup -> evac_v1_warmup_42),
    # and yields the requested {score_method}_{select_method}_{weighting}_{seed}
    # layout for the default selected-stage naming.
    if _seed_suffix:
        out_dir = ensure_dir(out_dir.parent / f"{out_dir.name}{_seed_suffix}")
        _out_dir_source = "config output_dir +seed" if retrain_cfg.get("output_dir") else "default +seed"
    else:
        _out_dir_source = "config retraining.output_dir" if retrain_cfg.get("output_dir") else "default"
    print(f"[al retrain] output_dir={out_dir} ({_out_dir_source})")
    selected: list[dict[str, Any]] = []
    if needs_selected:
        print(f"[al retrain] loading selected manifest: {selected_manifest}")
        selected = _load_selected(str(selected_manifest))
        print(f"[al retrain] {len(selected)} selected samples loaded")
    invalid_retrain_items: dict[str, Any] = {}
    if needs_selected and train_mode.startswith("external"):
        selected_for_filter: list[dict[str, Any]] = []
        for item in selected:
            row = dict(item)
            try:
                source_path = _source_path_from_selected(item)
                if str(source_path):
                    row.setdefault("source_path", str(source_path))
                    row.setdefault("episode_dir", str(source_path))
            except Exception as exc:
                row["_source_path_error"] = repr(exc)
            selected_for_filter.append(row)
        selected, invalid_selected, selected_filter_stats = filter_valid_external_worldmodel_items(
            selected_for_filter,
            min_frames=int(retrain_cfg.get("min_valid_frames", 1)),
            require_paths=retrain_cfg.get("require_paths", "auto"),
            require_training_files=True,
        )
        invalid_retrain_items["selected"] = {
            "items": invalid_selected,
            "filter_stats": selected_filter_stats,
        }
        if invalid_selected:
            print(
                f"[al retrain WARNING] filtered {len(invalid_selected)} invalid selected external samples "
                f"before retraining"
            )
        if not selected:
            raise ValueError("No valid selected external samples remain after retraining filter")
    replay_items: list[dict[str, Any]] = []
    replay_manifest = (
        retrain_cfg.get("replay_manifest")
        or _get(cfg, "manifests.score_method_train_split")
        or _get(cfg, "manifests.c3_train_split")
        or _get(cfg, "datasets.agibot.c3_train_manifest")
        or _get(cfg, "datasets.agibot.replay_manifest")
    )
    needs_replay = train_mode in {
        "replay_only",
        "external_replay_only",
        "replay_plus_selected",
        "external_replay_plus_selected",
        "simulated_seed_plus_selected",
    }
    if needs_replay:
        if not replay_manifest:
            raise ValueError(f"{train_mode} requires retraining.replay_manifest or manifests.score_method_train_split")
        print(f"[al retrain] loading replay manifest: {replay_manifest}")
        replay_items = flatten_manifest_items(load_json(replay_manifest))
        if not replay_items:
            raise ValueError(f"replay manifest is empty: {replay_manifest}")
        print(f"[al retrain] {len(replay_items)} replay samples loaded")
        if train_mode.startswith("external"):
            replay_items, invalid_replay, replay_filter_stats = filter_valid_external_worldmodel_items(
                replay_items,
                min_frames=int(retrain_cfg.get("min_valid_frames", 1)),
                require_paths=retrain_cfg.get("require_paths", "auto"),
                require_training_files=True,
            )
            invalid_retrain_items["replay"] = {
                "items": invalid_replay,
                "filter_stats": replay_filter_stats,
            }
            if invalid_replay:
                print(
                    f"[al retrain WARNING] filtered {len(invalid_replay)} invalid replay external samples "
                    f"before retraining"
                )
            if not replay_items:
                raise ValueError("No valid replay external samples remain after retraining filter")
            print(f"[al retrain] {len(replay_items)} valid replay samples after filtering")
    replay_manifest_used = replay_manifest if needs_replay else None
    selected_source_label = "external_selected" if train_mode.startswith("external") else "selected"
    effective_retrain_cfg = dict(retrain_cfg)
    if weighting_mode == "none":
        effective_retrain_cfg["lambda_conf"] = 0.0
    if weighting_mode != "confidence_guided_oversampling":
        effective_retrain_cfg["oversample_by_weight"] = False
    else:
        effective_retrain_cfg["oversample_by_weight"] = True
    print(f"[al retrain] building training metadata (mode={train_mode}, weighting={weighting_mode})")
    invalid_retrain_path = out_dir / "invalid_retraining_items.json"
    save_json(invalid_retrain_items, invalid_retrain_path)
    oversampling_rows, merged_items = _build_training_metadata(
        replay_items,
        selected,
        effective_retrain_cfg,
        dataset_name=dataset_name,
        selected_source_label=selected_source_label,
    )

    manifest_dir = ensure_dir(out_dir / "retrain_manifests")
    merged_manifest_path = manifest_dir / f"{dataset_name}_{_score_method}_{_select_method}_{train_mode}.json"
    save_json(
        {
            "score_method": _score_method,
            "select_method": _select_method,
            "method": _method,
            "weighting": weighting_manifest_name,
            "include_c3_train_split": bool(include_c3_train_split),
            "phase": _get(cfg, "phase.name", ""),
            "dataset": dataset_name,
            "train_mode": train_mode,
            "items": merged_items,
            "stats": {
                "replay_items": len(replay_items),
                "selected_items": len(selected),
                "manifest_items": len(merged_items),
                "invalid_retraining_items_path": str(invalid_retrain_path),
            },
        },
        merged_manifest_path,
    )

    # Derive data_root from replay items: source_path is .../WorldModel/train/{ep},
    # Dataset expects data_root/train/{ep}, so data_root = parent of train/.
    if replay_items:
        data_root = str(_source_path_from_replay(replay_items[0]).parent.parent)
    elif selected:
        data_root = str(_source_path_from_selected(selected[0]).parent.parent)
    else:
        data_root = str(_get(cfg, "data.worldmodel_root", "."))
    print(f"[al retrain] data_root={data_root}")
    _set_nested(
        train_cfg,
        ["data", "params", "train", "params", "data_roots"],
        [data_root],
    )
    _set_nested(
        train_cfg,
        ["data", "params", "train", "params", "split"],
        "train",
    )
    _set_nested(
        train_cfg,
        ["lightning", "trainer", "num_nodes"],
        1,
    )
    # Apply retraining overrides from active learning config (fine-tuning defaults)
    # max_steps is deferred to after nproc is known for multi-GPU scaling
    if retrain_cfg.get("base_learning_rate"):
        _set_nested(train_cfg, ["model", "base_learning_rate"], float(retrain_cfg["base_learning_rate"]))
    # Checkpoint naming + deduplication:
    # - save_last=False: never write last.ckpt. The rolling epoch-step ckpt
    #   (every_n_train_steps) or the periodic every-eval ckpts already capture
    #   the final step; FinalStepCheckpoint guarantees it for every nproc.
    # - filename "epoch={epoch}-step={step}" + auto_insert_metric_name=False:
    #   uniform naming with FinalStepCheckpoint / PeriodicCheckpointCallback so
    #   no duplicate-name files coexist at the same step. auto_insert_metric_name
    #   MUST be False — Lightning's default (True) re-prefixes {epoch}/{step},
    #   producing doubled names like epoch=epoch=8-step=step=4000.ckpt.
    _set_nested(train_cfg, ["lightning", "callbacks", "model_checkpoint", "params", "save_last"], False)
    _set_nested(train_cfg, ["lightning", "callbacks", "model_checkpoint", "params", "filename"], "epoch={epoch}-step={step}")
    _set_nested(train_cfg, ["lightning", "callbacks", "model_checkpoint", "params", "auto_insert_metric_name"], False)
    # Remove every_n_train_steps by default; re-added later only when periodic
    # every-eval saving is OFF (so the base model_checkpoint is the sole saver).
    _pop_nested_key(
        train_cfg.setdefault("lightning", {})
        .setdefault("callbacks", {})
        .setdefault("model_checkpoint", {})
        .setdefault("params", {}),
        "every_n_train_steps",
    )
    # Remove the duplicate metrics_over_trainsteps_checkpoint callback from
    # the base config entirely — we never want two ModelCheckpoint instances.
    train_cfg.setdefault("lightning", {}).setdefault("callbacks", {}).pop(
        "metrics_over_trainsteps_checkpoint", None
    )
    if retrain_cfg.get("batch_frequency"):
        _set_nested(train_cfg, ["lightning", "callbacks", "batch_logger", "params", "batch_frequency"], int(retrain_cfg["batch_frequency"]))
    traj_conditioning_cfg = dict(retrain_cfg.get("traj_conditioning", {}) or {})
    for key in ("traj_gripper_z_offset", "traj_keypoint_scale", "traj_radius"):
        if key in traj_conditioning_cfg:
            value = traj_conditioning_cfg[key]
            _set_nested(train_cfg, ["model", "params", key], value)
            _set_nested(train_cfg, ["data", "params", "train", "params", key], value)
    config_out = out_dir / "config.yaml"
    plan_out = out_dir / "oversampling_plan.json"
    conf_map_manifest_out = out_dir / "active_learning_conf_maps.json"
    _set_nested(
        train_cfg,
        ["data", "params", "train", "params", "manifest_path"],
        str(plan_out),
    )
    patch_weighting_cfg = dict(retrain_cfg.get("patch_weighting", {}) or {})
    frame_patch_alpha = float(
        patch_frame_hyp
        if patch_frame_hyp is not None
        else retrain_cfg.get(
            "patch_frame_hyp",
            retrain_cfg.get("frame_patch_alpha", patch_weighting_cfg.get("frame_patch_alpha", 0.5)),
        )
    )
    if not (0.0 <= frame_patch_alpha <= 1.0):
        raise ValueError(f"--patch_frame_hyp / frame_patch_alpha must be in [0, 1], got {frame_patch_alpha}")
    if weighting_mode == "frame_weight":
        frame_conf_dir = ensure_dir(out_dir / "frame_conf_maps")
        conf_rows = _build_frame_conf_maps(
            oversampling_rows=oversampling_rows,
            score_method=_score_method,
            scored_pool_dir=run_root / "pool_scores" / f"{_score_method}_{_select_method}_selected_scores",
            out_dir=frame_conf_dir,
            default_h=20,
            default_w=32,
        )
        note = "Frame-uniform conf_map (spatially averaged for C3, tiled GT-vs-pred delta confidence for non-C3)."
    elif weighting_mode == "frame_patch":
        frame_patch_dir = ensure_dir(out_dir / "frame_patch_conf_maps")
        conf_rows = _build_frame_patch_conf_maps(
            oversampling_rows=oversampling_rows,
            out_dir=frame_patch_dir,
            alpha=frame_patch_alpha,
        )
        note = (
            "Frame+residual-patch C3 conf_map. "
            f"conf_hybrid = frame_mean(conf) + alpha * (conf - frame_mean(conf)); alpha={frame_patch_alpha:.4f}."
        )
    else:
        conf_rows = [row for row in oversampling_rows if row.get("conf_map_path")]
        note = "Maps original episode source_paths to dense C3 conf_map.npy for patch-level loss weighting. No symlinks needed."

    save_json({"items": conf_rows, "note": note}, conf_map_manifest_out)
    if weighting_mode in LOSS_MAP_WEIGHTINGS and len(conf_rows) < len(oversampling_rows):
        print(
            "[al retrain WARNING] loss-map weighting is enabled but only "
            f"{len(conf_rows)}/{len(oversampling_rows)} training rows have confidence maps. "
            "Rows without maps will follow conf_map_missing behavior."
        )

    if weighting_mode in LOSS_MAP_WEIGHTINGS:
        train_params = train_cfg.setdefault("data", {}).setdefault("params", {}).setdefault("train", {}).setdefault("params", {})
        train_params["enable_conf_map"] = True
        train_params["conf_map_manifest"] = str(conf_map_manifest_out)
        train_params["conf_map_missing"] = str(retrain_cfg.get("conf_map_missing", "ones"))
        train_params["conf_map_time_mode"] = str(retrain_cfg.get("conf_map_time_mode", "head"))
        train_params["conf_map_aligned_sampling"] = bool(retrain_cfg.get("conf_map_aligned_sampling", True))

        patch_cfg = patch_weighting_cfg
        patch_cfg.setdefault("enabled", True)
        patch_cfg.setdefault("lambda_conf", float(retrain_cfg.get("lambda_conf", 1.0)))
        patch_cfg.setdefault("weight_clip_min", float(retrain_cfg.get("patch_weight_clip_min", 0.5)))
        patch_cfg.setdefault("weight_clip_max", float(retrain_cfg.get("patch_weight_clip_max", retrain_cfg.get("weight_clip_max", 3.0))))
        patch_cfg.setdefault("weight_warmup_steps", int(retrain_cfg.get("weight_warmup_steps", 1000)))
        patch_cfg.setdefault("confidence_format", str(retrain_cfg.get("confidence_format", "probability")))
        patch_cfg.setdefault("risk_quantile_low", 0.05)
        patch_cfg.setdefault("risk_quantile_high", 0.95)
        patch_cfg.setdefault("risk_gamma", 1.0)
        patch_cfg.setdefault("preserve_mean", True)
        patch_cfg["weighting_mode"] = weighting_mode
        patch_cfg["frame_patch_alpha"] = frame_patch_alpha if weighting_mode == "frame_patch" else None
        _set_nested(train_cfg, ["model", "params", "patch_weighting_config"], patch_cfg)
    else:
        _set_nested(train_cfg, ["model", "params", "patch_weighting_config"], {"enabled": False})

    resolved_save_every = (
        bool(save_every)
        if save_every is not None
        else bool(retrain_cfg.get("save_every", False))
    )

    train_cfg.setdefault("active_learning", {})
    train_cfg["active_learning"].update(
        {
            "enabled": True,
            "stage": stage,
            "score_method": _score_method,
            "select_method": _select_method,
            "train_mode": train_mode,
            "weighting_mode": weighting_mode,
            "include_c3_train_split": bool(include_c3_train_split),
            "selected_manifest": str(selected_manifest) if selected_manifest else None,
            "replay_manifest": str(replay_manifest_used) if replay_manifest_used else None,
            "merged_retraining_manifest": str(merged_manifest_path),
            "oversampling_plan_path": str(out_dir / "oversampling_plan.json"),
            "conf_map_manifest": str(conf_map_manifest_out),
            "lambda_conf": float(effective_retrain_cfg.get("lambda_conf", 1.0)),
            "weight_clip_min": float(retrain_cfg.get("weight_clip_min", 1.0)),
            "weight_clip_max": float(retrain_cfg.get("weight_clip_max", 3.0)),
            "weight_warmup_steps": int(retrain_cfg.get("weight_warmup_steps", 1000)),
            "frame_patch_alpha": frame_patch_alpha if weighting_mode == "frame_patch" else None,
            "oversample_by_weight": bool(effective_retrain_cfg.get("oversample_by_weight", False)),
            "save_every": bool(resolved_save_every),
            "stop_gradient_c3": weighting_mode in LOSS_MAP_WEIGHTINGS,
            "patch_weighting_status": (
                "enabled; Dataset returns confidence_map and ddpm3d.p_losses() applies stop-grad weights"
                if weighting_mode in LOSS_MAP_WEIGHTINGS
                else "disabled"
            ),
        }
    )

    # --- Resolve nproc before writing config (needed for max_steps scaling) ---
    gpu_ids: list[str] = []
    if gpus is not None:
        gpu_ids = [s.strip() for s in gpus.split(",") if s.strip()]
        if not gpu_ids:
            raise ValueError(f"Invalid --gpus value: {gpus!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        nproc = len(gpu_ids)
        print(f"[al retrain] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (nproc={nproc})")
    elif nproc is not None:
        nproc = nproc
        print(f"[al retrain] nproc_per_node={nproc} (from --nproc_per_node)")
    else:
        nproc = int(retrain_cfg.get("nproc_per_node", 1))
        print(f"[al retrain] nproc_per_node={nproc} (from config)")

    # --- Scale max_steps by GPU count (2-GPU baseline) ---
    base_max_steps = int(retrain_cfg.get("max_steps", 10000))
    scaled_max_steps = max(1, int(base_max_steps * 2 / nproc))
    _set_nested(train_cfg, ["lightning", "trainer", "max_steps"], scaled_max_steps)
    print(f"[al retrain] max_steps={scaled_max_steps} (base={base_max_steps}, nproc={nproc})")

    callbacks_cfg = train_cfg.setdefault("lightning", {}).setdefault("callbacks", {})
    resolved_save_best_loss = (
        save_best_loss
        if save_best_loss is not None
        else retrain_cfg.get("save_best_loss", "train/loss_simple_epoch")
    )
    save_best_loss_metric = None
    if resolved_save_best_loss is not None:
        save_best_loss_text = str(resolved_save_best_loss).strip()
        if save_best_loss_text and save_best_loss_text.lower() not in {"none", "null", "false", "0"}:
            save_best_loss_metric = save_best_loss_text
    if save_best_loss_metric and not resolved_save_every:
        callbacks_cfg["loss_best_checkpoint"] = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": str(out_dir / "logs" / "checkpoints"),
                "filename": "epoch={epoch}-step={step}(best_loss)",
                "monitor": save_best_loss_metric,
                "mode": "min",
                "save_top_k": 1,
                "save_weights_only": True,
                "auto_insert_metric_name": False,
                "verbose": True,
            },
        }
        print(f"[al retrain] loss-best checkpoint enabled: monitor={save_best_loss_metric}")
    elif save_best_loss_metric and resolved_save_every:
        callbacks_cfg.pop("loss_best_checkpoint", None)
        print(
            "[al retrain] loss-best checkpoint merged into every-eval checkpoints: "
            f"monitor={save_best_loss_metric}"
        )
    else:
        callbacks_cfg.pop("loss_best_checkpoint", None)
        print("[al retrain] loss-best checkpoint disabled")

    if resolved_save_every:
        every_eval_cfg = dict(retrain_cfg.get("every_eval", {}) or {})
        default_metrics = "pixel_mae,latent_loss,risk_reduction"
        eval_metrics = str(every_eval_cfg.get("metrics", default_metrics))
        # The base Lightning model_checkpoint is intentionally a NO-OP here:
        # PeriodicCheckpointCallback (below) is the sole saver and writes every
        # periodic ckpt + the final step directly into logs/checkpoints/. We
        # cannot pop model_checkpoint — utils_train merges a default back in —
        # so neutralize it with save_top_k=0 (PL 1.9.5: _save_topk_checkpoint
        # returns early at save_top_k==0) plus save_last=False.
        _set_nested(
            train_cfg,
            ["lightning", "callbacks", "model_checkpoint", "params", "save_top_k"],
            0,
        )
        # Periodic (save-only) checkpoint callback. Evaluation is deferred to
        # post-training: this launcher invokes eval/al_results/eval_periodic_checkpoints.py
        # after torchrun exits (see the launch block below), because the old
        # in-training async eval deadlocked when EWMBench inherited torchrun's
        # WORLD_SIZE.
        callbacks_cfg["every_eval_checkpoint"] = {
            "target": "callbacks.PeriodicCheckpointCallback",
            "params": {
                "output_dir": str(out_dir),
                "checkpoint_dir": str(out_dir / "logs" / "checkpoints"),
                "include_ewmbench": False,
                "max_eval_episodes": int(every_eval_cfg.get("max_eval_episodes", 50)),
                "start_fraction": float(every_eval_cfg.get("start_fraction", 0.75)),
                "interval_fraction": float(every_eval_cfg.get("interval_fraction", 0.025)),
                "metrics": eval_metrics,
                "loss_best_monitor": save_best_loss_metric,
                "loss_best_mode": str(every_eval_cfg.get("loss_best_mode", "min")),
            },
        }
        print(
            "[al retrain] periodic every-eval checkpoint saving enabled "
            "(include_ewmbench=False); checkpoints go to logs/checkpoints/. "
            "Run eval/al_results/eval_periodic_checkpoints.py after training "
            "to build the metric table."
        )
    else:
        callbacks_cfg.pop("every_eval_checkpoint", None)
        # Base model_checkpoint is the sole saver: a rolling epoch-step ckpt
        # (every_n_train_steps, save_top_k=1) plus FinalStepCheckpoint, which
        # guarantees the exact final step. No last.ckpt is written.
        ckpt_every = int(retrain_cfg.get("checkpoint_every", 2000))
        if ckpt_every > 0:
            _set_nested(
                train_cfg,
                ["lightning", "callbacks", "model_checkpoint", "params", "every_n_train_steps"],
                ckpt_every,
            )
        callbacks_cfg["final_step_checkpoint"] = {
            "target": "callbacks.FinalStepCheckpoint",
            "params": {
                "checkpoint_dir": str(out_dir / "logs" / "checkpoints"),
            },
        }
        print(
            "[al retrain] checkpoint saving: rolling epoch-step every "
            f"{ckpt_every} steps + final-step ckpt (no last.ckpt)"
        )

    with config_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False, allow_unicode=True)
    save_json({"items": oversampling_rows}, plan_out)

    if master_port is not None:
        base_master_port = int(master_port)
    elif gpu_ids:
        base_master_port = 29500 + int(gpu_ids[0])
    else:
        base_master_port = int(retrain_cfg.get("master_port", 29500))
    master_port = _resolve_master_port(base_master_port)
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "--node_rank=0",
        f"--master_port={master_port}",
        "trainer/trainer.py",
        "--base",
        str(config_out),
        "--train",
        "--name",
        "logs",
        "--logdir",
        str(out_dir),
        "--devices",
        str(nproc),
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    summary = {
        "stage": stage,
        "score_method": _score_method,
        "select_method": _select_method,
        "method": _method,
        "selected_manifest": str(selected_manifest) if selected_manifest else None,
        "replay_manifest": str(replay_manifest_used) if replay_manifest_used else None,
        "merged_retraining_manifest": str(merged_manifest_path),
        "output_dir": str(out_dir),
        "data_root": data_root,
        "phase": _get(cfg, "phase.name", ""),
        "dataset": dataset_name,
        "train_config": str(config_out),
        "oversampling_plan": str(plan_out),
        "conf_map_manifest": str(conf_map_manifest_out),
        "train_mode": train_mode,
        "weighting_mode": weighting_mode,
        "weighting": weighting_manifest_name,
        "frame_patch_alpha": frame_patch_alpha if weighting_mode == "frame_patch" else None,
        "include_c3_train_split": bool(include_c3_train_split),
        "save_best_loss": save_best_loss_metric,
        "save_every": bool(resolved_save_every),
        "master_port": master_port,
        "seed": seed,
        "launch_command": command,
        "caveat": (
            "No symlinks: DataLoader reads directly from the original data_root. "
            "conf_map matching uses episode source_path as the lookup key."
        ),
    }
    save_json(summary, out_dir / "retrain_summary.json")
    print(f"[al retrain] prepared config: {config_out}")
    print(f"[al retrain] data root: {data_root}")
    print(f"[al retrain] merged manifest: {merged_manifest_path}")
    print("[al retrain] launch command:")
    print(" ".join(command))

    if launch:
        subprocess.run(command, cwd=str(_REPO), check=True, env=os.environ.copy())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/launch EVAC retraining with active-learning samples")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["selected", "warmup"],
        default="selected",
        help="warmup trains EVAC v1 on score_method_train_split/replay only; selected trains EVAC v2 on finalized AL samples.",
    )
    parser.add_argument("--launch", action="store_true", help="Actually invoke trainer/trainer.py after preparing files")
    parser.add_argument("--gpus", type=str, default=None, metavar="0,1,2,3",
                        help="Comma-separated GPU IDs (sets CUDA_VISIBLE_DEVICES + nproc_per_node)")
    parser.add_argument("--nproc_per_node", type=int, default=None,
                        help="Override nproc_per_node (default: auto from --gpus count)")
    parser.add_argument("--method", type=str, default=None,
                        help="Deprecated alias for --select_method.")
    parser.add_argument("--score_method", "--score-method", dest="score_method", type=str, default=None,
                        help="Score method name, e.g. c3, random, robometer_prog.")
    parser.add_argument("--select_method", "--select-method", dest="select_method", type=str, default=None,
                        help="Selection method name, e.g. tail_risk, random. "
                             "Determines selection/{score_method}_{select_method}/selected.json.")
    parser.add_argument("--weighting", type=str, default=None,
                        help="Weighting mode: none | frame_patch | frame | patch_only | oversampling "
                             "(default: from config retraining.weighting_mode). "
                             "Legacy patch/patch_weight aliases now resolve to frame_patch.")
    parser.add_argument("--patch_frame_hyp", "--patch-frame-hyp", "--frame-patch-alpha",
                        dest="patch_frame_hyp", type=float, default=None,
                        help="Residual patch alpha for weighting=frame_patch. "
                             "0=pure frame, 1=old pure patch, default=0.5.")
    parser.add_argument("--save_best_loss", "--save-best-loss", dest="save_best_loss",
                        default=None,
                        help="Monitor metric for loss-best checkpoint. Use 'None' to disable. "
                             "With --save_every, the best marker is merged into the periodic ckpt filename.")
    parser.add_argument("--save_every", "--save-every",
                        dest="save_every", action="store_true", default=None,
                        help="Save periodic checkpoints (after 75%% of training) directly into logs/checkpoints/. "
                             "No EWMBench eval is run; do it manually via eval/al_results/eval_periodic_checkpoints.py.")
    parser.add_argument(
        "--include_c3_train_split",
        "--include-c3-train-split",
        dest="include_c3_train_split",
        action="store_true",
        help=(
            "Use the finalized selected manifest with the "
            f"_{C3_TRAIN_SPLIT_SUFFIX} suffix and add the same suffix to the "
            "default retrain output directory."
        ),
    )
    parser.add_argument("--master_port", "--master-port", dest="master_port", type=int, default=None,
                        help="Base torchrun master port. If busy, the launcher uses the next free port.")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed forwarded to trainer/trainer.py (overrides its default 123). "
                             "When set, the default retrain output dir is also suffixed with _{seed} "
                             "(e.g. c3_persistent_risk_frame_patch_42). When unset, behavior is unchanged: "
                             "trainer.py uses its default seed and no suffix is added.")
    args = parser.parse_args()
    prepare_retrain(
        args.config,
        stage=args.stage,
        launch=args.launch,
        gpus=args.gpus,
        nproc=args.nproc_per_node,
        method=args.method,
        score_method=args.score_method,
        select_method=args.select_method,
        weighting=args.weighting,
        include_c3_train_split=args.include_c3_train_split,
        patch_frame_hyp=args.patch_frame_hyp,
        save_best_loss=args.save_best_loss,
        save_every=args.save_every,
        master_port=args.master_port,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
