from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable

import yaml

from al_pipeline.utils import save_json
from eval.al_results.utils.video_utils import load_video_frames


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
EWMBENCH_TRAJ_MIN_VALID_RATIO = 0.20
# Optional (old_prefix, new_prefix) pairs for rewriting stale absolute paths
# recorded by EWMBench cluster configs from a previous server layout.
CLUSTER_PATH_REWRITES: tuple[tuple[str, str], ...] = ()


FindSourceFn = Callable[[Path, dict[str, Any]], Path | None]


def _normalize_cluster_paths(value: Any) -> Any:
    """Rewrite stale absolute paths from older training-server layouts."""
    if isinstance(value, str):
        out = value
        for old, new in CLUSTER_PATH_REWRITES:
            if out.startswith(old):
                out = new + out[len(old):]
        return out
    if isinstance(value, dict):
        return {key: _normalize_cluster_paths(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_cluster_paths(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_normalize_cluster_paths(val) for val in value)
    return value


_CV2_STUB_SOURCE = r'''
from __future__ import annotations

import numpy as np
from PIL import Image

__version__ = "evac-c3-pillow-fallback"
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3
INTER_LANCZOS4 = 4


def _uint8_image(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def resize(img, dsize, interpolation=INTER_LINEAR):
    width, height = int(dsize[0]), int(dsize[1])
    resampling = getattr(Image, "Resampling", Image)
    if interpolation == INTER_NEAREST:
        resample = resampling.NEAREST
    elif interpolation == INTER_CUBIC:
        resample = resampling.BICUBIC
    elif interpolation == INTER_LANCZOS4:
        resample = resampling.LANCZOS
    else:
        resample = resampling.BILINEAR
    return np.asarray(Image.fromarray(_uint8_image(img)).resize((width, height), resample=resample))


def setNumThreads(_num_threads):
    return None


def getBuildInformation():
    return "ConfAL-WM Pillow-backed cv2 fallback"
'''


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    raise TypeError(f"Expected a string/list for EWMBench dimensions, got {type(value)}")


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    return base / path


def _short_error(text: str, max_chars: int = 1200) -> str:
    text = str(text or "").strip()
    return text if len(text) <= max_chars else text[-max_chars:]


def _check_python_cv2_import(python_bin: str | Path, env: dict[str, str] | None = None) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [str(python_bin), "-c", "import cv2; print(getattr(cv2, '__version__', 'unknown'))"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=45,
        )
    except Exception as exc:
        return False, repr(exc)
    return proc.returncode == 0, _short_error(proc.stdout)


def _cv2_error_is_import_related(message: str) -> bool:
    msg = str(message)
    return (
        "cv2" in msg
        or "libGL" in msg
        or "No module named" in msg
        or "ImportError" in msg
        or "ModuleNotFoundError" in msg
    ) and "FileNotFoundError" not in msg


def _write_cv2_stub_dir(stub_dir: Path) -> Path:
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_path = stub_dir / "cv2.py"
    stub_path.write_text(_CV2_STUB_SOURCE, encoding="utf-8")
    return stub_path


def _install_cv2_stub_module() -> None:
    sys.modules.pop("cv2", None)
    module = types.ModuleType("cv2")
    exec(_CV2_STUB_SOURCE, module.__dict__)
    sys.modules["cv2"] = module


def _image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _count_image_frames(path: Path) -> int | None:
    files = _image_files(path)
    return len(files) if files else None


def _has_png_frames(path: Path) -> bool:
    return any(p.suffix.lower() == ".png" and p.name.startswith("frame_") for p in _image_files(path))


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _reset_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        _reset_path(path)
    path.mkdir(parents=True, exist_ok=True)


def _symlink(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        _reset_path(dest)
    os.symlink(source.resolve(), dest, target_is_directory=source.is_dir())


def _link_image_dir(
    source: Path,
    dest: Path,
    *,
    role: str,
    episode_id: str,
    force_png_names: bool = False,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> None:
    """Link a directory of frames into EWMBench's expected ``video/`` path.

    EWMBench's PSNR/SSIM code only scans GT files named ``frame_*.png``.
    When GT frames are JPGs (e.g. RoboTwin), create per-frame symlinks with
    PNG names. This avoids copying frames while keeping EWMBench untouched.
    """
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        raise ValueError(
            f"EWMBench expects an image directory for {role} episode {episode_id}, "
            f"but got a video file: {source}"
        )
    files = _image_files(source)
    if not files:
        raise FileNotFoundError(f"No image frames found for {role} episode {episode_id}: {source}")
    if start_frame > 0:
        files = files[start_frame:]
    if max_frames is not None:
        files = files[:max_frames]

    if max_frames is None and (not force_png_names or _has_png_frames(source)):
        _symlink(source, dest)
        return

    _reset_dir(dest)
    for idx, src in enumerate(files):
        # PIL decodes by file content, so a .png symlink to a .jpg file is OK.
        suffix = ".png" if force_png_names else src.suffix.lower()
        link_name = f"frame_{idx:05d}{suffix}"
        _symlink(src, dest / link_name)


def _decode_video_to_dir(
    source: Path,
    dest: Path,
    *,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> int:
    """Decode a video file into an EWMBench image-frame directory."""
    from PIL import Image

    _reset_dir(dest)
    read_max = None if max_frames is None else max_frames + max(0, start_frame)
    frames = load_video_frames(source, max_frames=read_max)
    if start_frame > 0:
        frames = frames[start_frame:]
    if max_frames is not None:
        frames = frames[:max_frames]
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(dest / f"frame_{idx:05d}.png")
    return int(frames.shape[0])


def _safe_component(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    if not text:
        text = fallback
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _tree_ids(ep: dict[str, Any], task_map: dict[str, str]) -> tuple[str, str]:
    """Return EWMBench-compatible ``task_id`` and ``episode_id``.

    EWMBench's CSV merger casts task_id to int, so non-numeric external task
    names are mapped to stable numeric ids and written to mapping.json.
    """
    ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
    task_raw = ep.get("task_id") or ep.get("task_name") or ep.get("dataset") or "task"
    task_key = str(task_raw)
    if task_key.isdigit():
        task_id = task_key
    else:
        task_id = task_map.setdefault(task_key, str(len(task_map) + 1))

    raw_ep = ep.get("raw_episode_id")
    seg = ep.get("segment_id")
    if str(task_raw).isdigit() and raw_ep is not None and seg is not None:
        episode_id = f"{raw_ep}{seg}"
    else:
        episode_id = _safe_component(ep_id, f"episode_{len(task_map)}")
    return task_id, episode_id


def _link_optional_dir(
    source: Path,
    dest: Path,
    *,
    required_file: str | None = None,
    require_eval_ready_traj: bool = False,
) -> bool:
    if source.is_dir():
        if required_file is not None and not _valid_traj_file(
            source / required_file,
            require_eval_ready=require_eval_ready_traj,
        ):
            return False
        _symlink(source, dest)
        return True
    return False


def _prepare_adapter_tree(
    *,
    episodes: list[dict[str, Any]],
    pred_root: Path,
    gt_root: Path,
    adapter_root: Path,
    dimensions: list[str],
    find_pred_source: FindSourceFn,
    find_gt_source: FindSourceFn,
    decode_video_files: bool = True,
    gt_start_frame: int = 0,
) -> dict[str, Any]:
    pred_base = adapter_root / "evac_c3_dataset"
    gt_base = adapter_root / "gt_dataset"
    _reset_dir(adapter_root)
    pred_base.mkdir(parents=True, exist_ok=True)
    gt_base.mkdir(parents=True, exist_ok=True)

    needs_png_gt = any(dim in {"psnr", "ssim", "psnr_ssim"} for dim in dimensions)
    needs_traj = "trajectory_consistency" in dimensions
    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    task_map: dict[str, str] = {}

    for ep in episodes:
        original_ep_id = str(ep.get("episode_id") or ep.get("ep_id") or ep.get("folder") or "")
        pred_source = find_pred_source(pred_root, ep)
        gt_source = find_gt_source(gt_root, ep)
        if pred_source is None:
            skipped.append({"episode_id": original_ep_id, "reason": "missing_pred"})
            continue
        if gt_source is None:
            skipped.append({"episode_id": original_ep_id, "reason": "missing_gt"})
            continue
        if pred_source.is_file() and not decode_video_files:
            skipped.append({"episode_id": original_ep_id, "reason": "pred_not_image_dir", "path": str(pred_source)})
            continue
        if gt_source.is_file() and not decode_video_files:
            skipped.append({"episode_id": original_ep_id, "reason": "gt_not_image_dir", "path": str(gt_source)})
            continue

        task_id, episode_id = _tree_ids(ep, task_map)
        pred_ep_dir = pred_base / task_id / episode_id / "1"
        gt_ep_dir = gt_base / task_id / episode_id
        pred_video_dir = pred_ep_dir / "video"
        gt_video_dir = gt_ep_dir / "video"
        pred_frame_count = _count_image_frames(pred_source) if pred_source.is_dir() else None
        try:
            if pred_source.is_file():
                pred_frame_count = _decode_video_to_dir(pred_source, pred_video_dir)
            else:
                _link_image_dir(
                    pred_source,
                    pred_video_dir,
                    role="prediction",
                    episode_id=original_ep_id,
                    force_png_names=False,
                )
            if gt_source.is_file():
                _decode_video_to_dir(
                    gt_source,
                    gt_video_dir,
                    max_frames=pred_frame_count,
                    start_frame=gt_start_frame,
                )
            else:
                _link_image_dir(
                    gt_source,
                    gt_video_dir,
                    role="ground truth",
                    episode_id=original_ep_id,
                    force_png_names=needs_png_gt,
                    max_frames=pred_frame_count,
                    start_frame=gt_start_frame,
                )
        except Exception as exc:
            skipped.append({"episode_id": original_ep_id, "reason": "link_or_decode_failed", "error": str(exc)})
            if pred_ep_dir.exists():
                _reset_path(pred_ep_dir)
            if gt_ep_dir.exists():
                _reset_path(gt_ep_dir)
            continue

        pred_episode_root = pred_source.parent if pred_source.name in {"pred_frames", "frames", "video"} else pred_source
        gt_episode_root = gt_source.parent if gt_source.name in {"frames", "video"} else gt_source
        pred_traj = _link_optional_dir(pred_episode_root / "traj", pred_ep_dir / "traj", required_file="traj.npy")
        gt_traj = _link_optional_dir(
            gt_episode_root / "traj",
            gt_ep_dir / "traj",
            required_file="traj.npy",
            require_eval_ready_traj=True,
        )
        pred_gripper = _link_optional_dir(pred_episode_root / "gripper_detection", pred_ep_dir / "gripper_detection")
        gt_gripper = _link_optional_dir(gt_episode_root / "gripper_detection", gt_ep_dir / "gripper_detection")

        pairs.append(
            {
                "episode_id": original_ep_id,
                "ewmbench_task_id": task_id,
                "ewmbench_episode_id": episode_id,
                "trial_id": "1",
                "pred_source": str(pred_source),
                "gt_source": str(gt_source),
                "has_pred_traj": pred_traj,
                "has_gt_traj": gt_traj,
                "has_pred_gripper_detection": pred_gripper,
                "has_gt_gripper_detection": gt_gripper,
            }
        )

    if not pairs:
        preview = skipped[:10]
        raise ValueError(
            "No EWMBench-compatible eval pairs were found. Upstream EWMBench "
            "requires image-frame directories under video/. Enable "
            "ewmbench.decode_video_files or provide image-frame GT. "
            f"pred_root={pred_root}, gt_root={gt_root}, "
            f"decode_video_files={decode_video_files}, skipped preview: {preview}"
        )

    if needs_traj:
        missing_traj = [
            p["episode_id"]
            for p in pairs
            if not bool(p["has_pred_traj"]) or not bool(p["has_gt_traj"])
        ]
        if missing_traj:
            raise ValueError(
                "EWMBench trajectory_consistency requires traj/traj.npy for both "
                f"pred and GT. Missing for {len(missing_traj)} episodes; examples: {missing_traj[:10]}"
            )

    mapping = {
        "pred_base": str(pred_base),
        "gt_base": str(gt_base),
        "pairs": pairs,
        "skipped": skipped,
        "task_name_to_numeric_id": task_map,
        "note": (
            "Adapter symlinks image-frame directories and decodes mp4-only "
            "sources into this cache when decode_video_files is enabled."
        ),
    }
    save_json(mapping, adapter_root / "mapping.json")
    return mapping


def _load_repo_config(repo: Path) -> dict[str, Any]:
    cfg_path = repo / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return _normalize_cluster_paths(data)


def _write_ewmbench_config(
    *,
    repo: Path,
    config_path: Path,
    pred_base: Path,
    gt_base: Path,
    save_path: Path,
    ewm_cfg: dict[str, Any],
) -> None:
    repo_cfg = _load_repo_config(repo)
    ckpt_cfg = dict(repo_cfg.get("ckpt", {}) or {})
    ckpt_cfg.update(_normalize_cluster_paths(ewm_cfg.get("ckpt", {}) or {}))
    ckpt_cfg = _normalize_cluster_paths(ckpt_cfg)

    config = _normalize_cluster_paths({
        "model_name": ewm_cfg.get("model_name", "evac_c3"),
        "data": {
            "gt_path": str(gt_base),
            "val_base": str(pred_base),
        },
        "save_path": str(save_path),
        "ckpt": ckpt_cfg,
    })
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _read_mean_row(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("task_id") != "MEAN":
                continue
            means: dict[str, float] = {}
            for key, value in row.items():
                if key in {"task_id", "episode_id", "trial_id"} or value in {None, ""}:
                    continue
                try:
                    means[key] = float(value)
                except (TypeError, ValueError):
                    pass
            return means
    return {}


def _trajectory_breakdown(traj_obj: Any) -> dict[str, float]:
    """Trajectory stats with detector-failure awareness.

    EWMBench's YOLO gripper detector often fails on RoboTwin predictions; when it
    does, that episode is stored as hsd=dyn=ndtw=0.000 (the worst score, since
    trajectory metrics are higher=better normalized similarities). Returns:
      - traj_hsd_clean / traj_dyn_clean / traj_ndtw_clean : mean over DETECTED
        episodes only (the 0.000 detector-failures excluded).
      - Motion_clean : mean over detected episodes of (hsd+dyn+ndtw).
      - eef_detection_rate : fraction of episodes successfully tracked.
      - n_trajectory_episodes / n_detected_episodes : raw counts.
    (The include-0 traj_hsd/dyn/ndtw + Motion are added by the caller.)
    """
    eps: list[tuple[float, float, float]] = []

    def _walk(obj):
        if isinstance(obj, dict):
            if {"hsd", "dyn", "ndtw"} <= set(obj.keys()):
                try:
                    eps.append((float(obj["hsd"]), float(obj["dyn"]), float(obj["ndtw"])))
                    return
                except (TypeError, ValueError):
                    pass
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(traj_obj)
    out: dict[str, float] = {}
    n_total = len(eps)
    if n_total == 0:
        return out
    detected = [(h, d, n) for (h, d, n) in eps if not (h == 0.0 and d == 0.0 and n == 0.0)]
    n_det = len(detected)
    out["eef_detection_rate"] = round(n_det / n_total, 6)
    out["n_trajectory_episodes"] = n_total
    out["n_detected_episodes"] = n_det
    if n_det:
        for k, idx in (("hsd", 0), ("dyn", 1), ("ndtw", 2)):
            out[f"traj_{k}_clean"] = round(sum(e[idx] for e in detected) / n_det, 6)
        out["Motion_clean"] = round(sum(h + d + n for (h, d, n) in detected) / n_det, 6)
    return out


def _compute_mean_from_results(results_json: Path) -> dict[str, float]:
    """Compute per‑dimension means directly from *evac_c3_results.json*.

    This is the primary aggregation path — it does not depend on the
    upstream EWMBench CSV (which may crash during merge).  Sub‑metrics
    of ``trajectory_consistency`` (hsd / dyn / ndtw) and ``semantics``
    (BLEUScore / CLIPScore) are extracted separately.
    """
    if not results_json.exists():
        return {}
    with results_json.open(encoding="utf-8") as f:
        data = json.load(f)

    # Compound metrics whose sub-keys should NOT be averaged together.
    _COMPOUND_METRICS = {"trajectory_consistency", "semantics"}

    def _collect(obj, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    try:
                        out.append(float(v))
                    except ValueError:
                        pass
                _collect(v, out)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v, out)
        elif isinstance(obj, (int, float)):
            out.append(float(obj))

    means: dict[str, float] = {}
    for metric in data:
        if metric in _COMPOUND_METRICS:
            continue  # handled separately below
        vals: list[float] = []
        _collect(data[metric], vals)
        if vals:
            means[metric] = round(float(sum(vals) / len(vals)), 6)

    # trajectory sub‑metrics
    traj = data.get("trajectory_consistency", {})
    subs: dict[str, list[float]] = {"hsd": [], "dyn": [], "ndtw": []}
    def _coll_subs(obj):
        if isinstance(obj, dict):
            for k in ("hsd", "dyn", "ndtw"):
                if k in obj:
                    try:
                        subs[k].append(float(obj[k]))
                    except (ValueError, TypeError):
                        pass
            for v in obj.values():
                _coll_subs(v)
    _coll_subs(traj)
    for k, vals in subs.items():
        if vals:
            means[f"traj_{k}"] = round(float(sum(vals) / len(vals)), 6)
    # Motion (include-0) = mean over all episodes of (hsd+dyn+ndtw). Trajectory
    # sub-metrics are higher=better; 0.000 = detector failure (worst).
    if subs["hsd"] and subs["dyn"] and subs["ndtw"] and len(subs["hsd"]) == len(subs["dyn"]) == len(subs["ndtw"]):
        means["Motion"] = round(float(sum(h + d + n for h, d, n in zip(subs["hsd"], subs["dyn"], subs["ndtw"])) / len(subs["hsd"])), 6)

    # Trajectory breakdown: the EWMBench YOLO gripper detector frequently fails
    # on RoboTwin predictions (stores hsd=dyn=ndtw=0.000 for that episode), so
    # the plain traj_* means above mix real scores with detector-failure zeros.
    # Also report (a) EEF Detection Rate = fraction of episodes successfully
    # tracked, and (b) "clean" means over detected episodes only. traj_* stay
    # include-0 (the canonical EWMBench numbers); *_clean are detected-only.
    means.update(_trajectory_breakdown(traj))

    # semantics sub‑metrics (BLEUScore / CLIPScore)
    sem = data.get("semantics", {})
    sem_subs: dict[str, list[float]] = {"BLEUScore": [], "CLIPScore": []}
    def _coll_sem(obj):
        if isinstance(obj, dict):
            for k in ("BLEUScore", "CLIPScore"):
                if k in obj:
                    try:
                        sem_subs[k].append(float(obj[k]))
                    except (ValueError, TypeError):
                        pass
            for v in obj.values():
                _coll_sem(v)
    _coll_sem(sem)
    for k, vals in sem_subs.items():
        if vals:
            means[f"semantics_{k}"] = round(float(sum(vals) / len(vals)), 6)

    return means


def _basic_metric_subset(dimensions: list[str]) -> list[str]:
    names: list[str] = []
    for dim in dimensions:
        if dim == "psnr_ssim":
            for name in ["psnr", "ssim"]:
                if name not in names:
                    names.append(name)
        elif dim in {"psnr", "ssim"} and dim not in names:
            names.append(dim)
    return names


def _direct_skip_reasons(dimensions: list[str]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for dim in dimensions:
        if dim in {"psnr", "ssim", "psnr_ssim"}:
            continue
        reasons[dim] = (
            "This dimension is only available through EWMBench's official "
            "evaluate.py path. Re-run with --use_ewmbench_evaluate_py after "
            "configuring ewmbench.python and the required checkpoints."
        )
    return reasons


def _load_basic_metrics_module(repo: Path):
    module_path = repo / "EWMBench" / "basic_metrics.py"
    if not module_path.exists():
        raise FileNotFoundError(f"EWMBench basic_metrics.py not found: {module_path}")
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        _install_cv2_stub_module()
        print(
            "[ewmbench WARNING] current Python cannot import cv2; "
            "using a Pillow-backed fallback for EWMBench basic_metrics.py. "
            f"Original error: {exc!r}"
        )

    spec = importlib.util.spec_from_file_location("evac_c3_ewmbench_basic_metrics", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load EWMBench basic_metrics.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_basic_metrics_csv(results: dict[str, Any], csv_path: Path) -> dict[str, float]:
    triplets: set[tuple[str, str, str]] = set()
    for metric_data in results.values():
        for task_id, episodes in metric_data.items():
            for episode_id, trials in episodes.items():
                for trial_id in trials:
                    triplets.add((str(task_id), str(episode_id), str(trial_id)))

    fields = ["task_id", "episode_id", "trial_id"] + sorted(results.keys())
    metric_values: dict[str, list[float]] = {metric: [] for metric in results}
    rows: list[dict[str, Any]] = []
    for task_id, episode_id, trial_id in sorted(triplets):
        row: dict[str, Any] = {
            "task_id": int(task_id) if task_id.isdigit() else task_id,
            "episode_id": episode_id,
            "trial_id": int(trial_id) if trial_id.isdigit() else trial_id,
        }
        for metric, metric_data in results.items():
            value = metric_data.get(task_id, {}).get(episode_id, {}).get(trial_id, "")
            row[metric] = value
            try:
                metric_values[metric].append(float(value))
            except (TypeError, ValueError):
                pass
        rows.append(row)

    mean_row: dict[str, Any] = {field: "" for field in fields}
    mean_row["task_id"] = "MEAN"
    means: dict[str, float] = {}
    for metric, values in metric_values.items():
        if values:
            means[metric] = round(sum(values) / len(values), 6)
            mean_row[metric] = means[metric]
    rows.append(mean_row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return means


def _run_basic_metrics_direct(
    *,
    repo: Path,
    pred_base: Path,
    gt_base: Path,
    save_path: Path,
    metric_names: list[str],
    log_path: Path,
) -> dict[str, Any]:
    module = _load_basic_metrics_module(repo)
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[ewmbench] running basic metrics directly from "
        f"{repo / 'EWMBench' / 'basic_metrics.py'}"
    )
    results = module.compute_basic_metrics(
        gt_path=str(gt_base),
        pd_path=str(pred_base),
        metric_names=metric_names,
    )
    results_json = save_path / "evac_c3_results.json"
    with results_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    csv_path = save_path / "ewmbm_final_table.csv"
    means = _write_basic_metrics_csv(results, csv_path)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Ran EWMBench basic_metrics.py directly for PSNR/SSIM.\n")
        f.write(f"repo={repo}\npred_base={pred_base}\ngt_base={gt_base}\n")
        f.write(f"metrics={metric_names}\n")
    return {
        "results_json": str(results_json),
        "csv": str(csv_path),
        "log": str(log_path),
        "mean": means,
    }


def _resolve_yolo_ckpt_internal(ewm_cfg: dict, ewmbench_repo: str | Path) -> str:
    """Resolve YOLO checkpoint: EVAC config first, then EWMBench default."""
    import yaml as _yaml
    ckpt = _normalize_cluster_paths(ewm_cfg.get("ckpt", {}).get("yolo_world_ckpt", ""))
    if isinstance(ckpt, str) and ckpt:
        p = Path(ckpt)
        if not p.is_absolute():
            p = Path(ewmbench_repo) / p if not p.exists() else p
        if p.exists():
            return str(p)
    repo_cfg_path = Path(ewmbench_repo) / "config.yaml"
    if repo_cfg_path.exists():
        repo_cfg = _yaml.safe_load(repo_cfg_path.open()) or {}
        repo_cfg = _normalize_cluster_paths(repo_cfg)
        ckpt = repo_cfg.get("ckpt", {}).get("yolo_world_ckpt", "")
        if isinstance(ckpt, str) and ckpt:
            p = Path(ckpt)
            if not p.is_absolute():
                p = Path(ewmbench_repo) / p
            if p.exists():
                return str(p)
    return ""


def _has_traj_in_adapter(adapter_root: Path, pair: dict) -> bool:
    """Check whether a pair has usable traj.npy files on both sides.

    A prediction trajectory full of ``-1`` means the detector failed to find
    grippers; that is still a valid EWMBench input and should be scored as poor
    trajectory consistency.  GT trajectories are stricter, but RoboTwin can be
    single-sided: at least one GT point track must be valid enough for EWMBench
    to choose it.  Episodes where both GT tracks are too sparse are skipped.
    """
    pred_traj = adapter_root / "evac_c3_dataset" / str(pair["ewmbench_task_id"]) / str(pair["ewmbench_episode_id"]) / "1" / "traj" / "traj.npy"
    gt_traj = adapter_root / "gt_dataset" / str(pair["ewmbench_task_id"]) / str(pair["ewmbench_episode_id"]) / "traj" / "traj.npy"
    return _valid_traj_file(pred_traj) and _valid_traj_file(gt_traj, require_eval_ready=True)


def _traj_file_status(path: Path, *, require_eval_ready: bool = False) -> tuple[bool, dict[str, Any]]:
    try:
        import numpy as _np

        if not path.exists():
            return False, {"path": str(path), "reason": "missing_file"}
        arr = _np.load(path)
        if arr.ndim != 3 or arr.shape[1:] != (2, 2):
            return False, {"path": str(path), "reason": "bad_shape", "shape": list(arr.shape)}
        if arr.shape[0] <= 0:
            return False, {"path": str(path), "reason": "empty_traj", "shape": list(arr.shape)}

        # EWMBench treats each point track independently and declares a point
        # invalid when more than 80% of its frames are [-1, -1].
        finite = _np.isfinite(arr).all(axis=-1)
        not_missing = _np.any(arr != -1.0, axis=-1)
        valid = finite & not_missing
        ratios = valid.mean(axis=0).astype(float).tolist()
        status: dict[str, Any] = {
            "path": str(path),
            "shape": list(arr.shape),
            "frames": int(arr.shape[0]),
            "valid_ratios": ratios,
            "left_valid_ratio": float(ratios[0]),
            "right_valid_ratio": float(ratios[1]),
            "any_valid_ratio": float(valid.any(axis=1).mean()),
            "both_valid_ratio": float(valid.all(axis=1).mean()),
        }
        invalid_indices = [
            idx
            for idx, ratio in enumerate(ratios)
            if ratio < EWMBENCH_TRAJ_MIN_VALID_RATIO
        ]
        valid_indices = [
            idx
            for idx, ratio in enumerate(ratios)
            if ratio >= EWMBENCH_TRAJ_MIN_VALID_RATIO
        ]
        status["invalid_indices"] = invalid_indices
        status["valid_indices"] = valid_indices
        status["min_required_valid_ratio"] = EWMBENCH_TRAJ_MIN_VALID_RATIO
        if require_eval_ready and not valid_indices:
            status.update(
                {
                    "reason": "gt_traj_all_tracks_too_sparse_for_ewmbench",
                }
            )
            return False, status
        return True, status
    except Exception as exc:
        return False, {"path": str(path), "reason": "load_failed", "error": repr(exc)}


def _valid_traj_file(path: Path, *, require_eval_ready: bool = False) -> bool:
    ok, _ = _traj_file_status(path, require_eval_ready=require_eval_ready)
    return ok


def _traj_status_in_adapter(adapter_root: Path, pair: dict) -> tuple[bool, dict[str, Any] | None]:
    pred_traj = adapter_root / "evac_c3_dataset" / str(pair["ewmbench_task_id"]) / str(pair["ewmbench_episode_id"]) / "1" / "traj" / "traj.npy"
    gt_traj = adapter_root / "gt_dataset" / str(pair["ewmbench_task_id"]) / str(pair["ewmbench_episode_id"]) / "traj" / "traj.npy"
    details: list[dict[str, Any]] = []
    for role, path in (("pred", pred_traj), ("gt", gt_traj)):
        ok, status = _traj_file_status(path, require_eval_ready=(role == "gt"))
        if not ok:
            status["role"] = role
            details.append(status)
    if not details:
        return True, None
    return False, {
        "episode_id": pair["episode_id"],
        "ewmbench_task_id": pair["ewmbench_task_id"],
        "ewmbench_episode_id": pair["ewmbench_episode_id"],
        "reason": "missing_or_invalid_traj",
        "details": details,
    }


def _drop_adapter_pairs(adapter: dict[str, Any], adapter_root: Path, skipped: list[dict[str, Any]]) -> None:
    if not skipped:
        return
    skipped_ids = {str(item["episode_id"]) for item in skipped}
    pred_base = Path(adapter["pred_base"])
    gt_base = Path(adapter["gt_base"])
    kept_pairs: list[dict[str, Any]] = []
    for pair in adapter["pairs"]:
        if str(pair["episode_id"]) not in skipped_ids:
            kept_pairs.append(pair)
            continue
        tid = str(pair["ewmbench_task_id"])
        eid = str(pair["ewmbench_episode_id"])
        for path in (pred_base / tid / eid, gt_base / tid / eid):
            if path.exists() or path.is_symlink():
                _reset_path(path)
        for task_dir in (pred_base / tid, gt_base / tid):
            try:
                if task_dir.is_dir() and not any(task_dir.iterdir()):
                    task_dir.rmdir()
            except Exception:
                pass
    adapter["pairs"] = kept_pairs
    adapter.setdefault("skipped", []).extend(skipped)
    adapter["trajectory_skipped_episodes"] = skipped
    save_json(adapter, adapter_root / "mapping.json")


def _seed_traj_cache_from_adapter(adapter_root: Path, traj_cache_dir: Path) -> int:
    mapping_path = adapter_root / "mapping.json"
    if not mapping_path.exists():
        return 0
    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception:
        return 0
    pairs = mapping.get("pairs", [])
    if not isinstance(pairs, list):
        return 0
    seeded = 0
    for pair in pairs:
        try:
            tid = str(pair["ewmbench_task_id"])
            eid = str(pair["ewmbench_episode_id"])
            key = _safe_component(pair.get("episode_id"), f"{tid}_{eid}")
            sources = {
                "pred": adapter_root / "evac_c3_dataset" / tid / eid / "1" / "traj" / "traj.npy",
                "gt": adapter_root / "gt_dataset" / tid / eid / "traj" / "traj.npy",
            }
            for role, source in sources.items():
                target = traj_cache_dir / role / key / "traj.npy"
                require_eval_ready = role == "gt"
                if _valid_traj_file(
                    target,
                    require_eval_ready=require_eval_ready,
                ) or not _valid_traj_file(
                    source,
                    require_eval_ready=require_eval_ready,
                ):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                seeded += 1
        except Exception:
            continue
    if seeded:
        print(f"[ewmbench] traj: seeded {seeded} cached traj files from previous adapter")
    return seeded


def _parse_yolo_video_limit(value: Any) -> int | str | None:
    if value in {None, False, 0, "0", "false", "False", "none", "None", ""}:
        return None
    if value is True:
        return 3
    text = str(value).strip().lower()
    if text == "all":
        return "all"
    limit = int(text)
    if limit <= 0:
        return None
    return limit


def _should_save_yolo_video(limit: int | str | None, pair_index: int) -> bool:
    if limit is None:
        return False
    if limit == "all":
        return True
    return pair_index < int(limit)


def _write_traj_overlay_video(frame_dir: Path, traj_path: Path, video_path: Path, stats_path: Path | None = None) -> bool:
    """Write a lightweight center-point overlay from an existing traj.npy cache."""
    try:
        import cv2
        import numpy as _np

        traj = _np.load(traj_path)
        frame_files = _image_files(frame_dir)
        if not frame_files or traj.ndim != 3 or traj.shape[1:] != (2, 2):
            return False
        video_path.parent.mkdir(parents=True, exist_ok=True)
        first = cv2.imread(str(frame_files[0]))
        if first is None:
            return False
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
        total = min(len(frame_files), int(traj.shape[0]))
        left_valid = right_valid = both_valid = any_valid = 0
        for idx, frame_path in enumerate(frame_files[:total]):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            left = traj[idx, 0]
            right = traj[idx, 1]
            l_ok = bool(_np.all(left >= 0))
            r_ok = bool(_np.all(right >= 0))
            left_valid += int(l_ok)
            right_valid += int(r_ok)
            both_valid += int(l_ok and r_ok)
            any_valid += int(l_ok or r_ok)
            for label, pt, color in (("L", left, (0, 255, 0)), ("R", right, (0, 0, 255))):
                if bool(_np.all(pt >= 0)):
                    x = int(float(pt[0]) * w)
                    y = int(float(pt[1]) * h)
                    cv2.circle(frame, (x, y), 5, color, -1)
                    cv2.putText(frame, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(
                frame,
                f"frame={idx} any={any_valid}/{idx + 1}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            writer.write(frame)
        writer.release()
        if stats_path is not None:
            stats = {
                "source": "cached_traj_overlay",
                "frames": total,
                "left_valid": left_valid,
                "right_valid": right_valid,
                "both_valid": both_valid,
                "any_valid": any_valid,
                "left_valid_ratio": left_valid / total if total else 0.0,
                "right_valid_ratio": right_valid / total if total else 0.0,
                "both_valid_ratio": both_valid / total if total else 0.0,
                "any_valid_ratio": any_valid / total if total else 0.0,
                "traj_path": str(traj_path),
                "video_path": str(video_path),
            }
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(stats, stats_path)
        return True
    except Exception as exc:
        print(f"[ewmbench traj WARNING] cached overlay failed for {traj_path}: {exc!r}")
        return False


def _generate_traj_in_adapter(
    adapter: dict,
    ewmbench_repo: str | Path,
    python_bin: str,
    yolo_ckpt: str,
    ewmbench_gpus: str = "",
    traj_cache_dir: str | Path | None = None,
    save_yolo_videos: Any = None,
    yolo_video_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run YOLO gripper detection on adapter tree frames, write traj.npy.

    This is called after the adapter symlink/decoding step so mp4 GT sources
    have already been expanded to frame directories.
    """
    import subprocess as _sp
    import tempfile as _tf
    from pathlib import Path as _Path

    # Try EVAC-configured model first, fall back to EWMBench default
    _primary_ckpt = str(yolo_ckpt or "")
    if _primary_ckpt and not _Path(_primary_ckpt).exists():
        _primary_ckpt = ""
    _fallback_ckpt = _resolve_yolo_ckpt_internal(
        {} if not yolo_ckpt else {"ckpt": {"yolo_world_ckpt": ""}}, ewmbench_repo
    )  # get EWMBench default
    if _fallback_ckpt == _primary_ckpt:
        _fallback_ckpt = ""

    if not _primary_ckpt and not _fallback_ckpt:
        print("[ewmbench traj] WARNING: no YOLO checkpoint found; skipping")
        return {"status": "skipped", "reason": "missing_yolo_checkpoint"}

    # collect (frame_dir, output_parent) pairs from the adapter
    pred_base = _Path(adapter["pred_base"])
    gt_base = _Path(adapter["gt_base"])
    traj_cache_root = _Path(traj_cache_dir) if traj_cache_dir else None
    if traj_cache_root is not None:
        traj_cache_root.mkdir(parents=True, exist_ok=True)
    yolo_video_limit = _parse_yolo_video_limit(save_yolo_videos)
    yolo_video_root = _Path(yolo_video_dir) if yolo_video_dir else None
    if yolo_video_limit is not None and yolo_video_root is not None:
        yolo_video_root.mkdir(parents=True, exist_ok=True)
    work_items: list[tuple[str, str, str, str, str, str]] = []
    cache_hits = 0

    def _traj_cache_path(role: str, pair: dict[str, Any]) -> _Path | None:
        if traj_cache_root is None:
            return None
        tid = str(pair["ewmbench_task_id"])
        eid = str(pair["ewmbench_episode_id"])
        key = _safe_component(pair.get("episode_id"), f"{tid}_{eid}")
        return traj_cache_root / role / key / "traj.npy"

    def _yolo_debug_paths(role: str, pair: dict[str, Any], pair_index: int) -> tuple[_Path | None, _Path | None]:
        if yolo_video_root is None or not _should_save_yolo_video(yolo_video_limit, pair_index):
            return None, None
        tid = str(pair["ewmbench_task_id"])
        eid = str(pair["ewmbench_episode_id"])
        key = _safe_component(pair.get("episode_id"), f"{tid}_{eid}")
        root = yolo_video_root / key
        return root / f"{role}_yolo.mp4", root / f"{role}_yolo_stats.json"

    def _restore_cached_traj(cache_path: _Path | None, output_parent: _Path, *, role: str) -> bool:
        require_eval_ready = role == "gt"
        if cache_path is None or not _valid_traj_file(
            cache_path,
            require_eval_ready=require_eval_ready,
        ):
            return False
        _symlink(cache_path, output_parent / "traj" / "traj.npy")
        return True

    for pair_index, pair in enumerate(adapter["pairs"]):
        tid = pair["ewmbench_task_id"]
        eid = pair["ewmbench_episode_id"]
        # pred: evac_c3_dataset/{task}/{ep}/1/video → output to {task}/{ep}/1/
        _pred_video = pred_base / str(tid) / str(eid) / "1" / "video"
        _pred_out = pred_base / str(tid) / str(eid) / "1"
        if _pred_video.is_dir() and not (_pred_out / "traj" / "traj.npy").exists():
            _pred_cache = _traj_cache_path("pred", pair)
            _pred_vis, _pred_stats = _yolo_debug_paths("pred", pair, pair_index)
            if _restore_cached_traj(_pred_cache, _pred_out, role="pred"):
                if _pred_vis is not None and not _pred_vis.exists() and _pred_cache is not None:
                    _write_traj_overlay_video(_pred_video, _pred_cache, _pred_vis, _pred_stats)
                cache_hits += 1
            else:
                work_items.append((
                    str(_pred_video),
                    str(_pred_out),
                    f"pred:{tid}/{eid}",
                    str(_pred_cache or ""),
                    str(_pred_vis or ""),
                    str(_pred_stats or ""),
                ))

        # gt: gt_dataset/{task}/{ep}/video → output to {task}/{ep}/
        _gt_video = gt_base / str(tid) / str(eid) / "video"
        _gt_out = gt_base / str(tid) / str(eid)
        if _gt_video.is_dir() and not (_gt_out / "traj" / "traj.npy").exists():
            _gt_cache = _traj_cache_path("gt", pair)
            _gt_vis, _gt_stats = _yolo_debug_paths("gt", pair, pair_index)
            if _restore_cached_traj(_gt_cache, _gt_out, role="gt"):
                if _gt_vis is not None and not _gt_vis.exists() and _gt_cache is not None:
                    _write_traj_overlay_video(_gt_video, _gt_cache, _gt_vis, _gt_stats)
                cache_hits += 1
            else:
                work_items.append((
                    str(_gt_video),
                    str(_gt_out),
                    f"gt:{tid}/{eid}",
                    str(_gt_cache or ""),
                    str(_gt_vis or ""),
                    str(_gt_stats or ""),
                ))

    if cache_hits:
        print(f"[ewmbench] traj: restored {cache_hits} cached traj files")
    if not work_items:
        print("[ewmbench] traj: all files already present")
        return {"status": "cached", "cache_hits": cache_hits, "work_items": 0}

    print(f"[ewmbench] traj: generating {len(work_items)} from adapter frames ...")
    _model_paths = [p for p in (_primary_ckpt, _fallback_ckpt) if p]
    _model_path = _model_paths[0] if _model_paths else ""
    _alt_model = _model_paths[1] if len(_model_paths) > 1 else ""

    env = os.environ.copy()
    if ewmbench_gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(ewmbench_gpus).strip()
    cv2_ok, cv2_msg = _check_python_cv2_import(python_bin, env=env)
    if not cv2_ok and _cv2_error_is_import_related(cv2_msg):
        print(
            "[ewmbench traj] WARNING: cv2 is not importable in the EWMBench "
            f"Python ({python_bin}); skipping YOLO trajectory generation. "
            "Install libGL.so.1 or opencv-python-headless in that environment "
            f"to enable trajectory_consistency. Original error: {cv2_msg}"
        )
        return {
            "status": "skipped",
            "reason": "cv2_import_failed",
            "python": str(python_bin),
            "error": cv2_msg,
            "work_items": len(work_items),
            "cache_hits": cache_hits,
        }

    script_lines = [
        "import os, sys, cv2, gc, json, numpy as np",
        "from tqdm import tqdm",
        "from ultralytics import YOLO",
        "import logging",
        "logging.getLogger('ultralytics').setLevel(logging.ERROR)",
        "logging.getLogger('ultralytics').propagate = False",
        "",
        f"_CONF = 0.05",
        f"_MODEL_PATH = {_model_path!r}",
        f"_ALT_MODEL_PATH = {_alt_model!r}",
        f"_WORK = {work_items!r}",
        "",
        "try:",
        "    import torch",
        "    _DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'",
        "    _CUDA_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'",
        "    print(f\"[ewmbench traj] device={_DEVICE} cuda_available={torch.cuda.is_available()} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')} name={_CUDA_NAME}\")",
        "except Exception as _e:",
        "    _DEVICE = 'cpu'",
        "    print(f'[ewmbench traj] device=cpu cuda_check_failed={_e!r}')",
        "_model = YOLO(_MODEL_PATH).to(_DEVICE)",
        "_alt_model = None",
        "",
        "",
        "def _run_detection(_work, _model):",
        "    _failed = 0",
        "    _any_valid = False",
        "    for _input, _output_dir, _label, _cache_path, _video_path, _stats_path in tqdm(_work, desc='[ewmbench traj]', unit='src', ncols=80):",
        "        try:",
        "            _traj_dir = os.path.join(_output_dir, 'traj')",
        "            os.makedirs(_traj_dir, exist_ok=True)",
        "            _image_files = sorted([f for f in os.listdir(_input) if f.lower().endswith(('.jpg','.jpeg','.png'))])",
        "            _traj = []",
        "            _writer = None",
        "            _total = _left_valid = _right_valid = _both_valid = _any_frame_valid = 0",
        "            for _fname in _image_files:",
        "                _img_bgr = cv2.imread(os.path.join(_input, _fname))",
        "                if _img_bgr is None: continue",
        "                _img = cv2.cvtColor(_img_bgr, cv2.COLOR_BGR2RGB)",
        "                _h, _w = _img.shape[:2]",
        "                if _video_path and _writer is None:",
        "                    os.makedirs(os.path.dirname(_video_path), exist_ok=True)",
        "                    _writer = cv2.VideoWriter(_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (_w, _h))",
        "                _results = _model.track(_img, persist=True, conf=_CONF, imgsz=640, verbose=False)",
        "                _boxes = _results[0].boxes",
        "                _clses = _boxes.cls.cpu().tolist() if _boxes.cls is not None else []",
        "                _confs = _boxes.conf.cpu().tolist() if _boxes.conf is not None else []",
        "                _best = {}",
        "                for _i, (_c, _f) in enumerate(zip(_clses, _confs)):",
        "                    if int(_c) not in _best or _f > _best[int(_c)][1]:",
        "                        _best[int(_c)] = (_i, _f)",
        "                _lx = _ly = _rx = _ry = -1.0",
        "                _vis = _img_bgr.copy() if _writer is not None else None",
        "                if 0 in _best and _best[0][1] > 0:",
        "                    _xywh = _boxes[_best[0][0]].xywh.cpu().numpy()[0]",
        "                    _xyxy = _boxes[_best[0][0]].xyxy.cpu().numpy()[0]",
        "                    _lx, _ly = float(_xywh[0]) / _w, float(_xywh[1]) / _h",
        "                    _any_valid = True",
        "                    _left_valid += 1",
        "                    if _vis is not None:",
        "                        cv2.rectangle(_vis, (int(_xyxy[0]), int(_xyxy[1])), (int(_xyxy[2]), int(_xyxy[3])), (0, 255, 0), 2)",
        "                        cv2.circle(_vis, (int(_xywh[0]), int(_xywh[1])), 5, (0, 255, 0), -1)",
        "                        cv2.putText(_vis, f'L {_best[0][1]:.2f}', (int(_xyxy[0]), max(15, int(_xyxy[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)",
        "                if 1 in _best and _best[1][1] > 0:",
        "                    _xywh = _boxes[_best[1][0]].xywh.cpu().numpy()[0]",
        "                    _xyxy = _boxes[_best[1][0]].xyxy.cpu().numpy()[0]",
        "                    _rx, _ry = float(_xywh[0]) / _w, float(_xywh[1]) / _h",
        "                    _any_valid = True",
        "                    _right_valid += 1",
        "                    if _vis is not None:",
        "                        cv2.rectangle(_vis, (int(_xyxy[0]), int(_xyxy[1])), (int(_xyxy[2]), int(_xyxy[3])), (0, 0, 255), 2)",
        "                        cv2.circle(_vis, (int(_xywh[0]), int(_xywh[1])), 5, (0, 0, 255), -1)",
        "                        cv2.putText(_vis, f'R {_best[1][1]:.2f}', (int(_xyxy[0]), max(15, int(_xyxy[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)",
        "                _l_ok = _lx >= 0 and _ly >= 0",
        "                _r_ok = _rx >= 0 and _ry >= 0",
        "                _both_valid += int(_l_ok and _r_ok)",
        "                _any_frame_valid += int(_l_ok or _r_ok)",
        "                _total += 1",
        "                if _vis is not None:",
        "                    cv2.putText(_vis, f'{_label} frame={_total - 1} any={_any_frame_valid}/{_total}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)",
        "                    _writer.write(_vis)",
        "                _traj.append([(_lx, _ly), (_rx, _ry)])",
        "            if _writer is not None:",
        "                _writer.release()",
        "            _traj_arr = np.array(_traj, dtype=np.float32).reshape(-1, 2, 2)",
        "            np.save(os.path.join(_traj_dir, 'traj.npy'), _traj_arr)",
        "            if _cache_path:",
        "                os.makedirs(os.path.dirname(_cache_path), exist_ok=True)",
        "                np.save(_cache_path, _traj_arr)",
        "            if _stats_path:",
        "                os.makedirs(os.path.dirname(_stats_path), exist_ok=True)",
        "                _stats = {",
        "                    'source': 'yolo_track',",
        "                    'label': _label,",
        "                    'input': _input,",
        "                    'frames': _total,",
        "                    'left_valid': _left_valid,",
        "                    'right_valid': _right_valid,",
        "                    'both_valid': _both_valid,",
        "                    'any_valid': _any_frame_valid,",
        "                    'left_valid_ratio': (_left_valid / _total) if _total else 0.0,",
        "                    'right_valid_ratio': (_right_valid / _total) if _total else 0.0,",
        "                    'both_valid_ratio': (_both_valid / _total) if _total else 0.0,",
        "                    'any_valid_ratio': (_any_frame_valid / _total) if _total else 0.0,",
        "                    'video_path': _video_path or None,",
        "                    'traj_path': os.path.join(_traj_dir, 'traj.npy'),",
        "                }",
        "                with open(_stats_path, 'w', encoding='utf-8') as _f:",
        "                    json.dump(_stats, _f, indent=2)",
        "        except Exception as _e:",
        "            _failed += 1",
        "            tqdm.write(f'[ewmbench traj]  FAILED {_label}: {_e}')",
        "    return _failed, _any_valid",
        "",
        "_failed, _valid = _run_detection(_WORK, _model)",
        f"print(f'[ewmbench traj] primary model: {{len(_WORK) - _failed}}/{{len(_WORK)}} ok, {{_failed}} failed, any_valid={{_valid}}')",
        "",
        "if not _valid and _ALT_MODEL_PATH:",
        "    print(f'[ewmbench traj] primary model had zero valid detections, falling back to {_ALT_MODEL_PATH}')",
        "    del _model  # free GPU memory before loading fallback",
        "    gc.collect()",
        "    try:",
        "        import torch",
        "        torch.cuda.empty_cache()",
        "    except Exception:",
        "        pass",
        "    _alt_model = YOLO(_ALT_MODEL_PATH).to(_DEVICE)",
        "    _model = _alt_model",
        "    _failed, _valid = _run_detection(_WORK, _model)",
        f"    print(f'[ewmbench traj] fallback model: {{len(_WORK) - _failed}}/{{len(_WORK)}} ok, any_valid={{_valid}}')",
    ]

    tmp = _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(script_lines))
        tmp.close()
        proc = _sp.run([str(python_bin), tmp.name], env=env, text=True, timeout=14400)
        if proc.returncode != 0:
            print(f"[ewmbench traj] WARNING: detection subprocess exited {proc.returncode}")
        return {
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "work_items": len(work_items),
            "cache_hits": cache_hits,
        }
    finally:
        os.unlink(tmp.name)


def compute_ewmbench(
    *,
    episodes: list[dict[str, Any]],
    pred_dir: str | Path,
    gt_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None,
    project_root: str | Path,
    find_pred_source: FindSourceFn,
    find_gt_source: FindSourceFn,
) -> dict[str, Any]:
    cfg = config or {}
    ewm_cfg = cfg.get("ewmbench", {}) or {}
    project_root = Path(project_root)
    repo = _resolve_path(ewm_cfg.get("repo", "third_party/EWMBench"), project_root)
    if not (repo / "evaluate.py").exists():
        raise FileNotFoundError(f"EWMBench repo/evaluate.py not found: {repo}")

    dimensions = _as_list(
        ewm_cfg.get("dimensions"),
        ["psnr", "ssim", "scene_consistency", "trajectory_consistency", "semantics", "diversity"],
    )
    decode_video_files = bool(ewm_cfg.get("decode_video_files", True))
    use_official_evaluate_py = bool(ewm_cfg.get("use_official_evaluate_py", False))
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    gt_start_frame = int(ewm_cfg.get("gt_start_frame", eval_cfg.get("gt_start_frame", cfg.get("gt_start_frame", cfg.get("n_previous", 4)))))
    output_dir = _resolve_path(output_dir, project_root) if not Path(output_dir).is_absolute() else Path(output_dir)
    adapter_root = _resolve_path(ewm_cfg.get("adapter_dir", output_dir / "adapter"), project_root)
    traj_cache_dir = _resolve_path(ewm_cfg.get("traj_cache_dir", output_dir / "traj_cache"), project_root)
    yolo_video_dir = _resolve_path(ewm_cfg.get("yolo_video_dir", output_dir / "yolo_videos"), project_root)
    save_path = _resolve_path(ewm_cfg.get("save_path", output_dir / "results"), project_root)
    config_path = _resolve_path(ewm_cfg.get("config_path", output_dir / "ewmbench_config.yaml"), project_root)
    log_path = _resolve_path(ewm_cfg.get("log_path", output_dir / "ewmbench.log"), project_root)

    print(
        "[ewmbench] adapter config: "
        f"repo={repo}, dimensions={dimensions}, decode_video_files={decode_video_files}, "
        f"use_official_evaluate_py={use_official_evaluate_py}, "
        f"pred_dir={pred_dir}, gt_dir={gt_dir}, adapter_dir={adapter_root}, "
        f"save_yolo_videos={ewm_cfg.get('save_yolo_videos')}"
    )
    _seed_traj_cache_from_adapter(adapter_root, traj_cache_dir)

    if not use_official_evaluate_py:
        basic_metric_names = _basic_metric_subset(dimensions)
        skipped_dimensions = _direct_skip_reasons(dimensions)
        if not basic_metric_names:
            return {
                "status": "skipped",
                "runner": "evac_c3_direct",
                "dimensions": dimensions,
                "num_eval_episodes": 0,
                "num_skipped_episodes": len(episodes),
                "decode_video_files": decode_video_files,
                "gt_start_frame": gt_start_frame,
                "repo": str(repo),
                "adapter_dir": str(adapter_root),
                "config_path": str(config_path),
                "output_dir": str(save_path),
                "skipped_dimensions": skipped_dimensions,
                "mean": {},
            }

        adapter = _prepare_adapter_tree(
            episodes=episodes,
            pred_root=Path(pred_dir),
            gt_root=Path(gt_dir),
            adapter_root=adapter_root,
            dimensions=basic_metric_names,
            find_pred_source=find_pred_source,
            find_gt_source=find_gt_source,
            decode_video_files=decode_video_files,
            gt_start_frame=gt_start_frame,
        )
        _write_ewmbench_config(
            repo=repo,
            config_path=config_path,
            pred_base=Path(adapter["pred_base"]),
            gt_base=Path(adapter["gt_base"]),
            save_path=save_path,
            ewm_cfg=ewm_cfg,
        )
        direct_result = _run_basic_metrics_direct(
            repo=repo,
            pred_base=Path(adapter["pred_base"]),
            gt_base=Path(adapter["gt_base"]),
            save_path=save_path,
            metric_names=basic_metric_names,
            log_path=log_path,
        )
        return {
            "status": "partial" if skipped_dimensions else "ok",
            "runner": "evac_c3_direct",
            "dimensions": dimensions,
            "computed_dimensions": basic_metric_names,
            "skipped_dimensions": skipped_dimensions,
            "num_eval_episodes": len(adapter["pairs"]),
            "num_skipped_episodes": len(adapter["skipped"]),
            "decode_video_files": decode_video_files,
            "gt_start_frame": gt_start_frame,
            "repo": str(repo),
            "adapter_dir": str(adapter_root),
            "config_path": str(config_path),
            "output_dir": str(save_path),
            **direct_result,
        }

    # build adapter WITHOUT trajectory check first — we may need to generate traj
    _needs_traj = "trajectory_consistency" in dimensions
    adapter = _prepare_adapter_tree(
        episodes=episodes,
        pred_root=Path(pred_dir),
        gt_root=Path(gt_dir),
        adapter_root=adapter_root,
        dimensions=[d for d in dimensions if d != "trajectory_consistency"],
        find_pred_source=find_pred_source,
        find_gt_source=find_gt_source,
        decode_video_files=decode_video_files,
        gt_start_frame=gt_start_frame,
    )
    _write_ewmbench_config(
        repo=repo,
        config_path=config_path,
        pred_base=Path(adapter["pred_base"]),
        gt_base=Path(adapter["gt_base"]),
        save_path=save_path,
        ewm_cfg=ewm_cfg,
    )

    trajectory_skipped_episodes: list[dict[str, Any]] = []
    skipped_dimensions: dict[str, str] = {}
    trajectory_preprocess: dict[str, Any] | None = None

    # ---- generate traj.npy inside the adapter tree if needed ----
    if _needs_traj:
        _yolo_ckpt = _normalize_cluster_paths(ewm_cfg.get("ckpt", {}).get("yolo_world_ckpt", ""))
        if isinstance(_yolo_ckpt, str) and _yolo_ckpt and not Path(_yolo_ckpt).is_absolute():
            _yolo_ckpt = str(project_root / _yolo_ckpt)
        trajectory_preprocess = _generate_traj_in_adapter(
            adapter=adapter,
            ewmbench_repo=repo,
            python_bin=ewm_cfg.get("python") or sys.executable,
            yolo_ckpt=_yolo_ckpt,
            ewmbench_gpus=ewm_cfg.get("gpus", ""),
            traj_cache_dir=traj_cache_dir,
            save_yolo_videos=ewm_cfg.get("save_yolo_videos"),
            yolo_video_dir=yolo_video_dir,
        )
        _missing_traj: list[dict[str, Any]] = []
        for p in adapter["pairs"]:
            ok, detail = _traj_status_in_adapter(adapter_root, p)
            if not ok and detail is not None:
                _missing_traj.append(detail)
        if _missing_traj:
            examples = [item["episode_id"] for item in _missing_traj[:10]]
            if (trajectory_preprocess or {}).get("reason") == "cv2_import_failed":
                dimensions = [d for d in dimensions if d != "trajectory_consistency"]
                skipped_dimensions["trajectory_consistency"] = (
                    "Skipped because YOLO trajectory preprocessing could not run: "
                    "the configured EWMBench Python cannot import cv2 "
                    "(commonly missing libGL.so.1). Other EWMBench dimensions "
                    "were kept on the full adapter set."
                )
                adapter["trajectory_skipped_episodes"] = _missing_traj
                save_json(adapter, adapter_root / "mapping.json")
                print(
                    "[ewmbench WARNING] trajectory_consistency skipped because "
                    "EWMBench Python cannot import cv2; keeping all episodes for "
                    f"other dimensions. examples={examples}"
                )
            elif len(_missing_traj) >= len(adapter["pairs"]):
                dimensions = [d for d in dimensions if d != "trajectory_consistency"]
                skipped_dimensions["trajectory_consistency"] = (
                    "Skipped because no eval pairs had usable traj/traj.npy after YOLO preprocessing."
                )
                adapter.setdefault("skipped", []).extend(_missing_traj)
                adapter["trajectory_skipped_episodes"] = _missing_traj
                save_json(adapter, adapter_root / "mapping.json")
                print(
                    "[ewmbench WARNING] trajectory_consistency skipped: "
                    f"no usable traj.npy pairs; examples={examples}"
                )
            else:
                _drop_adapter_pairs(adapter, adapter_root, _missing_traj)
                print(
                    "[ewmbench WARNING] trajectory_consistency skipped for "
                    f"{len(_missing_traj)} episodes with missing/invalid traj.npy; "
                    f"examples={examples}"
                )
            trajectory_skipped_episodes = _missing_traj

    if not dimensions:
        return {
            "status": "skipped",
            "runner": "ewmbench_official",
            "dimensions": [],
            "skipped_dimensions": skipped_dimensions,
            "num_eval_episodes": 0,
            "num_skipped_episodes": len(adapter["skipped"]),
            "trajectory_skipped_episodes": trajectory_skipped_episodes,
            "trajectory_preprocess": trajectory_preprocess,
            "decode_video_files": decode_video_files,
            "gt_start_frame": gt_start_frame,
            "repo": str(repo),
            "adapter_dir": str(adapter_root),
            "traj_cache_dir": str(traj_cache_dir),
            "yolo_video_dir": str(yolo_video_dir),
            "config_path": str(config_path),
            "output_dir": str(save_path),
            "mean": {},
        }

    python_bin = str(ewm_cfg.get("python") or sys.executable)
    cmd = [
        python_bin,
        "evaluate.py",
        "--dimension",
        *dimensions,
        "--config_path",
        str(config_path),
        "--overwrite",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    # EWMBench dist_init uses init_method='env://', so it reads WORLD_SIZE/RANK/
    # LOCAL_RANK/MASTER_ADDR/MASTER_PORT from the environment. EWMBench always
    # runs here as a single process, so force these explicitly — do NOT use
    # setdefault. If this is launched from inside a torchrun training process,
    # the parent env already contains WORLD_SIZE=N / RANK / LOCAL_RANK, and
    # setdefault would keep them, making the single EWMBench process wait for
    # N-1 peers at the rendezvous and time out after 600s. Assign a free port
    # and force localhost so concurrent runs don't collide either.
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        env["MASTER_PORT"] = str(_s.getsockname()[1])
    env["MASTER_ADDR"] = "127.0.0.1"
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    # pin to a specific GPU if configured
    _ewm_gpus = str(ewm_cfg.get("gpus", "")).strip()
    if _ewm_gpus:
        env["CUDA_VISIBLE_DEVICES"] = _ewm_gpus

    cv2_fallback: dict[str, Any] | None = None
    cv2_stub_tmp: tempfile.TemporaryDirectory[str] | None = None
    cv2_ok, cv2_msg = _check_python_cv2_import(python_bin, env=env)
    if not cv2_ok and _cv2_error_is_import_related(cv2_msg):
        cv2_stub_tmp = tempfile.TemporaryDirectory(prefix="evac_c3_cv2_stub_")
        stub_dir = Path(cv2_stub_tmp.name)
        _write_cv2_stub_dir(stub_dir)
        env["PYTHONPATH"] = str(stub_dir) + os.pathsep + env.get("PYTHONPATH", "")
        cv2_fallback = {
            "enabled": True,
            "reason": "cv2_import_failed",
            "python": python_bin,
            "error": cv2_msg,
        }
        print(
            "[ewmbench WARNING] EWMBench Python cannot import cv2; "
            "injecting a temporary Pillow-backed cv2 fallback for evaluate.py "
            f"import/basic resize. Original error: {cv2_msg}"
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ewmbench] running: {' '.join(cmd)} (cwd={repo})")
    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
    finally:
        if cv2_stub_tmp is not None:
            cv2_stub_tmp.cleanup()
    results_json = save_path / "evac_c3_results.json"
    csv_path = save_path / "ewmbm_final_table.csv"

    if proc.returncode != 0:
        # EWMBench may have crashed in the final CSV merge while all per‑dimension
        # results are already written — accept partial results when they exist.
        if results_json.exists():
            print(
                "[ewmbench] evaluate.py exited with code "
                f"{proc.returncode}, but per‑dimension results exist — "
                "treating as partial success (CSV merge may have failed)."
            )
            return {
                "status": "partial",
                "note": f"evaluate.py exit code {proc.returncode}; CSV merge may have failed",
                "dimensions": dimensions,
                "skipped_dimensions": skipped_dimensions,
                "num_eval_episodes": len(adapter["pairs"]),
                "num_skipped_episodes": len(adapter["skipped"]),
                "trajectory_skipped_episodes": trajectory_skipped_episodes,
                "trajectory_preprocess": trajectory_preprocess,
                "cv2_fallback": cv2_fallback,
                "decode_video_files": decode_video_files,
                "gt_start_frame": gt_start_frame,
                "repo": str(repo),
                "adapter_dir": str(adapter_root),
                "traj_cache_dir": str(traj_cache_dir),
                "yolo_video_dir": str(yolo_video_dir),
                "config_path": str(config_path),
                "output_dir": str(save_path),
                "results_json": str(results_json),
                "csv": str(csv_path) if csv_path.exists() else None,
                "log": str(log_path),
                "mean": _compute_mean_from_results(results_json),
            }
        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            lines = [
                _l for _l in lines
                if "This metric has all invalid values" not in _l
                and "Invalid gt trajectory encountered at index" not in _l
            ]
            tail = "\n".join(lines[-100:])
        except Exception:
            pass
        raise RuntimeError(
            f"EWMBench failed with exit code {proc.returncode}. "
            f"Log: {log_path}\n--- log tail ---\n{tail}"
        )

    return {
        "status": "partial" if trajectory_skipped_episodes or skipped_dimensions else "ok",
        "dimensions": dimensions,
        "skipped_dimensions": skipped_dimensions,
        "num_eval_episodes": len(adapter["pairs"]),
        "num_skipped_episodes": len(adapter["skipped"]),
        "trajectory_skipped_episodes": trajectory_skipped_episodes,
        "trajectory_preprocess": trajectory_preprocess,
        "cv2_fallback": cv2_fallback,
        "decode_video_files": decode_video_files,
        "gt_start_frame": gt_start_frame,
        "repo": str(repo),
        "adapter_dir": str(adapter_root),
        "traj_cache_dir": str(traj_cache_dir),
        "yolo_video_dir": str(yolo_video_dir),
        "config_path": str(config_path),
        "output_dir": str(save_path),
        "results_json": str(results_json),
        "csv": str(csv_path),
        "log": str(log_path),
        "mean": _compute_mean_from_results(results_json),
    }
