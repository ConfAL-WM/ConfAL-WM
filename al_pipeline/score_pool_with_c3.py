from __future__ import annotations

import argparse
import contextlib
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(1, str(_REPO / "evac"))

from al_pipeline.utils import (  # noqa: E402
    check_manifest_overlap,
    compute_risk_stats,
    ensure_dir,
    filter_valid_external_worldmodel_items,
    flatten_manifest_items,
    load_json,
    load_yaml,
    record_uid,
    save_json,
    validate_confidence_map,
)
from eval.confidence_eval.probe_inference import (  # noqa: E402
    encode_rgb_frames_to_latents,
    extract_probe_confidence,
    internal_samples_to_latent_array,
    load_action_h5,
    load_caminfo_json,
    load_generated_frames,
    load_model_with_probe,
)
from omegaconf import OmegaConf  # noqa: E402
from evac.utils.general_utils import instantiate_from_config, load_checkpoints  # noqa: E402
from eval.al_results.utils.video_utils import load_frames, normalize_frames, save_video  # noqa: E402
from lvdm.data.get_actions import get_actions  # noqa: E402
from lvdm.data.statistics import StatisticInfo  # noqa: E402

EXTERNAL_ACTION_TRANSFORM = "external_get_actions_sliced_normalized_v2"
EXTERNAL_CAMERA_TRANSFORM = "external_extrinsic_cv_w2c_v1"


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_n_frames_value(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        if text in {"auto", "dynamic", "episode"}:
            return "auto"
        value = text
    n = int(value)
    if n <= 0:
        raise ValueError(f"n_frames_to_generate must be positive or 'auto', got {value!r}")
    return n


def _n_frames_cli_value(value: str) -> int | str:
    try:
        parsed = _parse_n_frames_value(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed is None:
        raise argparse.ArgumentTypeError("--n-frames expects a positive integer or 'auto'")
    return parsed


def _resolve_n_frames_to_generate(
    value: Any,
    *,
    action_frames: int,
    n_previous: int,
    gt_frames: int | None = None,
) -> tuple[int, str]:
    parsed = _parse_n_frames_value(value)
    if parsed == "auto" or parsed is None:
        total_frames = int(action_frames)
        if gt_frames is not None and int(gt_frames) > 0:
            total_frames = min(total_frames, int(gt_frames))
        return max(1, total_frames - int(n_previous)), "auto"
    return int(parsed), "fixed"


def _trim_generated_frame_dir(pred_dir: Path, n_frames: int) -> None:
    frame_files = sorted(
        p for p in pred_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ) if pred_dir.is_dir() else []
    for path in frame_files[int(n_frames):]:
        path.unlink()


def _count_available_gt_frames(ep: dict[str, Any]) -> int | None:
    if ep.get("data_format") == "external_worldmodel":
        source = ep.get("frames_dir")
    else:
        source = ep.get("video_mp4")
    if not source:
        return None
    source_path = Path(source)
    if source_path.is_dir():
        return sum(
            1 for p in source_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    if source_path.exists():
        import cv2

        cap = cv2.VideoCapture(str(source_path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return count if count > 0 else None
    return None


def _read_condition_frames(video_path: str | os.PathLike[str], n_previous: int) -> torch.Tensor:
    """Read the first n_previous frames from head_color.mp4 as legal context.

    Selection must not use future GT error. Reading the observed prefix frames
    and the action sequence is allowed; no oracle / future reconstruction error
    is computed in this script.
    """
    video_path = Path(video_path)
    if video_path.is_dir():
        from PIL import Image

        frame_files = sorted(
            p for p in video_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not frame_files:
            raise RuntimeError(f"No frame images found in {video_path}")
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in frame_files[:n_previous]]
        while len(frames) < n_previous:
            frames.append(frames[-1].copy())
        arr = np.stack(frames[:n_previous], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()

    try:
        import imageio.v3 as iio

        frames = []
        for idx, frame in enumerate(iio.imiter(video_path)):
            frames.append(np.asarray(frame)[..., :3])
            if idx + 1 >= n_previous:
                break
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while len(frames) < n_previous:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any condition frames from {video_path}")
    while len(frames) < n_previous:
        frames.append(frames[-1].copy())
    arr = np.stack(frames[:n_previous], axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()


def _load_future_gt_frames(
    ep: dict[str, Any],
    *,
    n_previous: int,
    n_frames: int,
) -> np.ndarray | None:
    """Load GT future frames aligned with EVAC predictions.

    EVAC inference writes frames after the observed prefix, so latent_gt should
    be encoded from GT frames starting at n_previous, not from frame 0.
    """
    if ep.get("data_format") == "external_worldmodel":
        source = ep.get("frames_dir")
    else:
        source = ep.get("video_mp4")
    if not source:
        return None
    frames = normalize_frames(load_frames(source, max_frames=n_previous + n_frames))
    if frames.shape[0] <= n_previous:
        return None
    frames = frames[n_previous:n_previous + n_frames]
    return frames if frames.shape[0] > 0 else None


def _record_to_agibot_episode(record: dict[str, Any]) -> dict[str, Any]:
    root = Path(record["path"])
    files = record.get("files", {})
    return {
        "task_id": record["task_id"],
        "ep_id": record_uid(record),
        "episode_id": record_uid(record),
        "raw_episode_id": record.get("raw_episode_id"),
        "segment_id": record.get("segment_id"),
        "video_mp4": str(root / files.get("video", "head_color.mp4")),
        "action_h5": str(root / files.get("proprio", "proprio_stats.h5")),
        "extrinsic": str(root / files.get("extrinsic", "head_extrinsic_params_aligned.json")),
        "intrinsic": str(root / files.get("intrinsic", "head_intrinsic_params.json")),
        "source_record": record,
        "data_format": "agibot",
        "dataset": "agibot",
    }


def _record_to_external_episode(record: dict[str, Any]) -> dict[str, Any]:
    meta_path = record.get("meta_path")
    meta = load_json(meta_path) if meta_path and Path(meta_path).exists() else {}
    camera_path = record.get("camera_path") or meta.get("camera_path")
    # Prefer actions_evac.npy (explicitly EVAC-compatible); fallback to generic actions_path
    actions_path = record.get("actions_path")
    if actions_path and Path(actions_path).exists():
        pass  # use as-is
    elif meta.get("evac_action_compatible"):
        ep_dir = Path(record.get("episode_dir", ""))
        actions_path = str(ep_dir / "actions_evac.npy")
    evac_action_compatible = (
        record.get("evac_action_compatible")
        or meta.get("evac_action_compatible", False)
    )
    return {
        "task_id": record.get("task_name", "external_task"),
        "task_name": record.get("task_name", meta.get("task_name", "unknown")),
        "ep_id": record.get("episode_id"),
        "episode_id": record.get("episode_id"),
        "frames_dir": record.get("frames_dir") or record.get("video_or_frames_path"),
        "actions_path": actions_path,
        "actions_delta_path": record.get("actions_delta_path"),
        "proprio_path": record.get("proprio_path"),
        "camera_path": camera_path,
        "meta_path": meta_path,
        "meta": meta,
        "source_record": record,
        "data_format": "external_worldmodel",
        "dataset": record.get("dataset", meta.get("dataset", "external")),
        "evac_action_compatible": evac_action_compatible,
    }


def _record_to_episode(record: dict[str, Any], data_format: str) -> dict[str, Any]:
    if data_format == "agibot":
        return _record_to_agibot_episode(record)
    if data_format == "external_worldmodel":
        return _record_to_external_episode(record)
    raise ValueError(f"Unsupported scoring.data_format={data_format!r}")


def _load_external_action_arrays(ep: dict[str, Any], n_previous: int, chunk: int, *, allow_incompatible: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Load EVAC-compatible actions from external_worldmodel episode.

    For 16-dim EE actions, recomputes action/delta through EVAC's get_actions
    path instead of trusting converter-side actions_delta_evac.npy.  This keeps
    scoring/val inference aligned with the training dataset's action transform.

    Rejects incompatible actions (qpos, padded joints) unless
    allow_incompatible_actions_for_debug=true.
    """
    # --- Safety gate ---
    evac_compatible = ep.get("evac_action_compatible", False)
    if not evac_compatible and not allow_incompatible:
        raise ValueError(
            f"Episode {ep.get('episode_id')} is NOT evac_action_compatible. "
            f"action_representation={ep.get('meta', {}).get('action_representation', 'unknown')}. "
            "To force-run with incompatible actions (debug only!), set "
            "scoring.allow_incompatible_actions_for_debug=true in config."
        )
    if not evac_compatible:
        print(
            f"[al scoring ⚠ DEBUG] Allowing incompatible actions for {ep.get('episode_id')}. "
            "C3 scores will NOT be reliable for AL experiments."
        )

    abs_path = ep.get("actions_path")
    if not abs_path:
        raise FileNotFoundError(f"External episode {ep.get('episode_id')} has no actions_path")
    if not Path(abs_path).exists():
        raise FileNotFoundError(f"actions_evac.npy not found: {abs_path}")

    arr_abs = np.load(abs_path).astype(np.float32)
    if arr_abs.ndim != 2:
        raise ValueError(f"actions_evac.npy must be [T,D], got {arr_abs.shape}")

    mean_v = torch.tensor(StatisticInfo["agibotworld"]["mean"]).unsqueeze(0)
    std_v = torch.tensor(StatisticInfo["agibotworld"]["std"]).unsqueeze(0)

    def _normalize_delta(delta: torch.Tensor) -> torch.Tensor:
        delta = delta.clone()
        delta[:, :6] = (delta[:, :6] - mean_v[:, :6]) / std_v[:, :6]
        delta[:, 7:13] = (delta[:, 7:13] - mean_v[:, 6:]) / std_v[:, 6:]
        return delta

    # Match AgiBotWorldChallengeDataset.get_action_npy: recompute EVAC visual
    # quaternions and normalized delta_action from the absolute EE actions.
    action_dim = arr_abs.shape[-1]
    if action_dim == 16:
        slices = list(range(arr_abs.shape[0]))
        action_np, delta_np = get_actions(
            gripper=np.stack((arr_abs[:, 7], arr_abs[:, 15]), axis=1),
            all_ends_p=np.stack((arr_abs[:, 0:3], arr_abs[:, 8:11]), axis=1),
            all_ends_o=np.stack((arr_abs[:, 3:7], arr_abs[:, 11:15]), axis=1),
            slices=slices,
            delta_act_sidx=n_previous,
        )
        action = torch.from_numpy(action_np.astype(np.float32)).float()
        delta = _normalize_delta(torch.from_numpy(delta_np.astype(np.float32)).float())
    elif action_dim == 14:
        delta = _normalize_delta(torch.from_numpy(arr_abs).float())
        action = torch.zeros(delta.shape[0] + n_previous, 16, dtype=torch.float32)
    else:
        raise ValueError(
            f"actions_evac.npy has unsupported action_dim={action_dim} for {ep.get('episode_id')}. "
            "Expected 14 (delta) or 16 (absolute EE)."
        )
    return action, delta


def _external_camera_fallback(n_frames: int, meta: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n_frames, 1, 1)
    w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n_frames, 1, 1)
    intrinsic_vals = meta.get("intrinsic", {}) if isinstance(meta, dict) else {}
    fx = float(intrinsic_vals.get("fx", 1.0))
    fy = float(intrinsic_vals.get("fy", 1.0))
    ppx = float(intrinsic_vals.get("ppx", 0.5))
    ppy = float(intrinsic_vals.get("ppy", 0.5))
    intrinsic = torch.tensor([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return c2w, w2c, intrinsic


def _load_external_camera(ep: dict[str, Any], n_frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    camera_path = ep.get("camera_path")
    if not camera_path:
        return _external_camera_fallback(n_frames, ep.get("meta", {}))
    path = Path(camera_path)
    if not path.exists():
        raise FileNotFoundError(f"External camera_path does not exist: {path}")
    data = np.load(path)
    if "extrinsic_cv" in data:
        w2c_np = np.asarray(data["extrinsic_cv"], dtype=np.float32)
        if w2c_np.ndim == 2:
            w2c_np = w2c_np[None]
        if w2c_np.shape[-2:] == (3, 4):
            pad = np.zeros((*w2c_np.shape[:-2], 1, 4), dtype=np.float32)
            pad[..., 0, 3] = 1.0
            w2c_np = np.concatenate([w2c_np, pad], axis=-2)
        if w2c_np.shape[-2:] != (4, 4):
            raise ValueError(f"extrinsic_cv must have shape [T,3,4]/[T,4,4], got {w2c_np.shape} in {path}")
        if w2c_np.shape[0] < n_frames:
            w2c_np = np.concatenate([w2c_np, np.repeat(w2c_np[-1:], n_frames - w2c_np.shape[0], axis=0)], axis=0)
        w2c = torch.from_numpy(w2c_np[:n_frames]).float()
        c2w = torch.linalg.inv(w2c).float()
    elif "cam2world_gl" in data:
        c2w_np = np.asarray(data["cam2world_gl"], dtype=np.float32)
        if c2w_np.ndim == 2:
            c2w_np = c2w_np[None]
        if c2w_np.shape[-2:] != (4, 4):
            raise ValueError(f"cam2world_gl must have shape [T,4,4] or [4,4], got {c2w_np.shape} in {path}")
        if c2w_np.shape[0] < n_frames:
            c2w_np = np.concatenate([c2w_np, np.repeat(c2w_np[-1:], n_frames - c2w_np.shape[0], axis=0)], axis=0)
        c2w = torch.from_numpy(c2w_np[:n_frames]).float()
        w2c = torch.linalg.inv(c2w).float()
    else:
        c2w, w2c, _ = _external_camera_fallback(n_frames, ep.get("meta", {}))

    if "intrinsic_cv" in data:
        intrinsic_np = np.asarray(data["intrinsic_cv"], dtype=np.float32)
        if intrinsic_np.ndim == 3:
            intrinsic_np = intrinsic_np[0]
        if intrinsic_np.shape != (3, 3):
            raise ValueError(f"intrinsic_cv must have shape [T,3,3] or [3,3], got {intrinsic_np.shape} in {path}")
        intrinsic = torch.from_numpy(intrinsic_np).float()
    else:
        intrinsic = _external_camera_fallback(n_frames, ep.get("meta", {}))[2]
    return c2w, w2c, intrinsic


def _build_probe_args(scoring_cfg: dict[str, Any], device: str) -> SimpleNamespace:
    return SimpleNamespace(
        n_chunk=int(scoring_cfg.get("n_chunk", -1)),
        ddim_steps=int(scoring_cfg.get("ddim_steps", 27)),
        cfg=float(scoring_cfg.get("cfg", 1.0)),
        gr=float(scoring_cfg.get("gr", 0.7)),
        n_probe_ts=int(scoring_cfg.get("n_probe_ts", 3)),
        probe_ts_min=int(scoring_cfg.get("probe_ts_min", 50)),
        probe_ts_max=int(scoring_cfg.get("probe_ts_max", 200)),
        probe_ts_reduce=scoring_cfg.get("probe_ts_reduce", "mean"),
        probe_t_ref=scoring_cfg.get("probe_t_ref"),
        probe_chunk_stride=scoring_cfg.get("probe_chunk_stride"),
        probe_tau=scoring_cfg.get("probe_tau"),
        effective_probe_tau=None,
        save_probe_ts_stack=bool(scoring_cfg.get("save_probe_ts_stack", False)),
        save_latents=bool(scoring_cfg.get("save_latents", True)),
        save_hdec_embedding=True,
        n_frames_to_generate=_parse_n_frames_value(scoring_cfg.get("n_frames_to_generate", "auto")),
        extreme_length=None,
        force_low_quality=False,
        save_pred_video=False,
        random_seed=int(scoring_cfg.get("random_seed", 42)),
        confidence_format=scoring_cfg.get("confidence_format", "probability"),
        confidence_out_of_range=scoring_cfg.get("confidence_out_of_range", "error"),
        data_format=scoring_cfg.get("data_format", "agibot"),
        allow_incompatible_actions_for_debug=bool(scoring_cfg.get("allow_incompatible_actions_for_debug", False)),
        overwrite=bool(scoring_cfg.get("overwrite", False)),
        device=device,
    )


def _to_video_frames(frames: Any) -> np.ndarray:
    """Convert model-output traj tensor to uint8 numpy for save_video."""
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    return frames


def _apply_traj_conditioning_overrides(model: Any, cfg: dict[str, Any]) -> None:
    for key in ("traj_gripper_z_offset", "traj_keypoint_scale", "traj_radius"):
        if key in cfg:
            value = cfg[key]
            if key == "traj_radius":
                value = int(value)
            else:
                value = float(value)
            setattr(model, key, value)


def load_model_for_prediction(config_path: str, device: torch.device, evac_ckpt_path: str | None = None) -> tuple[Any, Any]:
    """Load only the EVAC backbone for prediction-cache generation."""
    config = OmegaConf.load(config_path)
    model_cfg = config.model
    if evac_ckpt_path:
        model_cfg.pretrained_checkpoint = str(evac_ckpt_path)
        print(f"[load] EVAC checkpoint override: {evac_ckpt_path}")
    try:
        model_cfg.params.enable_c3_probe = False
    except Exception:
        pass
    model = instantiate_from_config(model_cfg)
    model = load_checkpoints(model, model_cfg, ignore_mismatched_sizes=True)
    model = model.to(device).eval()
    return model, config


def run_pool_episode(
    model: Any,
    config: Any,
    ep: dict[str, Any],
    device: torch.device,
    args: SimpleNamespace,
    output_root: Path,
    score_root: Path | None = None,
) -> dict[str, Any] | None:
    chunk = int(config.chunk)
    n_previous = int(config.n_previous)
    if args.data_format == "agibot":
        img = _read_condition_frames(ep["video_mp4"], n_previous)
        action, delta_action = load_action_h5(ep["action_h5"], args.n_chunk, chunk, n_previous)
        n = int(action.shape[0])
        c2w, w2c, intrinsic = load_caminfo_json(ep["extrinsic"], ep["intrinsic"], n)
    elif args.data_format == "external_worldmodel":
        img = _read_condition_frames(ep["frames_dir"], n_previous)
        allow_incompat = bool(getattr(args, "allow_incompatible_actions_for_debug", False))
        action, delta_action = _load_external_action_arrays(ep, n_previous, chunk, allow_incompatible=allow_incompat)
        n = int(action.shape[0])
        c2w, w2c, intrinsic = _load_external_camera(ep, n)
    else:
        raise ValueError(f"Unsupported data_format={args.data_format}")

    n_frames_to_generate, n_frames_mode = _resolve_n_frames_to_generate(
        args.n_frames_to_generate,
        action_frames=n,
        n_previous=n_previous,
        gt_frames=_count_available_gt_frames(ep),
    )
    n_chunk_to_pred = int(math.ceil(float(n_frames_to_generate) / chunk))
    needed = n_previous + n_frames_to_generate
    if n < needed:
        pad_len = needed - n
        action = torch.cat([action, action[-1:].repeat(pad_len, 1)], dim=0)
        delta_action = torch.cat([delta_action, delta_action[-1:].repeat(pad_len, 1)], dim=0)
        c2w = torch.cat([c2w, c2w[-1:].repeat(pad_len, 1, 1)], dim=0)
        w2c = torch.cat([w2c, w2c[-1:].repeat(pad_len, 1, 1)], dim=0)
        n = int(action.shape[0])

    ep_id = ep["episode_id"]
    ep_out = output_root / ep_id
    _score_root = score_root or output_root
    score_ep_out = _score_root / ep_id
    if bool(getattr(args, "overwrite", False)):
        for d in (ep_out, score_ep_out):
            if d.exists():
                if d.is_dir():
                    shutil.rmtree(d)
                else:
                    d.unlink()
    ep_out = ensure_dir(ep_out)
    score_ep_out = ensure_dir(score_ep_out)
    pred_dir = ensure_dir(ep_out / "pred_frames")

    with (
        open(os.devnull, "w") as quiet,
        contextlib.redirect_stdout(quiet),
        contextlib.redirect_stderr(quiet),
    ):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            infer_out = model.inference(
                config,
                img.to(device),
                action,
                delta_action,
                c2w,
                w2c,
                intrinsic,
                str(pred_dir),
                n_chunk_to_pred,
                chunk=chunk,
                n_previous=n_previous,
                n_valid=n - n_previous,
                unconditional_guidance_scale=float(args.cfg),
                guidance_rescale=float(args.gr),
                ddim_steps=int(args.ddim_steps),
                saving_tag="",
                saving_video=False,
                video_dir=None,
                lambda_guide=0.0,
                return_latents=bool(args.save_latents),
            )

    internal_pred_latents_raw = None
    if args.save_latents and isinstance(infer_out, tuple) and len(infer_out) >= 3:
        internal_pred_latents_raw = infer_out[2]

    save_traj = bool(getattr(args, "save_traj_videos", False))
    if save_traj and isinstance(infer_out, tuple) and len(infer_out) >= 2:
        save_video(_to_video_frames(infer_out[1])[:n_frames_to_generate], ep_out / "traj_condition.mp4", fps=8)

    gen_frames = load_generated_frames(str(pred_dir))
    if gen_frames is None or len(gen_frames) == 0:
        return None
    if int(gen_frames.shape[0]) < n_frames_to_generate:
        raise RuntimeError(
            f"Generated only {int(gen_frames.shape[0])}/{n_frames_to_generate} frames for {ep['episode_id']}"
        )
    if int(gen_frames.shape[0]) > n_frames_to_generate:
        _trim_generated_frame_dir(pred_dir, n_frames_to_generate)
        gen_frames = gen_frames[:n_frames_to_generate]

    T = int(gen_frames.shape[0])
    if delta_action.shape[0] >= T:
        delta_action_out = delta_action[:T]
    elif delta_action.shape[0] > 0:
        delta_action_out = torch.cat([delta_action, delta_action[-1:].repeat(T - delta_action.shape[0], 1)], dim=0)
    else:
        delta_action_out = torch.zeros(T, 14, dtype=torch.float32)

    action_transform = EXTERNAL_ACTION_TRANSFORM if args.data_format == "external_worldmodel" else "agibot_h5_normalized"
    camera_transform = EXTERNAL_CAMERA_TRANSFORM if args.data_format == "external_worldmodel" else "agibot_json_c2w"

    if bool(getattr(args, "prediction_only", False)):
        actions_path = ep_out / "actions.npy"
        meta_pred_path = ep_out / "meta.json"
        np.save(actions_path, delta_action_out.cpu().numpy().astype(np.float32))
        meta_pred = {
            "episode_id": ep_id,
            "task_id": ep["task_id"],
            "task_name": ep.get("task_name", ep["task_id"]),
            "dataset": ep.get("dataset", "agibot"),
            "data_format": args.data_format,
            "source_path": ep["source_record"].get("path") or ep["source_record"].get("episode_dir"),
            "n_condition_frames": n_previous,
            "n_generated_frames": T,
            "n_frames_to_generate_requested": args.n_frames_to_generate,
            "n_frames_to_generate_resolved": int(n_frames_to_generate),
            "n_frames_to_generate_mode": n_frames_mode,
            "action_transform": action_transform,
            "camera_transform": camera_transform,
            "overwrite": bool(getattr(args, "overwrite", False)),
            "prediction_only": True,
            "save_latents": False,
            "pred_frames_dir": str(pred_dir),
            "actions_path": str(actions_path),
            "latent_pred_path": None,
            "latent_gt_path": None,
        }
        save_json(meta_pred, meta_pred_path)
        return {
            "episode_id": ep_id,
            "task_id": ep["task_id"],
            "task_name": ep.get("task_name", ep["task_id"]),
            "dataset": ep.get("dataset", "agibot"),
            "data_format": args.data_format,
            "raw_episode_id": ep.get("raw_episode_id"),
            "segment_id": ep.get("segment_id"),
            "split": "candidate_pool",
            "cache_dir": str(ep_out),
            "pred_frames_dir": str(pred_dir),
            "actions_path": str(actions_path),
            "meta_path": str(meta_pred_path),
            "action_transform": action_transform,
            "camera_transform": camera_transform,
            "prediction_ready": True,
            "score_ready": True,
        }

    internal_pred_latents = internal_samples_to_latent_array(
        internal_pred_latents_raw,
        n_previous=n_previous,
        n_frames=gen_frames.shape[0],
    )
    with (
        open(os.devnull, "w") as quiet2,
        contextlib.redirect_stdout(quiet2),
        contextlib.redirect_stderr(quiet2),
    ):
        probe_out = extract_probe_confidence(
            model,
            config,
            gen_frames,
            device,
            args,
            ref_img=img,
            action=action,
            delta_action=delta_action,
            c2w=c2w,
            w2c=w2c,
            intrinsic=intrinsic,
        )
    conf_map, conf_stack, pred_latents, hdec_embedding = probe_out
    conf_map, conf_format_info = validate_confidence_map(
        conf_map,
        confidence_format=args.confidence_format,
        out_of_range=args.confidence_out_of_range,
    )

    conf_path = score_ep_out / "conf_map.npy"
    actions_path = ep_out / "actions.npy"
    hdec_path = score_ep_out / "hdec_embedding.npy"
    risk_path = score_ep_out / "risk_stats.json"
    meta_path = score_ep_out / "meta.json"
    meta_pred_path = ep_out / "meta.json"

    np.save(conf_path, conf_map.astype(np.float32))
    np.save(actions_path, delta_action_out.cpu().numpy().astype(np.float32))
    if hdec_embedding is None:
        # Fallback keeps diverse selector runnable and records the source in meta.
        stats_for_fallback = compute_risk_stats(conf_map)
        hdec_embedding = np.asarray(
            [
                stats_for_fallback["mean_risk"],
                stats_for_fallback["tail_risk_top5"],
                stats_for_fallback["tail_risk_top10"],
                stats_for_fallback["persistent_risk"],
                stats_for_fallback["risk_area"],
            ],
            dtype=np.float32,
        )
        emb_source = "risk_stats_fallback"
    else:
        emb_source = "h_dec_mean_pool"
    np.save(hdec_path, hdec_embedding.astype(np.float32))

    if args.save_probe_ts_stack and conf_stack is not None:
        np.save(score_ep_out / "conf_map_by_ts.npy", conf_stack.astype(np.float32))
    if args.save_latents:
        latent = internal_pred_latents if internal_pred_latents is not None else pred_latents
        if latent is not None:
            np.save(ep_out / "latent_pred.npy", latent.astype(np.float32))
        gt_frames = _load_future_gt_frames(ep, n_previous=n_previous, n_frames=T)
        if gt_frames is not None and gt_frames.shape[0] == T:
            sample_size = tuple(config.data.params.train.params.sample_size)
            latent_gt = encode_rgb_frames_to_latents(
                model,
                gt_frames,
                device,
                sample_size=sample_size,
            )
            np.save(ep_out / "latent_gt.npy", latent_gt.astype(np.float32))
        missing_latents = [
            name for name in ("latent_pred.npy", "latent_gt.npy")
            if not (ep_out / name).exists()
        ]
        if missing_latents:
            raise RuntimeError(
                "save_latents requested but scorer did not produce: "
                + ", ".join(missing_latents)
            )

    risk_stats = compute_risk_stats(conf_map)
    save_json(risk_stats, risk_path)

    meta = {
        "episode_id": ep_id,
        "task_id": ep["task_id"],
        "task_name": ep.get("task_name", ep["task_id"]),
        "dataset": ep.get("dataset", "agibot"),
        "data_format": args.data_format,
        "raw_episode_id": ep.get("raw_episode_id"),
        "segment_id": ep.get("segment_id"),
        "source_path": ep["source_record"].get("path") or ep["source_record"].get("episode_dir"),
        "selection_protocol": "condition frames + actions + EVAC prediction + C3 confidence only; no future GT/oracle error.",
        "hdec_embedding_source": emb_source,
        "n_condition_frames": n_previous,
        "n_generated_frames": T,
        "n_frames_to_generate_requested": args.n_frames_to_generate,
        "n_frames_to_generate_resolved": int(n_frames_to_generate),
        "n_frames_to_generate_mode": n_frames_mode,
        "conf_map_shape": list(conf_map.shape),
        "hdec_embedding_shape": list(hdec_embedding.shape),
        "probe_tau": getattr(args, "effective_probe_tau", args.probe_tau),
        "confidence_format": conf_format_info,
        "action_transform": action_transform,
        "camera_transform": camera_transform,
        "overwrite": bool(getattr(args, "overwrite", False)),
        "save_latents": bool(args.save_latents),
        "pred_cache_dir": str(ep_out),
        "pred_frames_dir": str(pred_dir),
        "actions_path": str(actions_path),
        "latent_pred_path": str(ep_out / "latent_pred.npy") if (ep_out / "latent_pred.npy").exists() else None,
        "latent_gt_path": str(ep_out / "latent_gt.npy") if (ep_out / "latent_gt.npy").exists() else None,
        "score_cache_dir": str(score_ep_out),
        "conf_map_path": str(conf_path),
        "hdec_embedding_path": str(hdec_path),
        "risk_stats_path": str(risk_path),
    }
    save_json(meta, meta_path)

    # Pred-only meta (no C3 fields)
    meta_pred = {
        "episode_id": ep_id,
        "task_id": ep["task_id"],
        "task_name": ep.get("task_name", ep["task_id"]),
        "dataset": ep.get("dataset", "agibot"),
        "data_format": args.data_format,
        "source_path": ep["source_record"].get("path") or ep["source_record"].get("episode_dir"),
        "n_condition_frames": n_previous,
        "n_generated_frames": T,
        "n_frames_to_generate_requested": args.n_frames_to_generate,
        "n_frames_to_generate_resolved": int(n_frames_to_generate),
        "n_frames_to_generate_mode": n_frames_mode,
        "action_transform": action_transform,
        "camera_transform": camera_transform,
        "overwrite": bool(getattr(args, "overwrite", False)),
        "save_latents": bool(args.save_latents),
        "pred_frames_dir": str(pred_dir),
        "actions_path": str(actions_path),
        "latent_pred_path": str(ep_out / "latent_pred.npy") if (ep_out / "latent_pred.npy").exists() else None,
        "latent_gt_path": str(ep_out / "latent_gt.npy") if (ep_out / "latent_gt.npy").exists() else None,
        "score_cache_dir": str(score_ep_out),
    }
    save_json(meta_pred, meta_pred_path)

    return {
        "episode_id": ep_id,
        "task_id": ep["task_id"],
        "task_name": ep.get("task_name", ep["task_id"]),
        "dataset": ep.get("dataset", "agibot"),
        "data_format": args.data_format,
        "raw_episode_id": ep.get("raw_episode_id"),
        "segment_id": ep.get("segment_id"),
        "split": "candidate_pool",
        "cache_dir": str(score_ep_out),
        "pred_frames_dir": str(pred_dir),
        "conf_map_path": str(conf_path),
        "actions_path": str(actions_path),
        "hdec_embedding_path": str(hdec_path),
        "risk_stats_path": str(risk_path),
        "meta_path": str(meta_path),
        "action_transform": action_transform,
        "camera_transform": camera_transform,
        "latent_pred_path": str(ep_out / "latent_pred.npy") if (ep_out / "latent_pred.npy").exists() else None,
        "latent_gt_path": str(ep_out / "latent_gt.npy") if (ep_out / "latent_gt.npy").exists() else None,
        "score_ready": True,
        **risk_stats,
    }


def _has_generated_frames(pred_dir: Path) -> bool:
    return _count_generated_frames(pred_dir) > 0


def _count_generated_frames(pred_dir: Path) -> int:
    if not pred_dir.is_dir():
        return 0
    return sum(1 for p in pred_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _expected_action_transform(data_format: str) -> str:
    return EXTERNAL_ACTION_TRANSFORM if data_format == "external_worldmodel" else "agibot_h5_normalized"


def _expected_camera_transform(data_format: str) -> str:
    return EXTERNAL_CAMERA_TRANSFORM if data_format == "external_worldmodel" else "agibot_json_c2w"


def _episode_output_complete(ep_out: Path, args: SimpleNamespace, pred_ep_out: Path | None = None) -> bool:
    _pred = pred_ep_out or ep_out
    required = [
        ep_out / "meta.json",
        ep_out / "risk_stats.json",
        ep_out / "conf_map.npy",
        _pred / "actions.npy",
        ep_out / "hdec_embedding.npy",
    ]
    if any(not path.exists() for path in required):
        return False
    try:
        meta = load_json(str(ep_out / "meta.json"))
    except Exception:
        return False
    if getattr(args, "data_format", None) == "external_worldmodel":
        if meta.get("action_transform") != EXTERNAL_ACTION_TRANSFORM:
            return False
        if meta.get("camera_transform") != EXTERNAL_CAMERA_TRANSFORM:
            return False
    actual_frames = _count_generated_frames(_pred / "pred_frames")
    if actual_frames <= 0:
        return False
    expected_frames = getattr(args, "n_frames_to_generate", None)
    if expected_frames == "auto":
        if meta.get("n_frames_to_generate_mode") != "auto":
            return False
        resolved = int(meta.get("n_frames_to_generate_resolved") or meta.get("n_generated_frames") or 0)
        if resolved <= 0:
            return False
        if actual_frames != resolved or int(meta.get("n_generated_frames", 0) or 0) != resolved:
            return False
    elif expected_frames is not None:
        expected_frames = int(expected_frames)
        if actual_frames != expected_frames:
            return False
        if int(meta.get("n_generated_frames", 0) or 0) != expected_frames:
            return False
    if bool(getattr(args, "save_probe_ts_stack", False)) and not (ep_out / "conf_map_by_ts.npy").exists():
        return False
    if bool(getattr(args, "save_latents", False)):
        if not (_pred / "latent_pred.npy").exists():
            return False
        if not (_pred / "latent_gt.npy").exists():
            return False
    return True


def _try_backfill_latents(
    ep_out: Path,
    args: SimpleNamespace,
    model: torch.nn.Module | None,
    config: Any | None,
    device: torch.device | None,
    pred_ep_out: Path | None = None,
) -> bool:
    _pred = pred_ep_out or ep_out
    if not bool(getattr(args, "save_latents", False)):
        return False
    if (_pred / "latent_pred.npy").exists() and (_pred / "latent_gt.npy").exists():
        return True
    if model is None or config is None or device is None:
        return False
    meta_path = ep_out / "meta.json"
    pred_frames_dir = _pred / "pred_frames"
    if not meta_path.exists() or not _has_generated_frames(pred_frames_dir):
        return False
    try:
        meta = load_json(str(meta_path))
        n_previous = int(meta.get("n_condition_frames", 4))
        n_frames = int(meta.get("n_generated_frames", 0))
        pred_frames = load_generated_frames(pred_frames_dir)
        if pred_frames is None or pred_frames.shape[0] == 0:
            return False
        if n_frames <= 0:
            n_frames = int(pred_frames.shape[0])
        pred_frames = pred_frames[:n_frames]
        if pred_frames.shape[0] != n_frames:
            return False

        source_path = meta.get("source_path")
        data_format = meta.get("data_format", getattr(args, "data_format", "agibot"))
        ep: dict[str, Any] = {"data_format": data_format}
        if data_format == "external_worldmodel":
            ep["frames_dir"] = str(Path(source_path) / "frames") if source_path else None
        else:
            ep["video_mp4"] = str(Path(source_path) / "head_color.mp4") if source_path else None
        gt_frames = _load_future_gt_frames(ep, n_previous=n_previous, n_frames=n_frames)
        if gt_frames is None or gt_frames.shape[0] != n_frames:
            return False

        sample_size = tuple(config.data.params.train.params.sample_size)
        if not (_pred / "latent_pred.npy").exists():
            latent_pred = encode_rgb_frames_to_latents(model, pred_frames, device, sample_size=sample_size)
            np.save(_pred / "latent_pred.npy", latent_pred.astype(np.float32))
        if not (_pred / "latent_gt.npy").exists():
            latent_gt = encode_rgb_frames_to_latents(model, gt_frames, device, sample_size=sample_size)
            np.save(_pred / "latent_gt.npy", latent_gt.astype(np.float32))

        meta["save_latents"] = True
        meta["latent_pred_path"] = str(_pred / "latent_pred.npy")
        meta["latent_gt_path"] = str(_pred / "latent_gt.npy")
        meta["latent_backfilled"] = True
        meta["pred_cache_dir"] = str(_pred)
        meta.setdefault(
            "action_transform",
            EXTERNAL_ACTION_TRANSFORM if data_format == "external_worldmodel" else "agibot_h5_normalized",
        )
        meta.setdefault(
            "camera_transform",
            EXTERNAL_CAMERA_TRANSFORM if data_format == "external_worldmodel" else "agibot_json_c2w",
        )
        save_json(meta, meta_path)
        print(f"[al scoring] backfilled latents for {ep_out.name}")
        return True
    except Exception as exc:
        print(f"[al scoring WARNING] latent backfill failed for {ep_out.name}: {exc!r}")
        return False


def _try_load_cached_row(ep_out: Path, ep: dict[str, Any], data_format: str, args: SimpleNamespace, pred_ep_out: Path | None = None) -> dict[str, Any] | None:
    """If the episode was already fully scored, reconstruct its row from saved files.

    Checks for meta.json (the last file written per episode) as the completion
    marker.  Returns None when the episode still needs scoring.
    """
    _pred = pred_ep_out or ep_out
    if not _episode_output_complete(ep_out, args, pred_ep_out=_pred):
        return None
    meta_path = ep_out / "meta.json"
    risk_path = ep_out / "risk_stats.json"
    meta = load_json(meta_path)
    risk_stats = load_json(risk_path)
    return {
        "episode_id": meta.get("episode_id", ep["episode_id"]),
        "task_id": meta.get("task_id", ep["task_id"]),
        "task_name": meta.get("task_name", ep.get("task_name", ep["task_id"])),
        "dataset": meta.get("dataset", ep.get("dataset", "agibot")),
        "data_format": meta.get("data_format", data_format),
        "raw_episode_id": meta.get("raw_episode_id", ep.get("raw_episode_id")),
        "segment_id": meta.get("segment_id", ep.get("segment_id")),
        "split": "candidate_pool",
        "cache_dir": str(ep_out),
        "pred_frames_dir": str(_pred / "pred_frames"),
        "conf_map_path": str(ep_out / "conf_map.npy"),
        "actions_path": str(_pred / "actions.npy"),
        "hdec_embedding_path": str(ep_out / "hdec_embedding.npy"),
        "risk_stats_path": str(risk_path),
        "meta_path": str(meta_path),
        "action_transform": meta.get("action_transform"),
        "camera_transform": meta.get("camera_transform"),
        "latent_pred_path": str(_pred / "latent_pred.npy") if (_pred / "latent_pred.npy").exists() else None,
        "latent_gt_path": str(_pred / "latent_gt.npy") if (_pred / "latent_gt.npy").exists() else None,
        "score_ready": True,
        **risk_stats,
    }


def _shard_file_path(
    output_dir: Path,
    shard_id: int,
    worker_id: int,
    num_shards: int,
    workers_per_gpu: int,
) -> Path:
    """Return the per-worker shard JSON path."""
    if num_shards > 1 or workers_per_gpu > 1:
        parts: list[str] = []
        if num_shards > 1:
            parts.append(f"shard{shard_id}")
        if workers_per_gpu > 1:
            parts.append(f"worker{worker_id}")
        return output_dir / f"scored_pool_{'_'.join(parts)}.json"
    return output_dir / "scored_pool.json"


def _reconstruct_item_from_dir(
    ep_dir: Path,
    args: SimpleNamespace,
    *,
    model: torch.nn.Module | None = None,
    config: Any | None = None,
    device: torch.device | None = None,
    pred_ep_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Reconstruct a scored-pool item from an episode output directory.

    Returns ``None`` when the directory doesn't contain a completed meta.json.
    """
    _pred = pred_ep_dir or ep_dir
    if not _episode_output_complete(ep_dir, args, pred_ep_out=_pred):
        _try_backfill_latents(ep_dir, args, model, config, device, pred_ep_out=_pred)
    if not _episode_output_complete(ep_dir, args, pred_ep_out=_pred):
        return None
    meta_path = ep_dir / "meta.json"
    risk_path = ep_dir / "risk_stats.json"
    try:
        meta = load_json(str(meta_path))
        risk_stats = load_json(str(risk_path))
        return {
            "episode_id": meta.get("episode_id", ep_dir.name),
            "task_id": meta.get("task_id", ""),
            "task_name": meta.get("task_name", ""),
            "dataset": meta.get("dataset", "agibot"),
            "data_format": meta.get("data_format", "agibot"),
            "raw_episode_id": meta.get("raw_episode_id"),
            "segment_id": meta.get("segment_id"),
            "split": "candidate_pool",
            "cache_dir": str(ep_dir),
            "pred_frames_dir": str(_pred / "pred_frames"),
            "conf_map_path": str(ep_dir / "conf_map.npy"),
            "actions_path": str(_pred / "actions.npy"),
            "hdec_embedding_path": str(ep_dir / "hdec_embedding.npy"),
            "risk_stats_path": str(risk_path),
            "meta_path": str(meta_path),
            "action_transform": meta.get("action_transform"),
            "camera_transform": meta.get("camera_transform"),
            "latent_pred_path": str(_pred / "latent_pred.npy") if (_pred / "latent_pred.npy").exists() else None,
            "latent_gt_path": str(_pred / "latent_gt.npy") if (_pred / "latent_gt.npy").exists() else None,
            "score_ready": True,
            **risk_stats,
        }
    except Exception:
        return None


def _recover_existing(
    output_dir: Path,
    shard_path: Path,
    args: SimpleNamespace,
    *,
    assigned_ids: set[str] | None = None,
    model: torch.nn.Module | None = None,
    config: Any | None = None,
    device: torch.device | None = None,
    pred_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], set[str]]:
    """Recover already-scored episodes from a previous run.

    1. If *shard_path* exists, load items/failures from it.
    2. Otherwise scan *output_dir* (score dir) for episode subdirectories with a
       completed ``meta.json``, reconstruct items, and persist them to
       *shard_path* (so the next resume is instantaneous).

    Returns ``(scored_items, failures, done_episode_ids)``.
    """
    _pred = pred_dir or output_dir
    if shard_path.exists():
        data = load_json(str(shard_path))
        items: list[dict[str, Any]] = []
        stale_items = 0
        for item in data.get("items", []):
            if assigned_ids is not None and item.get("episode_id") not in assigned_ids:
                continue
            cache_dir = Path(item.get("cache_dir") or output_dir / str(item.get("episode_id", "")))
            item_pred = Path(item.get("pred_frames_dir", "")).parent if item.get("pred_frames_dir") else _pred / str(item.get("episode_id", ""))
            refreshed = _reconstruct_item_from_dir(cache_dir, args, model=model, config=config, device=device, pred_ep_dir=item_pred)
            if refreshed is None:
                stale_items += 1
                continue
            items.append(refreshed)
        fails: list[dict[str, str]] = []
        done: set[str] = {it["episode_id"] for it in items}
        print(
            f"[al scoring] recovered {len(items)} complete cached episodes "
            f"({stale_items} stale/incomplete will be rescored) "
            f"from {shard_path.name}"
        )
        return items, fails, done

    # No shard file yet — scan individual episode directories (score dir)
    items = []
    for ep_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        if assigned_ids is not None and ep_dir.name not in assigned_ids:
            continue
        item_pred = _pred / ep_dir.name
        item = _reconstruct_item_from_dir(ep_dir, args, model=model, config=config, device=device, pred_ep_dir=item_pred)
        if item is not None:
            items.append(item)
    fails: list[dict[str, str]] = []

    done: set[str] = {it["episode_id"] for it in items}

    if items or fails:
        print(
            f"[al scoring] scanned episode dirs → {len(items)} scored + {len(fails)} failed; "
            f"persisting to {shard_path.name}"
        )
        # Immediately persist so next resume is fast
        _save_shard_file(shard_path, items, fails, len(items) + len(fails), 0, 0, len(items))

    return items, fails, done


def _save_shard_file(
    shard_path: Path,
    scored: list[dict[str, Any]],
    failures: list[dict[str, str]],
    total_assigned: int,
    cached_count: int,
    newly_scored: int,
    total_scored: int,
    *,
    source_manifest: str = "",
    probe_step: str = "",
    protocol: dict[str, Any] | None = None,
) -> None:
    """Atomically write the current shard state (crash-resilient)."""
    tmp_path = shard_path.with_suffix(".tmp")
    payload: dict[str, Any] = {
        "source_manifest": source_manifest,
        "probe_step": probe_step,
        "items": scored,
        "failures": failures,
        "stats": {
            "requested": total_assigned,
            "scored": total_scored,
            "failed": len(failures),
            "cached": cached_count,
            "newly_scored_this_run": newly_scored,
        },
    }
    if protocol is not None:
        payload["protocol"] = protocol
    save_json(payload, tmp_path)
    tmp_path.replace(shard_path)


def _try_merge_all_shards(output_dir: Path) -> dict[str, Any] | None:
    """Merge all available shard files into ``scored_pool.json``.

    Called after every episode so the merged file always reflects the latest
    state.  Shard files are *not* deleted — they remain as the source of truth
    for crash recovery.
    """
    shard_files = sorted(output_dir.glob("scored_pool_*.json"))
    if not shard_files:
        return None
    merged_path = output_dir / "scored_pool.json"
    shard_payloads = []
    for p in shard_files:
        try:
            shard_payloads.append(load_json(str(p)))
        except Exception:
            pass  # skip corrupt/incomplete shard files from crashed runs
    if not shard_payloads:
        return None
    merged_items = []
    merged_failures = []
    for p in shard_payloads:
        merged_items.extend(p.get("items", []))
        merged_failures.extend(p.get("failures", []))
    merged = dict(shard_payloads[0])
    merged["items"] = merged_items
    merged["failures"] = merged_failures
    merged["stats"]["scored"] = len(merged_items)
    merged["stats"]["failed"] = len(merged_failures)
    merged["stats"]["requested"] = sum(p["stats"].get("requested", 0) for p in shard_payloads)
    merged["stats"]["cached"] = sum(p["stats"].get("cached", 0) for p in shard_payloads)
    merged.setdefault("protocol", {})
    merged["protocol"]["shard_merged"] = True
    merged["protocol"]["source_shard_files"] = [str(p.name) for p in shard_files]
    save_json(merged, merged_path)
    return merged


def _merge_shard_outputs(output_dir: Path, scored_path: Path) -> dict[str, Any]:
    """Merge all shard files and delete them (called only when fully done)."""
    merged = _try_merge_all_shards(score_dir)
    if merged is None:
        raise FileNotFoundError(f"No shard outputs found in {output_dir}")
    # Only delete shards when we know all workers have finished
    for sf in sorted(output_dir.glob("scored_pool_*.json")):
        sf.unlink()
    return merged


def run_from_config(
    config_path: str,
    *,
    max_episodes: int | None = None,
    manifest_override: str | None = None,
    output_dir_override: str | None = None,
    score_dir_override: str | None = None,
    evac_checkpoint_override: str | None = None,
    device_override: str | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    worker_id: int = 0,
    workers_per_gpu: int = 1,
    overwrite: bool = False,
    n_frames_to_generate_override: int | str | None = "auto",
    save_traj_videos: bool = False,
    prediction_only: bool = False,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    run_name = _get(cfg, "project.run_name", _get(cfg, "run.name", "debug_al"))
    root_dir = _get(cfg, "project.root_dir", _get(cfg, "run.root", "al_runs"))
    dataset_name = _get(cfg, "phase.dataset", "agibot")
    run_root = Path(root_dir) / run_name
    scoring_cfg = dict(_get(cfg, "scoring", {}) or {})
    retraining_cfg = dict(_get(cfg, "retraining", {}) or {})
    if n_frames_to_generate_override is not None:
        scoring_cfg["n_frames_to_generate"] = _parse_n_frames_value(n_frames_to_generate_override)
    c3_probe_cfg = _get(cfg, "c3_probe", {})
    manifests_cfg = _get(cfg, "manifests", {})
    data_format = scoring_cfg.get("data_format") or _get(cfg, "evac.data_format", "agibot")
    scoring_cfg["data_format"] = data_format
    candidate_pool_path = (
        manifest_override
        or scoring_cfg.get("candidate_pool_manifest")
        or manifests_cfg.get("candidate_pool")
        or str(run_root / "manifests" / "candidate_pool.json")
    )
    output_dir = ensure_dir(output_dir_override or scoring_cfg.get("output_dir") or str(run_root / "pool_scores"))
    score_dir = ensure_dir(score_dir_override) if score_dir_override else output_dir
    if device_override is not None:
        device_str = device_override
    elif num_shards > 1:
        device_str = f"cuda:{shard_id}"
    else:
        device_str = scoring_cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model_cfg = _get(cfg, "model", {})
    evac_config = scoring_cfg.get("evac_config") or model_cfg.get("evac_config")
    evac_checkpoint = (
        evac_checkpoint_override
        or scoring_cfg.get("evac_checkpoint")
        or model_cfg.get("evac_v1_checkpoint")
        or model_cfg.get("evac_checkpoint")
    )
    if bool(c3_probe_cfg.get("train_on_external", False)):
        c3_probe = (
            scoring_cfg.get("external_c3_probe_checkpoint")
            or c3_probe_cfg.get("external_checkpoint")
            or scoring_cfg.get("c3_probe_checkpoint")
            or model_cfg.get("c3_probe_checkpoint")
        )
    else:
        c3_probe = scoring_cfg.get("c3_probe_checkpoint") or c3_probe_cfg.get("checkpoint") or model_cfg.get("c3_probe_checkpoint")
    if not evac_config:
        raise ValueError("Config must define model.evac_config")
    if not prediction_only and not c3_probe:
        raise ValueError("Config must define model.c3_probe_checkpoint unless --prediction-only is used")
    if evac_checkpoint:
        print(f"[al scoring] EVAC checkpoint override: {evac_checkpoint}")

    pool_payload = load_json(candidate_pool_path)
    if bool(c3_probe_cfg.get("train_on_external", False)):
        trained_manifest_path = (
            c3_probe_cfg.get("c3_train_split_manifest")
            or c3_probe_cfg.get("external_trained_split_manifest")
            or c3_probe_cfg.get("external_train_manifest")
            or c3_probe_cfg.get("trained_split_manifest")
        )
    else:
        trained_manifest_path = c3_probe_cfg.get("trained_split_manifest") or c3_probe_cfg.get("c3_train_split_manifest")
    leakage_check = None
    if prediction_only:
        trained_manifest_path = None
    if trained_manifest_path:
        allow_seen_pool = bool(c3_probe_cfg.get("allow_seen_pool", False))
        trained_payload = load_json(trained_manifest_path)
        leakage_check = check_manifest_overlap(
            trained_payload,
            pool_payload,
            left_name="c3_probe.trained_split_manifest",
            right_name="candidate_pool",
            allow_overlap=allow_seen_pool,
        )
        if leakage_check["overlap_count"] > 0 and allow_seen_pool:
            print(
                "[al scoring WARNING] C3 probe trained split overlaps candidate_pool "
                f"({leakage_check['overlap_count']} episode keys). Continuing because "
                "c3_probe.allow_seen_pool=true; treat this only as an ablation."
            )
    elif not prediction_only:
        print("[al scoring WARNING] c3_probe.trained_split_manifest is not set; leakage check skipped.")
    records = flatten_manifest_items(pool_payload)
    external_filter_stats: dict[str, Any] | None = None
    invalid_external_items: list[dict[str, Any]] = []
    if data_format == "external_worldmodel":
        min_valid_frames = int(scoring_cfg.get("min_valid_frames", 1))
        records, invalid_external_items, external_filter_stats = filter_valid_external_worldmodel_items(
            records,
            min_frames=min_valid_frames,
            require_paths=scoring_cfg.get("require_paths", "auto"),
            require_training_files=False,
        )
        invalid_path = output_dir / "invalid_candidate_pool_items.json"
        save_json(
            {
                "items": invalid_external_items,
                "count": len(invalid_external_items),
                "filter_stats": external_filter_stats,
                "source_manifest": str(candidate_pool_path),
            },
            invalid_path,
        )
        if invalid_external_items:
            print(
                f"[al scoring WARNING] filtered {len(invalid_external_items)} invalid external candidate items "
                f"before scoring; details: {invalid_path}"
            )
    if max_episodes is None:
        max_episodes = scoring_cfg.get("max_episodes")
    if max_episodes is not None:
        records = records[: int(max_episodes)]
    if not records:
        raise ValueError(f"No records to score in {candidate_pool_path}")

    total_workers = num_shards * workers_per_gpu
    overwrite_requested = bool(overwrite or scoring_cfg.get("overwrite", False))

    # Clean stale shard files from previous single-process runs.  In
    # multi-process mode the parent process cleans these before spawning.
    if total_workers == 1 and shard_id == 0 and worker_id == 0:
        for stale in sorted(score_dir.glob("scored_pool_*.json")):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        if overwrite_requested:
            scored_pool_path = score_dir / "scored_pool.json"
            if scored_pool_path.exists():
                scored_pool_path.unlink()

    if num_shards > 1 or workers_per_gpu > 1:
        records = [r for i, r in enumerate(records) if i % total_workers == shard_id * workers_per_gpu + worker_id]
        parts = []
        if num_shards > 1:
            parts.append(f"shard {shard_id}/{num_shards}")
        if workers_per_gpu > 1:
            parts.append(f"worker {worker_id}/{workers_per_gpu}")
        print(f"[al scoring] {' '.join(parts)}: {len(records)} episodes on {device_str}")

    if prediction_only:
        print(f"[al scoring] loading EVAC prediction-only model on {device}")
        model, model_config = load_model_for_prediction(evac_config, device, evac_checkpoint)
        probe_step = "prediction_only"
    else:
        print(f"[al scoring] loading EVAC+C3 on {device}")
        model, model_config, probe_step = load_model_with_probe(evac_config, c3_probe, device, evac_checkpoint)
    traj_conditioning_cfg = dict(retraining_cfg.get("traj_conditioning", {}) or {})
    traj_conditioning_cfg.update(dict(scoring_cfg.get("traj_conditioning", {}) or {}))
    if traj_conditioning_cfg:
        _apply_traj_conditioning_overrides(model, traj_conditioning_cfg)
        print(f"[al scoring] traj_conditioning overrides: {traj_conditioning_cfg}")
    args = _build_probe_args(scoring_cfg, str(device))
    args.overwrite = bool(args.overwrite or overwrite_requested)
    args.save_traj_videos = bool(save_traj_videos)
    args.prediction_only = bool(prediction_only)
    if prediction_only:
        args.save_latents = False
    print(
        f"[al scoring] data_format={data_format} "
        f"n_frames_to_generate={args.n_frames_to_generate} "
        f"save_latents={args.save_latents} "
        f"save_probe_ts_stack={args.save_probe_ts_stack} "
        f"save_traj_videos={args.save_traj_videos} "
        f"overwrite={args.overwrite} "
        f"prediction_only={args.prediction_only}"
    )
    if not prediction_only and getattr(model.c3_probe, "use_tau_cond", False):
        if args.probe_tau is None:
            args.effective_probe_tau = float(((model.c3_probe.thresh_low + model.c3_probe.thresh_high) * 0.5).item())
        else:
            args.effective_probe_tau = float(args.probe_tau)

    # --- Recovery: resume from previous (possibly interrupted) runs ---
    total_workers = num_shards * workers_per_gpu
    shard_path = _shard_file_path(score_dir, shard_id, worker_id, num_shards, workers_per_gpu)
    episode_by_id: dict[str, dict[str, Any]] = {}
    for r in records:
        ep_record = _record_to_episode(r, data_format)
        episode_by_id[ep_record["episode_id"]] = ep_record
    assigned_ids: set[str] = set(episode_by_id)
    if args.overwrite:
        scored, failures, done_ids = [], [], set()
    else:
        scored, failures, done_ids = _recover_existing(
            score_dir,
            shard_path,
            args,
            assigned_ids=assigned_ids,
            model=model,
            config=model_config,
            device=device,
            pred_dir=output_dir,
        )

    # Persist per-worker shard state immediately so later merges see
    # consistent data even before this worker scores a new episode.
    scored = [it for it in scored if it["episode_id"] in assigned_ids]
    failures = [f for f in failures if f["episode_id"] in assigned_ids]
    done_ids = assigned_ids & done_ids  # only consider our own episodes "done"
    cached_count = len(scored)  # all recovered items are "cached" (scored in a previous run)

    # Persist corrected per-worker shard file so merge sees consistent data
    _save_shard_file(
        shard_path,
        scored,
        failures,
        total_assigned=len(records),
        cached_count=cached_count,
        newly_scored=0,
        total_scored=len(scored),
        source_manifest=str(candidate_pool_path),
        probe_step=probe_step,
        protocol={
            "phase": _get(cfg, "phase.name", ""),
            "dataset": dataset_name,
            "data_format": data_format,
            "no_future_gt_for_selection": True,
            "risk_definition": "risk = 1 - confidence",
            "confidence_source": "C3 probe on EVAC decoder features",
            "action_transform": _expected_action_transform(data_format),
            "camera_transform": _expected_camera_transform(data_format),
            "overwrite": bool(args.overwrite),
            "confidence_format": scoring_cfg.get("confidence_format", "probability"),
            "evac_checkpoint": str(evac_checkpoint) if evac_checkpoint else None,
            "leakage_check": leakage_check,
            "external_filter": external_filter_stats,
        },
    )

    # Filter out already-done episodes
    total_assigned = len(records)
    remaining = [r for r in records if _record_to_episode(r, data_format)["episode_id"] not in done_ids]
    if len(remaining) < len(records):
        print(
            f"[al scoring] {len(records) - len(remaining)} episodes already done "
            f"→ {len(remaining)} remaining to score"
        )
        records = remaining

    pbar = tqdm(records, desc="[al scoring]", unit="ep", dynamic_ncols=True)
    newly_scored = 0
    for record in pbar:
        ep = _record_to_episode(record, data_format)
        ep_id = ep["episode_id"]
        pbar.set_postfix_str(ep_id)

        try:
            row = run_pool_episode(model, model_config, ep, device, args, output_dir, score_root=score_dir)
            if row is not None:
                scored.append(row)
                newly_scored += 1
            else:
                failures.append({"episode_id": ep_id, "error": "no generated frames"})
        except Exception as exc:
            traceback.print_exc()
            failures.append({"episode_id": ep_id, "error": repr(exc)})

        # --- Incremental save: survive Ctrl+C ---
        _save_shard_file(
            shard_path,
            scored,
            failures,
            total_assigned=total_assigned,
            cached_count=cached_count,
            newly_scored=newly_scored,
            total_scored=len(scored),
            source_manifest=str(candidate_pool_path),
            probe_step=probe_step,
            protocol={
                "phase": _get(cfg, "phase.name", ""),
                "dataset": dataset_name,
                "data_format": data_format,
                "no_future_gt_for_selection": True,
                "risk_definition": "risk = 1 - confidence",
                "confidence_source": "C3 probe on EVAC decoder features",
                "action_transform": _expected_action_transform(data_format),
                "camera_transform": _expected_camera_transform(data_format),
                "overwrite": bool(args.overwrite),
                "confidence_format": scoring_cfg.get("confidence_format", "probability"),
                "evac_checkpoint": str(evac_checkpoint) if evac_checkpoint else None,
                "leakage_check": leakage_check,
                "external_filter": external_filter_stats,
            },
        )
        # Keep merged scored_pool.json up-to-date for downstream consumers
        _try_merge_all_shards(score_dir)

    # Final atomic write (guarantees metadata is present even if no episodes were scored)
    _save_shard_file(
        shard_path,
        scored,
        failures,
        total_assigned=total_assigned,
        cached_count=cached_count,
        newly_scored=newly_scored,
        total_scored=len(scored),
        source_manifest=str(candidate_pool_path),
        probe_step=probe_step,
        protocol={
            "phase": _get(cfg, "phase.name", ""),
            "dataset": dataset_name,
            "data_format": data_format,
            "no_future_gt_for_selection": True,
            "risk_definition": "risk = 1 - confidence",
            "confidence_source": "C3 probe on EVAC decoder features",
            "action_transform": _expected_action_transform(data_format),
            "camera_transform": _expected_camera_transform(data_format),
            "overwrite": bool(args.overwrite),
            "confidence_format": scoring_cfg.get("confidence_format", "probability"),
            "evac_checkpoint": str(evac_checkpoint) if evac_checkpoint else None,
            "leakage_check": leakage_check,
            "external_filter": external_filter_stats,
        },
    )
    merged = _try_merge_all_shards(score_dir)

    print(
        f"[al scoring] wrote {shard_path} "
        f"({len(scored)} scored, {cached_count} cached, {newly_scored} new, {len(failures)} failed)"
    )

    # Build return payload (same structure as before for API compatibility)
    protocol_block = {
        "phase": _get(cfg, "phase.name", ""),
        "dataset": dataset_name,
        "data_format": data_format,
        "no_future_gt_for_selection": True,
        "risk_definition": "risk = 1 - confidence",
        "confidence_source": "C3 probe on EVAC decoder features",
        "action_transform": _expected_action_transform(data_format),
        "camera_transform": _expected_camera_transform(data_format),
        "overwrite": bool(args.overwrite),
        "confidence_format": scoring_cfg.get("confidence_format", "probability"),
        "evac_checkpoint": str(evac_checkpoint) if evac_checkpoint else None,
        "leakage_check": leakage_check,
        "external_filter": external_filter_stats,
    }
    payload: dict[str, Any] = {
        "source_manifest": str(candidate_pool_path),
        "probe_step": probe_step,
        "items": scored,
        "failures": failures,
        "stats": {
            "requested": total_assigned,
            "scored": len(scored),
            "failed": len(failures),
            "cached": cached_count,
            "newly_scored_this_run": newly_scored,
        },
        "protocol": protocol_block,
    }

    if total_workers > 1:
        if merged is not None:
            n_shard_files = len(list(score_dir.glob("scored_pool_*.json")))
            print(
                f"[al scoring] merged {n_shard_files} shard(s) → {score_dir / 'scored_pool.json'} "
                f"({merged['stats']['scored']} scored, {merged['stats'].get('cached', 0)} cached, "
                f"{merged['stats']['failed']} failed)"
            )
            return merged
        else:
            print("[al scoring] merge skipped (no shard files found)")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AL candidate pool with EVAC+C3 confidence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None, help="Override scoring.candidate_pool_manifest")
    parser.add_argument("--output-dir", default=None, help="Override EVAC prediction output dir (pred_frames, latents)")
    parser.add_argument("--score-dir", default=None, help="Override C3 scoring output dir (conf_map, risk_stats, scored_pool.json)")
    parser.add_argument(
        "--evac_checkpoint",
        "--evac-checkpoint",
        dest="evac_checkpoint",
        default=None,
        help="Override the EVAC backbone checkpoint, e.g. a RoboTwin warmup v1 checkpoint.",
    )
    parser.add_argument("--max_episodes", "--max-episodes", dest="max_episodes", type=int, default=None)
    parser.add_argument(
        "--n_frames_to_generate",
        "--n-frames-to-generate",
        "--n_frames",
        "--n-frames",
        "--num_frames",
        "--num-frames",
        "--num_inference_frames",
        "--num-inference-frames",
        dest="n_frames_to_generate",
        type=_n_frames_cli_value,
        default="auto",
        help=(
            "Frames to generate per episode: positive integer or 'auto'. "
            "Default: auto (episode num_frames - n_previous; ignores legacy config 64/200)."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard_id", type=int, default=0, help="0-indexed shard id (for multi-GPU)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (e.g., 8 for 8 GPUs)")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Workers per GPU (each runs an independent subprocess)")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs, e.g. '2,3'. Overrides auto cuda:{shard_id} mapping.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate every assigned episode and replace existing per-episode score outputs "
            "instead of resuming from cached pred_frames/conf_map."
        ),
    )
    parser.add_argument(
        "--save_traj_videos",
        "--save-traj-videos",
        action="store_true",
        default=False,
        help="Save per-episode traj_condition.mp4 for action/camera alignment debugging.",
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        default=False,
        help="Only generate EVAC pred_frames/actions/meta; skip C3 probe scoring.",
    )
    parser.add_argument("--_worker_id", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Resolve --gpus into a list of GPU IDs
    gpu_list = None
    if args.gpus is not None:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",")]
        if not gpu_list:
            parser.error("--gpus must contain at least one GPU ID")
        # If --num_shards is still at the default (1), auto-set from GPU count
        if args.num_shards == 1:
            args.num_shards = len(gpu_list)
        if args.num_shards > len(gpu_list):
            parser.error(
                f"--num_shards ({args.num_shards}) exceeds number of GPUs "
                f"specified via --gpus ({len(gpu_list)})"
            )
        # Resolve device for direct (non-spawning) mode
        if args.device is None:
            args.device = f"cuda:{gpu_list[args.shard_id]}"

    workers_per_gpu = max(1, args.workers_per_gpu)
    worker_id = args._worker_id if args._worker_id is not None else 0
    total_procs = args.num_shards * workers_per_gpu

    if total_procs > 1 and args._worker_id is None:
        print(f"[al scoring] spawning {workers_per_gpu} workers × {args.num_shards} GPUs = {total_procs} total processes")
        cfg = load_yaml(args.config)
        root_dir = _get(cfg, "project.root_dir", _get(cfg, "run.root", "al_runs"))
        run_name = _get(cfg, "project.run_name", _get(cfg, "run.name", "debug_al"))
        scoring_cfg = _get(cfg, "scoring", {})
        output_dir = Path(args.output_dir or scoring_cfg.get("output_dir") or str(Path(root_dir) / run_name / "pool_scores"))
        score_dir = Path(args.score_dir) if args.score_dir else output_dir
        for stale in sorted(score_dir.glob("scored_pool_*.json")):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        if args.overwrite:
            scored_pool_path = score_dir / "scored_pool.json"
            if scored_pool_path.exists():
                scored_pool_path.unlink()
        procs: list[subprocess.Popen] = []
        for sid in range(args.num_shards):
            for wid in range(workers_per_gpu):
                cmd = [
                    sys.executable, __file__,
                    "--config", args.config,
                    "--shard_id", str(sid),
                    "--num_shards", str(args.num_shards),
                    "--workers_per_gpu", str(workers_per_gpu),
                    "--_worker_id", str(wid),
                ]
                if args.manifest is not None:
                    cmd += ["--manifest", args.manifest]
                if args.output_dir is not None:
                    cmd += ["--output-dir", args.output_dir]
                if args.score_dir is not None:
                    cmd += ["--score-dir", args.score_dir]
                if args.evac_checkpoint is not None:
                    cmd += ["--evac-checkpoint", args.evac_checkpoint]
                if args.max_episodes is not None:
                    cmd += ["--max_episodes", str(args.max_episodes)]
                if args.n_frames_to_generate is not None:
                    cmd += ["--n-frames-to-generate", str(args.n_frames_to_generate)]
                if args.overwrite:
                    cmd += ["--overwrite"]
                if args.save_traj_videos:
                    cmd += ["--save-traj-videos"]
                if args.prediction_only:
                    cmd += ["--prediction-only"]
                if gpu_list is not None:
                    cmd += ["--device", f"cuda:{gpu_list[sid]}"]
                elif args.device is not None:
                    cmd += ["--device", args.device]
                p = subprocess.Popen(cmd)
                procs.append(p)
                print(f"[al scoring]  started shard {sid} worker {wid} (pid={p.pid})")
        for p in procs:
            p.wait()

        # Final merge + cleanup: all workers have finished, so it's safe to
        # merge shard files and remove them.
        merged = _try_merge_all_shards(score_dir)
        expected = args.num_shards * workers_per_gpu
        shard_files = sorted(score_dir.glob("scored_pool_*.json"))
        if len(shard_files) == expected and merged is not None:
            for sf in shard_files:
                sf.unlink()
            print(
                f"[al scoring] final merge complete → {score_dir / 'scored_pool.json'} "
                f"({merged['stats']['scored']} scored, {merged['stats'].get('cached', 0)} cached, "
                f"{merged['stats']['failed']} failed)"
            )
        print("[al scoring] all workers finished")
        return

    run_from_config(
        args.config,
        max_episodes=args.max_episodes,
        manifest_override=args.manifest,
        output_dir_override=args.output_dir,
        score_dir_override=args.score_dir,
        evac_checkpoint_override=args.evac_checkpoint,
        device_override=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        worker_id=worker_id,
        workers_per_gpu=workers_per_gpu,
        overwrite=args.overwrite,
        n_frames_to_generate_override=args.n_frames_to_generate,
        save_traj_videos=args.save_traj_videos,
        prediction_only=args.prediction_only,
    )


if __name__ == "__main__":
    main()
