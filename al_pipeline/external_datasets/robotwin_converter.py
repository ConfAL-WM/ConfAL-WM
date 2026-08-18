"""RoboTwin2.0 → AgiBot-compatible training format converter.

Converts raw RoboTwin2.0 ZIP archives directly to a flat directory structure
that both EVAC training (AgiBotWorldICRA26Challenge Dataset) and the C3 scoring
pipeline can consume without any intermediate conversion step.

Output layout::

    {converted_root}/
      manifests/all.json
      {task_name}-{safe_episode_id}-0/
        head_color.mp4              ← AgiBot training
        head_intrinsic_params.json  ← AgiBot training
        head_extrinsic_params_aligned.json ← AgiBot training
        proprio_stats.h5            ← AgiBot training
        actions_evac.npy            ← C3 scoring
        actions_delta_evac.npy      ← C3 scoring
        camera.npz                  ← C3 scoring
        meta.json                   ← metadata (both)

Usage::

    python al_pipeline/external_datasets/robotwin_converter.py \\
      --raw_root datasets/RoboTwin2.0/dataset \\
      --converted_root datasets/RoboTwin2.0/aloha-agilex_rand_500 \\
      --robot aloha-agilex --variant randomized_500 --min_frames 68 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
import zipfile
from io import BytesIO
import multiprocessing as mp

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import h5py  # noqa: F401
except ImportError:
    h5py = None

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(1, str(_REPO / "evac"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_angles(radius: np.ndarray) -> np.ndarray:
    return np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)


def _parse_list_arg(val: str | list | None) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if val.startswith("["):
        return json.loads(val)
    return [val]


def _which_ffmpeg() -> str:
    """Find a working ffmpeg binary."""
    candidates = [
        os.path.expanduser("~/miniconda3/envs/enerverse/bin/ffmpeg"),
        "ffmpeg",
    ]
    for c in candidates:
        try:
            subprocess.run([c, "-version"], capture_output=True, check=True)
            return c
        except Exception:
            continue
    return "ffmpeg"  # fallback, will fail with clear error if missing


# ---------------------------------------------------------------------------
# Episode-chunk worker (for multiprocessing)
# ---------------------------------------------------------------------------

class _EpisodeWorker:
    """Convert a chunk of pre-scanned episodes. Used by multiprocessing."""

    def __init__(self, raw_root: str, converted_root: str, config: dict[str, Any]) -> None:
        self.raw_root = raw_root
        self.converted_root = converted_root
        self.config = config

    def __call__(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        import h5py  # noqa: F401
        from al_pipeline.external_datasets.robotwin_converter import RoboTwinConverter
        converter = RoboTwinConverter(self.raw_root, self.converted_root, self.config)
        items = []
        failures = []
        for ep in tqdm(episodes, desc=f"[convert:worker]", unit="ep", position=0, leave=False):
            try:
                items.append(converter.convert_episode(ep))
            except Exception:
                failures.append({
                    "episode_id": ep.get("episode_idx", "?"),
                    "error": traceback.format_exc()[-200:],
                })
        from al_pipeline.utils import ensure_dir, save_json
        wid = episodes[0].get("episode_idx", 0) if episodes else 0
        worker_dir = ensure_dir(Path(self.converted_root) / ".workers")
        save_json({"items": items, "failures": failures, "count": len(items)},
                   worker_dir / f"worker_{wid:06d}.json")
        return {"ok": True, "converted": len(items), "failed": len(failures)}


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class RoboTwinConverter:
    """RoboTwin2.0 → AgiBot-compatible format (+ C3 scoring files)."""

    dataset_name = "robotwin"

    def __init__(self, raw_root: str | Path, converted_root: str | Path,
                 config: dict[str, Any] | None = None) -> None:
        self.raw_root = Path(raw_root)
        self.converted_root = Path(converted_root)
        self.config = config or {}
        self.manifests_root = self.converted_root.parent / "manifests"
        self.ffmpeg_bin = _which_ffmpeg()

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def scan_raw_dataset(self) -> list[dict[str, Any]]:
        zip_glob = self.config.get("zip_glob", "*/*.zip")
        zip_files = sorted(self.raw_root.glob(zip_glob))
        if not zip_files:
            return self._scan_loose()

        robot_inc = set(_parse_list_arg(self.config.get("robot_include", [])))
        robot_exc = set(_parse_list_arg(self.config.get("robot_exclude", [])))
        task_inc = set(_parse_list_arg(self.config.get("task_include", [])))
        task_exc = set(_parse_list_arg(self.config.get("task_exclude", [])))
        variant_inc = set(_parse_list_arg(self.config.get("variant_include", [])))
        max_episodes = self.config.get("max_scan_episodes")

        out = []
        for zip_path in zip_files:
            task_name = zip_path.parent.name
            if task_inc and task_name not in task_inc:
                continue
            if task_exc and task_name in task_exc:
                continue

            fname = zip_path.stem
            parts = fname.split("_")
            robot = "_".join(parts[:-2]) if len(parts) >= 2 else fname
            variant = "_".join(parts[-2:])
            clean_or_randomized = "randomized" if "randomized" in variant else "clean"

            if robot_inc and robot not in robot_inc:
                continue
            if robot_exc and robot in robot_exc:
                continue
            if variant_inc and variant not in variant_inc:
                continue

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    names = zf.namelist()
                    prefixes = {n.split("/")[0] for n in names if "/" in n}
                    prefix = sorted(prefixes)[0] if prefixes else ""
                    has_hdf5 = any(f"{prefix}/data/episode" in n and n.endswith(".hdf5") for n in names)
                    episodes = set()
                    for n in names:
                        src = f"{prefix}/data/episode" if has_hdf5 else f"{prefix}/_traj_data/episode"
                        if src in n and (n.endswith(".hdf5") or n.endswith(".pkl")):
                            try:
                                ep_num = int(Path(n).stem.replace("episode", ""))
                                episodes.add(ep_num)
                            except ValueError:
                                pass
                    for ep_idx in sorted(episodes):
                        out.append({
                            "source_type": "robotwin2_zip",
                            "zip_path": str(zip_path), "prefix": prefix,
                            "episode_idx": ep_idx, "task_name": task_name,
                            "robot": robot, "variant": variant,
                            "clean_or_randomized": clean_or_randomized,
                            "has_hdf5": has_hdf5,
                        })
                        if max_episodes and len(out) >= max_episodes:
                            break
            except Exception:
                continue
            if max_episodes and len(out) >= max_episodes:
                break
        return out

    def _scan_loose(self) -> list[dict[str, Any]]:
        out = []
        for hdf5_path in sorted(self.raw_root.glob("data/*.hdf5")):
            if hdf5_path.is_file():
                out.append({"episode_dir": hdf5_path.parent, "hdf5_path": hdf5_path, "has_hdf5": True})
        return out

    # ------------------------------------------------------------------
    # Build manifest
    # ------------------------------------------------------------------

    def build_manifest(self, max_episodes: int | None = None) -> dict[str, Any]:
        from al_pipeline.utils import ensure_dir, save_json

        raw_eps = self.scan_raw_dataset()
        if max_episodes is not None:
            raw_eps = raw_eps[:max_episodes]
        ensure_dir(self.manifests_root)
        items = []
        failures = []
        for raw_ep in tqdm(raw_eps, desc=f"[convert:{self.dataset_name}]", unit="ep"):
            try:
                items.append(self.convert_episode(raw_ep))
            except Exception as exc:
                failures.append({
                    "raw_episode": {k: str(v) for k, v in raw_ep.items() if k != "zip_path"},
                    "error": repr(exc),
                })
        manifest = {
            "dataset": self.dataset_name,
            "items": items, "failures": failures,
            "stats": {"raw_episodes": len(raw_eps), "converted": len(items), "failed": len(failures)},
        }
        save_json(manifest, self.manifests_root / "all.json")
        save_json({"dataset": self.dataset_name, **manifest["stats"]},
                   self.manifests_root / "split_summary.json")
        return manifest

    # ------------------------------------------------------------------
    # Convert single episode
    # ------------------------------------------------------------------

    def convert_episode(self, raw_episode: dict[str, Any]) -> dict[str, Any]:
        if raw_episode.get("source_type") == "robotwin2_zip":
            if raw_episode.get("has_hdf5"):
                return self._convert_zip_hdf5_episode(raw_episode)
            return self._convert_zip_pkl_fallback(raw_episode)
        if raw_episode.get("hdf5_path"):
            return self._convert_hdf5_episode(raw_episode)
        return self._convert_loose_episode(raw_episode)

    # ---- ZIP + HDF5 (primary path) -------------------------------------------

    def _convert_zip_hdf5_episode(self, raw_episode: dict[str, Any]) -> dict[str, Any]:
        import h5py
        from al_pipeline.utils import ensure_dir, save_json

        zip_path = Path(raw_episode["zip_path"])
        prefix = raw_episode["prefix"]
        ep_idx = raw_episode["episode_idx"]
        task_name = raw_episode["task_name"]
        robot = raw_episode["robot"]
        variant = raw_episode.get("variant", "unknown")
        clean_or_randomized = raw_episode.get("clean_or_randomized", "unknown")

        # episode_id uses underscores to avoid breaking AgiBot {task}-{ep}-{step} parsing
        episode_id = f"{task_name}_{robot}_{variant}_ep{ep_idx:03d}"
        # Directory name: {task_name}-{safe_ep}-0  (AgiBot convention)
        safe_ep = episode_id.replace("-", "_")
        dir_name = f"{task_name}-{safe_ep}-0"
        ep_dir = ensure_dir(self.converted_root / dir_name)

        # Reuse already-converted episodes without poisoning the manifest.
        # Older code returned skipped=True here, which rewrote valid rows as
        # num_frames=0/_skipped=true on every converter rerun.
        if self._is_converted(ep_dir):
            return self._build_manifest_item_from_converted(
                ep_dir,
                task_name,
                robot,
                variant,
                clean_or_randomized,
                episode_id,
            )

        with zipfile.ZipFile(zip_path, "r") as zf:
            hdf5_name = f"{prefix}/data/episode{ep_idx}.hdf5"
            hdf5_bytes = zf.read(hdf5_name)
            tmp_h5 = ep_dir / "_temp.hdf5"
            tmp_h5.write_bytes(hdf5_bytes)

            with h5py.File(tmp_h5, "r") as f:
                # --- Extract frames ---
                rgb_key = self.config.get("rgb_key", "observation/head_camera/rgb")
                frames_dir = ensure_dir(ep_dir / "frames")
                if rgb_key in f:
                    rgb = f[rgb_key]
                    for idx, blob in enumerate(rgb):
                        img = Image.open(BytesIO(bytes(blob))).convert("RGB")
                        img.save(frames_dir / f"frame_{idx:06d}.jpg", quality=95)
                    n_frames = len(rgb)
                else:
                    n_frames = self._extract_video_from_zip(zf, prefix, ep_idx, frames_dir)

                # --- Extract actions ---
                gripper_mode = self.config.get("gripper_mode", "normalized_01")
                has_endpose = all(k in f for k in [
                    "endpose/left_endpose", "endpose/right_endpose",
                    "endpose/left_gripper", "endpose/right_gripper",
                ])
                if has_endpose:
                    evac_actions, delta_actions, evac_meta = _endpose_to_evac_actions(
                        f, gripper_mode=gripper_mode)
                    np.save(ep_dir / "actions_evac.npy", evac_actions)
                    np.save(ep_dir / "actions_delta_evac.npy", delta_actions)
                    evac_action_compatible = True
                    action_dim = 16
                else:
                    evac_actions = np.zeros((n_frames, 16), dtype=np.float32)
                    delta_actions = np.zeros((max(1, n_frames - 1), 14), dtype=np.float32)
                    evac_meta = {"error": "no endpose in hdf5"}
                    evac_action_compatible = False
                    action_dim = 16

                # --- Alignment check ---
                num_rgb = n_frames
                num_left_ep = int(f["endpose/left_endpose"].shape[0]) if "endpose/left_endpose" in f else 0
                num_right_ep = int(f["endpose/right_endpose"].shape[0]) if "endpose/right_endpose" in f else 0
                num_lg = int(f["endpose/left_gripper"].shape[0]) if "endpose/left_gripper" in f else 0
                num_rg = int(f["endpose/right_gripper"].shape[0]) if "endpose/right_gripper" in f else 0
                num_intrin = int(f["observation/head_camera/intrinsic_cv"].shape[0]) if "observation/head_camera/intrinsic_cv" in f else 0
                num_extrin = int(f["observation/head_camera/extrinsic_cv"].shape[0]) if "observation/head_camera/extrinsic_cv" in f else 0
                alignment_ok = (num_rgb == num_left_ep == num_right_ep == num_lg == num_rg == num_intrin == num_extrin)
                num_frames_used = min(num_rgb, num_left_ep, num_right_ep, num_lg, num_rg, num_intrin, num_extrin)
                if not alignment_ok:
                    print(f"  [WARN alignment] {episode_id}: rgb={num_rgb} left_ep={num_left_ep} "
                          f"right_ep={num_right_ep} lg={num_lg} rg={num_rg} "
                          f"intrin={num_intrin} extrin={num_extrin} → using {num_frames_used}")

                is_dual_arm = evac_meta.get("is_dual_arm", False)

                # Skip short episodes
                min_frames = int(self.config.get("min_frames", 0))
                if min_frames > 0 and num_frames_used < min_frames:
                    if tmp_h5.exists():
                        tmp_h5.unlink()
                    raise ValueError(f"Episode too short: {num_frames_used} < {min_frames} frames")

                # Save qpos as reference
                has_qpos = "joint_action/vector" in f
                if has_qpos:
                    qpos = np.asarray(f["joint_action/vector"], dtype=np.float32)
                    np.save(ep_dir / "actions_qpos_raw.npy", qpos)

                # Extract camera
                cam = _extract_camera_from_hdf5(f)
                if cam:
                    np.savez(ep_dir / "camera.npz", **cam)

                # --- AgiBot training format ---
                self._write_agibot_format(ep_dir, evac_actions, cam, num_frames_used)

            if tmp_h5.exists():
                tmp_h5.unlink()

        # Write meta.json
        meta = {
            "dataset": self.dataset_name,
            "episode_id": episode_id,
            "task_name": task_name, "robot": robot, "variant": variant,
            "clean_or_randomized": clean_or_randomized,
            "is_dual_arm": is_dual_arm,
            "active_arm": evac_meta.get("active_arm"),
            "left_arm_moving": evac_meta.get("left_arm_moving"),
            "right_arm_moving": evac_meta.get("right_arm_moving"),
            "left_move_sum": evac_meta.get("left_move_sum"),
            "right_move_sum": evac_meta.get("right_move_sum"),
            "camera": self.config.get("camera", "head_camera"),
            "num_frames_raw": num_rgb,
            "num_endpose_left_raw": num_left_ep, "num_endpose_right_raw": num_right_ep,
            "num_gripper_left_raw": num_lg, "num_gripper_right_raw": num_rg,
            "num_intrinsic_raw": num_intrin, "num_extrinsic_raw": num_extrin,
            "num_frames_used": num_frames_used,
            "alignment_ok": alignment_ok,
            "action_dim": action_dim,
            "action_source": "endpose_from_hdf5",
            "action_representation": "evac_absolute_ee_xyz_quat_xyzw_gripper_16dim",
            "evac_action_compatible": evac_action_compatible,
            "has_endpose": has_endpose, "has_qpos": has_qpos,
            "has_camera": cam is not None,
            "gripper_mode": gripper_mode,
            "source_zip": str(zip_path), "source_prefix": prefix, "source_episode_idx": ep_idx,
            "source_format": "robotwin2_hdf5_endpose",
        }
        save_json(meta, ep_dir / "meta.json")
        self.validate_converted_episode(ep_dir)

        return self._build_manifest_item(ep_dir, task_name, robot, variant,
                                          clean_or_randomized, episode_id,
                                          num_frames=num_frames_used, action_dim=action_dim,
                                          evac_action_compatible=evac_action_compatible,
                                          is_dual_arm=is_dual_arm, meta=meta)

    def _write_agibot_format(self, ep_dir: Path, evac_actions: np.ndarray,
                              cam: dict | None, num_frames: int) -> None:
        """Write AgiBot training files alongside the existing scoring files."""
        # 1. Convert frames to mp4
        frames_dir = ep_dir / "frames"
        mp4_path = ep_dir / "head_color.mp4"
        if frames_dir.is_dir() and not mp4_path.exists():
            ffmpeg_cmd = [
                self.ffmpeg_bin, "-y", "-nostdin",
                "-framerate", "10",
                "-i", str(frames_dir / "frame_%06d.jpg"),
                "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                str(mp4_path),
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Fallback: glob pattern
                ffmpeg_cmd2 = [
                    self.ffmpeg_bin, "-y", "-nostdin",
                    "-framerate", "10",
                    "-pattern_type", "glob",
                    "-i", str(frames_dir / "*.jpg"),
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-loglevel", "error",
                    str(mp4_path),
                ]
                subprocess.run(ffmpeg_cmd2, capture_output=True, check=True)

        # 2. Write camera JSON
        if cam is not None and "intrinsic_cv" in cam:
            intr = cam["intrinsic_cv"]
            intrinsic_dict = {
                "intrinsic": {
                    "fx": float(intr[0, 0, 0]),
                    "fy": float(intr[0, 1, 1]),
                    "ppx": float(intr[0, 0, 2]),
                    "ppy": float(intr[0, 1, 2]),
                }
            }
            with open(str(ep_dir / "head_intrinsic_params.json"), "w") as f:
                json.dump(intrinsic_dict, f)

            if "extrinsic_cv" in cam:
                w2c = cam["extrinsic_cv"]
                if w2c.shape[-2:] == (3, 4):
                    pad = np.zeros((*w2c.shape[:-2], 1, 4), dtype=np.float32)
                    pad[..., 0, 3] = 1.0
                    w2c = np.concatenate([w2c, pad], axis=-2)
                c2w = np.linalg.inv(w2c).astype(np.float32)
            else:
                c2w = cam["cam2world_gl"]
            extrinsic_list = []
            for i in range(min(c2w.shape[0], num_frames)):
                extrinsic_list.append({
                    "extrinsic": {
                        "rotation_matrix": c2w[i, :3, :3].tolist(),
                        "translation_vector": c2w[i, :3, 3].tolist(),
                    }
                })
            with open(str(ep_dir / "head_extrinsic_params_aligned.json"), "w") as f:
                json.dump(extrinsic_list, f)

        # 3. Write actions as h5
        T = evac_actions.shape[0]
        with h5py.File(str(ep_dir / "proprio_stats.h5"), "w") as f:
            gripper = np.stack([
                evac_actions[:, 7],   # left gripper
                evac_actions[:, 15],  # right gripper
            ], axis=-1).astype(np.float32)
            f.create_dataset("state/effector/position", data=gripper[:num_frames])
            end_pos = np.stack([
                evac_actions[:num_frames, 0:3],   # left xyz
                evac_actions[:num_frames, 8:11],  # right xyz
            ], axis=1).astype(np.float32)
            f.create_dataset("state/end/position", data=end_pos)
            end_ori = np.stack([
                evac_actions[:num_frames, 3:7],    # left quat xyzw
                evac_actions[:num_frames, 11:15],  # right quat xyzw
            ], axis=1).astype(np.float32)
            f.create_dataset("state/end/orientation", data=end_ori)

    def _is_converted(self, ep_dir: Path) -> bool:
        return all((ep_dir / f).exists() for f in [
            "head_color.mp4", "head_intrinsic_params.json",
            "head_extrinsic_params_aligned.json", "proprio_stats.h5",
            "actions_evac.npy", "camera.npz", "meta.json",
        ])

    def _build_manifest_item_from_converted(self, ep_dir: Path, task_name: str, robot: str,
                                            variant: str, clean_or_randomized: str,
                                            episode_id: str) -> dict[str, Any]:
        from al_pipeline.utils import load_json, save_json

        meta = load_json(ep_dir / "meta.json")
        validation = self.validate_converted_episode(ep_dir)
        try:
            actions = np.load(ep_dir / "actions_evac.npy")
            meta.update(_motion_meta_from_abs_actions(actions))
            save_json(meta, ep_dir / "meta.json")
        except Exception:
            pass
        num_frames = int(
            meta.get("num_frames_used")
            or meta.get("num_frames_raw")
            or validation.get("num_frames", 0)
        )
        action_dim = int(meta.get("action_dim") or validation.get("action_dim", 16))
        return self._build_manifest_item(
            ep_dir,
            task_name,
            robot,
            variant,
            clean_or_randomized,
            episode_id,
            num_frames=num_frames,
            action_dim=action_dim,
            evac_action_compatible=bool(meta.get("evac_action_compatible", True)),
            is_dual_arm=bool(meta.get("is_dual_arm", False)),
            meta=meta,
        )

    def _build_manifest_item(self, ep_dir: Path, task_name: str, robot: str,
                              variant: str, clean_or_randomized: str,
                              episode_id: str, **kwargs) -> dict[str, Any]:
        skipped = kwargs.pop("skipped", False)
        num_frames = kwargs.pop("num_frames", 0)
        action_dim = kwargs.pop("action_dim", 16)
        evac_action_compatible = kwargs.pop("evac_action_compatible", True)
        is_dual_arm = kwargs.pop("is_dual_arm", False)
        meta = kwargs.pop("meta", None)
        return {
            "dataset": self.dataset_name,
            "episode_id": episode_id,
            "task_name": task_name, "robot": robot, "variant": variant,
            "clean_or_randomized": clean_or_randomized,
            "is_dual_arm": is_dual_arm,
            "active_arm": meta.get("active_arm") if meta else None,
            "left_arm_moving": meta.get("left_arm_moving") if meta else None,
            "right_arm_moving": meta.get("right_arm_moving") if meta else None,
            "left_move_sum": meta.get("left_move_sum") if meta else None,
            "right_move_sum": meta.get("right_move_sum") if meta else None,
            "camera": self.config.get("camera", "head_camera"),
            "episode_dir": str(ep_dir),
            "frames_dir": str(ep_dir / "frames"),
            "actions_path": str(ep_dir / "actions_evac.npy"),
            "actions_delta_path": str(ep_dir / "actions_delta_evac.npy"),
            "proprio_path": str(ep_dir / "actions_evac.npy"),
            "camera_path": str(ep_dir / "camera.npz") if (ep_dir / "camera.npz").exists() else None,
            "meta_path": str(ep_dir / "meta.json"),
            "num_frames": num_frames, "action_dim": action_dim,
            "action_representation": meta.get("action_representation", "") if meta else "",
            "evac_action_compatible": evac_action_compatible,
            "format": "external_worldmodel",
            "_skipped": skipped,
        }

    def validate_converted_episode(self, episode_dir: Path) -> dict[str, Any]:
        actions_path = episode_dir / "actions_evac.npy"
        meta_path = episode_dir / "meta.json"
        mp4_path = episode_dir / "head_color.mp4"
        if not actions_path.exists():
            raise FileNotFoundError(f"Missing actions: {episode_dir}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json: {meta_path}")
        if not mp4_path.exists():
            raise FileNotFoundError(f"Missing mp4: {mp4_path}")
        actions = np.load(actions_path)
        return {"num_frames": int(actions.shape[0]),
                "action_shape": list(actions.shape),
                "action_dim": int(actions.shape[-1])}

    # ---- Helpers ----

    @staticmethod
    def _extract_video_from_zip(zf: zipfile.ZipFile, prefix: str, ep_idx: int,
                                 dst_dir: Path) -> int:
        from al_pipeline.utils import ensure_dir
        video_name = f"{prefix}/video/episode{ep_idx}.mp4"
        if video_name not in zf.namelist():
            return 0
        ensure_dir(dst_dir)
        tmp_path = dst_dir / "_temp_video.mp4"
        tmp_path.write_bytes(zf.read(video_name))
        count = 0
        try:
            import imageio.v3 as iio
            for idx, frame in enumerate(iio.imiter(str(tmp_path))):
                rgb = np.asarray(frame)[..., :3]
                Image.fromarray(rgb).save(dst_dir / f"frame_{idx:06d}.jpg", quality=95)
                count += 1
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return count

    # ---- Fallback paths ----

    def _convert_zip_pkl_fallback(self, raw_episode: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("PKL-only episodes not supported.")

    def _convert_hdf5_episode(self, raw_episode: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Use ZIP path instead.")

    def _convert_loose_episode(self, raw_episode: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Use ZIP path instead.")


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

def _endpose_to_evac_actions(h5_file, gripper_mode: str = "normalized_01") -> tuple[np.ndarray, np.ndarray, dict]:
    left = np.asarray(h5_file["endpose/left_endpose"], dtype=np.float32)
    right = np.asarray(h5_file["endpose/right_endpose"], dtype=np.float32)
    lg_raw = np.asarray(h5_file["endpose/left_gripper"], dtype=np.float32)
    rg_raw = np.asarray(h5_file["endpose/right_gripper"], dtype=np.float32)

    T = min(left.shape[0], right.shape[0], lg_raw.shape[0], rg_raw.shape[0])

    lg_min, lg_max = float(lg_raw[:T].min()), float(lg_raw[:T].max())
    rg_min, rg_max = float(rg_raw[:T].min()), float(rg_raw[:T].max())

    if gripper_mode == "agibot_120":
        lg = lg_raw[:T] * 120.0
        rg = rg_raw[:T] * 120.0
        gripper_norm_note = "multiplied by 120 (agibot_120 mode)"
    elif gripper_mode == "normalized_01":
        lg = lg_raw[:T].astype(np.float32)
        rg = rg_raw[:T].astype(np.float32)
        gripper_norm_note = "kept as-is [0,1] (RoboTwin default)"
    else:
        lg = lg_raw[:T].astype(np.float32)
        rg = rg_raw[:T].astype(np.float32)
        gripper_norm_note = "raw value, unnormalized"

    abs_act = np.zeros((T, 16), dtype=np.float32)
    abs_act[:, 0:3] = left[:T, :3]
    abs_act[:, 3:7] = left[:T, 3:7]
    abs_act[:, 7] = lg
    abs_act[:, 8:11] = right[:T, :3]
    abs_act[:, 11:15] = right[:T, 3:7]
    abs_act[:, 15] = rg

    left_q_norms = np.linalg.norm(left[:T, 3:7], axis=1)
    right_q_norms = np.linalg.norm(right[:T, 3:7], axis=1)
    q_norm_ok = bool(np.allclose(left_q_norms, 1.0, atol=1e-3) and np.allclose(right_q_norms, 1.0, atol=1e-3))

    left_rpy = np.zeros((T, 6), dtype=np.float32)
    right_rpy = np.zeros((T, 6), dtype=np.float32)
    left_rpy[:, :3] = left[:T, :3]
    right_rpy[:, :3] = right[:T, :3]
    for t in range(T):
        left_rpy[t, 3:6] = Rotation.from_quat(left[t, 3:7]).as_euler("xyz", degrees=False)
        right_rpy[t, 3:6] = Rotation.from_quat(right[t, 3:7]).as_euler("xyz", degrees=False)

    delta = np.zeros((max(1, T - 1), 14), dtype=np.float32)
    for t in range(1, T):
        delta[t - 1, 0:6] = left_rpy[t] - left_rpy[t - 1]
        delta[t - 1, 3:6] = _normalize_angles(delta[t - 1, 3:6])
        delta[t - 1, 6] = lg[t]
        delta[t - 1, 7:13] = right_rpy[t] - right_rpy[t - 1]
        delta[t - 1, 10:13] = _normalize_angles(delta[t - 1, 10:13])
        delta[t - 1, 13] = rg[t]

    motion_meta = _motion_meta_from_abs_actions(abs_act)

    return abs_act.astype(np.float32), delta.astype(np.float32), {
        "abs_action_layout": "left_xyz+quat_xyzw+gripper+right_xyz+quat_xyzw+gripper=16",
        "delta_action_layout": "left_dxyz+drpy+gripper+right_dxyz+drpy+gripper=14",
        "quaternion_source": "SAPIEN endpose (xyzw)", "quaternion_norms_ok": q_norm_ok,
        "euler_convention": "xyz (scipy Rotation default)", "delta_rpy_normalized_to_pi": True,
        "gripper_mode": gripper_mode, "gripper_note": gripper_norm_note,
        "gripper_left_range": [lg_min, lg_max], "gripper_right_range": [rg_min, rg_max],
        **motion_meta,
        "domain_normalization": "NOT applied (no RoboTwin domain stats)",
        "raw_lengths": {
            "left_endpose": int(left.shape[0]), "right_endpose": int(right.shape[0]),
            "left_gripper": int(lg_raw.shape[0]), "right_gripper": int(rg_raw.shape[0]),
        },
        "used_length": T, "alignment_policy": "truncate_to_min_length",
    }


def _motion_meta_from_abs_actions(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 11:
        return {
            "left_arm_moving": False,
            "right_arm_moving": False,
            "active_arm": "unknown",
            "left_move_sum": 0.0,
            "right_move_sum": 0.0,
            "is_dual_arm": False,
        }
    left_step = np.linalg.norm(np.diff(actions[:, 0:3], axis=0), axis=1) if actions.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    right_step = np.linalg.norm(np.diff(actions[:, 8:11], axis=0), axis=1) if actions.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    left_move_sum = float(left_step.sum())
    right_move_sum = float(right_step.sum())
    move_eps = 1e-6
    left_moving = bool(left_move_sum > move_eps)
    right_moving = bool(right_move_sum > move_eps)
    if left_moving and right_moving:
        active_arm = "both"
    elif left_moving:
        active_arm = "left"
    elif right_moving:
        active_arm = "right"
    else:
        active_arm = "none"
    return {
        "left_arm_moving": left_moving,
        "right_arm_moving": right_moving,
        "active_arm": active_arm,
        "left_move_sum": left_move_sum,
        "right_move_sum": right_move_sum,
        "is_dual_arm": bool(left_moving and right_moving),
    }


def _extract_camera_from_hdf5(h5_file) -> dict[str, np.ndarray] | None:
    cam = {}
    for key in ["intrinsic_cv", "extrinsic_cv", "cam2world_gl"]:
        path = f"observation/head_camera/{key}"
        if path in h5_file:
            cam[key] = np.asarray(h5_file[path], dtype=np.float32)
    return cam if cam else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RoboTwin2.0 → AgiBot-compatible format converter")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--converted_root", required=True)
    parser.add_argument("--robot", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--min_frames", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config_json", default=None)
    args = parser.parse_args()

    cfg: dict[str, Any] = {"gripper_mode": "normalized_01"}
    if args.robot:
        cfg["robot_include"] = _parse_list_arg(args.robot)
    if args.variant:
        cfg["variant_include"] = _parse_list_arg(args.variant)
    if args.min_frames > 0:
        cfg["min_frames"] = args.min_frames
    if args.config_json:
        override = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if Path(args.config_json).exists() else json.loads(args.config_json)
        cfg.update(override)

    print(f"[convert] robot={cfg.get('robot_include','all')} variant={cfg.get('variant_include','all')} "
          f"min_frames={cfg.get('min_frames',0)} workers={args.workers}")

    raw_root = Path(args.raw_root)
    converted_root = Path(args.converted_root)

    if args.workers <= 1:
        converter = RoboTwinConverter(str(raw_root), str(converted_root), cfg)
        manifest = converter.build_manifest(max_episodes=args.max_episodes)
        print(f"[convert:robotwin] converted={manifest['stats']['converted']} failed={manifest['stats']['failed']}")
        sys.exit(0)

    if h5py is None:
        print("[convert] WARNING: h5py not available, falling back to single-process")
        converter = RoboTwinConverter(str(raw_root), str(converted_root), cfg)
        manifest = converter.build_manifest(max_episodes=args.max_episodes)
        print(f"[convert:robotwin] converted={manifest['stats']['converted']} failed={manifest['stats']['failed']}")
        sys.exit(0)

    converter = RoboTwinConverter(str(raw_root), str(converted_root), cfg)
    all_eps = converter.scan_raw_dataset()
    if args.max_episodes:
        all_eps = all_eps[:args.max_episodes]
    print(f"[convert] {len(all_eps)} episodes scanned, {args.workers} workers")

    chunk_size = max(1, len(all_eps) // args.workers)
    chunks = [all_eps[i:i + chunk_size] for i in range(0, len(all_eps), chunk_size)]
    chunks = chunks[:args.workers]

    worker_fn = _EpisodeWorker(str(raw_root), str(converted_root), cfg)
    with mp.get_context("fork").Pool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(worker_fn, chunks),
                  total=len(chunks), desc="[convert:workers]", unit="chunk"))

    from al_pipeline.utils import ensure_dir, save_json
    all_items = []
    total_ok = total_fail = 0
    worker_dir = converted_root / ".workers"
    if worker_dir.is_dir():
        for wf in sorted(worker_dir.glob("worker_*.json")):
            wdata = json.loads(wf.read_text())
            all_items.extend(wdata.get("items", []))
            total_ok += wdata.get("count", 0)
            total_fail += len(wdata.get("failures", []))

    ensure_dir(converted_root.parent / "manifests")
    save_json({
        "dataset": "robotwin",
        "items": all_items,
        "stats": {"converted": total_ok, "failed": total_fail},
    }, converted_root.parent / "manifests" / "all.json")
    print(f"[convert:robotwin] {len(all_eps)} episodes → {total_ok} converted, {total_fail} failed")
