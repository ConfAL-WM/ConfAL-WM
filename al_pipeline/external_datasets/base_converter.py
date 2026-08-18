from __future__ import annotations

import argparse
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from al_pipeline.utils import ensure_dir, save_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


class ExternalDatasetConverter(ABC):
    """Base converter into ConfAL-WM external_worldmodel_dataset format."""

    dataset_name: str = "external"

    def __init__(self, raw_root: str | Path, converted_root: str | Path, config: dict[str, Any] | None = None) -> None:
        self.raw_root = Path(raw_root)
        self.converted_root = Path(converted_root)
        self.config = config or {}
        self.dataset_root = self.converted_root / self.dataset_name
        self.episodes_root = self.dataset_root / "episodes"
        self.manifests_root = self.dataset_root / "manifests"

    @abstractmethod
    def scan_raw_dataset(self) -> list[dict[str, Any]]:
        """Return raw episode descriptors."""

    @abstractmethod
    def convert_episode(self, raw_episode: dict[str, Any]) -> dict[str, Any]:
        """Convert one raw episode and return manifest record."""

    def validate_converted_episode(self, episode_dir: Path) -> dict[str, Any]:
        frames_dir = episode_dir / "frames"
        # Prefer actions_evac.npy, fallback to actions.npy
        actions_path = episode_dir / "actions_evac.npy"
        if not actions_path.exists():
            actions_path = episode_dir / "actions.npy"
        meta_path = episode_dir / "meta.json"
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"Missing frames directory: {frames_dir}")
        frame_files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not frame_files:
            raise FileNotFoundError(f"No image frames found in {frames_dir}")
        if not actions_path.exists():
            raise FileNotFoundError(f"Missing actions_evac.npy or actions.npy: {episode_dir}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json: {meta_path}")
        actions = np.load(actions_path)
        return {
            "num_frames": len(frame_files),
            "action_shape": list(actions.shape),
            "action_dim": int(actions.shape[-1]) if actions.ndim >= 2 else 1,
        }

    def build_manifest(self, max_episodes: int | None = None) -> dict[str, Any]:
        raw_eps = self.scan_raw_dataset()
        if max_episodes is not None:
            raw_eps = raw_eps[:max_episodes]
        ensure_dir(self.episodes_root)
        ensure_dir(self.manifests_root)
        items = []
        failures = []
        for raw_ep in tqdm(raw_eps, desc=f"[convert:{self.dataset_name}]", unit="ep"):
            try:
                items.append(self.convert_episode(raw_ep))
            except Exception as exc:
                failures.append({
                    "raw_episode": {k: str(v) for k, v in raw_ep.items()},
                    "error": repr(exc),
                })
        manifest = {
            "dataset": self.dataset_name,
            "format": "external_worldmodel",
            "items": items,
            "failures": failures,
            "stats": {
                "raw_episodes": len(raw_eps),
                "converted": len(items),
                "failed": len(failures),
            },
        }
        save_json(manifest, self.manifests_root / "all.json")
        save_json({"dataset": self.dataset_name, **manifest["stats"]}, self.manifests_root / "split_summary.json")
        return manifest

    def _copy_frame_dir(self, src_dir: Path, dst_dir: Path) -> int:
        ensure_dir(dst_dir)
        frames = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for idx, frame in enumerate(frames):
            dst = dst_dir / f"frame_{idx:06d}{frame.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(frame, dst)
        return len(frames)

    def _extract_video_frames(self, video_path: Path, dst_dir: Path) -> int:
        ensure_dir(dst_dir)
        try:
            import imageio.v3 as iio
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("video conversion requires imageio and pillow") from exc
        count = 0
        for idx, frame in enumerate(iio.imiter(video_path)):
            Image.fromarray(np.asarray(frame)[..., :3]).save(dst_dir / f"frame_{idx:06d}.jpg", quality=95)
            count += 1
        return count

    def _write_episode(
        self,
        *,
        episode_id: str,
        frames_source: Path,
        actions: np.ndarray,
        meta: dict[str, Any],
        proprio: np.ndarray | None = None,
        camera: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        ep_dir = ensure_dir(self.episodes_root / episode_id)
        frames_dir = ep_dir / "frames"
        if frames_source.is_dir():
            n_frames = self._copy_frame_dir(frames_source, frames_dir)
        elif frames_source.suffix.lower() in VIDEO_EXTS:
            n_frames = self._extract_video_frames(frames_source, frames_dir)
        else:
            raise ValueError(f"Unsupported frames_source: {frames_source}")

        actions = np.asarray(actions, dtype=np.float32)
        np.save(ep_dir / "actions.npy", actions)
        proprio_path = None
        if proprio is not None:
            proprio_path = ep_dir / "proprio.npy"
            np.save(proprio_path, np.asarray(proprio, dtype=np.float32))
        camera_path = None
        if camera is not None:
            camera_path = ep_dir / "camera.npz"
            np.savez(camera_path, **{k: np.asarray(v, dtype=np.float32) for k, v in camera.items()})

        meta = {
            "dataset": self.dataset_name,
            "episode_id": episode_id,
            "num_frames": int(n_frames),
            "action_dim": int(actions.shape[-1]) if actions.ndim >= 2 else 1,
            "camera_path": str(camera_path) if camera_path else None,
            **meta,
        }
        save_json(meta, ep_dir / "meta.json")
        validation = self.validate_converted_episode(ep_dir)
        return {
            "dataset": self.dataset_name,
            "episode_id": episode_id,
            "task_name": meta.get("task_name", "unknown"),
            "robot": meta.get("robot", "unknown"),
            "camera": meta.get("camera", "unknown"),
            "episode_dir": str(ep_dir),
            "video_or_frames_path": str(frames_dir),
            "frames_dir": str(frames_dir),
            "actions_path": str(ep_dir / "actions.npy"),
            "proprio_path": str(proprio_path) if proprio_path else None,
            "camera_path": str(camera_path) if camera_path else None,
            "meta_path": str(ep_dir / "meta.json"),
            "num_frames": validation["num_frames"],
            "action_dim": validation["action_dim"],
            "format": "external_worldmodel",
        }


def converter_cli(converter_cls: type[ExternalDatasetConverter]) -> None:
    parser = argparse.ArgumentParser(description=f"Convert {converter_cls.dataset_name} to external_worldmodel format")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--converted_root", required=True)
    parser.add_argument("--config_json", default=None, help="Optional JSON config for dataset-specific field names")
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    cfg = {}
    if args.config_json:
        cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    converter = converter_cls(args.raw_root, args.converted_root, cfg)
    manifest = converter.build_manifest(max_episodes=args.max_episodes)
    print(f"[convert:{converter.dataset_name}] converted={manifest['stats']['converted']} failed={manifest['stats']['failed']}")
