from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def _read_image_dir(dir_path: Path, max_frames: int | None = None) -> np.ndarray:
    """Read sorted image files from a directory into a [T,H,W,C] uint8 array."""
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No image files found in {dir_path}")
    if max_frames is not None:
        files = files[:max_frames]
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in files]
    return np.stack(frames, axis=0)


def load_video_frames(path: Union[str, Path], max_frames: int | None = None) -> np.ndarray:
    """Load frames from an mp4/avi/mkv video as a [T,H,W,C] uint8 array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    try:
        import imageio.v3 as iio

        frames = []
        for idx, frame in enumerate(iio.imiter(path)):
            frames.append(np.asarray(frame)[..., :3])
            if max_frames is not None and idx + 1 >= max_frames:
                break
        if frames:
            return np.stack(frames, axis=0)
    except Exception:
        pass

    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")
    return np.stack(frames, axis=0)


def load_frames(path: Union[str, Path], max_frames: int | None = None) -> np.ndarray:
    """Load frames from a directory of images or a video file.

    Returns [T,H,W,C] uint8 array.
    """
    path = Path(path)
    if path.is_dir():
        return _read_image_dir(path, max_frames)
    return load_video_frames(path, max_frames)


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Convert uint8 [T,H,W,C] to float32 in [0,1]."""
    if frames.dtype == np.uint8:
        return frames.astype(np.float32) / 255.0
    return frames.astype(np.float32)


def save_video(frames: np.ndarray, path: Union[str, Path], fps: int = 15) -> None:
    """Save [T,H,W,C] float32/uint8 frames as mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
