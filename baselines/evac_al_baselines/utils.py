import csv
import base64
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def safe_slug(value: Any, default: str = "item") -> str:
    text = str(value or default)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text or default


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> Dict[str, Any]:
    """Small YAML fallback for the shipped config shape."""
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_yaml_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _strip_yaml_comment(raw: str) -> str:
    quote = None
    escaped = False
    for idx, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        if char == "#" and quote is None:
            return raw[:idx]
    return raw


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def stable_random_score(example: Dict[str, Any], salt: str = "evac") -> float:
    key = str(example.get("episode_id") or example.get("video_path") or json.dumps(example, sort_keys=True))
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def coerce_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def format_command(template: str, **kwargs: Any) -> str:
    values = {k: str(v) for k, v in kwargs.items()}
    return template.format(**values)


def run_command(command: str, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    print(f"[evac_al_baselines] external command: {command}")
    return subprocess.run(command, cwd=cwd, shell=True, check=True, text=True)


def post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach LRMs server at {url}: {exc}") from exc


def encode_ndarray_image(image: Any) -> Dict[str, Any]:
    import numpy as np

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _load_numpy_frames(path: Path) -> Any:
    import numpy as np

    if path.suffix.lower() == ".npy":
        frames = np.load(str(path))
    else:
        with np.load(str(path), allow_pickle=False) as npz:
            if "frames" in npz:
                frames = npz["frames"].copy()
            elif "arr_0" in npz:
                frames = npz["arr_0"].copy()
            else:
                frames = next(iter(npz.values())).copy()
    if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
        frames = frames.transpose(0, 2, 3, 1)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def load_video_frames(
    video_path: str,
    *,
    fps: Optional[float] = None,
    max_frames: int = 16,
    include_last: bool = True,
) -> Any:
    """Load a compact RGB uint8 frame array from a video or .npy/.npz file."""
    import numpy as np

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video_path not found: {video_path}")
    if path.is_dir():
        from PIL import Image

        frame_paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not frame_paths:
            raise RuntimeError(f"No image frames found in directory: {video_path}")
        if max_frames and len(frame_paths) > max_frames:
            idxs = np.linspace(0, len(frame_paths) - 1, int(max_frames), dtype=int)
            frame_paths = [frame_paths[int(i)] for i in idxs]
        return np.asarray([np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], dtype=np.uint8)
    if path.suffix.lower() in {".npy", ".npz"}:
        frames = _load_numpy_frames(path)
        if max_frames and frames.shape[0] > max_frames:
            idxs = np.linspace(0, frames.shape[0] - 1, int(max_frames), dtype=int)
            frames = frames[idxs]
        return frames

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0:
            indices = []
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                indices.append(len(indices))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            arr = np.asarray(frames, dtype=np.uint8)
            if arr.size == 0:
                raise RuntimeError(f"No frames decoded from {video_path}")
            if max_frames and arr.shape[0] > max_frames:
                idxs = np.linspace(0, arr.shape[0] - 1, int(max_frames), dtype=int)
                arr = arr[idxs]
            return arr

        if fps and native_fps > 0:
            step = max(1, int(round(native_fps / float(fps))))
            indices = list(range(0, total, step))
            if include_last and indices[-1] != total - 1:
                indices.append(total - 1)
            if max_frames and len(indices) > max_frames:
                indices = np.linspace(0, total - 1, int(max_frames), dtype=int).tolist()
        else:
            count = min(int(max_frames or total), total)
            indices = np.linspace(0, total - 1, count, dtype=int).tolist()

        frames = []
        for idx in sorted(set(int(i) for i in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")
        return np.asarray(frames, dtype=np.uint8)
    except ImportError:
        import imageio.v3 as iio

        frames = []
        for frame in iio.imiter(str(path)):
            frames.append(np.asarray(frame))
        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")
        arr = np.asarray(frames)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if max_frames and arr.shape[0] > max_frames:
            idxs = np.linspace(0, arr.shape[0] - 1, int(max_frames), dtype=int)
            arr = arr[idxs]
        return arr


def post_robometer_npy(
    server_url: str,
    *,
    frames: Any,
    task: str,
    sample_id: str,
    timeout: float = 120.0,
    use_frame_steps: bool = False,
) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Robometer/GVL server scoring requires `requests` in the ConfAL-WM environment.") from exc

    import tempfile

    import numpy as np
    from PIL import Image as _PILImage

    frames = np.asarray(frames)
    tmpdir = tempfile.mkdtemp(prefix="robometer_frames_")
    frame_paths = []
    try:
        for i in range(frames.shape[0]):
            img = _PILImage.fromarray(frames[i])
            path = f"{tmpdir}/frame_{i:06d}.jpg"
            img.save(path, "JPEG", quality=95)
            frame_paths.append(path)

        sample = {
            "sample_type": "progress",
            "trajectory": {
                "frames": frame_paths,
                "frames_shape": [int(x) for x in frames.shape],
                "task": task,
                "id": sample_id,
                "metadata": {"subsequence_length": int(frames.shape[0])},
                "video_embeddings": None,
            },
        }
        payload = {
            "samples": [sample],
            "use_frame_steps": use_frame_steps,
        }
        url = server_url.rstrip("/") + "/evaluate_batch"
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def read_json_or_csv(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".jsonl":
        return list(iter_jsonl(str(p)))
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    return []
