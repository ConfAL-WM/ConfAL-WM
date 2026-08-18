import base64
import io
import json
import os
import random
import re
import ssl
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from PIL import Image

from adapters.base import BaseScorer
from metrics.acquisition import stagnation_score
from metrics.voc import compute_voc
from utils import load_video_frames


def _clip01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    if x > 1.0:
        x = x / 100.0
    return max(0.0, min(1.0, x))


def _openai_chat_url(base_url: str) -> str:
    url = str(base_url or "http://localhost:8000/v1").rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _build_ssl_context() -> "ssl.SSLContext":
    """A TLS context with a usable CA bundle.

    urllib's default context inherits OpenSSL's compiled-in ``cafile`` path,
    which on some conda envs points at a non-existent prefix and breaks cert
    verification (``SSL: CERTIFICATE_VERIFY_FAILED ... unable to get local
    issuer certificate``). Prefer certifi's bundle in that case; honour the
    user's own ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` if set.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return ssl.create_default_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class GVLQwenScorer(BaseScorer):
    """GVL-style shuffled-frame value prediction through an OpenAI-compatible VLM API.

    Talks to any OpenAI-compatible vision-language endpoint — a cloud platform
    API (e.g. Alibaba DashScope ``qwen-vl-max``) or a local vLLM server. Set
    ``base_url`` / ``api_key`` / ``model_name`` in the baseline config
    (``methods.gvl`` or the ``gvl_qwen_*`` defaults); ``api_key`` may also come
    from the ``OPENAI_API_KEY`` (or ``GVL_API_KEY``) env var.
    """

    def __init__(self, method: str, config: Dict[str, Any], method_config: Dict[str, Any]):
        super().__init__(method, config, method_config)
        self.base_url = _openai_chat_url(
            method_config.get("base_url")
            or config.get("gvl_qwen_base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = str(
            method_config.get("model_name")
            or config.get("gvl_qwen_model_name")
            or "qwen-vl-max"
        )
        self.api_key = str(
            method_config.get("api_key")
            or config.get("gvl_qwen_api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GVL_API_KEY")
            or "EMPTY"
        )
        self.timeout = float(method_config.get("timeout", config.get("timeout", 120.0)))
        self.max_frames = int(method_config.get("max_frames", config.get("max_video_frames", 15)))
        self.max_tokens = int(method_config.get("max_tokens", 4096))
        self.temperature = float(method_config.get("temperature", 0.0))
        self.max_retries = int(method_config.get("max_retries", 3))
        self.retry_delay = float(method_config.get("retry_delay", 2.0))
        self.max_image_side = int(method_config.get("max_image_side", 512))
        self.seed = int(method_config.get("seed", 42))
        self._ssl_context = _build_ssl_context()

    def score_episode(self, example: dict) -> dict:
        video_path = example.get("video_path")
        if not video_path:
            raise ValueError(f"Missing video_path for GVL-Qwen scoring: {example.get('episode_id')}")
        frames = load_video_frames(str(video_path), fps=None, max_frames=self.max_frames)
        if frames.shape[0] == 0:
            raise RuntimeError(f"No frames loaded for {example.get('episode_id')}: {video_path}")

        task = str(example.get("task") or example.get("task_name") or "robot task")
        episode_id = str(example.get("episode_id") or "episode")
        content, frame_map = self._build_prompt_content(frames, task, episode_id)
        raw_response = self._request(content)
        frame_scores = self._parse_frame_scores(raw_response, frame_map)
        episode_score = frame_scores[-1] if frame_scores else 0.0
        acquisition = stagnation_score(frame_scores, episode_score)
        return {
            "episode_id": example.get("episode_id"),
            "method": self.method,
            "frame_scores": frame_scores,
            "episode_score": episode_score,
            "acquisition_score": acquisition,
            "extra": {
                "source": "gvl_qwen_openai_compatible",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "max_frames": self.max_frames,
                "max_image_side": self.max_image_side,
                "raw_response": raw_response,
                "frame_map": frame_map,
                "voc": compute_voc(frame_scores),
            },
        }

    def _build_prompt_content(self, frames: Any, task: str, episode_id: str) -> tuple[list[dict], list[dict]]:
        n = int(frames.shape[0])
        indices = list(range(n))
        rng = random.Random(f"{self.seed}:{episode_id}")
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if 0 in shuffled:
            shuffled.remove(0)

        prompt = (
            "You are an expert roboticist. Estimate task completion percentages for robot frames.\n"
            f"Task: {task}\n\n"
            "The initial scene is shown first and has task completion 0.\n"
            "The query frames are shown in random order. Judge each frame independently by visual task progress, "
            "not by the order in which frames are presented.\n"
            "Return only a JSON array. Each item must be:\n"
            "{\"frame_number\": <integer>, \"frame_description\": \"...\", "
            "\"task_completion_percentage\": <number from 0 to 100>}\n"
            "Include exactly one item for every query frame."
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.append({"type": "text", "text": "Initial robot scene:"})
        content.append({"type": "image_url", "image_url": {"url": self._image_data_url(frames[0]), "detail": "low"}})

        frame_map: list[dict] = []
        for display_idx, original_idx in enumerate(shuffled, start=1):
            content.append({"type": "text", "text": f"Frame {display_idx}:"})
            content.append(
                {"type": "image_url", "image_url": {"url": self._image_data_url(frames[original_idx]), "detail": "low"}}
            )
            frame_map.append(
                {
                    "frame_number": display_idx,
                    "sampled_frame_index": int(original_idx),
                    "temporal_order": int(original_idx),
                }
            )
        return content, frame_map

    def _image_data_url(self, frame: Any) -> str:
        img = Image.fromarray(frame).convert("RGB")
        if self.max_image_side > 0:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
            img.thumbnail((self.max_image_side, self.max_image_side), resample)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"

    def _request(self, content: list[dict]) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urlrequest.Request(self.base_url, data=data, headers=headers, method="POST")
            try:
                with urlrequest.urlopen(req, timeout=self.timeout, context=self._ssl_context) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                return self._extract_text(out)
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} from {self.base_url}: {body}")
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError(f"GVL-Qwen request failed after retries: {last_error}")

    @staticmethod
    def _extract_text(response: dict) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return str(content)

    def _parse_frame_scores(self, text: str, frame_map: list[dict]) -> list[float]:
        parsed = self._extract_json_array(text)
        by_display: dict[int, float] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                frame_no = item.get("frame_number", item.get("frame", item.get("index")))
                value = item.get(
                    "task_completion_percentage",
                    item.get("completion_percentage", item.get("progress", item.get("score"))),
                )
                try:
                    by_display[int(frame_no)] = _clip01(value)
                except Exception:
                    continue
        if not by_display:
            numbers = re.findall(r"[-+]?\d*\.?\d+", text or "")
            for frame, value in zip(frame_map, numbers):
                by_display[int(frame["frame_number"])] = _clip01(value)

        temporal_scores = {0: 0.0}
        for frame in frame_map:
            display = int(frame["frame_number"])
            temporal = int(frame["temporal_order"])
            temporal_scores[temporal] = by_display.get(display, 0.0)
        return [temporal_scores.get(i, 0.0) for i in sorted(temporal_scores)]

    @staticmethod
    def _extract_json_array(text: str) -> Any:
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
