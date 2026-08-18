from typing import Any, Dict

from adapters.base import BaseScorer
from metrics.acquisition import combined_score, near_miss_score, progress_low_score
from utils import coerce_float_list, encode_ndarray_image, load_video_frames, post_json


class LRMScorer(BaseScorer):
    """HTTP wrapper for a user-started Large Reward Models server."""

    DEFAULT_ENDPOINTS = {
        "progress": "/compute_reward",
        "completion": "/compute_completion",
        "contrastive": "/compute_comparison",
        "roboreward": "/compute_roboreward",
    }

    def __init__(self, method: str, config: Dict[str, Any], method_config: Dict[str, Any]):
        super().__init__(method, config, method_config)
        self.mode = method_config.get("mode", "progress")
        self.server_url = method_config.get("server_url", config.get("lrm_server_url", "http://localhost:5002")).rstrip("/")
        self.endpoint = method_config.get("endpoint", self.DEFAULT_ENDPOINTS.get(self.mode, "/compute_reward"))
        self.timeout = float(method_config.get("timeout", 120.0))
        self.fps = method_config.get("fps", config.get("video_fps", 1.0))
        self.max_frames = int(method_config.get("max_frames", config.get("max_video_frames", 16)))

    def score_episode(self, example: dict) -> dict:
        video_path = example.get("video_path")
        if not video_path:
            raise ValueError(f"Missing video_path for LRMs scoring: {example.get('episode_id')}")
        frames = load_video_frames(
            str(video_path),
            fps=float(self.fps) if self.fps is not None else None,
            max_frames=self.max_frames,
        )
        task = str(example.get("task") or example.get("task_name") or "")
        result, frame_scores, episode_score = self._score_frames(frames, task)
        acquisition = self._acquisition(frame_scores, episode_score, result)
        return {
            "episode_id": example.get("episode_id"),
            "method": self.method,
            "frame_scores": frame_scores,
            "episode_score": episode_score,
            "acquisition_score": acquisition,
            "extra": {"mode": self.mode, "server_result": result},
        }

    def _score_frames(self, frames: Any, task: str) -> tuple[Dict[str, Any], list[float], float]:
        if self.mode == "completion":
            return self._score_completion(frames, task)
        if self.mode in {"contrastive", "comparison"}:
            return self._score_contrastive(frames, task)
        if self.mode == "roboreward":
            return self._score_roboreward(frames, task)
        return self._score_progress(frames, task)

    def _score_progress(self, frames: Any, task: str) -> tuple[Dict[str, Any], list[float], float]:
        payload = {
            "images": [encode_ndarray_image(frame) for frame in frames],
            "task_descriptions": [task] * len(frames),
            "reward_type": "progress",
        }
        url = self.server_url + self.method_config.get("batch_endpoint", "/compute_rewards_batch")
        result = post_json(url, payload, timeout=self.timeout * max(1, len(frames)))
        rows = result.get("results", []) if isinstance(result, dict) else []
        frame_scores = []
        for row in rows:
            vals = coerce_float_list(row.get("score") if isinstance(row, dict) else row)
            if vals:
                frame_scores.append(vals[-1])
        episode_score = frame_scores[-1] if frame_scores else 0.0
        return result, frame_scores, episode_score

    def _score_completion(self, frames: Any, task: str) -> tuple[Dict[str, Any], list[float], float]:
        payload = {"image": encode_ndarray_image(frames[-1]), "task_description": task}
        url = self.server_url + self.method_config.get("endpoint", "/compute_completion")
        result = post_json(url, payload, timeout=self.timeout)
        episode_score = self._extract_episode_score(result, [])
        return result, [episode_score], episode_score

    def _score_contrastive(self, frames: Any, task: str) -> tuple[Dict[str, Any], list[float], float]:
        url = self.server_url + self.method_config.get("endpoint", "/compute_comparison")
        progress = [0.0]
        pair_results = []
        if len(frames) < 2:
            return {"results": pair_results}, progress, 0.0
        denom = max(1, len(frames) - 1)
        current = 0.0
        for prev, cur in zip(frames[:-1], frames[1:]):
            payload = {
                "image_a": encode_ndarray_image(prev)["data"],
                "image_a_shape": list(prev.shape),
                "image_a_dtype": str(prev.dtype),
                "image_b": encode_ndarray_image(cur)["data"],
                "image_b_shape": list(cur.shape),
                "image_b_dtype": str(cur.dtype),
                "task_description": task,
            }
            result = post_json(url, payload, timeout=self.timeout)
            pair_results.append(result)
            delta = float(result.get("score", 0.0)) if isinstance(result, dict) else 0.0
            current = max(0.0, min(1.0, current + max(-1.0, min(1.0, delta)) / denom))
            progress.append(current)
        return {"results": pair_results}, progress, progress[-1] if progress else 0.0

    def _score_roboreward(self, frames: Any, task: str) -> tuple[Dict[str, Any], list[float], float]:
        payload = {
            "frames": [encode_ndarray_image(frame) for frame in frames],
            "task_description": task,
        }
        url = self.server_url + self.method_config.get("endpoint", "/compute_roboreward")
        result = post_json(url, payload, timeout=self.timeout)
        episode_score = self._extract_episode_score(result, [])
        frame_scores = [episode_score] * len(frames)
        return result, frame_scores, episode_score

    def _extract_frame_scores(self, result: Dict[str, Any]) -> list[float]:
        for key in ("frame_scores", "scores", "progress", "completion_percentages"):
            vals = coerce_float_list(result.get(key))
            if vals:
                return vals
        return []

    def _extract_episode_score(self, result: Dict[str, Any], frame_scores: list[float]) -> float:
        for key in ("episode_score", "score", "reward", "completion_score"):
            vals = coerce_float_list(result.get(key))
            if vals:
                return vals[-1]
        return frame_scores[-1] if frame_scores else 0.0

    def _acquisition(self, frame_scores: list[float], episode_score: float, result: Dict[str, Any]) -> float:
        if self.mode == "progress":
            return progress_low_score(frame_scores, episode_score)
        if self.mode == "completion":
            return near_miss_score(frame_scores, episode_score)
        return combined_score(frame_scores, episode_score, {"server_result": result})
