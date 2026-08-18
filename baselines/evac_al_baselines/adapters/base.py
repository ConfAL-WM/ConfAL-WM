from abc import ABC, abstractmethod
from typing import Any, Dict

from tqdm import tqdm

from metrics.acquisition import compute_acquisition
from utils import iter_jsonl, stable_random_score, write_jsonl


class BaseScorer(ABC):
    def __init__(self, method: str, config: Dict[str, Any], method_config: Dict[str, Any]):
        self.method = method
        self.config = config
        self.method_config = method_config

    @abstractmethod
    def score_episode(self, example: dict) -> dict:
        raise NotImplementedError

    def score_manifest(self, manifest_path: str, output_path: str) -> None:
        examples = list(iter_jsonl(manifest_path))
        total = len(examples)

        def rows():
            for example in tqdm(examples, desc=f"[{self.method}] scoring", unit="ep"):
                try:
                    scored = self.score_episode(example)
                except Exception as exc:
                    scored = {
                        "episode_id": example.get("episode_id"),
                        "method": self.method,
                        "score_ready": False,
                        "error": repr(exc),
                        "frame_scores": [],
                        "episode_score": 0.0,
                        "acquisition_score": 0.0,
                        "extra": {"source": "exception"},
                    }
                scored.setdefault("episode_id", example.get("episode_id"))
                scored.setdefault("method", self.method)
                scored.setdefault("extra", {})
                for key in (
                    "video_path",
                    "score_video_path",
                    "gt_video_path",
                    "source_video_path",
                    "pred_frames_dir",
                ):
                    if example.get(key) is not None:
                        scored.setdefault(key, example.get(key))
                if "acquisition_score" not in scored:
                    scored["acquisition_score"] = self.default_acquisition(scored, example)
                yield scored

        write_jsonl(output_path, rows())
        print(f"[evac_al_baselines] wrote {output_path} ({total} episodes)")

    def default_acquisition(self, scored: Dict[str, Any], example: Dict[str, Any]) -> float:
        strategy = self.method_config.get("acquisition", self.method_config.get("type", "combined"))
        return compute_acquisition(
            strategy=strategy,
            frame_scores=scored.get("frame_scores", []),
            episode_score=scored.get("episode_score"),
            extra=scored.get("extra", {}),
            random_value=stable_random_score(example, salt=self.method),
        )


class RandomScorer(BaseScorer):
    def score_episode(self, example: dict) -> dict:
        score = stable_random_score(example, salt=self.method)
        return {
            "episode_id": example.get("episode_id"),
            "method": self.method,
            "frame_scores": [],
            "episode_score": score,
            "acquisition_score": score,
            "extra": {"source": "stable_hash"},
        }


class C3ManifestScorer(BaseScorer):
    """Use C3/Ours scores already materialized in the candidate manifest."""

    def score_episode(self, example: dict) -> dict:
        frame_scores = example.get("c3_frame_scores", example.get("frame_scores", []))
        episode_score = example.get(
            "c3_score",
            example.get("episode_score", example.get("success_probability", example.get("confidence", 0.0))),
        )
        try:
            episode_score = float(episode_score)
        except (TypeError, ValueError):
            episode_score = 0.0
        extra = {
            "source": "manifest_fields",
            "success_probability": example.get("success_probability", episode_score),
            "confidence": example.get("confidence", episode_score),
        }
        return {
            "episode_id": example.get("episode_id"),
            "method": self.method,
            "frame_scores": frame_scores if isinstance(frame_scores, list) else [],
            "episode_score": episode_score,
            "extra": extra,
        }
