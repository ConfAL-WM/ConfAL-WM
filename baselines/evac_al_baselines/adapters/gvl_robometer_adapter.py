from pathlib import Path
from typing import Any, Dict

from adapters.robometer_adapter import RobometerScorer
from metrics.acquisition import stagnation_score
from metrics.voc import compute_voc
from utils import format_command


class GVLRobometerScorer(RobometerScorer):
    """GVL via Robometer baseline eval, without a separate GVL deployment."""

    def __init__(self, method: str, config: Dict[str, Any], method_config: Dict[str, Any]):
        method_config = dict(method_config)
        method_config.setdefault("mode", "gvl")
        method_config.setdefault("server_url", config.get("gvl_server_url", "http://localhost:8001"))
        super().__init__(method, config, method_config)

    def _build_command(self, example: dict, output_path: Path) -> str:
        template = self.method_config.get(
            "command_template",
            "python robometer/evals/run_baseline_eval.py "
            "reward_model=gvl "
            "custom_eval.eval_types=[reward_alignment] "
            "custom_eval.use_frame_steps=true "
            "max_frames=8 "
            "# TODO: adapt GVL dataset/video inputs: {video_path} task={task} output={output_path}",
        )
        return format_command(
            template,
            repo=self.repo,
            video_path=example.get("video_path", ""),
            task=example.get("task", ""),
            mode=self.mode,
            output_path=output_path,
        )

    def score_episode(self, example: dict) -> dict:
        scored = super().score_episode(example)
        frame_scores = scored.get("frame_scores", [])
        scored["acquisition_score"] = stagnation_score(frame_scores, scored.get("episode_score"))
        scored["extra"]["voc"] = compute_voc(frame_scores)
        return scored
