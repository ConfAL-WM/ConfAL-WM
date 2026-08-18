from pathlib import Path
from typing import Any, Dict

from adapters.base import BaseScorer
from metrics.acquisition import near_miss_score, progress_low_score
from utils import (
    coerce_float_list,
    format_command,
    load_video_frames,
    post_robometer_npy,
    project_path,
    read_json_or_csv,
    run_command,
)


class RobometerScorer(BaseScorer):
    """Score a trajectory with a user-started Robometer eval server."""

    def __init__(self, method: str, config: Dict[str, Any], method_config: Dict[str, Any]):
        super().__init__(method, config, method_config)
        self.mode = method_config.get("mode", "progress")
        self.repo = project_path(config["robometer_repo"])
        self.server_url = method_config.get(
            "server_url",
            config.get("robometer_server_url", "http://localhost:8000"),
        ).rstrip("/")
        self.timeout = float(method_config.get("timeout", config.get("timeout", 120.0)))
        self.fps = method_config.get("fps", config.get("video_fps", 1.0))
        self.max_frames = int(method_config.get("max_frames", config.get("max_video_frames", 16)))
        self.use_frame_steps = bool(method_config.get("use_frame_steps", False))

    def score_episode(self, example: dict) -> dict:
        if self.method_config.get("backend", "server") == "server":
            return self._score_episode_with_server(example)
        return self._score_episode_with_command(example)

    def _score_episode_with_server(self, example: dict) -> dict:
        video_path = example.get("video_path")
        if not video_path:
            raise ValueError(f"Missing video_path for Robometer scoring: {example.get('episode_id')}")
        frames = load_video_frames(
            str(video_path),
            fps=float(self.fps) if self.fps is not None else None,
            max_frames=self.max_frames,
        )
        task = str(example.get("task") or example.get("task_name") or "")
        result = post_robometer_npy(
            self.server_url,
            frames=frames,
            task=task,
            sample_id=str(example.get("episode_id") or "episode"),
            timeout=self.timeout,
            use_frame_steps=self.use_frame_steps,
        )
        frame_scores, success_probs = self._parse_server_scores(result)
        episode_score = frame_scores[-1] if frame_scores else 0.0
        if self.mode == "preference":
            acquisition = near_miss_score(frame_scores, episode_score)
        else:
            acquisition = progress_low_score(frame_scores, episode_score)
        return {
            "episode_id": example.get("episode_id"),
            "method": self.method,
            "frame_scores": frame_scores,
            "episode_score": episode_score,
            "acquisition_score": acquisition,
            "extra": {
                "mode": self.mode,
                "server_url": self.server_url,
                "success_probs": success_probs,
            },
        }

    def _score_episode_with_command(self, example: dict) -> dict:
        output_path = self._episode_output_path(example)
        command = self._build_command(example, output_path)
        print(f"[evac_al_baselines] Robometer command template: {command}")
        if self.method_config.get("run_command", False):
            run_command(command, cwd=str(self.repo))
        else:
            raise RuntimeError(
                "Robometer command is a TODO template by default. Set methods."
                f"{self.method}.run_command=true and verify the command_template first."
            )
        frame_scores = self._parse_frame_scores(output_path)
        episode_score = frame_scores[-1] if frame_scores else 0.0
        return {
            "episode_id": example.get("episode_id"),
            "method": self.method,
            "frame_scores": frame_scores,
            "episode_score": episode_score,
            "acquisition_score": (
                near_miss_score(frame_scores, episode_score)
                if self.mode == "preference"
                else progress_low_score(frame_scores, episode_score)
            ),
            "extra": {"mode": self.mode, "output_path": str(output_path)},
        }

    def _episode_output_path(self, example: dict) -> Path:
        result_dir = project_path(self.config.get("result_dir", "baselines/results"))
        episode_id = str(example.get("episode_id", "episode"))
        return result_dir / self.method / f"{episode_id}.json"

    def _build_command(self, example: dict, output_path: Path) -> str:
        template = self.method_config.get(
            "command_template",
            "python robometer/evals/run_baseline_eval.py "
            "reward_model=rbm "
            "custom_eval.eval_types=[reward_alignment] "
            "custom_eval.use_frame_steps=true "
            "max_frames=8 "
            "# TODO: adapt dataset/video inputs: {video_path} task={task} output={output_path}",
        )
        return format_command(
            template,
            repo=self.repo,
            video_path=example.get("video_path", ""),
            task=example.get("task", ""),
            mode=self.mode,
            output_path=output_path,
        )

    def _parse_frame_scores(self, output_path: Path) -> list[float]:
        rows = read_json_or_csv(str(output_path))
        for row in rows:
            for key in ("frame_scores", "progress", "scores", "completion_percentage"):
                vals = coerce_float_list(row.get(key))
                if vals:
                    return vals
            if "episode_score" in row:
                return coerce_float_list(row.get("episode_score"))
        return []

    def _parse_server_scores(self, result: Dict[str, Any]) -> tuple[list[float], list[float]]:
        outputs_progress = result.get("outputs_progress") or result
        progress_pred = outputs_progress.get("progress_pred", [])
        frame_scores = progress_pred[0] if progress_pred and isinstance(progress_pred[0], list) else progress_pred
        frame_scores = coerce_float_list(frame_scores)

        success_container = (
            result.get("outputs_success")
            or outputs_progress.get("outputs_success")
            or outputs_progress
        )
        success_probs_raw = success_container.get("success_probs", []) if isinstance(success_container, dict) else []
        success_probs = (
            success_probs_raw[0]
            if success_probs_raw and isinstance(success_probs_raw[0], list)
            else success_probs_raw
        )
        return frame_scores, coerce_float_list(success_probs)
