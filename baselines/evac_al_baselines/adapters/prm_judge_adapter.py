from pathlib import Path
from typing import Any, Dict
import json
import os
import re
import shutil
import subprocess
import sys

from tqdm import tqdm

from adapters.base import BaseScorer
from metrics.acquisition import combined_score
from metrics.opd import OPD_KEYS, compute_opd_metrics, read_opd_metrics
from utils import (
    coerce_float_list,
    format_command,
    iter_jsonl,
    project_path,
    read_json_or_csv,
    run_command,
    safe_slug,
    write_jsonl,
)


class PRMJudgeScorer(BaseScorer):
    """Thin wrapper for PRM-as-a-Judge eval outputs."""

    def __init__(self, method: str, config: Dict[str, Any], method_config: Dict[str, Any]):
        super().__init__(method, config, method_config)
        self.repo = project_path(config["prm_judge_repo"])

    def score_episode(self, example: dict) -> dict:
        if self.method_config.get("batch_runner", True):
            raise RuntimeError("PRMJudgeScorer scores manifests in batch; call score_manifest().")
        output_path = self._episode_output_path(example)
        command = self._build_command(example, output_path)
        print(f"[evac_al_baselines] PRM-as-a-Judge command template: {command}")
        if self.method_config.get("run_command", False):
            run_command(command, cwd=str(self.repo / "eval"))
        else:
            raise RuntimeError(
                "PRM-as-a-Judge command is a TODO template by default. Set methods."
                f"{self.method}.run_command=true and verify command_template/output_path first."
            )
        frame_scores = self._parse_frame_scores(output_path)
        opd = read_opd_metrics(str(output_path)) or compute_opd_metrics(frame_scores)
        episode_score = opd.get("MC", frame_scores[-1] if frame_scores else 0.0)
        return {
            "episode_id": example.get("episode_id"),
            "method": self.method,
            "frame_scores": frame_scores,
            "episode_score": episode_score,
            "acquisition_score": combined_score(frame_scores, episode_score, {"opd": opd}),
            "extra": {"opd": {key: opd.get(key, 0.0) for key in OPD_KEYS}, "output_path": str(output_path)},
        }

    def score_manifest(self, manifest_path: str, output_path: str) -> None:
        examples = list(iter_jsonl(manifest_path))
        if not examples:
            write_jsonl(output_path, [])
            return
        self._manifest_path = manifest_path
        prepared = self._prepare_batch_inputs(examples)
        if self.method_config.get("run_command", True):
            self._run_batch_judge(prepared)
        else:
            raise RuntimeError(
                "PRM-as-a-Judge requires running eval/run_judge.py. "
                f"Set methods.{self.method}.run_command=true after installing the PRM environment."
            )
        records = self._load_latest_records(prepared["output_root"])
        rows = list(self._rows_from_records(examples, records))
        write_jsonl(output_path, rows)
        print(f"[evac_al_baselines] wrote PRM-as-a-Judge scores: {output_path}")

    @staticmethod
    def _attach_input_paths(row: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "video_path",
            "score_video_path",
            "gt_video_path",
            "source_video_path",
            "pred_frames_dir",
        ):
            if example.get(key) is not None:
                row.setdefault(key, example.get(key))
        return row

    def _episode_output_path(self, example: dict) -> Path:
        result_dir = project_path(self.config.get("result_dir", "baselines/results"))
        episode_id = str(example.get("episode_id", "episode"))
        return result_dir / self.method / f"{episode_id}.json"

    def _build_command(self, example: dict, output_path: Path) -> str:
        template = self.method_config.get(
            "command_template",
            "{python} run_judge.py "
            "--video-root TODO_VIDEO_ROOT "
            "--task-filter TODO_TASK "
            "--output-root {output_path} "
            "# TODO: map EVAC candidate video to PRM eval sample layout: {video_path}",
        )
        return format_command(
            template,
            python=self._python_bin(),
            repo=self.repo,
            video_path=example.get("video_path", ""),
            task=example.get("task", ""),
            output_path=output_path,
        )

    def _parse_frame_scores(self, output_path: Path) -> list[float]:
        rows = read_json_or_csv(str(output_path))
        for row in rows:
            for key in ("frame_scores", "progress", "scores", "step_scores"):
                vals = coerce_float_list(row.get(key))
                if vals:
                    return vals
        return []

    def _prepare_batch_inputs(self, examples: list[dict]) -> dict[str, Any]:
        cache_dir = project_path(self.config.get("cache_dir", "baselines/cache"))
        result_dir = project_path(self.config.get("result_dir", "baselines/results"))
        benchmark = safe_slug(self.method_config.get("benchmark", "evac_c3"))
        model_name = safe_slug(self.method_config.get("model_name", "candidate"))
        # Append manifest stem for multi-shard isolation (videos/tasks/goals/output).
        manifest_suffix = ""
        if hasattr(self, "_manifest_path"):
            stem = Path(self._manifest_path).stem
            if stem.startswith("baseline_manifest_"):
                manifest_suffix = "_" + stem[len("baseline_manifest_"):]
        videos_root = cache_dir / (self.method + manifest_suffix) / "videos"
        tasks_root = cache_dir / (self.method + manifest_suffix) / "tasks"
        goals_root = cache_dir / (self.method + manifest_suffix) / "goals"
        output_root = result_dir / (self.method + manifest_suffix)
        videos_root.mkdir(parents=True, exist_ok=True)
        tasks_root.mkdir(parents=True, exist_ok=True)
        goals_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

        task_prompts: dict[str, str] = {}
        sample_ids: dict[str, str] = {}
        # Track which sample_dirs we touch and the expected symlink names,
        # so we can clean stale symlinks from previous runs afterwards.
        expected_in_dir: dict[Path, set[str]] = {}
        for example in examples:
            episode_id = str(example.get("episode_id") or "episode")
            sample_id = safe_slug(episode_id)
            sample_ids[episode_id] = sample_id
            task_text = str(example.get("task") or example.get("task_name") or "robot task")
            task_key = safe_slug(example.get("task_name") or task_text)
            task_prompts[task_key] = task_text
            sample_dir = videos_root / benchmark / task_key / model_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            views = self._resolve_prm_views(example)
            for view, src in views.items():
                dst = sample_dir / f"{sample_id}_{view}.mp4"
                expected_in_dir.setdefault(sample_dir, set()).add(dst.name)

                # If the source is a directory (e.g. EVAC pred_frames/), convert
                # the frames to a real .mp4 so PRM-as-a-Judge can decode it.
                src_path = Path(src)
                if src_path.is_dir():
                    video_src = sample_dir / f"{sample_id}_{view}_gen.mp4"
                    if not video_src.exists():
                        self._frames_to_mp4(src_path, video_src)
                    src_path = video_src
                src = str(src_path)

                if dst.is_symlink():
                    try:
                        if Path(os.readlink(str(dst))) == src_path.resolve():
                            continue  # already points to the right source
                    except OSError:
                        pass
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                elif dst.exists():
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                try:
                    dst.symlink_to(src_path)
                except FileExistsError:
                    pass  # race with another worker

        # Remove stale symlinks and generated videos from previous runs.
        # Keeps: (a) expected symlinks, (b) _gen.mp4 videos referenced by
        # those symlinks.  Everything else is stale.
        gen_suffix = "_gen.mp4"
        for sample_dir, expected_names in expected_in_dir.items():
            if not sample_dir.exists():
                continue
            for existing in list(sample_dir.iterdir()):
                name = existing.name
                if name in expected_names:
                    continue
                # Keep _gen.mp4 files whose corresponding symlink is expected
                if name.endswith(gen_suffix):
                    link_name = name[: -len(gen_suffix)] + ".mp4"
                    if link_name in expected_names:
                        continue
                if existing.is_symlink() or existing.suffix == ".mp4":
                    try:
                        existing.unlink()
                    except OSError:
                        pass

        (tasks_root / f"{benchmark}.json").write_text(
            json.dumps(task_prompts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "benchmark": benchmark,
            "model_name": model_name,
            "videos_root": videos_root,
            "tasks_root": tasks_root,
            "goals_root": goals_root,
            "output_root": output_root,
            "sample_ids": sample_ids,
        }

    def _resolve_prm_views(self, example: dict) -> dict[str, str]:
        input_videos = example.get("input_videos") if isinstance(example.get("input_videos"), dict) else {}
        prm_videos = example.get("prm_videos") if isinstance(example.get("prm_videos"), dict) else {}
        merged = {**input_videos, **prm_videos}
        view_map = {
            "high": merged.get("cam_high") or merged.get("high") or example.get("high_video_path"),
            "left": merged.get("cam_left_wrist") or merged.get("left") or example.get("left_video_path"),
            "right": merged.get("cam_right_wrist") or merged.get("right") or example.get("right_video_path"),
        }
        if all(view_map.values()):
            return {k: str(v) for k, v in view_map.items()}
        if self.method_config.get("allow_single_view", True):
            video_path = example.get("video_path")
            if video_path:
                return {"high": str(video_path), "left": str(video_path), "right": str(video_path)}
        missing = [k for k, v in view_map.items() if not v]
        raise ValueError(
            f"PRM-as-a-Judge needs high/left/right videos for {example.get('episode_id')}; "
            f"missing={missing}. Set allow_single_view=true to reuse video_path for all views."
        )

    @staticmethod
    def _frames_to_mp4(frame_dir: Path, output: Path) -> None:
        """Convert a directory of JPEG/PNG frames to an MP4 video via ffmpeg."""
        import subprocess as _subprocess

        frame_dir = Path(frame_dir)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Find the first image to detect glob pattern
        img_files = sorted(
            p for p in frame_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not img_files:
            raise FileNotFoundError(f"No image frames found in {frame_dir}")

        # Use ffmpeg to create video at 1 fps (matching video_fps config default)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "1",
            "-pattern_type", "glob",
            "-i", str(frame_dir / "*.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output),
        ]
        result = _subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback: try explicit file list
            list_file = output.with_suffix(".txt")
            list_file.write_text(
                "\n".join(f"file '{p.resolve()}'" for p in img_files),
                encoding="utf-8",
            )
            cmd2 = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output),
            ]
            result2 = _subprocess.run(cmd2, capture_output=True, text=True)
            list_file.unlink(missing_ok=True)
            if result2.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed to convert frames to video:\n{result.stderr}\n{result2.stderr}"
                )

    def _run_batch_judge(self, prepared: dict[str, Any]) -> None:
        grm_path = project_path(
            self.method_config.get(
                "prm_path",
                self.method_config.get("grm_path", "baselines/PRM-as-a-Judge/PRM/Robo-Dopamine-GRM-8B-Pro"),
            )
        )
        goal_fallback = project_path(
            self.method_config.get(
                "goal_fallback",
                "baselines/PRM-as-a-Judge/eval/examples/blank.png",
            )
        )
        cmd = [
            self._python_bin(),
            "run_judge.py",
            "--benchmark",
            prepared["benchmark"],
            "--videos-root",
            str(prepared["videos_root"]),
            "--tasks-root",
            str(prepared["tasks_root"]),
            "--goals-root",
            str(prepared["goals_root"]),
            "--goal-fallback",
            str(goal_fallback),
            "--output-root",
            str(prepared["output_root"]),
            "--model-filter",
            prepared["model_name"],
            "--grm-path",
            str(grm_path),
            "--frame-interval",
            str(int(self.method_config.get("frame_interval", 5))),
            "--batch-size",
            str(int(self.method_config.get("batch_size", 8))),
            "--tensor-parallel-size",
            str(int(self.method_config.get("tensor_parallel_size", 1))),
            "--eval-mode",
            str(self.method_config.get("eval_mode", "backward")),
        ]
        if self.method_config.get("visualize", False):
            cmd.append("--visualize")
        if self.method_config.get("keep_cache", False):
            cmd.append("--keep-cache")
        if self.method_config.get("skip_existing", True):
            cmd.append("--skip-existing")
        print("[evac_al_baselines] PRM-as-a-Judge command: " + " ".join(cmd))

        # Temporarily swap in the transformers-based inference backend
        # when vllm is not available (e.g. CUDA driver too old for Qwen3VL).
        upstream_inference = self.repo / "eval" / "examples" / "inference.py"
        hf_inference = Path(__file__).resolve().parent.parent / "prm_judge_hf_inference.py"
        backup_path = upstream_inference.with_suffix(".py.vllm_backup")
        use_hf_inference = hf_inference.exists() and not (
            self.method_config.get("use_vllm", False)
        )
        if use_hf_inference:
            shutil.copy2(str(upstream_inference), str(backup_path))
            shutil.copy2(str(hf_inference), str(upstream_inference))
            print("[evac_al_baselines] Using transformers backend for PRM-as-a-Judge")

        try:
            progress_re = re.compile(r"^\[(\d+)/(\d+)\]")
            log_path = self.repo / "eval" / f"prm_judge_run_{prepared.get('output_root', Path('run')).name}.log"
            with open(str(log_path), "w") as log_f, \
                 subprocess.Popen(
                     cmd, cwd=str(self.repo / "eval"),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1,
                 ) as proc:
                with tqdm(desc="[prm_judge]", unit="ep", dynamic_ncols=True) as pbar:
                    for line in proc.stdout:
                        log_f.write(line)
                        line = line.rstrip()
                        m = progress_re.match(line)
                        if m:
                            current, total = int(m.group(1)), int(m.group(2))
                            pbar.total = total
                            pbar.n = current
                            pbar.refresh()
                        elif any(tag in line for tag in ("WARN", "ERROR", "Traceback", "Error:", "FAIL")):
                            tqdm.write(line, file=sys.stderr)
                if proc.wait() != 0:
                    tqdm.write(f"[prm_judge] subprocess failed. Full log: {log_path}", file=sys.stderr)
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
        finally:
            if use_hf_inference and backup_path.exists():
                shutil.move(str(backup_path), str(upstream_inference))

    def _python_bin(self) -> str:
        python_bin = self.method_config.get("python") or self.config.get("prm_judge_python")
        if not python_bin:
            raise RuntimeError(
                "PRM-as-a-Judge must run in its own conda environment. "
                "Set methods.prm_judge.python or top-level prm_judge_python "
                "in baselines/evac_al_baselines/configs/baselines.yaml."
            )
        path = Path(str(python_bin)).expanduser()
        if not path.is_absolute():
            path = project_path(str(path))
        if not path.exists():
            raise FileNotFoundError(
                f"Configured PRM python does not exist: {path}. "
                "Create the prm_judge conda environment and update baselines.yaml."
            )
        return str(path)

    def _load_latest_records(self, output_root: Path) -> dict[str, dict[str, Any]]:
        runs = sorted(output_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise FileNotFoundError(f"No PRM-as-a-Judge run_* directory under {output_root}")
        results_path = runs[-1] / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing PRM-as-a-Judge results: {results_path}")
        records = json.loads(results_path.read_text(encoding="utf-8"))
        return {str(row.get("sample_id")): row for row in records if isinstance(row, dict)}

    def _rows_from_records(self, examples: list[dict], records: dict[str, dict[str, Any]]):
        for example in examples:
            episode_id = str(example.get("episode_id") or "episode")
            sample_id = safe_slug(episode_id)
            record = records.get(sample_id)
            if not record:
                yield self._attach_input_paths({
                    "episode_id": episode_id,
                    "method": self.method,
                    "score_ready": False,
                    "error": f"missing PRM record for sample_id={sample_id}",
                    "extra": {},
                }, example)
                continue
            if record.get("status") != "ok":
                yield self._attach_input_paths({
                    "episode_id": episode_id,
                    "method": self.method,
                    "score_ready": False,
                    "error": record.get("error", "PRM record status is not ok"),
                    "extra": {"record": record},
                }, example)
                continue
            frame_scores = self._frame_scores_from_record(record)
            opd = compute_opd_metrics(frame_scores)
            episode_score = float(opd.get("MC", frame_scores[-1] if frame_scores else 0.0))
            yield self._attach_input_paths({
                "episode_id": episode_id,
                "method": self.method,
                "frame_scores": frame_scores,
                "episode_score": episode_score,
                "acquisition_score": combined_score(frame_scores, episode_score, {"opd": opd}),
                "extra": {
                    "opd": {key: opd.get(key, 0.0) for key in OPD_KEYS},
                    "output_dir": record.get("output_dir"),
                    "pred_path": record.get("pred_path"),
                },
            }, example)

    def _frame_scores_from_record(self, record: dict[str, Any]) -> list[float]:
        pred_path = record.get("pred_path")
        if pred_path and Path(str(pred_path)).exists():
            try:
                data = json.loads(Path(str(pred_path)).read_text(encoding="utf-8"))
                vals = coerce_float_list([row.get("progress") for row in data if isinstance(row, dict)])
                if vals:
                    return vals
            except Exception:
                pass
        summary = record.get("summary", {}) if isinstance(record.get("summary"), dict) else {}
        return coerce_float_list(summary.get("final_progress", 0.0))
