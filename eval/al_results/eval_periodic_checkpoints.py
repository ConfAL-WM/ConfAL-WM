#!/usr/bin/env python3
"""Post-training evaluator for periodic checkpoints.

Training with ``--save_every_with_ewmbench`` SAVES periodic checkpoints directly
into ``logs/checkpoints/epoch=*-step=*.ckpt`` and, once torchrun exits,
``train_evac_with_al.py`` invokes this script automatically (EWMBench is no
longer run during training — that deadlocked when eval subprocesses inherited
torchrun's ``WORLD_SIZE``). This script may also be run manually. Evaluation
runs in parallel across checkpoints: one isolated subprocess per checkpoint,
each pinned to its own GPU, then writes the
``checkpoint_metrics_with_ewmbench.{json,csv,png}`` summary table under
``logs/periodic_eval/``.

Example::

    python eval/al_results/eval_periodic_checkpoints.py \
        --run-dir al_runs/robotwin_al/retrain/c3_persistent_risk_frame_patch \
        --config configs/agibotworld/al_robotwin.yaml \
        --gpus 0,1,2,3

Reuses ``run_async_ewmbench_job.py`` (inference + evaluate_al_round per ckpt) and
``periodic_eval_summary.write_summary`` for the table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import load_json  # noqa: E402
from eval.al_results import periodic_eval_summary  # noqa: E402


_STEP_RE = re.compile(r"step=(\d+)")


def _parse_steps(text: str | None) -> set[int] | None:
    if not text:
        return None
    out = set()
    for part in text.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out or None


def _discover_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] sorted by step; one ckpt per step."""
    by_step: dict[int, Path] = {}
    for path in sorted(ckpt_dir.glob("epoch=*-step=*.ckpt")):
        m = _STEP_RE.search(path.name)
        if not m:
            continue
        step = int(m.group(1))
        # If both a plain and a (best_loss) variant ever coexist, keep the
        # (best_loss) one (it is the canonical best for that step).
        prev = by_step.get(step)
        if prev is None or ("(best_loss)" in path.name and "(best_loss)" not in prev.name):
            by_step[step] = path
    return [(step, p) for step, p in sorted(by_step.items())]


def _infer_config(run_dir: Path) -> str | None:
    # run_dir = <run_root>/retrain/<name>; run_root parent dir names the track.
    parts = run_dir.parts
    if "robotwin_al" in parts:
        return "configs/agibotworld/al_robotwin.yaml"
    if "agibot_al" in parts:
        return "configs/agibotworld/al_agibot.yaml"
    return None


def _resolve_run_meta(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = run_dir / "retrain_summary.json"
    retrain_summary = load_json(str(summary_path)) if summary_path.exists() else {}

    config = args.config or retrain_summary.get("pipeline_config") or _infer_config(run_dir)
    if config is None:
        raise SystemExit(
            f"Could not infer pipeline config for {run_dir}. Pass --config <pipeline.yaml>."
        )
    if not Path(config).is_absolute() and not Path(config).exists():
        config = str(_REPO / config)

    run_root = run_dir.parent.parent
    val_manifest = args.val_manifest or str(run_root / "manifests" / "al_val.json")

    score_method = args.score_method or retrain_summary.get("score_method")
    select_method = args.select_method or retrain_summary.get("select_method")
    weighting = args.weighting or retrain_summary.get("weighting")

    include_ewmbench = not args.no_ewmbench
    if not args.no_ewmbench and retrain_summary:
        # Honour what the run was actually trained with.
        include_ewmbench = bool(retrain_summary.get("save_every_with_ewmbench", True))

    return {
        "config": config,
        "val_manifest": val_manifest,
        "score_method": score_method,
        "select_method": select_method,
        "weighting": weighting,
        "include_ewmbench": include_ewmbench,
    }


def _build_job(
    *,
    step: int,
    ckpt_path: Path,
    run_dir: Path,
    report_dir: Path,
    meta: dict[str, Any],
    max_eval_episodes: int,
    metrics: str,
    use_ewmbench_evaluate_py: bool,
    overwrite_infer: bool,
    skip_infer: bool,
) -> dict[str, Any]:
    job_dir = report_dir / "jobs" / f"step_{step:08d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return {
        "step": int(step),
        "checkpoint": str(ckpt_path),
        "project_root": str(_REPO),
        "python_executable": sys.executable,
        "pipeline_config": meta["config"],
        "val_manifest": meta["val_manifest"],
        "output_dir": str(run_dir),
        "job_dir": str(job_dir),
        "score_method": meta["score_method"],
        "select_method": meta["select_method"],
        "weighting": meta["weighting"],
        "max_eval_episodes": int(max_eval_episodes),
        "metrics": metrics,
        "include_ewmbench": bool(meta["include_ewmbench"]),
        "use_ewmbench_evaluate_py": bool(use_ewmbench_evaluate_py),
        "overwrite_infer": bool(overwrite_infer),
        "skip_infer": bool(skip_infer),
        "result_path": str(job_dir / "job_result.json"),
    }


def _launch(job: dict[str, Any], gpu: str, helper: Path) -> tuple[Any, Any]:
    job = dict(job)
    job["gpu"] = gpu
    job_dir = Path(job["job_dir"])
    job_config = job_dir / "job_config.json"
    with job_config.open("w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    log_path = job_dir / "job.log"
    log_f = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    # EWMBench dist_init must NOT inherit any torchrun WORLD_SIZE/RANK. The worker
    # chain (run_async_ewmbench_job -> evaluate_al_round -> compute_ewmbench) now
    # forces these explicitly, but scrub them here too for belt-and-suspenders.
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    cmd = [sys.executable, str(helper), "--job", str(job_config)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    return proc, log_f


def _read_result(path: str, job: dict[str, Any], gpu: str, returncode: int) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception as exc:
        result = {"status": "failed", "error": repr(exc), "result_path": path}
    result.setdefault("step", job["step"])
    result.setdefault("checkpoint", job["checkpoint"])
    result["gpu"] = gpu
    result["returncode"] = returncode
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate periodic checkpoints after training (parallel across GPUs).")
    parser.add_argument("--run-dir", required=True, help="Retrain run dir, e.g. al_runs/robotwin_al/retrain/c3_persistent_risk_frame_patch")
    parser.add_argument("--config", default=None, help="Pipeline config YAML (al_robotwin.yaml). Inferred from run-dir if omitted.")
    parser.add_argument("--val-manifest", default=None, help="Val manifest (default: {run_root}/manifests/al_val.json)")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs to parallelize across (default: 0)")
    parser.add_argument("--max-episodes", type=int, default=50, help="Val episodes per checkpoint (default 50)")
    parser.add_argument("--steps", default=None, help="Optional comma-separated step list to evaluate (default: all)")
    parser.add_argument("--metrics", default=None, help="Metric list (default: pixel_mae,latent_loss,risk_reduction[,ewmbench])")
    parser.add_argument("--score-method", dest="score_method", default=None)
    parser.add_argument("--select-method", dest="select_method", default=None)
    parser.add_argument("--weighting", default=None)
    parser.add_argument("--no-ewmbench", action="store_true", help="Skip the EWMBench metric (faster): only compute pixel_mae/latent_loss/risk_reduction. Checkpoints are read from the same logs/checkpoints/ dir either way.")
    parser.add_argument("--no-use-ewmbench-evaluate-py", dest="use_ewmbench_evaluate_py", action="store_false", help="Use the ConfAL-WM direct basic-metric runner instead of official EWMBench evaluate.py")
    parser.set_defaults(use_ewmbench_evaluate_py=True)
    parser.add_argument("--overwrite-infer", dest="overwrite_infer", action="store_true", help="Regenerate prediction frames even if they already exist")
    parser.add_argument("--skip-infer", dest="skip_infer", action="store_true", help="Skip run_val_inference and reuse existing val_infer_tmp predictions (use only when preds are already complete for every checkpoint)")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between completion polls (default 5)")
    args = parser.parse_args()

    # Line-buffer stdout so progress is visible when run via nohup/background/redirect.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    meta = _resolve_run_meta(run_dir, args)
    ckpt_dir = run_dir / "logs" / "checkpoints"
    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint dir not found: {ckpt_dir} (did training run with --save_every_with_ewmbench?)")

    discovered = _discover_checkpoints(ckpt_dir)
    step_filter = _parse_steps(args.steps)
    if step_filter is not None:
        discovered = [(s, p) for s, p in discovered if s in step_filter]
    if not discovered:
        raise SystemExit(f"No checkpoints to evaluate under {ckpt_dir}")

    metrics = args.metrics
    if metrics is None:
        metrics = "pixel_mae,latent_loss,risk_reduction"
        if meta["include_ewmbench"]:
            metrics = f"{metrics},ewmbench"

    print(f"[every_eval] run_dir={run_dir}")
    print(f"[every_eval] config={meta['config']} val_manifest={meta['val_manifest']}")
    print(f"[every_eval] score={meta['score_method']} select={meta['select_method']} weighting={meta['weighting']} include_ewmbench={meta['include_ewmbench']}")
    print(f"[every_eval] {len(discovered)} checkpoints to evaluate: {[s for s, _ in discovered]}")

    report_dir = run_dir / "logs" / "periodic_eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    helper = _REPO / "eval" / "al_results" / "run_async_ewmbench_job.py"

    jobs = [
        _build_job(
            step=step,
            ckpt_path=path,
            run_dir=run_dir,
            report_dir=report_dir,
            meta=meta,
            max_eval_episodes=args.max_episodes,
            metrics=metrics,
            use_ewmbench_evaluate_py=args.use_ewmbench_evaluate_py,
            overwrite_infer=args.overwrite_infer,
            skip_infer=args.skip_infer,
        )
        for step, path in discovered
    ]

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    pending = list(jobs)
    running: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    step_to_ckpt_path = {int(j["step"]): j["checkpoint"] for j in jobs}

    def _free_gpus() -> list[str]:
        return [g for g in gpus if g not in running]

    while pending or running:
        # Launch on free GPUs.
        while pending and _free_gpus():
            gpu = _free_gpus()[0]
            job = pending.pop(0)
            proc, log_f = _launch(job, gpu, helper)
            running[gpu] = {"proc": proc, "job": job, "log": log_f}
            print(f"[every_eval] launched step={job['step']} on gpu={gpu}")

        # Poll for completion.
        finished = [(gpu, state) for gpu, state in list(running.items()) if state["proc"].poll() is not None]
        for gpu, state in finished:
            running.pop(gpu, None)
            try:
                state["log"].close()
            except Exception:
                pass
            result = _read_result(state["job"]["result_path"], state["job"], gpu, state["proc"].returncode)
            results.append(result)
            if state["proc"].returncode == 0 and result.get("status") == "ok":
                print(f"[every_eval] finished step={state['job']['step']} (gpu={gpu}) OK")
            else:
                print(
                    f"[every_eval] FAILED step={state['job']['step']} (gpu={gpu}) "
                    f"returncode={state['proc'].returncode} status={result.get('status')}"
                )

        if pending or running:
            time.sleep(args.poll_interval)

    json_path, csv_path, png_path = periodic_eval_summary.write_summary(
        report_dir=report_dir,
        results=results,
        step_to_ckpt_path=step_to_ckpt_path,
        include_ewmbench=meta["include_ewmbench"],
        metrics=metrics,
    )
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"[every_eval] done: {n_ok}/{len(results)} OK")
    print(f"[every_eval] summary: {json_path}\n             {csv_path}\n             {png_path}")
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
