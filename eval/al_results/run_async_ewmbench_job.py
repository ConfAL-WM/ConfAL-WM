#!/usr/bin/env python3
"""Run one async periodic checkpoint eval job.

This helper is launched by trainer.callbacks.AsyncEveryEvalCheckpoint.
It first generates temporary predictions for a small validation subset, then
runs evaluate_al_round.py and writes a compact job_result.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import flatten_manifest_items, load_json, save_json


def _write_subset_manifest(source: str, out_path: Path, max_items: int) -> Path:
    payload = load_json(source)
    items = flatten_manifest_items(payload)[: max(1, int(max_items))]
    out = {
        "source_manifest": source,
        "split": "async_ewmbench_subset",
        "max_items": int(max_items),
        "items": items,
        "stats": {
            "source_items": len(flatten_manifest_items(payload)),
            "subset_items": len(items),
        },
    }
    save_json(out, out_path)
    return out_path


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[async job] running: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[async job] returncode={proc.returncode}\n")
        return int(proc.returncode)


def run_job(job_path: str) -> dict[str, Any]:
    job = load_json(job_path)
    project_root = Path(job.get("project_root") or _REPO)
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(job.get("result_path") or job_dir / "job_result.json")
    log_path = job_dir / "job.log"
    py = str(job.get("python_executable") or sys.executable)
    gpu = str(job.get("gpu", "0"))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")

    subset_manifest = _write_subset_manifest(
        str(job["val_manifest"]),
        job_dir / "val_first_subset.json",
        int(job.get("max_eval_episodes", 50)),
    )
    pred_dir = job_dir / "val_infer_tmp"
    metrics_path = job_dir / "metrics.json"

    infer_cmd = [
        py,
        "eval/al_results/run_val_inference.py",
        "--checkpoint",
        str(job["checkpoint"]),
        "--config",
        str(job["pipeline_config"]),
        "--manifest",
        str(subset_manifest),
        "--output",
        str(pred_dir),
        "--num-shards",
        "1",
        "--workers-per-gpu",
        "1",
        "--gpus",
        "0",
    ]
    if bool(job.get("overwrite_infer", False)):
        infer_cmd.append("--overwrite")
    if bool(job.get("skip_infer", False)):
        # Reuse existing predictions (assume complete) and skip the EVAC inference
        # pass — avoids the slow model load when val_infer_tmp already holds all
        # episodes' pred frames.
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n[async job] skip_infer=True: reusing existing predictions, skipping run_val_inference\n")
        infer_rc = 0
    else:
        infer_rc = _run(infer_cmd, cwd=project_root, env=env, log_path=log_path)

    eval_rc = None
    metrics: dict[str, Any] = {}
    if infer_rc == 0:
        eval_cmd = [
            py,
            "eval/al_results/evaluate_al_round.py",
            "--checkpoint",
            str(job["checkpoint"]),
            "--score_method",
            str(job["score_method"]),
            "--select_method",
            str(job["select_method"]),
            "--weighting",
            str(job["weighting"]),
            "--val-manifest",
            str(subset_manifest),
            "--pred-dir",
            str(pred_dir),
            "--output",
            str(metrics_path),
            "--metrics",
            str(job.get("metrics", "pixel_mae,latent_loss,risk_reduction,ewmbench")),
            "--config",
            str(job["pipeline_config"]),
            "--ewmbench-gpus",
            "0",
        ]
        if bool(job.get("use_ewmbench_evaluate_py", True)):
            eval_cmd.append("--use_ewmbench_evaluate_py")
        eval_rc = _run(eval_cmd, cwd=project_root, env=env, log_path=log_path)
        if eval_rc == 0 and metrics_path.exists():
            metrics = load_json(str(metrics_path))

    status = "ok" if infer_rc == 0 and eval_rc == 0 and bool(metrics) else "failed"
    result = {
        "status": status,
        "step": int(job.get("step", -1)),
        "gpu": gpu,
        "checkpoint": str(job["checkpoint"]),
        "subset_manifest": str(subset_manifest),
        "pred_dir": str(pred_dir),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "infer_returncode": infer_rc,
        "eval_returncode": eval_rc,
        "log_path": str(log_path),
    }
    save_json(result, result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one async EWMBench checkpoint-selection job")
    parser.add_argument("--job", required=True, help="Path to job_config.json")
    args = parser.parse_args()
    result = run_job(args.job)
    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
