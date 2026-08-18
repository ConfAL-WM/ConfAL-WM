# EVAC Active Learning Baselines

This folder is ConfAL-WM's local comparison layer for the paper's active-learning scoring baselines. It gives each method a shared JSONL input/output contract so the selection step can be compared without modifying third-party repositories.

The cloned third-party repositories under `baselines/robometer`, `baselines/Large-Reward-Models`, and `baselines/PRM-as-a-Judge` are dependencies only. Do not edit them for EVAC changes and do not commit them into the main ConfAL-WM repository. Outputs should go under `baselines/results`, `baselines/cache`, or `baselines/logs`.

## Candidate Pool Format

Prepare a JSONL manifest where each line is one candidate episode:

```json
{"episode_id":"xxx","video_path":"/path/to/video.mp4","task":"open the drawer","success_label":0,"split":"train"}
```

The shared scoring output is also JSONL:

```json
{"episode_id":"xxx","method":"robometer_prog","frame_scores":[0.1,0.2,0.4,0.7],"episode_score":0.7,"acquisition_score":0.35,"extra":{}}
```

## Score A Baseline

Random is fully runnable and is useful for smoke testing:

```bash
python baselines/evac_al_baselines/run_score.py \
  --manifest data/candidate_pool.jsonl \
  --method random \
  --output baselines/results/random_scores.jsonl
```

Other methods are runnable scoring wrappers around local repos or a
user-started server. They score `video_path` plus `task` and write
`acquisition_score`; the main ConfAL-WM `score_pool.py` converts that JSONL back
to `scored_pool.json`.

```bash
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method robometer_prog --output baselines/results/robometer_prog_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method robometer_pref --output baselines/results/robometer_pref_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method gvl --output baselines/results/gvl_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method lrms --output baselines/results/lrms_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method lrm --output baselines/results/lrm_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method lrm_progress --output baselines/results/lrm_progress_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method lrm_completion --output baselines/results/lrm_completion_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method lrm_contrastive --output baselines/results/lrm_contrastive_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method roboreward --output baselines/results/roboreward_scores.jsonl
python baselines/evac_al_baselines/run_score.py --manifest data/candidate_pool.jsonl --method prm_judge --output baselines/results/prm_judge_scores.jsonl
```

Robometer progress/preference assumes `baselines/robometer`'s
`eval_server` is already running. Start it through
`servers/robometer_eval_server_dtype_patch.py` from the Robometer uv
environment; this keeps the Robometer checkout unchanged while applying the
Qwen visual-input dtype compatibility patch at runtime. The default URL is
`http://localhost:8000`. `robometer_pref` uses the same single-trajectory model
output as `robometer_prog` but applies a near-miss acquisition rule because the
ConfAL-WM AL pool is not paired by default.

GVL defaults to the ConfAL-WM local Qwen3-VL adapter (`type: gvl_qwen`). Start a
Qwen3-VL OpenAI-compatible server, usually with vLLM, and point
`gvl_qwen_base_url` / `methods.gvl.base_url` at it. This path does not require
the Robometer GVL server or provider API keys. The original GVL paper used
Gemini-1.5-Pro, so report this local baseline as GVL-Qwen3-VL if precision
matters.

Legacy API-backed GVL is still available through `type: robometer_gvl` or the
`methods.gvl_api_server` entry. That path assumes `baselines/robometer`'s
`baseline_eval_server.py` is running with `reward_model=gvl`, default URL
`http://localhost:8001`, and `GEMINI_API_KEY` or `OPENAI_API_KEY`.

LRMs assumes the reward server has already been manually started. The default
URL is `http://localhost:5002`; update `lrm_server_url`, `endpoint`, and
`batch_endpoint` in `configs/baselines.yaml` to match the actual LRMs server.
The `lrm` and `lrms` methods are progress-mode aliases for the main AL
comparison.

RoboReward is not deployed separately here. It is routed through LRMs with `mode: roboreward` and `/compute_roboreward`.

PRM-as-a-Judge uses `eval/run_judge.py` in batch. The adapter builds the
expected `eval/videos/{benchmark}/{task}/{model}` layout under
`baselines/cache/prm_judge`. ConfAL-WM RoboTwin currently has one
`head_color.mp4`, so `allow_single_view: true` reuses that video as
high/left/right unless explicit `prm_videos` or `input_videos` are provided.
Set `methods.prm_judge.python` to the separate PRM conda environment python and
`methods.prm_judge.prm_path` to the downloaded GRM checkpoint.

## Environment Boundary

Run this wrapper from ConfAL-WM's normal `enerverse` environment. Robometer,
GVL-Qwen, LRMs, and RoboReward are HTTP server integrations: start their
servers in separate environments, then point `robometer_server_url`,
`gvl_qwen_base_url`, or `lrm_server_url` at them. The wrapper does not import
those model packages.

PRM-as-a-Judge is the only non-server integration here. It is launched as a
subprocess using the configured `methods.prm_judge.python`, so it also stays
outside `enerverse`.

## Select Episodes

Select the highest acquisition scores by count, ratio, or percentage:

```bash
python baselines/evac_al_baselines/run_select.py \
  --scores baselines/results/random_scores.jsonl \
  --budget 100 \
  --strategy top \
  --output baselines/results/random_selected.jsonl
```

Examples: `--budget 0.1` selects 10%, `--budget 10%` selects 10%, and `--strategy random` shuffles with a fixed seed.

## What Each Baseline Replaces

Random, GVL, RoboReward, Robometer-Prog, Robometer-Pref, PRM-as-a-Judge, LRMs, and C3 Probe/Ours are used here as alternatives for the active learning select-dataset stage: they assign an `acquisition_score`, then `run_select.py` chooses episodes under a budget.

Weighted training is a separate stage. The score files can be reused for
trajectory-level selection/oversampling, but this folder does not start EVAC
fine-tuning and does not produce C3 `conf_map.npy` files.

The `c3` method expects C3/Ours scores to already be present in the manifest fields, for example `c3_score`, `c3_frame_scores`, `success_probability`, or `confidence`.

For ConfAL-WM retraining, the main repository now supports two runnable
confidence-guided mechanisms outside this baseline folder:

- `retraining.weighting_mode=oversampling`: trajectory-level risk repeats
  selected samples through symlinks.
- `retraining.weighting_mode=patch_weight`: dense C3 `conf_map.npy` is passed
  through the Dataset batch as `confidence_map` and used by
  `ddpm3d.p_losses()` as stop-gradient patch/time loss weights.

This baseline folder still only covers selector scoring and selection. It does
not launch EWMBench, EVAC fine-tuning, or patch-weighted training. Use C3
scoring for any `patch_weight` experiment.

## Files

- `adapters/base.py`: shared `BaseScorer` interface and runnable random scorer.
- `adapters/robometer_adapter.py`: Robometer progress/preference wrapper.
- `adapters/lrm_adapter.py`: HTTP wrapper for LRMs modes.
- `adapters/roboreward_lrm_adapter.py`: RoboReward through LRMs.
- `adapters/gvl_qwen_adapter.py`: local Qwen3-VL GVL-style scorer.
- `adapters/gvl_robometer_adapter.py`: legacy GVL through Robometer.
- `adapters/prm_judge_adapter.py`: PRM-as-a-Judge wrapper and OPD parser hook.
- `metrics/acquisition.py`: AL acquisition aggregators.
- `metrics/opd.py`: MC, MP, PPL, CRA, STR readers/approximations.
- `metrics/voc.py`: Spearman, Kendall, and VOC helpers with scipy fallback.
