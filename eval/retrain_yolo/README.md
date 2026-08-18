# RoboTwin Gripper YOLO Fine-Tuning

EWMBench `trajectory_consistency` needs `traj/traj.npy`, which depends on a
gripper detector. The released EWMBench YOLO checkpoint is AgiBot-tuned, so it
usually fails on RoboTwin. This folder provides a lightweight RoboTwin
pseudo-label pipeline.

## 1. Build Pseudo Labels

```bash
python eval/retrain_yolo/prepare_robotwin_yolo_dataset.py \
  --config configs/agibotworld/al_robotwin.yaml \
  --output_dir eval/retrain_yolo/robotwin_gripper_yolo \
  --frame_stride 10 \
  --workers 16 \
  --overwrite
```

`--max_frames` caps how many image frames are written into the YOLO dataset
after projection filtering. Default: no cap, so all frames surviving
`--frame_stride` and projection filtering are written. `--frame_stride` samples
every Nth frame from each episode before projection. Default: `10`.
`--workers` controls episode-level parallelism. Default: `1`. `--vis_examples`
selects the first N episodes for bbox visual checks. Default: `5`.
`--vis_frames_per_episode` uses this many labeled frames per selected episode.
Default: `3`. The selected samples are written as one contact sheet at
`output_dir/visualization.jpg`.

The default stride is a practical first pass: RoboTwin videos have strong
frame-to-frame redundancy, so stride 10 reduces near-duplicate pseudo-labels
while preserving episode/task diversity. For a denser detector run, reduce
`--frame_stride` after inspecting pseudo-label quality.

The script is resume-safe when `--overwrite` is omitted: existing image/label
pairs are kept and counted, while missing pairs are generated. Use
`--overwrite` only when you want to rebuild the dataset directory from scratch.
For full-data runs, file existence checks, image links/copies, and label writes
are performed inside the episode workers. If `--max_frames` is set, the script
keeps exact frame capping by doing the final writes in the main process.

The script reads converted RoboTwin episodes, projects `actions_evac.npy`
left/right end-effector positions through `camera.npz`
`intrinsic_cv`/`extrinsic_cv`, and writes YOLO labels for:

- `0`: left gripper
- `1`: right gripper

Outputs stay under `eval/retrain_yolo/robotwin_gripper_yolo/`:

- `images/{train,val}/`
- `labels/{train,val}/`
- `data.yaml`
- `pseudo_label_manifest.json`
- `projection_report.json`

## 2. Train YOLO

Run this in the YOLO/EWMBench training environment with Ultralytics installed:

```bash
python eval/retrain_yolo/train_yolo_robotwin.py \
  --dataset_dir eval/retrain_yolo/robotwin_gripper_yolo \
  --base_model eval/retrain_yolo/base_models/yolo26s.pt \
  --epochs 5 \
  --imgsz 640 \
  --batch 256 \
  --workers 16 \
  --cache disk \
  --gpus 0,1
```

Per‑step training metrics (box/cls/dfl loss + learning rate) are logged to
`per_step_metrics.csv` inside the run directory for fine‑grained curve plotting.

Weights are saved under `eval/retrain_yolo/runs/<name>/weights/`.
If `--base_model` points to an existing local file, that file is used directly;
if the path is missing and its filename is a known Ultralytics model such as
`yolo26s.pt`, the script downloads it to that path. For YOLO26 weights, download
tries Hugging Face first using `HF_ENDPOINT` if set, otherwise
`https://hf-mirror.com`, then official Hugging Face, then GitHub assets. You can
also pass a known model name or a direct URL.

AMP is disabled by default because Ultralytics may run an online AMP check that
downloads a separate model such as `yolo26n.pt`. Add `--amp` only in an
environment that can reach the required assets.

For A800‑class GPUs (80 GB), `--batch 256` keeps per‑GPU memory around
55‑60 GB, leaving headroom for AMP or larger image sizes. Avoid `--cache ram`
for this dataset unless the job has enough host memory for the train/val images
multiplied by the DDP worker processes; otherwise the OS may kill a rank with
SIGKILL.

## 3. Evaluate

```bash
python eval/retrain_yolo/eval_yolo_robotwin.py \
  --weights eval/retrain_yolo/runs/robotwin_gripper_yolo26s-3/weights/best.pt \
  --dataset_dir eval/retrain_yolo/robotwin_gripper_yolo \
  --split val \
  --batch 256
```

Outputs `eval_summary.json` (mAP50, mAP50‑95, precision, recall per‑class) and
PR‑curve PNG under `eval/retrain_yolo/eval_runs/<name>/`.

## 4. Plot Training Curves

```bash
python eval/retrain_yolo/plot_yolo_training.py \
  --csv eval/retrain_yolo/runs/<name>/per_step_metrics.csv
```

Produces a single multi‑panel figure with training loss (raw + smoothed),
validation loss, and learning rate over global steps, saved as
`training_curves.png` alongside the CSV.

## Notes

- `extrinsic_cv` is treated as world-to-camera, matching the ConfAL-WM RoboTwin
  scoring/inference path.
- The default boxes are fixed-size pseudo boxes around projected 3D EE centers.
  Inspect a small subset before trusting the detector.
- The intended checkpoint is RoboTwin/`aloha-agilex_rand_500`-specific and may
  not generalize to the full RoboTwin collection without more data.
