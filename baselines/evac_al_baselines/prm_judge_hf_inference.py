"""Transformers-based GRMInference (drop-in replacement for the vllm version).

Used by PRM-as-a-Judge when the system CUDA driver is too old for vllm's
required PyTorch build.  The adapter temporarily swaps this file in place of
``examples/inference.py`` before launching ``run_judge.py``.
"""

from __future__ import annotations

import os
from typing import Dict, List

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Global Environment Settings
os.environ.setdefault('LOCAL_RANK', '0')
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ---- prompt (identical to upstream) ----

SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = "just beginning"; REFERENCE END = "fully completed."
   - Visually estimate how far BEFORE and AFTER are along this START->END continuum.
2) Direction:
   - AFTER better than BEFORE -> positive score.
   - AFTER worse than BEFORE -> negative score.
   - Essentially the same -> 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged -> 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""


# ---- file / video utilities (identical to upstream) ----

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def list_pngs_sorted(dir_path):
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"])


def get_frame_count(path):
    if path.is_dir():
        files = list_pngs_sorted(path)
        if not files:
            raise RuntimeError(f"No PNG frames found in directory: {path}")
        return "dir", len(files)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Invalid frame count from video: {path}")
    return "video", n


def make_sample_indices_by_interval(num_frames, interval):
    if num_frames < 1:
        return []
    indices = list(range(0, num_frames, interval))
    last_idx = num_frames - 1
    if not indices or indices[-1] != last_idx:
        indices.append(last_idx)
    return indices


def save_frames(src_path, out_dir, indices, src_type):
    ensure_dir(out_dir)
    if src_type == "video":
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src_path}")
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame {idx} from {src_path}")
                continue
            out_path = out_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cap.release()
    elif src_type == "dir":
        files = list_pngs_sorted(src_path)
        n = len(files)
        for idx in indices:
            if not (0 <= idx < n):
                continue
            src = files[idx]
            dst = out_dir / f"frame_{idx:06d}.png"
            import shutil
            shutil.copyfile(src, dst)


def build_samples_json(run_root, task, indices, ref_end_path, mode="incremental"):
    timestamp_name = run_root.name
    cache_root = run_root / ".cache"
    items = []
    if len(indices) < 2:
        return items
    for k in range(len(indices) - 1):
        af = indices[k + 1]
        bf_images = []
        bf_id_str = ""
        if mode == "incremental":
            bf = indices[k]
            bf_id_str = f"bf_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "forward":
            bf = indices[0]
            bf_id_str = f"start_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "backward":
            bf_id_str = "goal"
            bf_images = [ref_end_path, ref_end_path, ref_end_path]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        af_images = [
            str(cache_root / "cam_high" / f"frame_{af:06d}.png"),
            str(cache_root / "cam_left_wrist" / f"frame_{af:06d}.png"),
            str(cache_root / "cam_right_wrist" / f"frame_{af:06d}.png"),
        ]
        items.append({
            "id": f"step-{timestamp_name}-{k:04d}-{bf_id_str}-af_{af:06d}",
            "task": task,
            "image": [
                str(cache_root / "cam_high" / f"frame_{0:06d}.png"),
                ref_end_path,
                bf_images[0], bf_images[1], bf_images[2],
                af_images[0], af_images[1], af_images[2],
            ],
        })
    return items


# ---- visualization (identical to upstream) ----

def plot_video_reward(episode_root):
    import json as _json
    import re as _re

    pred_path = episode_root / "pred_vllm.json"
    if not pred_path.exists():
        print(f"[WARN] No prediction file found at {pred_path}, skipping visualization.")
        return
    with open(pred_path, "r", encoding="utf-8") as f:
        items = _json.load(f)
    if not items:
        return

    def get_fid(p):
        m = _re.search(r"frame_(\d+)\.png", os.path.basename(p))
        return int(m.group(1)) if m else 0

    out_path = episode_root / "reward_vis.mp4"
    W_panel, H_panel = 384, 288
    H_plot = 260
    W_total = 3 * W_panel
    H_total = H_panel + H_plot
    xs_step, ys_hop, c_hop = [], [], []
    xs_frame, ys_prog, c_prog = [], [], []
    frames_buffer = []

    def draw_plots():
        dpi = 100
        fig = plt.figure(figsize=(W_total / dpi, H_plot / dpi), dpi=dpi)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        if xs_step:
            if len(xs_step) > 1:
                ax1.plot(xs_step, ys_hop, "-", color="#999999", alpha=0.6)
            ax1.scatter(xs_step, ys_hop, c=c_hop, s=36, edgecolors="k", linewidths=0.5)
        ax1.set_yscale('symlog', linthresh=0.05)
        ax1.set_ylim(-1.5, 1.5)
        ticks = [-1.0, -0.1, 0, 0.1, 1.0]
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(["-100%", "-10%", "0", "10%", "100%"])
        max_step = len(items) + 1
        ax1.set_xlim(left=0, right=max(5, max_step))
        ax1.set_title("Instant Hop/Reward (SymLog)")
        ax1.grid(True, which="major", linestyle="-", alpha=0.4)
        if xs_frame:
            if len(xs_frame) > 1:
                ax2.plot(xs_frame, ys_prog, "-", color="#999999", alpha=0.6)
            ax2.scatter(xs_frame, ys_prog, c=c_prog, s=36, edgecolors="k", linewidths=0.5)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title("Accumulated Progress")
        ax2.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout(pad=1.0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.resize(img, (W_total, H_plot), interpolation=cv2.INTER_AREA)

    def read_images_rgb(paths):
        imgs = []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                return None
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs.append(im)
        return imgs

    try:
        first = items[0]
        imgs_top = read_images_rgb([first["image"][i] for i in [3, 2, 4]])
        if imgs_top:
            top_row = np.hstack([cv2.resize(im, (W_panel, H_panel)) for im in imgs_top])
            xs_step.append(0)
            ys_hop.append(0.0)
            c_hop.append("#808080")
            xs_frame.append(0)
            ys_prog.append(0.0)
            c_prog.append("#00A000")
            bottom_plot = draw_plots()
            frame_comp = np.vstack([top_row, bottom_plot])
            if frame_comp.shape[:2] != (H_total, W_total):
                frame_comp = cv2.resize(frame_comp, (W_total, H_total))
            cv2.putText(frame_comp, "Start", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frames_buffer.append(frame_comp)
        prev_prog = 0.0
        cum_frame = 0
        for i, item in enumerate(items):
            imgs_top = read_images_rgb([item["image"][idx] for idx in [6, 5, 7]])
            if not imgs_top:
                continue
            top_row = np.hstack([cv2.resize(im, (W_panel, H_panel)) for im in imgs_top])
            hop = float(item.get("hop", 0))
            prog = float(item.get("progress", 0))
            step_id = i + 1
            cum_frame += 10
            try:
                cum_frame = get_fid(item["image"][5])
            except Exception:
                pass
            xs_step.append(step_id)
            ys_hop.append(hop)
            c_hop.append("#00A000" if hop > 0 else ("#CC0000" if hop < 0 else "#808080"))
            xs_frame.append(cum_frame)
            ys_prog.append(prog)
            c_prog.append("#00A000" if prog >= prev_prog else "#CC0000")
            prev_prog = prog
            bottom_plot = draw_plots()
            frame_comp = np.vstack([top_row, bottom_plot])
            if frame_comp.shape[:2] != (H_total, W_total):
                frame_comp = cv2.resize(frame_comp, (W_total, H_total))
            info_txt = f"Step={step_id} Hop={hop:.2f} Prog={prog:.2f}"
            cv2.putText(frame_comp, info_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frames_buffer.append(frame_comp)
        if frames_buffer:
            print(f"Generating video with {len(frames_buffer)} frames...")
            clip = ImageSequenceClip(frames_buffer, fps=4.0)
            clip.write_videofile(str(out_path), logger=None)
            print(f"Video saved to: {out_path}")
        else:
            print("[WARN] No frames generated.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] {e}")


# ---- main inference class (transformers backend) ----

class GRMInference:
    def __init__(
        self,
        model_path: str,
        max_image_num=8,
        min_pixels=12544,
        max_pixels=76800,
        tensor_parallel_size: int = 1,
    ):
        print(f"Loading model from {model_path} ...")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if hasattr(self.processor, 'image_processor'):
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = min_pixels

        self._gen_kwargs = dict(
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            do_sample=True,
        )

    def inference_batch(self, batch_data: List[Dict]) -> List[Dict]:
        results = []
        for item in batch_data:
            images = [Image.open(p).convert("RGB") for p in item["image"]]
            prompt_parts = SYSTEM_PROMPT.format(task=item["task"]).split("<image>")

            content = []
            for i in range(8):
                content.append({"type": "text", "text": prompt_parts[i]})
                content.append({"type": "image", "image": images[i]})
            content.append({"type": "text", "text": prompt_parts[8]})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=images, return_tensors="pt"
            ).to(self.model.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **self._gen_kwargs)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            pred_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            res_item = item.copy()
            res_item["pred"] = pred_text
            results.append(res_item)
        return results

    # ---- run_pipeline (identical logic to upstream, kept inline for drop-in compat) ----

    def run_pipeline(
        self,
        cam_high_path: str,
        cam_left_path: str,
        cam_right_path: str,
        out_root: str,
        task: str,
        frame_interval: int = 10,
        batch_size: int = 1,
        goal_image: str = None,
        eval_mode: str = "incremental",
        visualize: bool = False,
    ) -> str:
        import json as _json
        import re as _re
        import shutil
        from datetime import datetime
        from pathlib import Path as _Path

        valid_modes = ["incremental", "forward", "backward"]
        if eval_mode not in valid_modes:
            raise ValueError(f"Invalid eval_mode '{eval_mode}'. Must be one of {valid_modes}")
        assert goal_image is not None, (
            "Goal image path must be provided. If no goal image is available, "
            "please use a blank image as a placeholder."
        )

        out_root = _Path(out_root)
        ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        safe_task = _re.sub(r'[^A-Za-z0-9_]', '_', task)
        run_root = out_root / f"{ts}_{eval_mode}_mode_{safe_task}"
        cache_root = run_root / ".cache"

        cam_dirs = {
            "cam_high": cache_root / "cam_high",
            "cam_left_wrist": cache_root / "cam_left_wrist",
            "cam_right_wrist": cache_root / "cam_right_wrist",
        }
        for d in cam_dirs.values():
            ensure_dir(d)

        paths = [_Path(cam_high_path), _Path(cam_left_path), _Path(cam_right_path)]
        types_counts = [get_frame_count(p) for p in paths]
        counts = [tc[1] for tc in types_counts]
        if len(set(counts)) != 1:
            print(
                f"[WARN] Frame count mismatch among cameras: {counts}. "
                f"Using the shortest length {min(counts)}."
            )
        total_frames = min(counts)
        indices = make_sample_indices_by_interval(total_frames, frame_interval)
        print(f"Frames: {total_frames}, Int: {frame_interval}, Mode: {eval_mode}, Indices: {len(indices)}")

        for p, key, (stype, _) in zip(paths, cam_dirs.keys(), types_counts):
            save_frames(p, cam_dirs[key], indices, stype)

        if goal_image is not None and os.path.exists(goal_image):
            ref_end_path = cache_root / "ref_end.png"
            shutil.copy(goal_image, ref_end_path)
            ref_end_path_str = str(ref_end_path)
            print(f"Using Goal Image: {goal_image}")
        else:
            ref_end_path_str = str(cam_dirs["cam_high"] / f"frame_{total_frames - 1:06d}.png")
            print("No Goal Image provided. Using last frame.")

        samples = build_samples_json(run_root, task, indices, ref_end_path_str, mode=eval_mode)
        json_path = run_root / "sample.json"
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(samples, f, indent=2)

        print(f"Running inference on {len(samples)} samples...")
        results = []
        for i in tqdm(range(0, len(samples), batch_size)):
            batch = samples[i:i + batch_size]
            results.extend(self.inference_batch(batch))

        prev_prog = 0.0
        for idx, item in enumerate(results):
            raw = item.get("pred", "")
            try:
                val_str = raw.split("<score>")[-1].split("</score>")[0].replace("%", "").strip()
                raw_score = max(-100.0, min(100.0, float(val_str))) / 100.0
            except Exception:
                print(f"[ERR] Failed to parse score for item {idx}: {raw}")
                raw_score = 0.0

            curr_progress = 0.0
            hop = 0.0
            if eval_mode == "incremental":
                if idx == 0:
                    curr_progress = raw_score
                else:
                    if raw_score >= 0:
                        curr_progress = prev_prog + (1 - prev_prog) * raw_score
                    else:
                        curr_progress = prev_prog + prev_prog * raw_score
                hop = raw_score
            elif eval_mode == "forward":
                curr_progress = raw_score
                hop = curr_progress - prev_prog
            elif eval_mode == "backward":
                curr_progress = 1.0 + raw_score
                hop = curr_progress - prev_prog

            item["hop"] = hop
            item["progress"] = curr_progress
            prev_prog = curr_progress

        result_path = run_root / "pred_vllm.json"
        with open(result_path, "w", encoding="utf-8") as f:
            _json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Done. Results saved to {result_path}")

        if visualize:
            print("Generating visualization video...")
            plot_video_reward(run_root)

        return str(run_root)


# ---- entry point (identical to upstream) ----

if __name__ == "__main__":
    model = GRMInference("tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview")

    TASK_INSTRUCTION = "organize the table"
    BASE_DEMO_PATH = "./examples/demo_table"
    OUTPUT_ROOT = "./results"
    GOAL_IMAGE_PATH = "./examples/demo_table/goal_image.png"
    PREDICTION_MODE = "forward"

    output_dir = model.run_pipeline(
        cam_high_path=os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
        cam_left_path=os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
        cam_right_path=os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
        out_root=OUTPUT_ROOT,
        task=TASK_INSTRUCTION,
        frame_interval=10,
        batch_size=1,
        goal_image=GOAL_IMAGE_PATH,
        eval_mode=PREDICTION_MODE,
        visualize=True,
    )
    print(f"Episode ({BASE_DEMO_PATH}) processed with multi-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

    output_dir = model.run_pipeline(
        cam_high_path=os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
        cam_left_path=os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
        cam_right_path=os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
        out_root=OUTPUT_ROOT,
        task=TASK_INSTRUCTION,
        frame_interval=10,
        batch_size=1,
        goal_image=GOAL_IMAGE_PATH,
        eval_mode=PREDICTION_MODE,
        visualize=True,
    )
    print(f"Episode ({BASE_DEMO_PATH}) processed with single-view {PREDICTION_MODE}-mode. Output at: {output_dir}")
