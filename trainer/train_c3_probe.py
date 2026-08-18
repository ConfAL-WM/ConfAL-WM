"""
trainer/train_c3_probe_v2.py
=============================
Unified C3ConfidenceProbe v2 training script (merges the pixel and decoder versions).

Main arguments (new/changed vs v1):
  --feat_mode     dec_only | mid_only | mid_dec
                  selects which UNet features the probe consumes
  --probe_block_idx   which output_block h_dec is taken from (default 5)
  --probe_block_channels  matching channel count (default 1280)
  --target_space  pixel | latent
                  loss space: pixel needs a VAE decode (slow), latent does not (fast)
  --threshold_mode  ema_warmup | fixed
  --threshold_warmup_steps  active EMA steps (default 2000), then frozen
  --threshold_low   initial/fixed lower threshold (see c3_probe.py suggestions)
  --threshold_high  initial/fixed upper threshold

Fixes (vs v1):
  * EMA cold start: thresholds init at the center of a reasonable range; EMA updates only for the first N steps
  * evaluate() swallowed exceptions silently: now prints the first error and returns -1.0 when count==0
  * T_VIS out of the training range: now computed dynamically from t_high_ratio (always inside the training distribution)
  * pixel-vis GT row: uses the raw batch["video"] frames instead of decoding latents
  * decoder target direction: target_space="latent" uses x0_pred MAE (correct direction)

Usage examples:
  # dec_only + latent space (recommended: no VAE decode, fast, correct direction)
  torchrun --nproc_per_node=2 --master_port=29502 \
      trainer/train_c3_probe.py \
      --config configs/agibotworld/train_c3_probe.yaml \
      --index  data_index.json \
      --output_dir logs/c3_dec_latent_temb \
      --feat_mode dec_only \
      --probe_block_idx 5 \
      --probe_block_channels 1280 \
      --target_space latent \
      --threshold_mode ema_warmup \
      --threshold_low 0.20 \
      --threshold_high 0.70 \
      --t_low_ratio 0.15 --t_high_ratio 0.60 \
      --max_steps 8000 --bf16 --batch_size 16 --lr 2e-3 --warmup_steps 300 \
      --device 6,7

  # dec_only + pixel space (T range [300,900], covers the model's uncertain region)
  torchrun --nproc_per_node=2 --master_port=29503 \
      trainer/train_c3_probe.py \
      --config configs/agibotworld/train_c3_probe.yaml \
      --index  data_index.json \
      --output_dir logs/c3_dec_pixel_temb \
      --feat_mode dec_only \
      --probe_block_idx 5 \
      --probe_block_channels 1280 \
      --target_space pixel \
      --threshold_mode ema_warmup \
      --threshold_warmup_steps 300 \
      --threshold_low 0.15 \
      --threshold_high 0.45 \
      --t_low_ratio 0.15 --t_high_ratio 0.60 \
      --max_steps 4000 --bf16 --batch_size 16 --lr 2e-3 --warmup_steps 300 \
      --device 0,1
"""

import argparse, os, random, sys, math, json, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm

# -- paths --------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.join(_REPO, "evac"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

from evac.utils.general_utils import instantiate_from_config, load_checkpoints
from lvdm.common import extract_into_tensor
from evac.lvdm.data.get_actions import parse_h5, get_actions
from evac.lvdm.data.statistics import StatisticInfo
from evac.lvdm.data.traj_vis_statistics import (
    ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight,
    EndEffectorPts, Gripper2EEFCvt,
)
from evac.lvdm.data.utils import get_transformation_matrix_from_quat, intrinsic_transform
from evac.lvdm.modules.c3_probe import C3ConfidenceProbe


# =============================================================================
# logging / DDP
# =============================================================================

def setup_logging(rank, output_dir):
    logger = logging.getLogger("c3_v2_train")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger


def is_dist():   return dist.is_available() and dist.is_initialized()
def get_rank():  return dist.get_rank()  if is_dist() else 0
def get_world_size(): return dist.get_world_size() if is_dist() else 1


def init_dist(device_ids=None):
    if "LOCAL_RANK" not in os.environ:
        return 0
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id  = device_ids[local_rank % len(device_ids)] if device_ids else local_rank
    torch.cuda.set_device(device_id)
    dist.init_process_group(backend="nccl")
    return local_rank


# =============================================================================
# model construction
# =============================================================================

def build_model(evac_cfg_path, evac_ckpt_path, probe_kwargs,
                probe_block_idx, device):
    """
    Load EVAC (frozen) + instantiate C3ConfidenceProbe v2.

    Checkpoint loading logic (in order of precedence):
      1. explicit --evac_ckpt arg -> written into model_cfg.pretrained_checkpoint
      2. existing model.pretrained_checkpoint field in the config yaml
      3. neither -> load_checkpoints prints a WARNING and the UNet keeps random weights

    Uses load_checkpoints (from evac.utils.general_utils) instead of manual
    torch.load + load_state_dict, to correctly handle deepspeed-format checkpoints.
    """
    config    = OmegaConf.load(evac_cfg_path)
    model_cfg = config.model
    model_cfg.params.enable_c3_probe = False

    # -- If evac_ckpt_path was passed explicitly, write it into model_cfg so that
    #    load_checkpoints can find it (it reads the pretrained_checkpoint field)
    if evac_ckpt_path and os.path.exists(str(evac_ckpt_path)):
        model_cfg.pretrained_checkpoint = str(evac_ckpt_path)
        print(f"[build_model] Using explicit evac_ckpt: {evac_ckpt_path}")

    model = instantiate_from_config(model_cfg)

    # -- Same loading path as eval/confidence_eval/probe_inference.py; handles deepspeed format correctly
    print("[build_model] Loading EVAC weights via load_checkpoints ...")
    model = load_checkpoints(model, model_cfg, ignore_mismatched_sizes=True)
    print("[build_model] EVAC weights loaded successfully.")

    # select the UNet feature extraction block
    unet = model.model.diffusion_model
    unet.probe_feat_decoder_idx = probe_block_idx
    print(f"[build_model] probe_feat_decoder_idx = {probe_block_idx} "
          f"(dec_channels={probe_kwargs.get('dec_channels', probe_kwargs.get('feat_channels', '?'))})")

    # freeze EVAC
    for param in model.parameters():
        param.requires_grad_(False)

    probe = C3ConfidenceProbe(**probe_kwargs)
    model.c3_probe = probe

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in probe.parameters())
    print(f"[build_model] total={total/1e6:.1f}M  trainable(probe)={trainable/1e6:.2f}M")
    print(f"[build_model] feat_mode={probe.feat_mode}  "
          f"target_space={probe.target_space}  "
          f"threshold_mode={probe.threshold_mode}  "
          f"thresh=[{probe.thresh_low.item():.3f}, {probe.thresh_high.item():.3f}]")

    model = model.to(device)
    model.eval()
    probe.train()
    return model


# =============================================================================
# dataset
# =============================================================================

class ProbeValDataset(Dataset):
    """Read GT episodes from val_root (unused when val_root is not set)."""

    def __init__(self, val_root, sample_size, chunk, n_previous, domain_name="agibotworld"):
        self.sample_size = tuple(sample_size)
        self.chunk = chunk; self.n_previous = n_previous
        self.T = n_previous + chunk; self.fps = 2.0
        self.mean_v = torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0)
        self.std_v  = torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0)
        self.domain_id_val = 0
        info_root = os.path.join(val_root, "info_dataset")
        gt_root   = os.path.join(val_root, "gt_dataset")
        self._episodes = []
        for task_id in sorted(os.listdir(info_root)):
            task_info = os.path.join(info_root, task_id)
            if not os.path.isdir(task_info): continue
            for ep_id in sorted(os.listdir(task_info)):
                ep_info   = os.path.join(task_info, ep_id)
                ep_gt_vid = os.path.join(gt_root, task_id, ep_id, "video")
                if not (os.path.isdir(ep_info) and os.path.isdir(ep_gt_vid)): continue
                pngs = sorted(Path(ep_gt_vid).glob("frame_*.png"))
                if len(pngs) >= self.T:
                    self._episodes.append({"info_dir": ep_info, "pngs": pngs})
        self._items = [(ei, s) for ei, ep in enumerate(self._episodes)
                       for s in range(0, len(ep["pngs"]) - self.T + 1, self.chunk)]
        print(f"[ProbeValDataset] {len(self._episodes)} eps, {len(self._items)} chunks")

    def __len__(self): return len(self._items)

    def __getitem__(self, idx):
        ei, frame_start = self._items[idx]
        ep = self._episodes[ei]; H, W = self.sample_size
        pngs = ep["pngs"][frame_start: frame_start + self.T]
        frames = [np.array(Image.open(str(p)).convert("RGB")) for p in pngs]
        orig_h, orig_w = frames[0].shape[:2]
        frames = [np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR)) for f in frames]
        video = (torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
                 .permute(3, 0, 1, 2) * 2.0 - 1.0)
        info_dir = ep["info_dir"]
        with open(os.path.join(info_dir, "head_intrinsic_params.json")) as f:
            intr = json.load(f)["intrinsic"]
        intrinsic = torch.eye(3, dtype=torch.float)
        intrinsic[0, 0] = intr["fx"]; intrinsic[0, 2] = intr["ppx"]
        intrinsic[1, 1] = intr["fy"]; intrinsic[1, 2] = intr["ppy"]
        intrinsic = intrinsic_transform(intrinsic, (orig_h, orig_w), (H, W), 'resize')
        with open(os.path.join(info_dir, "head_extrinsic_params_aligned.json")) as f:
            ext_list = json.load(f)
        ext_info = ext_list[0]["extrinsic"]
        c2w = torch.eye(4, dtype=torch.float)
        c2w[:3, :3] = torch.FloatTensor(ext_info["rotation_matrix"])
        c2w[:3,  3] = torch.FloatTensor(ext_info["translation_vector"])
        c2ws = c2w.unsqueeze(0).expand(self.T, -1, -1).clone()
        w2cs = torch.linalg.inv(c2w).unsqueeze(0).expand(self.T, -1, -1).clone()
        h5_path = os.path.join(info_dir, "proprio_stats.h5")
        action, delta_action = parse_h5(
            h5_path, slices=list(range(frame_start, frame_start + self.T)),
            delta_act_sidx=self.n_previous)
        action = torch.FloatTensor(action); delta_action = torch.FloatTensor(delta_action)
        delta_action[:, :6]   = (delta_action[:, :6]   - self.mean_v[:, :6])  / self.std_v[:, :6]
        delta_action[:, 7:13] = (delta_action[:, 7:13] - self.mean_v[:, 6:])  / self.std_v[:, 6:]
        traj = self._get_traj(action, w2cs, c2ws, intrinsic) * 2.0 - 1.0
        return dict(video=video, cond_id=-(self.n_previous + self.chunk),
                    intrinsic=intrinsic, extrinsic=c2ws, domain_id=self.domain_id_val,
                    action=action, traj=traj, delta_action=delta_action, fps=self.fps)

    def _get_traj(self, pose, w2c, c2w, intrinsic, radius=50):
        H, W = self.sample_size
        if isinstance(pose, np.ndarray): pose = torch.tensor(pose, dtype=torch.float32)
        ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32).view(1, 4, 4).permute(0, 2, 1)
        pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7])
        pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15])
        cvt = torch.tensor(Gripper2EEFCvt, dtype=torch.float32).view(1, 4, 4)
        ee2cam_l = torch.matmul(torch.matmul(w2c, pose_l_mat), cvt)
        ee2cam_r = torch.matmul(torch.matmul(w2c, pose_r_mat), cvt)
        pts_l = torch.matmul(ee2cam_l, ee_key_pts)
        pts_r = torch.matmul(ee2cam_r, ee_key_pts)
        K = intrinsic.unsqueeze(0)
        uvs_l = ((torch.matmul(K, pts_l[:, :3, :]) / pts_l[:, 2:3, :])[:, :2, :]
                 .permute(0, 2, 1).to(dtype=torch.int64))
        uvs_r = ((torch.matmul(K, pts_r[:, :3, :]) / pts_r[:, 2:3, :])[:, :2, :]
                 .permute(0, 2, 1).to(dtype=torch.int64))
        img_list = []
        for i in range(pose.shape[0]):
            img = np.zeros((H, W, 3), dtype=np.uint8) + 50
            col_l = tuple(int(c * 255) for c in ColorMapLeft(pose[i, 7].item() / 120)[:3])
            col_r = tuple(int(c * 255) for c in ColorMapRight(pose[i, 15].item() / 120)[:3])
            for pts, color in zip([uvs_l[i], uvs_r[i]], [col_l, col_r]):
                base = np.array(pts[0])
                if 0 <= base[0] < W and 0 <= base[1] < H:
                    cv2.circle(img, tuple(base[:2]), radius, color, -1)
            img_list.append(img / 255.0)
        return rearrange(torch.tensor(np.stack(img_list), dtype=torch.float32), "t h w c -> c t h w")


class ProbeExternalWorldModelDataset(Dataset):
    """Train C3 on converted external_worldmodel manifests.

    The sample contract matches the AgiBot loader: video/traj/delta_action plus
    camera fields. RoboTwin conversion should provide proprio.npy with EVAC-style
    absolute end-effector actions [T,16]; actions.npy may store [T,14] deltas.
    """

    def __init__(self, manifest_path, sample_size, chunk, n_previous, domain_name="agibotworld"):
        self.sample_size = tuple(sample_size)
        self.chunk = int(chunk)
        self.n_previous = int(n_previous)
        self.T = self.chunk + self.n_previous
        self.fps = 2.0
        self.mean_v = torch.tensor(StatisticInfo[domain_name]["mean"]).unsqueeze(0)
        self.std_v = torch.tensor(StatisticInfo[domain_name]["std"]).unsqueeze(0)
        self.domain_id_val = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        items = payload.get("items", payload if isinstance(payload, list) else [])
        self.records = []
        for item in items:
            frames_dir = item.get("frames_dir") or item.get("video_or_frames_path")
            proprio_path = item.get("proprio_path")
            camera_path = item.get("camera_path")
            if not frames_dir or not proprio_path or not camera_path:
                continue
            frame_files = sorted(Path(frames_dir).glob("frame_*.jpg")) + sorted(Path(frames_dir).glob("frame_*.png"))
            if len(frame_files) >= self.T:
                row = dict(item)
                row["_frames"] = frame_files
                self.records.append(row)
        if not self.records:
            raise ValueError(f"No usable external_worldmodel records in {manifest_path}")
        print(f"[ProbeExternalWorldModelDataset] {len(self.records)} episodes from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def _frame_indexes(self, total_frames):
        import random as _random
        if total_frames > self.T:
            end = _random.randint(self.T, total_frames)
        else:
            end = total_frames
        start = max(0, end - self.T)
        return list(range(start, end))

    def __getitem__(self, idx):
        rec = self.records[idx]
        frame_files = rec["_frames"]
        indexes = self._frame_indexes(len(frame_files))
        H, W = self.sample_size
        frames = [np.array(Image.open(frame_files[i]).convert("RGB")) for i in indexes]
        orig_h, orig_w = frames[0].shape[:2]
        frames = [np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR)) for f in frames]
        video = (torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
                 .permute(3, 0, 1, 2) * 2.0 - 1.0)

        proprio = np.load(rec["proprio_path"]).astype(np.float32)
        if proprio.ndim != 2 or proprio.shape[-1] != 16:
            raise ValueError(f"external proprio.npy must be [T,16], got {proprio.shape}: {rec['proprio_path']}")
        idx_np = np.asarray(indexes, dtype=np.int64)
        action, delta_action = get_actions(
            gripper=np.stack([proprio[:, 7], proprio[:, 15]], axis=1),
            all_ends_p=np.stack([proprio[:, 0:3], proprio[:, 8:11]], axis=1),
            all_ends_o=np.stack([proprio[:, 3:7], proprio[:, 11:15]], axis=1),
            slices=idx_np.tolist(),
            delta_act_sidx=self.n_previous,
        )
        action = torch.FloatTensor(action)
        delta_action = torch.FloatTensor(delta_action)
        delta_action[:, :6] = (delta_action[:, :6] - self.mean_v[:, :6]) / self.std_v[:, :6]
        delta_action[:, 7:13] = (delta_action[:, 7:13] - self.mean_v[:, 6:]) / self.std_v[:, 6:]

        cam = np.load(rec["camera_path"])
        intrinsic_all = np.asarray(cam["intrinsic_cv"], dtype=np.float32)
        if intrinsic_all.ndim == 3:
            intrinsic = torch.from_numpy(intrinsic_all[indexes[0]]).float()
        else:
            intrinsic = torch.from_numpy(intrinsic_all).float()
        intrinsic = intrinsic_transform(intrinsic, (orig_h, orig_w), (H, W), "resize")

        if "extrinsic_cv" in cam:
            ext = np.asarray(cam["extrinsic_cv"], dtype=np.float32)[idx_np]
            if ext.shape[-2:] == (3, 4):
                pad = np.zeros((*ext.shape[:-2], 1, 4), dtype=np.float32)
                pad[..., 0, 3] = 1.0
                ext = np.concatenate([ext, pad], axis=-2)
            w2cs = torch.from_numpy(ext).float()
            c2ws = torch.linalg.inv(w2cs)
        elif "cam2world_gl" in cam:
            c2ws_np = np.asarray(cam["cam2world_gl"], dtype=np.float32)
            c2ws = torch.from_numpy(c2ws_np[idx_np]).float()
            w2cs = torch.linalg.inv(c2ws)
        else:
            c2ws = torch.eye(4).unsqueeze(0).repeat(self.T, 1, 1)
            w2cs = torch.eye(4).unsqueeze(0).repeat(self.T, 1, 1)

        traj = self._get_traj(action, w2cs, c2ws, intrinsic) * 2.0 - 1.0
        return dict(
            video=video,
            cond_id=-(self.n_previous + self.chunk),
            intrinsic=intrinsic,
            extrinsic=c2ws,
            domain_id=self.domain_id_val,
            action=action,
            traj=traj,
            delta_action=delta_action,
            fps=self.fps,
        )

    def _get_traj(self, pose, w2c, c2w, intrinsic, radius=50):
        H, W = self.sample_size
        ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32).view(1, 4, 4).permute(0, 2, 1)
        pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7])
        pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15])
        cvt = torch.tensor(Gripper2EEFCvt, dtype=torch.float32).view(1, 4, 4)
        ee2cam_l = torch.matmul(torch.matmul(w2c, pose_l_mat), cvt)
        ee2cam_r = torch.matmul(torch.matmul(w2c, pose_r_mat), cvt)
        pts_l = torch.matmul(ee2cam_l, ee_key_pts)
        pts_r = torch.matmul(ee2cam_r, ee_key_pts)
        K = intrinsic.unsqueeze(0)
        uvs_l = ((torch.matmul(K, pts_l[:, :3, :]) / pts_l[:, 2:3, :])[:, :2, :]
                 .permute(0, 2, 1).to(dtype=torch.int64))
        uvs_r = ((torch.matmul(K, pts_r[:, :3, :]) / pts_r[:, 2:3, :])[:, :2, :]
                 .permute(0, 2, 1).to(dtype=torch.int64))
        img_list = []
        for i in range(pose.shape[0]):
            img = np.zeros((H, W, 3), dtype=np.uint8) + 50
            col_l = tuple(int(c * 255) for c in ColorMapLeft(pose[i, 7].item() / 120)[:3])
            col_r = tuple(int(c * 255) for c in ColorMapRight(pose[i, 15].item() / 120)[:3])
            for pts, color in zip([uvs_l[i], uvs_r[i]], [col_l, col_r]):
                base = np.array(pts[0])
                if 0 <= base[0] < W and 0 <= base[1] < H:
                    cv2.circle(img, tuple(base[:2]), radius, color, -1)
            img_list.append(img / 255.0)
        return rearrange(torch.tensor(np.stack(img_list), dtype=torch.float32), "t h w c -> c t h w")


def _detect_data_format(index, requested):
    if requested != "auto":
        return requested
    if index.get("format") == "external_worldmodel":
        return "external_worldmodel"
    items = index.get("items") or []
    if items and items[0].get("format") == "external_worldmodel":
        return "external_worldmodel"
    return "agibot"


def build_dataloader(index_path, batch_size, num_workers, evac_cfg_path, data_format="auto"):
    with open(index_path) as f: index = json.load(f)
    data_format = _detect_data_format(index, data_format)
    if data_format == "external_worldmodel":
        config = OmegaConf.load(evac_cfg_path)
        dataset = ProbeExternalWorldModelDataset(
            index_path,
            sample_size=[320, 512],
            chunk=config.get("chunk", 16),
            n_previous=config.get("n_previous", 4),
        )
        sampler = DistributedSampler(dataset, shuffle=True) if is_dist() else None
        return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                          sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)

    meta       = index.get("meta", {})
    items      = index.get("items") or []
    train_root = meta.get("train_root") or meta.get("data_root")
    allowed_paths = set()
    allowed_names = set()
    if items:
        for item in items:
            item_path = item.get("path") or item.get("source_path")
            if item_path:
                allowed_paths.add(os.path.realpath(item_path))
                allowed_names.add(os.path.basename(item_path))
        if train_root is None and allowed_paths:
            first_path = next(iter(allowed_paths))
            train_root = os.path.dirname(first_path)
    if not train_root:
        raise ValueError(
            f"AgiBot C3 training index must contain meta.train_root/data_root or items[*].path: {index_path}"
        )
    data_root  = os.path.dirname(train_root.rstrip("/"))
    split_name = os.path.basename(train_root.rstrip("/"))
    config     = OmegaConf.load(evac_cfg_path)
    ds_cfg = OmegaConf.create({
        "target": "dataset.agibotworld_challenge_dataset.AgiBotWorldICRA26Challenge",
        "params": {
            "data_roots": [data_root], "domains": ["agibotworld"],
            "split": split_name, "sample_size": [320, 512],
            "sample_n_frames": config.get("chunk", 16) + config.get("n_previous", 4),
            "preprocess": "resize", "valid_cam": "head",
            "chunk": config.get("chunk", 16), "n_previous": config.get("n_previous", 4),
            "random_crop": True, "min_sep": 1, "max_sep": 3,
        },
    })
    dataset = instantiate_from_config(ds_cfg)
    if allowed_paths or allowed_names:
        before = len(dataset.dataset)
        dataset.dataset = [
            row for row in dataset.dataset
            if os.path.realpath(row[0]) in allowed_paths or os.path.basename(row[0]) in allowed_names
        ]
        dataset.length = len(dataset.dataset)
        if dataset.length == 0:
            raise ValueError(
                f"No AgiBot episodes from manifest {index_path} matched dataset root {data_root}/{split_name}. "
                "Check that c3_train_split.json was generated from the same train_root."
            )
        print(f"[build_dataloader] AgiBot manifest subset: {dataset.length}/{before} segments from {index_path}")
    sampler = DistributedSampler(dataset, shuffle=True) if is_dist() else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                      sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)


# =============================================================================
# helpers
# =============================================================================

def _unwrap_probe(model):
    p = model.c3_probe
    return p.module if isinstance(p, DDP) else p


def decode_latents_to_rgb(model, z, device):
    """z: [B, 4, T, h, w] -> rgb: [B, 3, T, H, W] in [-1, 1]."""
    B, C, T, h, w = z.shape
    z_flat = rearrange(z, 'b c t h w -> (b t) c h w').to(device)
    ae_bs = getattr(model, 'ae_batch_size', 2)
    frames = []
    for i in range(0, z_flat.shape[0], ae_bs):
        chunk = z_flat[i:i + ae_bs]
        out = model.first_stage_model.decode((1.0 / model.scale_factor) * chunk)
        if hasattr(out, 'sample'): out = out.sample
        frames.append(out.float())
    rgb_flat = torch.cat(frames, dim=0)
    return rearrange(rgb_flat, '(b t) c h w -> b c t h w', b=B, t=T).clamp(-1, 1)


def prepare_batch(model, batch, device, t_low_ratio=0.30, t_high_ratio=0.90):
    """Return (x_noisy, x_start_scaled, noise, t, c, fs, did).
    x_start_scaled = z_clean * scale_arr[t] (already dynamically scaled).
    """
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    batch_inputs = model.get_batch_input(
        batch, random_uncond=False, return_fs=True, return_did=True, return_traj=True)
    x, c, fs, did, traj = batch_inputs[:5]

    t_low  = max(50, int(model.num_timesteps * t_low_ratio))
    t_high = min(int(model.num_timesteps * t_high_ratio), model.num_timesteps)
    t = torch.randint(t_low, t_high, (x.shape[0] // model.n_view,), device=device).long()
    t = rearrange(t.unsqueeze(1).repeat(1, model.n_view), "b v -> (b v)")

    x[:, :, :-model.chunk] = model.teacher_forcing_aug(x[:, :, :-model.chunk])
    c["c_concat"][0] = torch.cat([
        x[:, :, :-model.chunk],
        x[:, :, -(model.chunk + 1):-model.chunk].repeat(1, 1, model.chunk, 1, 1),
    ], dim=2)
    if model.use_cat_mask:
        c["c_concat"][0] = c["c_concat"][0] * c["c_concat"][1]

    if model.use_dynamic_rescale:
        x = x * extract_into_tensor(model.scale_arr, t, x.shape)

    if model.noise_strength > 0:
        b, ch, f, hh, ww = x.shape
        noise = torch.randn_like(x) + model.noise_strength * torch.randn(b, ch, f, 1, 1, device=device)
    else:
        noise = torch.randn_like(x)

    x_noisy = model.q_sample(x_start=x, t=t, noise=noise)
    x_noisy[:, :, :-model.chunk] = x[:, :, :-model.chunk]
    return x_noisy, x, noise, t, c, fs, did


def _get_x0_from_vpred(model, x_noisy, x_recon, t):
    """
    Recover x0_pred from v_pred (x_recon), then undo the dynamic scaling.
    Returns (x0_pred_unscaled, x0_gt_unscaled), both in VAE latent space (z_clean units).
    x_noisy and x_recon are both in the scaled space.
    """
    sqrt_a  = extract_into_tensor(model.sqrt_alphas_cumprod,           t, x_noisy.shape)
    sqrt_oa = extract_into_tensor(model.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
    x0_pred_scaled = sqrt_a * x_noisy.float() - sqrt_oa * x_recon.float()   # scaled space

    return x0_pred_scaled   # the caller undoes the scale


def _undo_scale(model, x_scaled, t):
    """Undo the dynamic scaling back into VAE latent (z_clean) space."""
    if model.use_dynamic_rescale:
        scale = extract_into_tensor(model.scale_arr, t, x_scaled.shape)
        return x_scaled / scale.clamp(min=1e-6)
    return x_scaled


def _build_tau_full(sampled_tau_bt, total_T, n_cond_frames):
    """Expand the future-only tau [B, chunk] into the probe full-sequence condition [B, T_total]."""
    if sampled_tau_bt is None:
        return None
    if n_cond_frames <= 0:
        return sampled_tau_bt
    prefix = sampled_tau_bt[:, :1].expand(-1, n_cond_frames)
    return torch.cat([prefix, sampled_tau_bt], dim=1)


def _assert_tau_target_aligned(tau_full, sampled_tau_bt, n_cond_frames):
    """Ensure the tau used by the BCE future logits exactly matches the sampled target threshold."""
    if tau_full is None or sampled_tau_bt is None:
        raise RuntimeError("use_tau_cond=True requires both sampled_tau and tau_full")
    tau_future = tau_full[:, n_cond_frames:]
    if tau_future.shape != sampled_tau_bt.shape:
        raise RuntimeError(
            f"tau condition and target threshold shape mismatch: tau_future={tuple(tau_future.shape)}, "
            f"sampled_tau={tuple(sampled_tau_bt.shape)}"
        )
    if not torch.equal(tau_future, sampled_tau_bt):
        max_abs = (tau_future - sampled_tau_bt).abs().max().item()
        raise RuntimeError(f"tau condition and target threshold value mismatch: max_abs_diff={max_abs:.6g}")


def _call_probe(probe, h_mid, h_dec, emb, T, tau=None):
    """Pick the correct probe forward call for feat_mode."""
    fm = probe.feat_mode
    if fm == "dec_only":
        return probe(h_dec=h_dec, T=T, h_mid=None, emb=emb, tau=tau)
    elif fm == "mid_only":
        return probe(h_dec=h_mid, T=T, h_mid=None, emb=emb, tau=tau)
    else:  # mid_dec
        return probe(h_dec=h_dec, T=T, h_mid=h_mid, emb=emb, tau=tau)


# =============================================================================
# visualization
# =============================================================================

@torch.no_grad()
def visualize_step(model, probe, batch, device, save_path, step,
                   t_low_ratio=0.30, t_high_ratio=0.90, amp_dtype=None):
    """
    4 columns = the same frame at 4 noise timesteps (all inside the training t range).
    Rows: x_t decoded | MAE heatmap | Target Conf | Oracle Conf(from |x0_pred-x0_gt|)
          | Probe Conf | GT+overlay
    """
    # T_VIS is computed dynamically and always stays inside the [t_low, t_high] training range
    t_min = max(1, int(model.num_timesteps * t_low_ratio))
    t_max = min(int(model.num_timesteps * t_high_ratio), model.num_timesteps - 1)
    step_sz = (t_max - t_min) // 3
    T_VIS = [t_min, t_min + step_sz, t_min + 2 * step_sz, t_max]

    probe.eval()
    try:
        def _add_compact_cbar(fig, im, ax):
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015, shrink=0.72, aspect=24)
            cbar.ax.tick_params(labelsize=7, length=2)
            return cbar

        _, x_start_base, _, t_base, c, fs, did = prepare_batch(
            model, batch, device, t_low_ratio=t_low_ratio, t_high_ratio=t_high_ratio)
        T      = x_start_base.shape[2]
        n_cond = T - model.chunk
        fidx   = T - 1
        unet_kwargs = {"fs": fs.long(), "domain_id": did.long()}

        # z_clean: unscaled VAE latent
        z_clean = _undo_scale(model, x_start_base, t_base)

        # GT RGB: prefer raw frames from batch["video"] (avoids VAE decode error)
        gt_frame_np = None
        gt_video_raw = batch.get("video")
        if gt_video_raw is not None:
            gf = gt_video_raw[0].float()[:, fidx]          # [C, H, W]
            gt_frame_np = ((gf.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
        H_px = W_px = None
        if gt_frame_np is not None:
            H_px, W_px = gt_frame_np.shape[:2]

        ph_saved = pw_saved = None
        results = []

        for t_val_int in T_VIS:
            BV = z_clean.shape[0]
            t_v = torch.full((BV,), t_val_int, device=device, dtype=torch.long)

            # scale + noise
            x_scaled = z_clean * extract_into_tensor(model.scale_arr, t_v, z_clean.shape) \
                if model.use_dynamic_rescale else z_clean.clone()
            noise   = torch.randn_like(x_scaled)
            x_noisy = model.q_sample(x_start=x_scaled, t=t_v, noise=noise)
            x_noisy[:, :, :-model.chunk] = x_scaled[:, :, :-model.chunk]

            # UNet forward: same AMP precision as training (avoids FP16/BF16 numeric mismatch)
            vis_amp = (torch.autocast("cuda", dtype=amp_dtype)
                       if amp_dtype is not None else torch.cuda.amp.autocast())
            with torch.inference_mode():
                with vis_amp:
                    x_recon_raw, (h_mid_raw, h_dec_raw, emb_raw) = model.model(
                        x_noisy, t_v, return_probe_feat=True, **c, **unet_kwargs)

            x_recon = x_recon_raw.detach().float()
            h_mid   = h_mid_raw.detach()
            h_dec   = h_dec_raw.detach()
            emb     = emb_raw.detach()

            ph, pw = h_dec.shape[-2], h_dec.shape[-1]
            if ph_saved is None: ph_saved, pw_saved = ph, pw

            # x0_pred → undo scale → unscaled latent
            x0_pred_unscaled = _undo_scale(model, _get_x0_from_vpred(model, x_noisy, x_recon, t_v), t_v)
            x0_gt_unscaled   = z_clean.detach()

            # compute target confidence (different inputs per target_space)
            if probe.target_space == "pixel":
                pred_rgb = decode_latents_to_rgb(model, x0_pred_unscaled[0:1, :, n_cond:], device)
                gt_rgb   = decode_latents_to_rgb(model, x0_gt_unscaled  [0:1, :, n_cond:], device)
                if probe.use_tau_cond:
                    target_conf_full, mae_stats, sampled_tau = probe.compute_target_confidence(
                        pred_rgb, gt_rgb, probe_h=ph, probe_w=pw, return_threshold=True)
                else:
                    target_conf_full, mae_stats = probe.compute_target_confidence(
                        pred_rgb, gt_rgb, probe_h=ph, probe_w=pw)
                    sampled_tau = None
            else:  # latent
                pred_lat = x0_pred_unscaled[0:1, :, n_cond:]
                gt_lat   = x0_gt_unscaled  [0:1, :, n_cond:]
                if probe.use_tau_cond:
                    target_conf_full, mae_stats, sampled_tau = probe.compute_target_confidence(
                        pred_lat, gt_lat, probe_h=ph, probe_w=pw, return_threshold=True)
                else:
                    target_conf_full, mae_stats = probe.compute_target_confidence(
                        pred_lat, gt_lat, probe_h=ph, probe_w=pw)
                    sampled_tau = None

            # probe forward
            tau_full = _build_tau_full(sampled_tau, total_T=T, n_cond_frames=n_cond)
            with vis_amp:
                logits = _call_probe(probe, h_mid[:T], h_dec[:T], emb[:T], T, tau=tau_full)
            conf      = np.nan_to_num(torch.sigmoid(logits[0]).cpu().float().numpy(), nan=0.5)
            # target_conf_full only contains the chunk frames (the last future frame)
            chunk_fidx = fidx - n_cond   # index within target_conf
            target_np  = target_conf_full[0, chunk_fidx].cpu().float().numpy()
            # decode x_t for visualization row 0
            noisy_img_np = None
            try:
                x_t_unscaled = _undo_scale(model, x_noisy[0:1, :, fidx:fidx+1].float(), t_v[:1])
                noisy_rgb_vis = decode_latents_to_rgb(model, x_t_unscaled, device)
                noisy_img_np = ((noisy_rgb_vis[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
            except Exception:
                pass

            # MAE heatmap (in the space of target_space)
            if probe.target_space == "pixel":
                mae_full = (pred_rgb - gt_rgb).abs().mean(dim=1)[0]  # [chunk, H_rgb, W_rgb]
                # downsample to probe resolution
                mae_down_np = F.adaptive_avg_pool2d(
                    mae_full.reshape(mae_full.shape[0], 1, *mae_full.shape[-2:]),
                    (ph, pw)).squeeze(1)[chunk_fidx].cpu().numpy()
            else:
                mae_full = (pred_lat - gt_lat).abs().mean(dim=1)[0]  # [chunk, h_lat, w_lat]
                mae_down_np = F.adaptive_avg_pool2d(
                    mae_full.reshape(mae_full.shape[0], 1, *mae_full.shape[-2:]),
                    (ph, pw)).squeeze(1)[chunk_fidx].cpu().numpy()

            # continuous oracle confidence: expected accuracy under the current threshold band,
            # semantically aligned with the predicted probe confidence (high = confident).
            tau_span = max(mae_stats["thresh_high"] - mae_stats["thresh_low"], 1e-6)
            oracle_conf_np = np.clip((mae_stats["thresh_high"] - mae_down_np) / tau_span, 0.0, 1.0)

            results.append(dict(
                t=t_val_int, noisy_img=noisy_img_np, mae=mae_down_np,
                target=target_np, oracle_conf=oracle_conf_np, conf=conf[fidx],
                mae_stats=mae_stats,
            ))

        # -- plotting ---------------------------------------------------------
        ncols = len(T_VIS)
        fig, axes = plt.subplots(6, ncols, figsize=(5.2 * ncols, 22))
        fig.suptitle(
            f"Step {step}  |  feat={probe.feat_mode}  target={probe.target_space}  "
            f"ph×pw={ph_saved}×{pw_saved}  |  cols: last future frame @ T_VIS={T_VIS}",
            fontsize=15, fontweight="bold", y=0.985)

        row_labels = [
            "1. Noisy input x_t decoded",
            f"2. MAE heatmap in {probe.target_space} space",
            f"3. Binary target confidence  th=[{results[0]['mae_stats']['thresh_low']:.2f},"
            f"{results[0]['mae_stats']['thresh_high']:.2f}]",
            "4. Continuous oracle confidence from |x0_pred - x0_gt|",
            "5. Probe confidence prediction",
            "6. GT with low-confidence overlay",
        ]

        for col, res in enumerate(results):
            t_tag = f"t={res['t']}"

            # Row 0: x_t decoded
            if res['noisy_img'] is not None:
                axes[0, col].imshow(res['noisy_img'])
            axes[0, col].set_title(t_tag, fontsize=12, fontweight='bold', pad=8)
            axes[0, col].axis("off")

            # Row 1: MAE heatmap
            im = axes[1, col].imshow(res['mae'], cmap="hot")
            _add_compact_cbar(fig, im, axes[1, col])
            axes[1, col].set_title(
                f"mean={res['mae'].mean():.3f}", fontsize=10, pad=6)
            axes[1, col].axis("off")

            # Row 2: Target Conf
            im = axes[2, col].imshow(res['target'], cmap="RdBu", vmin=0, vmax=1)
            _add_compact_cbar(fig, im, axes[2, col])
            axes[2, col].set_title(f"mean={res['target'].mean():.2f}", fontsize=10, pad=6)
            axes[2, col].axis("off")

            # Row 3: Oracle Conf
            im = axes[3, col].imshow(res['oracle_conf'], cmap="RdBu", vmin=0, vmax=1)
            _add_compact_cbar(fig, im, axes[3, col])
            axes[3, col].set_title(f"mean={res['oracle_conf'].mean():.2f}", fontsize=10, pad=6)
            axes[3, col].axis("off")

            # Row 4: Probe Conf
            im = axes[4, col].imshow(res['conf'], cmap="RdBu", vmin=0, vmax=1)
            _add_compact_cbar(fig, im, axes[4, col])
            axes[4, col].set_title(f"mean={res['conf'].mean():.2f}", fontsize=10, pad=6)
            axes[4, col].axis("off")

            # Row 5: GT + overlay
            if gt_frame_np is not None and H_px is not None:
                c_up = F.interpolate(
                    torch.from_numpy(res['conf']).unsqueeze(0).unsqueeze(0).float(),
                    size=(H_px, W_px), mode="bilinear", align_corners=False
                ).squeeze().numpy()
                red_i = 1.0 - c_up
                alpha = (red_i ** 0.5) * 0.75
                red_rgb = np.stack([red_i, red_i * 0.3, red_i * 0.1], axis=-1)
                blend = np.clip(alpha[:, :, None] * red_rgb +
                                (1 - alpha[:, :, None]) * gt_frame_np, 0, 1)
                axes[5, col].imshow(blend)
                axes[5, col].set_title(f"conf={res['conf'].mean():.2f}", fontsize=10, pad=6)
            else:
                axes[5, col].text(0.5, 0.5, "no video", ha="center", va="center",
                                  transform=axes[5, col].transAxes, fontsize=8)
            axes[5, col].axis("off")

        fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.03, wspace=0.28, hspace=0.42)
        for row, label in enumerate(row_labels):
            bbox = axes[row, 0].get_position()
            fig.text(
                bbox.x0, bbox.y1 + 0.012, label,
                ha="left", va="bottom",
                fontsize=12, fontweight="bold",
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[vis] saved → {save_path}")

    except Exception as e:
        print(f"[vis] failed at step {step}: {e}")
        import traceback; traceback.print_exc()
    finally:
        probe.train()


# =============================================================================
# single training step
# =============================================================================

def probe_step(model, probe, batch, device, scaler=None, amp_dtype=None,
               t_low_ratio=0.30, t_high_ratio=0.90):
    """
    One training step.
    Whether to VAE-decode depends on probe.target_space:
      "latent" -> MAE between x0_pred / x0_gt computed directly in VAE latent space (fast)
      "pixel"  -> decode to RGB first, then pixel MAE (slow, more intuitive)
    """
    if amp_dtype is not None:
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype)
    elif scaler is not None:
        amp_ctx = torch.cuda.amp.autocast()
    else:
        amp_ctx = torch.autocast("cuda", enabled=False)

    x_noisy, x_start, noise, t, c, fs, did = prepare_batch(
        model, batch, device, t_low_ratio=t_low_ratio, t_high_ratio=t_high_ratio)
    T      = x_noisy.shape[2]
    n_cond = T - model.chunk
    unet_kwargs = {"fs": fs.long(), "domain_id": did.long()}

    # Phase 1: UNet (no grad). Use no_grad (NOT inference_mode): inference_mode
    # tags tensors so they can't be saved for backward in torch>=2, breaking the
    # probe backward below even after .clone(). no_grad tensors are normal leaves.
    with torch.no_grad():
        with amp_ctx:
            x_recon_raw, (h_mid_raw, h_dec_raw, emb_raw) = model.model(
                x_noisy, t, return_probe_feat=True, **c, **unet_kwargs)

    x_recon = x_recon_raw.detach().float()
    h_mid   = h_mid_raw.detach().clone()
    h_dec   = h_dec_raw.detach().clone()
    emb     = emb_raw.detach()
    ph, pw  = h_dec.shape[-2], h_dec.shape[-1]

    # recover x0_pred and x0_gt (VAE latent space)
    with torch.no_grad():
        x0_pred_unscaled = _undo_scale(
            model, _get_x0_from_vpred(model, x_noisy, x_recon, t), t)
        x0_gt_unscaled   = _undo_scale(model, x_start, t)

        # compute the target for future frames only (saves VAE decode cost)
        if probe.target_space == "pixel":
            rgb_pred = decode_latents_to_rgb(model, x0_pred_unscaled[:, :, n_cond:], device)
            rgb_gt   = decode_latents_to_rgb(model, x0_gt_unscaled  [:, :, n_cond:], device)
            if probe.use_tau_cond:
                target_conf, mae_stats, sampled_tau = probe.compute_target_confidence(
                    rgb_pred, rgb_gt, probe_h=ph, probe_w=pw, return_threshold=True)
            else:
                target_conf, mae_stats = probe.compute_target_confidence(
                    rgb_pred, rgb_gt, probe_h=ph, probe_w=pw)
                sampled_tau = None
        else:  # latent
            if probe.use_tau_cond:
                target_conf, mae_stats, sampled_tau = probe.compute_target_confidence(
                    x0_pred_unscaled[:, :, n_cond:],
                    x0_gt_unscaled  [:, :, n_cond:],
                    probe_h=ph, probe_w=pw, return_threshold=True)
            else:
                target_conf, mae_stats = probe.compute_target_confidence(
                    x0_pred_unscaled[:, :, n_cond:],
                    x0_gt_unscaled  [:, :, n_cond:],
                    probe_h=ph, probe_w=pw)
                sampled_tau = None

    # Phase 2: probe forward (with gradients)
    tau_full = _build_tau_full(sampled_tau, total_T=T, n_cond_frames=n_cond)
    if probe.use_tau_cond:
        _assert_tau_target_aligned(tau_full, sampled_tau, n_cond)
    with amp_ctx:
        logits = _call_probe(probe, h_mid, h_dec, emb, T, tau=tau_full)  # [B, T, ph, pw]
        loss   = probe.compute_loss(logits[:, n_cond:], target_conf, n_cond_frames=0)

    label_stats = dict(mae_stats)   # already contains label_mean, label_std, thresh_low, thresh_high
    return loss, label_stats


# =============================================================================
# checkpoints
# =============================================================================

def save_probe_ckpt(model, optimizer, step, val_loss, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {"probe": _unwrap_probe(model).state_dict(), "step": step,
            "val_loss": val_loss, "opt_state": optimizer.state_dict()}
    path = os.path.join(output_dir, f"probe_step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(output_dir, "probe_latest.pt")
    if os.path.islink(latest): os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    return path


def load_probe_ckpt(model, optimizer, ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("probe", ckpt.get("probe_state"))
    _unwrap_probe(model).load_state_dict(state, strict=False)
    if "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    print(f"[load_probe_ckpt] Resumed from step={ckpt.get('step', 'unknown')}")
    return ckpt.get("step", 0)


# =============================================================================
# validation
# =============================================================================

def evaluate(model, probe, loader, device, n_batches=20,
             t_low_ratio=0.30, t_high_ratio=0.90):
    """
    Compute val_loss over the first n_batches batches.
    Fixed: the first exception prints details; returns -1.0 when every batch fails (distinct from a real 0.0 loss).
    """
    _unwrap_probe(model).eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        try:
            with torch.no_grad():
                loss, _ = probe_step(model, probe, batch, device,
                                     scaler=None,
                                     t_low_ratio=t_low_ratio, t_high_ratio=t_high_ratio)
            total += loss.item(); count += 1
        except Exception as e:
            if i == 0:
                import traceback
                print(f"[evaluate] batch {i} failed: {e}")
                traceback.print_exc()
            continue
    _unwrap_probe(model).train()
    if count == 0:
        print("[evaluate] WARNING: all batches failed, returning -1.0")
        return -1.0
    return total / count


# =============================================================================
# main training loop
# =============================================================================

def train(args):
    device_ids = [int(x.strip()) for x in args.device.split(',')] if args.device else None
    local_rank = init_dist(device_ids=device_ids)
    rank       = get_rank()
    device_id  = device_ids[local_rank % len(device_ids)] if device_ids else local_rank
    device     = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(rank, args.output_dir)

    probe_kwargs = dict(
        feat_mode              = args.feat_mode,
        mid_channels           = 1280,
        dec_channels           = args.probe_block_channels,
        emb_dim                = 1280,
        probe_dim              = args.probe_dim,
        n_heads                = args.n_heads,
        n_spatial_layers       = args.n_spatial_layers,
        n_temporal_layers      = args.n_temporal_layers,
        max_h                  = 64,
        max_w                  = 64,
        max_T                  = 32,
        dropout                = 0.1,
        use_emb_cond           = args.use_emb_cond,
        use_tau_cond           = args.use_tau_cond,
        target_space           = args.target_space,
        threshold_mode         = args.threshold_mode,
        threshold_warmup_steps = args.threshold_warmup_steps,
        lr_warmup_steps        = args.warmup_steps,
        threshold_low          = args.threshold_low,
        threshold_high         = args.threshold_high,
        ema_p_low              = args.ema_p_low,
        ema_p_high             = args.ema_p_high,
    )

    model = build_model(args.evac_cfg, args.evac_ckpt, probe_kwargs,
                        args.probe_block_idx, device)
    probe = _unwrap_probe(model)

    # -- mixed precision ------------------------------------------------------
    if args.bf16:
        amp_dtype = torch.bfloat16; scaler = None
        logger.info("AMP: BF16 (no GradScaler)")
    elif args.fp16:
        amp_dtype = None; scaler = torch.cuda.amp.GradScaler()
        logger.info("AMP: FP16 + GradScaler")
    else:
        amp_dtype = None; scaler = None
        logger.info("AMP: disabled (FP32)")

    model.ae_batch_size = args.ae_batch_size
    logger.info(f"ae_batch_size={args.ae_batch_size}")

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        s / max(1, args.warmup_steps) if s < args.warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - args.warmup_steps)
                                 / max(1, args.max_steps - args.warmup_steps)))))

    start_step = 0
    if args.resume:
        start_step = load_probe_ckpt(model, optimizer, args.resume, device)
        logger.info(f"Resumed from {args.resume}, step={start_step}")

    if is_dist():
        ddp_dev = device_ids[local_rank % len(device_ids)] if device_ids else local_rank
        model.c3_probe = DDP(model.c3_probe, device_ids=[ddp_dev], output_device=ddp_dev)

    train_loader = build_dataloader(args.index, args.batch_size, args.num_workers, args.evac_cfg, args.data_format)
    val_loader   = train_loader   # periodic eval on the first N training batches

    logger.info(
        f"Training: max_steps={args.max_steps}  feat_mode={args.feat_mode}  "
        f"target_space={args.target_space}  threshold_mode={args.threshold_mode}  "
        f"use_tau_cond={args.use_tau_cond}  "
        f"thresh=[{args.threshold_low},{args.threshold_high}]  "
        f"ema_p=[{args.ema_p_low},{args.ema_p_high}]  "
        f"threshold_warmup_steps={args.threshold_warmup_steps}  "
        f"lr_warmup_steps={args.warmup_steps}")

    global_step  = start_step
    running_loss = 0.0
    last_val_loss = -1.0
    model.eval()
    _unwrap_probe(model).train()

    pbar = tqdm(total=args.max_steps, initial=start_step, desc="C3-v2 probe",
                dynamic_ncols=True, disable=(rank != 0))

    while global_step < args.max_steps:
        if is_dist() and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(global_step)

        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            # visualization
            if rank == 0 and (global_step == start_step or
                              (args.vis_every > 0 and global_step % args.vis_every == 0)):
                vis_path = os.path.join(args.output_dir, "vis",
                                        f"vis_step_{global_step:06d}.png")
                visualize_step(model, _unwrap_probe(model), batch, device,
                               vis_path, global_step,
                               t_low_ratio=args.t_low_ratio,
                               t_high_ratio=args.t_high_ratio, amp_dtype=amp_dtype)

            optimizer.zero_grad(set_to_none=True)
            try:
                loss, label_stats = probe_step(
                    model, _unwrap_probe(model), batch, device,
                    scaler=scaler, amp_dtype=amp_dtype,
                    t_low_ratio=args.t_low_ratio, t_high_ratio=args.t_high_ratio)
            except Exception as e:
                logger.warning(f"step {global_step} error: {e}")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"step {global_step}: loss={loss.item():.4f}, skip")
                continue

            probe_params = list(_unwrap_probe(model).parameters())
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(probe_params, args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(probe_params, args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step  += 1
            running_loss += loss.item()

            if rank == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.5f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False)
                pbar.update(1)

                # per-step EMA-bound logger (gated by $EMA_CURVE_CSV; default off so
                # normal runs are unaffected). Used for the appendix evolution curve.
                _csv_path = os.environ.get("EMA_CURVE_CSV")
                if _csv_path:
                    import csv as _csvmod
                    _new = not os.path.exists(_csv_path)
                    with open(_csv_path, "a", newline="") as _f:
                        _w = _csvmod.writer(_f)
                        if _new:
                            _w.writerow(["step", "thresh_low", "thresh_high", "ema_step",
                                         "mae_p_low", "mae_p50", "mae_p_high", "lr", "loss"])
                        _w.writerow([global_step,
                                     label_stats.get("thresh_low", float("nan")),
                                     label_stats.get("thresh_high", float("nan")),
                                     label_stats.get("ema_step", 0),
                                     label_stats.get("mae_p_low", float("nan")),
                                     label_stats.get("mae_p50", float("nan")),
                                     label_stats.get("mae_p_high", float("nan")),
                                     optimizer.param_groups[0]["lr"],
                                     loss.item()])

            if rank == 0 and global_step % args.log_every == 0:
                avg = running_loss / args.log_every
                lr  = optimizer.param_groups[0]["lr"]
                ema_step = label_stats.get("ema_step", 0)
                if args.threshold_mode == "fixed":
                    thresh_tag = " [FIXED]"
                elif args.threshold_warmup_steps is None:
                    # full fine-tuning: show the current rate stage
                    thresh_tag = (" [FAST]" if ema_step < args.warmup_steps
                                  else " [SLOW]")
                else:
                    thresh_tag = ("" if ema_step < args.threshold_warmup_steps
                                  else " [FROZEN]")
                logger.info(
                    f"step {global_step:6d} | loss={avg:.5f} | lr={lr:.2e} | "
                    f"label_mean={label_stats['label_mean']:.3f} "
                    f"label_std={label_stats['label_std']:.3f} | "
                    f"mae_p{int(label_stats.get('ema_p_low', args.ema_p_low) * 100):02d}="
                    f"{label_stats.get('mae_p_low', 0):.4f} "
                    f"mae_p50={label_stats.get('mae_p50', 0):.4f} "
                    f"mae_p{int(label_stats.get('ema_p_high', args.ema_p_high) * 100):02d}="
                    f"{label_stats.get('mae_p_high', 0):.4f} "
                    f"thresh=[{label_stats.get('thresh_low', 0):.4f},"
                    f"{label_stats.get('thresh_high', 0):.4f}]{thresh_tag}"
                )
                running_loss = 0.0

            if rank == 0 and args.val_every > 0 and global_step % args.val_every == 0:
                last_val_loss = evaluate(
                    model, _unwrap_probe(model), val_loader, device,
                    t_low_ratio=args.t_low_ratio, t_high_ratio=args.t_high_ratio)
                logger.info(f"step {global_step:6d} | val_loss={last_val_loss:.5f}")

            if rank == 0 and global_step % args.save_every == 0:
                path = save_probe_ckpt(model, optimizer, global_step,
                                       last_val_loss, args.output_dir)
                logger.info(f"Saved → {path}")

    pbar.close()
    if rank == 0:
        path = save_probe_ckpt(model, optimizer, global_step, last_val_loss, args.output_dir)
        logger.info(f"Training done. Final ckpt → {path}")

    if is_dist():
        dist.destroy_process_group()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train C3ConfidenceProbe v2 (unified)")

    # ── EVAC ──────────────────────────────────────────────────────────────────
    p.add_argument("--config",   required=True, help="EVAC config yaml")
    p.add_argument("--index",    required=True, help="Data index JSON")
    p.add_argument("--data_format", default="auto", choices=["auto", "agibot", "external_worldmodel"],
                   help="Dataset format for --index. external_worldmodel uses converted RoboTwin/LIBERO manifests.")
    p.add_argument("--evac_cfg", default=None,  help="Alias for --config")
    p.add_argument("--evac_ckpt", default=None, help="EVAC checkpoint (.pt/.ckpt)")

    # ── Probe feature ─────────────────────────────────────────────────────────
    p.add_argument("--feat_mode", default="dec_only",
                   choices=["dec_only", "mid_only", "mid_dec"],
                   help="Which UNet features to use")
    p.add_argument("--probe_block_idx",      type=int,   default=5,
                   help="UNet output_block index for h_dec (default 5 → 20×32, ch=1280)")
    p.add_argument("--probe_block_channels", type=int,   default=1280,
                   help="Channels at probe_block_idx")

    # ── Probe architecture ────────────────────────────────────────────────────
    p.add_argument("--probe_dim",         type=int,   default=256)
    p.add_argument("--n_heads",           type=int,   default=4)
    p.add_argument("--n_spatial_layers",  type=int,   default=3)
    p.add_argument("--n_temporal_layers", type=int,   default=3)
    p.add_argument("--use_emb_cond",      action="store_true", default=True,
                   help="Enable AdaLN timestep conditioning")
    p.add_argument("--use_tau_cond",      action="store_true", default=True,
                   help="condition the probe on the tau sampled during training, learning q(x,tau) instead of the tau-marginalized signal")

    # ── Loss target space ─────────────────────────────────────────────────────
    p.add_argument("--target_space", default="latent", choices=["pixel", "latent"],
                   help="Compute MAE in pixel (RGB) or latent (z) space")

    # ── Threshold ─────────────────────────────────────────────────────────────
    p.add_argument("--threshold_mode", default="ema_warmup",
                   choices=["ema_warmup", "fixed"],
                   help="ema_warmup: EMA updates thresh (freeze after N steps if "
                        "--threshold_warmup_steps is set, else run until end); fixed: no EMA")
    p.add_argument("--threshold_warmup_steps", type=int,   default=None,
                   help="Steps during which EMA actively updates then freezes "
                        "(ema_warmup mode only). Omit to run EMA for the full training "
                        "with LR-coupled adaptive rate (fast during warmup, slow after).")
    p.add_argument("--threshold_low",          type=float, default=0.20,
                   help="Initial/fixed threshold lower bound (pixel≈0.15, latent≈0.20)")
    p.add_argument("--threshold_high",         type=float, default=0.70,
                   help="Initial/fixed threshold upper bound (pixel≈0.35, latent≈0.70)")
    p.add_argument("--ema_p_low",             type=float, default=0.10,
                   help="Lower quantile used by EMA threshold tracking (default: 0.10)")
    p.add_argument("--ema_p_high",            type=float, default=0.90,
                   help="Upper quantile used by EMA threshold tracking (default: 0.90)")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--max_steps",     type=int,   default=20000)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--warmup_steps",  type=int,   default=200,
                   help="Optimizer LR warmup steps")
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--t_low_ratio",   type=float, default=0.15,
                   help="Min diffusion T fraction. Default 0.15 (T=150 for 1000-step model). "
                        "Lower T → x0_pred too clean → zero gradient.")
    p.add_argument("--t_high_ratio",  type=float, default=0.60,
                   help="Max diffusion T fraction. Default 0.60 (T=600 for 1000-step model). "
                        "T_VIS is computed from [t_low_ratio, t_high_ratio] automatically.")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--ae_batch_size", type=int,   default=8,
                   help="VAE decode batch size (only used when target_space=pixel)")
    p.add_argument("--bf16",   action="store_true")
    p.add_argument("--fp16",   action="store_true")
    p.add_argument("--device", default=None,
                   help="GPU ids, e.g. '0,1'")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log_every",  type=int, default=50)
    p.add_argument("--val_every",  type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--vis_every",  type=int, default=1000)

    args = p.parse_args()
    # accept both --config and --evac_cfg
    if args.evac_cfg is None:
        args.evac_cfg = args.config
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
