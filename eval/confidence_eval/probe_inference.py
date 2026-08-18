#!/usr/bin/env python3
"""
probe_inference.py
==================
Shared C3 probe / EVAC inference library used by the active-learning pipeline.

This module holds the reusable pieces of C3 probe inference:

  - AgiBot-format data loading helpers (joint trajectories, camera parameters)
    that mirror ``evac/main/infer_all.py`` exactly, so scoring and validation
    inference see the same inputs as the original EVAC pipeline.
  - ``load_model_with_probe``: instantiate the EVAC latent-diffusion model,
    load EVAC weights, auto-detect the probe architecture from its checkpoint,
    and load the unified ``C3ConfidenceProbe`` weights.
  - ``extract_probe_confidence``: run the probe chunk-by-chunk with the same
    conditioning EVAC used at inference time, producing dense per-patch
    confidence maps ``[T, h, w]`` over predicted future frames.
  - Latent helpers used to build oracle targets and latent metrics
    (``encode_rgb_frames_to_latents``, ``internal_samples_to_latent_array``).

Entry points that consume this library:

  - ``al_pipeline/score_pool_with_c3.py`` -- pool scoring / conf_map export
  - ``eval/al_results/run_val_inference.py`` -- validation-set inference
  - ``eval/al_results/visualize_val_results.py`` -- GT/pred/confidence sheets
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(1, str(_REPO / "evac"))

from omegaconf import OmegaConf

from evac.utils.general_utils import instantiate_from_config, load_checkpoints
from evac.lvdm.data.get_actions import parse_h5
from evac.lvdm.data.statistics import StatisticInfo

# Optional video/frame decoding backends.
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

__all__ = [
    "get_action_bias_std",
    "load_action_h5",
    "load_caminfo_json",
    "load_generated_frames",
    "encode_rgb_frames_to_latents",
    "internal_samples_to_latent_array",
    "load_model_with_probe",
    "extract_probe_confidence",
    "reduce_probe_conf_stack",
    "build_probe_window_starts",
    "build_probe_window_weights",
]


# ---------------------------------------------------------------------------
# Data-loading helpers (identical to evac/main/infer_all.py)
# ---------------------------------------------------------------------------

def get_action_bias_std(domain_name="agibotworld"):
    return (torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0),
            torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0))


def load_action_h5(action_path, n_chunk, chunk, n_previous, domain_name="agibotworld"):
    """Load a proprio_stats.h5 exactly like infer_all.py get_action_h5."""
    if n_chunk > 0:
        slices = list(range(0, n_chunk * chunk))
        slices = [0] * (n_previous - 1) + slices
    else:
        slices = None
    action, delta_action = parse_h5(action_path, slices=slices, delta_act_sidx=n_previous)
    action       = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    mean_v, std_v = get_action_bias_std(domain_name)
    delta_action[:, :6]  = (delta_action[:, :6]  - mean_v[:, :6])  / std_v[:, :6]
    delta_action[:, 7:13] = (delta_action[:, 7:13] - mean_v[:, 6:]) / std_v[:, 6:]
    return action, delta_action


def load_caminfo_json(extrinsic_path, intrinsic_path, n_frames):
    """Load camera extrinsics/intrinsics exactly like infer_all.py get_caminfo_json."""
    with open(extrinsic_path) as f:
        info = json.load(f)[0]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(info["extrinsic"]["rotation_matrix"])
    c2w[:3,  3] = np.array(info["extrinsic"]["translation_vector"])
    c2w = torch.from_numpy(c2w).float()
    w2c = torch.linalg.inv(c2w).float()
    c2w = c2w.unsqueeze(0).repeat(n_frames, 1, 1)
    w2c = w2c.unsqueeze(0).repeat(n_frames, 1, 1)
    with open(intrinsic_path) as f:
        info = json.load(f)["intrinsic"]
    K = np.eye(3)
    K[0, 0] = info["fx"]; K[0, 2] = info["ppx"]
    K[1, 1] = info["fy"]; K[1, 2] = info["ppy"]
    intrinsic = torch.from_numpy(K).float()
    return c2w, w2c, intrinsic


# ---------------------------------------------------------------------------
# Model loading (EVAC + unified C3ConfidenceProbe)
# ---------------------------------------------------------------------------

def _detect_probe_type(state):
    """
    Classify the probe architecture from the checkpoint state_dict keys.

    Returns:
      'standard'   -> C3ConfidenceProbe (fusion_conv + adaLN, latent MAE target)
      'pixel'      -> C3ConfidenceProbePixel (fusion_conv, pixel MAE target)
      'decoder'    -> C3ConfidenceProbeDecoder (feat_proj, h_dec-only features)

    Notes:
      - 'decoder' is identified by keys starting with 'feat_proj.' (with or
        without AdaLN).
      - 'pixel' has 'fusion_conv.' plus 'norm1.weight' (use_emb_cond=False),
        or 'fusion_conv.' plus 'adaLN_attn' (use_emb_cond=True).
      - 'standard' has 'fusion_conv.' + 'adaLN_attn' but no 'norm1.weight'
        (pixel+AdaLN and standard both have adaLN_attn; they are separated by
        the absence of norm1 and feat_proj).
    """
    has_feat_proj   = any(k.startswith("feat_proj.")   for k in state)
    has_fusion_conv = any(k.startswith("fusion_conv.") for k in state)
    has_norm1       = any(k.endswith(".norm1.weight")  for k in state)

    if has_feat_proj:
        return "decoder"           # decoder probe, regardless of use_emb_cond
    if has_fusion_conv and has_norm1:
        return "pixel"             # pixel probe, use_emb_cond=False
    if has_fusion_conv:
        # fusion_conv without norm1 -> standard (AdaLN) or pixel+AdaLN.
        # Both carry cond_proj; be conservative and use the cond_proj key.
        return "pixel_adaLN" if any(k.startswith("cond_proj.") for k in state) else "standard"
    return "standard"


def _count_layer_blocks(state, prefix):
    return len({
        int(k.split(".")[1]) for k in state
        if k.startswith(prefix)
    })


def _infer_unified_probe_kwargs(state, probe_kwargs, ckpt):
    """
    Map legacy / unified checkpoints onto the constructor kwargs of the
    current unified C3ConfidenceProbe.
    """
    import inspect
    from evac.lvdm.modules.c3_probe import C3ConfidenceProbe

    valid_keys = set(inspect.signature(C3ConfidenceProbe.__init__).parameters) - {"self"}
    kw = {k: v for k, v in probe_kwargs.items() if k in valid_keys}

    has_feat_proj = any(k.startswith("feat_proj.") for k in state)
    has_fusion_conv = any(k.startswith("fusion_conv.") for k in state)
    use_emb_cond = any(".adaLN" in k for k in state) or any(k.startswith("cond_proj.") for k in state)
    use_tau_cond = any(k.startswith("tau_proj.") for k in state)

    if has_feat_proj:
        in_channels = state["feat_proj.0.weight"].shape[1]
        probe_dim = state["feat_proj.0.weight"].shape[0]
        feat_mode = ckpt.get("feat_mode", kw.get("feat_mode", "dec_only"))
    elif has_fusion_conv:
        in_channels = state["fusion_conv.0.weight"].shape[1]
        probe_dim = state["fusion_conv.0.weight"].shape[0]
        # Legacy pixel / standard probes correspond to mid_dec two-branch fusion.
        feat_mode = "mid_dec"
    else:
        raise ValueError("Cannot infer the probe input structure from the checkpoint state_dict")

    sp_shape = state["spatial_pe"].shape
    tp_shape = state["temporal_pe"].shape
    max_h, max_w = int(sp_shape[2]), int(sp_shape[3])
    max_T = int(tp_shape[1])
    n_spatial_layers = _count_layer_blocks(state, "spatial_layers.")
    n_temporal_layers = _count_layer_blocks(state, "temporal_layers.")

    kw.update({
        "feat_mode": feat_mode,
        "probe_dim": probe_dim,
        "max_h": max_h,
        "max_w": max_w,
        "max_T": max_T,
        "n_spatial_layers": n_spatial_layers,
        "n_temporal_layers": n_temporal_layers,
        "use_emb_cond": use_emb_cond,
        "use_tau_cond": use_tau_cond,
    })

    if feat_mode == "dec_only":
        kw["dec_channels"] = in_channels
    elif feat_mode == "mid_only":
        kw["mid_channels"] = in_channels
    else:  # mid_dec
        cfg_mid = int(kw.get("mid_channels", 1280))
        cfg_dec = int(kw.get("dec_channels", 1280))
        if cfg_mid + cfg_dec != in_channels:
            if 0 < cfg_mid < in_channels:
                cfg_dec = in_channels - cfg_mid
            else:
                cfg_mid = in_channels // 2
                cfg_dec = in_channels - cfg_mid
        kw["mid_channels"] = cfg_mid
        kw["dec_channels"] = cfg_dec

    return kw


def _remap_probe_state_to_unified(state, probe):
    """
    Remap legacy checkpoint keys onto keys loadable by the unified
    C3ConfidenceProbe, and backfill buffers that exist in the new version
    but are missing from old checkpoints.
    """
    remapped = {}
    for key, value in state.items():
        if key.startswith("fusion_conv."):
            remapped["feat_proj." + key[len("fusion_conv."):]] = value
        else:
            remapped[key] = value

    current_state = probe.state_dict()
    for buffer_name in ("thresh_low", "thresh_high", "ema_step", "lr_warmup_steps_buf"):
        if buffer_name not in remapped and buffer_name in current_state:
            remapped[buffer_name] = current_state[buffer_name]
    return remapped


def load_model_with_probe(config_path, probe_ckpt_path, device, evac_ckpt_path=None):
    """
    1. Instantiate ACWMLatentDiffusion from the YAML (enable_c3_probe=True).
    2. Load EVAC weights (from config.model.pretrained_checkpoint).
    3. Auto-detect the probe architecture and replace model.c3_probe
       (keeping the RNG state unpolluted).
    4. Load the C3 probe weights.

    Legacy pixel / decoder probes have been merged into the unified
    C3ConfidenceProbe, so the constructor kwargs are inferred from the
    checkpoint key structure and restored onto the unified implementation.
    """
    config    = OmegaConf.load(config_path)
    model_cfg = config.model
    if evac_ckpt_path:
        model_cfg.pretrained_checkpoint = str(evac_ckpt_path)
        print(f"[load] EVAC checkpoint override: {evac_ckpt_path}")
    model_cfg.params.enable_c3_probe = True

    model = instantiate_from_config(model_cfg)
    model = load_checkpoints(model, model_cfg, ignore_mismatched_sizes=True)
    model = model.to(device).eval()

    # Load the probe checkpoint.
    ckpt  = torch.load(probe_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("probe", ckpt.get("probe_state"))
    if state is None:
        raise KeyError(
            f"Probe checkpoint '{probe_ckpt_path}' contains neither 'probe' nor "
            f"'probe_state' key. Available keys: {list(ckpt.keys())}"
        )

    probe_type = _detect_probe_type(state)
    print(f"[load] Detected probe architecture: '{probe_type}'")

    if any(k.startswith("feat_proj.") or k.startswith("fusion_conv.") for k in state):
        # Save the PyTorch RNG state so probe init does not pollute DDIM noise.
        rng_state     = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None

        probe_kwargs = dict(model_cfg.params.get("c3_probe_config", {}))
        from evac.lvdm.modules.c3_probe import C3ConfidenceProbe

        kw = _infer_unified_probe_kwargs(state, probe_kwargs, ckpt)
        new_probe = C3ConfidenceProbe(**kw)

        if kw["feat_mode"] == "dec_only":
            block_idx = ckpt.get("probe_block_idx", None)
            if block_idx is None:
                import re
                m = re.search(r"idx(\d+)", probe_ckpt_path)
                block_idx = int(m.group(1)) if m else 5
            model.model.diffusion_model.probe_feat_decoder_idx = block_idx
        else:
            block_idx = None

        print(f"[load] Replaced c3_probe -> unified C3ConfidenceProbe "
              f"(feat_mode={kw['feat_mode']}, "
              f"mid_channels={kw.get('mid_channels', 'NA')}, "
              f"dec_channels={kw.get('dec_channels', 'NA')}, "
              f"probe_dim={kw['probe_dim']}, max_h={kw['max_h']}, max_w={kw['max_w']}, max_T={kw['max_T']}, "
              f"n_spatial_layers={kw['n_spatial_layers']}, n_temporal_layers={kw['n_temporal_layers']}, "
              f"block_idx={block_idx}, use_emb_cond={kw['use_emb_cond']}, "
              f"{sum(p.numel() for p in new_probe.parameters())/1e6:.2f}M params)")

        model.c3_probe = new_probe.to(device).eval()

        # Restore the RNG state so DDIM sampling noise matches the standard probe.
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    state = _remap_probe_state_to_unified(state, model.c3_probe)
    incompatible = model.c3_probe.load_state_dict(state, strict=False)
    critical_missing = [
        k for k in incompatible.missing_keys
        if k not in {
            "thresh_low", "thresh_high", "ema_step", "lr_warmup_steps_buf",
            "tau_proj.0.weight", "tau_proj.0.bias", "tau_proj.2.weight", "tau_proj.2.bias",
        }
    ]
    if critical_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Probe state_dict still has incompatible keys: "
            f" missing={critical_missing}, unexpected={incompatible.unexpected_keys}"
        )
    step  = ckpt.get("step", 0)
    print(f"[load] Probe loaded from {probe_ckpt_path} (step={step})")
    return model, config, step


# ---------------------------------------------------------------------------
# Frame / latent helpers
# ---------------------------------------------------------------------------

def load_generated_frames(save_path):
    """
    Read the frame sequence saved by model.inference() from a frame directory.
    inference() writes frame_{:05d}.jpg (BGR via cv2.imwrite).
    Returns [T, H, W, 3] float32 in [0, 1], sorted by file name.
    """
    # inference() saves .jpg with cv2.imwrite (BGR on disk, read back as RGB).
    imgs = sorted(Path(save_path).glob("frame_*.jpg"))
    if not imgs:
        imgs = sorted(Path(save_path).glob("frame_*.png"))
    if not imgs:
        # Frames may also be stored directly as an mp4.
        mp4s = list(Path(save_path).glob("*.mp4"))
        if mp4s and HAS_CV2:
            return _read_mp4(str(mp4s[0]))
        return None
    frames = []
    for p in imgs:
        if HAS_CV2:
            # Read with cv2 and convert BGR -> RGB.
            bgr = cv2.imread(str(p))
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            img = np.array(Image.open(p)).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        frames.append(img[:, :, :3])
    return np.stack(frames, axis=0)


def _read_mp4(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


@torch.no_grad()
def encode_rgb_frames_to_latents(model, frames_np, device, sample_size):
    """
    Encode an RGB frame sequence into VAE latent space.
    Uses mode=True (posterior mode) so the latent oracle adds no sampling noise.
    """
    from torchvision import transforms

    frames = np.asarray(frames_np, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames_np expected [T,H,W,3], got {frames.shape}")

    trans_resize = transforms.Resize(tuple(sample_size))
    trans_norm   = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    video_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    video_t = trans_norm(trans_resize(video_t))
    with torch.amp.autocast("cuda"):
        latents = model.encode_first_stage(video_t.to(device), mode=True)
    return latents.detach().cpu().float().numpy()


def internal_samples_to_latent_array(samples, n_previous, n_frames):
    """
    Normalize the internal all_samples returned by ddpm3d.inference() to
    [T, C, H, W]. all_samples usually contains the conditioning-frame prefix,
    so the first n_previous frames are dropped when the sequence is long
    enough.
    """
    if samples is None:
        return None
    if isinstance(samples, np.ndarray):
        z = torch.from_numpy(samples)
    elif torch.is_tensor(samples):
        z = samples.detach().cpu()
    else:
        return None

    z = z.float()
    if z.dim() == 6:  # [B, C, V, T, H, W] -> [B*V, C, T, H, W]
        from einops import rearrange
        z = rearrange(z, "b c v t h w -> (b v) c t h w")
    if z.dim() != 5:
        return None

    # Evaluation output stitches views horizontally; the standard config is
    # single-view. For multi-view, keep the first view's latent.
    z = z[0]  # [C, T_all, H, W]
    t_all = z.shape[1]
    start = n_previous if t_all >= n_previous + n_frames else 0
    end = min(start + n_frames, t_all)
    z = z[:, start:end]
    return z.permute(1, 0, 2, 3).contiguous().numpy()


# ---------------------------------------------------------------------------
# Probe confidence extraction
# ---------------------------------------------------------------------------

def reduce_probe_conf_stack(conf_stack, reduce_mode):
    if reduce_mode == "mean":
        return conf_stack.mean(axis=0)
    if reduce_mode == "median":
        return np.median(conf_stack, axis=0)
    if reduce_mode == "single":
        return conf_stack[0]
    raise ValueError(f"Unsupported probe_ts_reduce: {reduce_mode}")


def build_probe_window_starts(total_len, chunk, stride):
    """Build sliding-window start indices for probe evaluation; the last frame is always covered."""
    if total_len <= 0:
        return [0]
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if total_len <= chunk:
        return [0]
    starts = list(range(0, total_len - chunk + 1, stride))
    last_start = total_len - chunk
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_probe_window_weights(length, chunk, stride):
    """
    Build smooth fusion weights for overlapping windows.
    Degenerates to all-ones when stride >= chunk; with overlap, edge weights
    are slightly lower to soften chunk-boundary transitions.
    """
    if length <= 0:
        return np.zeros((0, 1, 1), dtype=np.float32)
    if stride >= chunk or length == 1:
        return np.ones((length, 1, 1), dtype=np.float32)

    overlap = max(1, chunk - stride)
    fade = min(overlap, max(1, length // 2))
    w = np.ones(length, dtype=np.float32)
    ramp = np.linspace(0.5, 1.0, fade, dtype=np.float32)
    w[:fade] = np.minimum(w[:fade], ramp)
    w[-fade:] = np.minimum(w[-fade:], ramp[::-1])
    return w[:, None, None]


def _hdec_to_trajectory_embedding(h_dec, n_previous):
    """
    Convert a decoder feature tensor into one trajectory-level embedding.

    Expected EVAC h_dec shape is usually [B, C, T, H, W]. We keep the channel
    axis and mean-pool over batch/time/space. This is used only as an AL
    diversity feature; it is detached and never participates in optimization.
    """
    if h_dec is None:
        return None
    feat = h_dec.detach().float()
    if feat.ndim == 5:
        future = feat[:, :, n_previous:]
        emb = future.mean(dim=(0, 2, 3, 4))
    elif feat.ndim == 4:
        # Conservative fallback for [B, C, H, W].
        emb = feat.mean(dim=(0, 2, 3))
    elif feat.ndim == 3:
        emb = feat.mean(dim=(0, 2))
    else:
        emb = feat.reshape(-1)
    return emb.cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_probe_confidence(model, config, gen_frames_np, device, args,
                              ref_img, action, delta_action, c2w, w2c, intrinsic):
    """
    Run the C3 probe with the same EVAC conditioning used at training time and
    return per-frame confidence maps.

    Per-chunk flow:
      1. Rebuild the batch dict from episode data (video + traj + delta_action + cam).
      2. Call model.get_batch_input() for z + cond (mirrors inference internals).
      3. Encode generated frames to latents and pass them as pre_z.
      4. Noise the latents and extract features via model.model(return_probe_feat=True).
      5. Run the probe and aggregate over the configured noise timesteps.

    gen_frames_np: [T, H, W, 3] float32 in [0, 1]
    ref_img: [3, n_previous, H, W] reference frames (CPU, [0,1])
    """
    from torchvision import transforms
    from einops import rearrange

    chunk      = config.chunk
    n_previous = config.n_previous
    sample_size = tuple(config.data.params.train.params.sample_size)
    T_total = gen_frames_np.shape[0]

    trans_resize = transforms.Resize(sample_size)
    trans_norm   = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    fps_t     = 2.0 * torch.ones(1, device=device)          # matches inference(fps=2) default
    domain_id = torch.zeros(1, dtype=torch.long, device=device)

    def _to_device(obj, dev):
        """Recursively move dict/list/tuple/Tensor values onto dev."""
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        elif isinstance(obj, dict):
            return {k: _to_device(v, dev) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_to_device(v, dev) for v in obj)
        return obj

    # Reference frames (same as inference()).
    ref_t = rearrange(ref_img, 'c t h w -> t c h w')   # [n_prev, 3, H, W]
    ref_t = trans_norm(trans_resize(ref_t))              # [n_prev, 3, h, w] in [-1,1]

    # Encode all generated frames to latents.
    gen_t = torch.from_numpy(gen_frames_np).permute(3, 0, 1, 2).float()  # [3, T, H, W]
    gen_t = trans_norm(trans_resize(rearrange(gen_t, 'c t h w -> t c h w')))  # [T, 3, h, w]
    with torch.amp.autocast('cuda'):
        z_gen_all = model.encode_first_stage(gen_t.to(device), mode=True)  # [T, 4, h_lat, w_lat]
    _, _, h_lat, w_lat = z_gen_all.shape

    # Encode reference-frame latents (once).
    ref_video = ref_t.to(device)   # [n_prev, 3, h, w]
    with torch.amp.autocast('cuda'):
        z_ref = model.encode_first_stage(ref_video, mode=True)  # [n_prev, 4, h_lat, w_lat]

    # Run the probe chunk by chunk.
    stride = getattr(args, "probe_chunk_stride", None)
    if stride is None:
        stride = max(1, chunk // 2)
    stride = int(max(1, min(stride, chunk)))
    chunk_starts = build_probe_window_starts(T_total, chunk, stride)
    n_chunks   = len(chunk_starts)
    conf_sum = None
    conf_weight = None
    conf_stack_sum = None
    hdec_embedding_parts = []
    save_hdec_embedding = bool(getattr(args, "save_hdec_embedding", False))
    noise_bank = {}
    if args.probe_ts_reduce == "single":
        probe_t_ref = args.probe_t_ref
        if probe_t_ref is None:
            probe_t_ref = int(round((args.probe_ts_min + args.probe_ts_max) / 2.0))
        ts_vals = torch.tensor([probe_t_ref], device=device, dtype=torch.long)
    else:
        ts_vals = torch.linspace(args.probe_ts_min, args.probe_ts_max,
                                 args.n_probe_ts).long().to(device)

    if getattr(model.c3_probe, "use_tau_cond", False):
        probe_tau = getattr(args, "effective_probe_tau", args.probe_tau)
        if probe_tau is None:
            default_tau = float(((model.c3_probe.thresh_low + model.c3_probe.thresh_high) * 0.5).item())
            print(f"  [probe_tau] use_tau_cond=True but --probe_tau not set, fallback to checkpoint midpoint {default_tau:.4f}")
            probe_tau = default_tau
        probe_tau = float(probe_tau)
    else:
        probe_tau = None

    for t_val in ts_vals:
        noise_bank[int(t_val.item())] = torch.randn_like(z_gen_all)

    for i_chunk, t_start in enumerate(chunk_starts):
        t_end   = min(t_start + chunk, T_total)
        t_len   = t_end - t_start

        # Generated-frame latents (padded to chunk length).
        z_chunk = z_gen_all[t_start:t_end]
        if t_len < chunk:
            z_chunk = torch.cat([z_chunk,
                                 z_chunk[-1:].repeat(chunk - t_len, 1, 1, 1)], dim=0)

        # Conditioning latents: chunk 0 uses the GT reference; chunk i > 0 uses
        # the last n_previous generated frames, so temporal context carries
        # across chunks and removes per-chunk reset artifacts.
        if i_chunk == 0:
            cond_latents = z_ref   # [n_prev, 4, h_lat, w_lat]
        else:
            ctx_end   = t_start
            ctx_start = max(0, ctx_end - n_previous)
            cond_latents = z_gen_all[ctx_start:ctx_end]   # [<=n_prev, 4, h_lat, w_lat]
            if cond_latents.shape[0] < n_previous:
                # Pad with the tail of z_ref when too short.
                pad_len  = n_previous - cond_latents.shape[0]
                cond_latents = torch.cat([z_ref[-pad_len:], cond_latents], dim=0)

        # Build the batch dict (mirrors inference() step i_chunk).
        t_total_seq = n_previous + chunk
        t_off = t_start   # start offset of the current chunk in the full sequence

        # video: reference frames (n_prev) + generated-frame placeholders (chunk).
        # get_batch_input only reads the cond_id=0 frame for img_emb, so the
        # placeholder part is filled by repeating the last reference frame.
        video_seq = torch.cat([
            ref_video,
            ref_video[-1:].repeat(chunk, 1, 1, 1)
        ], dim=0)                           # [t_total_seq, 3, h, w]
        # Back to [1, 3, 1, t_total_seq, h, w] (batch=1, channel=3, view=1).
        video_b = rearrange(video_seq, 't c h w -> 1 c 1 t h w')

        # traj: computed via model.get_traj (action / camera matrices must be
        # sliced from the current chunk's start frame). get_traj runs numpy/cv2
        # internally, so these tensors stay on CPU exactly like model.inference()
        # does; the result is moved to CUDA afterwards.
        act_sub = action[t_off : t_off + t_total_seq]          # CPU
        w2c_sub = w2c[t_off : t_off + t_total_seq].unsqueeze(0)   # [1, t, 4, 4] CPU
        c2w_sub = c2w[t_off : t_off + t_total_seq].unsqueeze(0)   # [1, t, 4, 4] CPU
        K_sub   = intrinsic.unsqueeze(0)                           # [1, 3, 3]  CPU
        # The last chunk may be shorter than t_total_seq; pad to the standard length.
        actual_t = act_sub.shape[0]
        if actual_t < t_total_seq:
            pad = t_total_seq - actual_t
            act_sub = torch.cat([act_sub, act_sub[-1:].repeat(pad, 1)], dim=0)
            w2c_sub = torch.cat([w2c_sub, w2c_sub[:, -1:].repeat(1, pad, 1, 1)], dim=1)
            c2w_sub = torch.cat([c2w_sub, c2w_sub[:, -1:].repeat(1, pad, 1, 1)], dim=1)
        traj_cv = model.get_traj(sample_size, act_sub, w2c_sub, c2w_sub, K_sub).to(device)
        # traj_cv: [3, 1, t_total_seq, h, w] in [0,1] -> normalize
        traj_cv = trans_norm(
            rearrange(traj_cv.float(), 'c v t h w -> (v t) c h w')
        )                              # [(v*t), 3, h, w] in [-1,1]
        traj_b = rearrange(traj_cv, '(v t) c h w -> 1 c v t h w',
                           v=1, t=t_total_seq)

        # delta_action for this chunk: [1, chunk, action_dim]
        da_b = delta_action[i_chunk * chunk: i_chunk * chunk + chunk].unsqueeze(0).to(device)
        if da_b.shape[1] < chunk:
            pad = da_b[:, -1:].repeat(1, chunk - da_b.shape[1], 1)
            da_b = torch.cat([da_b, pad], dim=1)

        # cond_id = -(n_previous + chunk)  ->  conditioning frame index = 0
        cond_id_b = torch.tensor([-(n_previous + chunk)], dtype=torch.int64, device=device)

        batch = {
            "video":        video_b,
            "traj":         traj_b,
            "delta_action": da_b,
            "domain_id":    domain_id,
            "fps":          fps_t,
            "intrinsic":    K_sub,
            "extrinsic":    c2w_sub,
            "caption":      [""],
            "cond_id":      cond_id_b,
        }

        # pre_z: conditioning latents (including the previous chunk's tail as
        # context) + this chunk's generated-frame latents.
        pre_z = rearrange(
            torch.cat([cond_latents, z_chunk], dim=0),
            't c h w -> 1 c t h w'           # [1, 4, t_total_seq, h_lat, w_lat]
        )

        # Call get_batch_input for cond (identical to inference).
        try:
            with torch.amp.autocast('cuda'):
                gbi_out = model.get_batch_input(
                    batch, random_uncond=False,
                    return_fs=True, return_did=True,
                    return_traj=False, return_img_emb=False,
                    pre_z=pre_z,
                )
                z_b   = gbi_out[0].to(device)
                cond  = _to_device(gbi_out[1], device)
                fs_b  = gbi_out[2].to(device)
                did_b = gbi_out[3].to(device)
        except Exception as e:
            print(f"  [probe] get_batch_input failed for chunk {i_chunk}: {e}")
            continue

        # z_b: [b*v=1, 4, t_total_seq, h_lat, w_lat]
        x0_chunk = z_b[:, :, n_previous:]   # [1, 4, chunk, h_lat, w_lat]  -- clean generated latents
        unet_kwargs = {"fs": fs_b.long(), "domain_id": did_b.long()}

        probe_confs_chunk = []
        for t_val in ts_vals:
            t_bv = t_val.expand(x0_chunk.shape[0])  # [b*v=1]

            # Dynamic rescaling (same as p_losses forward; scale varies with t).
            if model.use_dynamic_rescale:
                scale = model.scale_arr.to(device)[t_val.item()].item()
                x0_scaled = x0_chunk * scale
            else:
                x0_scaled = x0_chunk

            noise_full = noise_bank[int(t_val.item())][t_start:t_end]
            if t_len < chunk:
                noise_full = torch.cat([noise_full, noise_full[-1:].repeat(chunk - t_len, 1, 1, 1)], dim=0)
            noise = rearrange(noise_full, 't c h w -> 1 c t h w').to(device=device, dtype=x0_scaled.dtype)
            x_noisy = model.q_sample(x0_scaled, t_bv, noise=noise)  # [1, 4, chunk, h, w]

            # Reference frames stay clean; only the chunk part is noised
            # (same as train p_losses).
            x_in = torch.cat([z_b[:, :, :n_previous], x_noisy], dim=2)
            # x_in: [b*v=1, 4, t_total_seq, h_lat, w_lat]

            try:
                with torch.amp.autocast('cuda'):
                    unet_out = model.model(
                        x_in, t_bv,
                        return_probe_feat=True,
                        **cond, **unet_kwargs,
                    )
                    _, (h_mid, h_dec, emb) = unet_out

                if save_hdec_embedding:
                    hdec_emb = _hdec_to_trajectory_embedding(h_dec, n_previous)
                    if hdec_emb is not None:
                        hdec_embedding_parts.append(hdec_emb)

                # Probe output: [B=1, T=t_total_seq, h, w]; keep the chunk part.
                # Note: autocast returns float16 intermediate features while the
                # probe parameters are float32; some PyTorch versions raise a
                # device error inside TransformerEncoder under mixed precision,
                # so cast explicitly to float32 before calling the probe.
                if getattr(model.c3_probe, "feat_mode", None) == "dec_only":
                    # Unified decoder path: feat_mode='dec_only' consumes only
                    # h_dec; if use_emb_cond=True, emb is passed as well.
                    tau_seq = None
                    if probe_tau is not None:
                        tau_seq = torch.full((1, t_total_seq), probe_tau, device=device, dtype=h_dec.dtype)
                    conf_all = torch.sigmoid(
                        model.c3_probe(
                            h_dec.float().to(device),
                            T=t_total_seq,
                            emb=emb.float().to(device),
                            tau=tau_seq,
                        )
                    )
                else:
                    # Other paths still use the two-branch feature interface.
                    tau_seq = None
                    if probe_tau is not None:
                        tau_seq = torch.full((1, t_total_seq), probe_tau, device=device, dtype=h_dec.dtype)
                    conf_all = torch.sigmoid(
                        model.c3_probe(
                            h_dec.float().to(device),
                            T=t_total_seq,
                            h_mid=h_mid.float().to(device),
                            emb=emb.float().to(device),
                            tau=tau_seq,
                        )
                    )
                conf_chunk = conf_all[0, n_previous:].cpu().float()  # [chunk, 10, 16]
                probe_confs_chunk.append(conf_chunk)
            except Exception as e:
                print(f"  [probe] failed at t={t_val.item()}: {e}")
                continue

        if probe_confs_chunk:
            conf_stack_chunk = torch.stack(probe_confs_chunk).cpu().float().numpy()[:, :t_len]  # [n_ts, t_len, h, w]
            avg_chunk = reduce_probe_conf_stack(conf_stack_chunk, args.probe_ts_reduce)
        else:
            conf_stack_chunk = np.full((len(ts_vals), t_len, 10, 16), 0.5, dtype=np.float32)
            avg_chunk = np.full((t_len, 10, 16), 0.5, dtype=np.float32)

        if conf_sum is None:
            _, h_probe, w_probe = avg_chunk.shape
            conf_sum = np.zeros((T_total, h_probe, w_probe), dtype=np.float32)
            conf_weight = np.zeros((T_total, 1, 1), dtype=np.float32)
            if args.save_probe_ts_stack:
                conf_stack_sum = np.zeros((len(ts_vals), T_total, h_probe, w_probe), dtype=np.float32)

        w = build_probe_window_weights(t_len, chunk=chunk, stride=stride)
        conf_sum[t_start:t_end] += avg_chunk * w
        conf_weight[t_start:t_end] += w
        if conf_stack_sum is not None:
            conf_stack_sum[:, t_start:t_end] += conf_stack_chunk * w[None, ...]

    if conf_sum is None:
        conf_map = np.full((T_total, 10, 16), 0.5, dtype=np.float32)
        conf_stack = None if conf_stack_sum is None else np.full((len(ts_vals), T_total, 10, 16), 0.5, dtype=np.float32)
        if save_hdec_embedding:
            return conf_map, conf_stack, z_gen_all.detach().cpu().float().numpy(), None
        return conf_map, conf_stack, z_gen_all.detach().cpu().float().numpy()

    conf_map = conf_sum / np.clip(conf_weight, 1e-6, None)
    conf_stack = None
    if conf_stack_sum is not None:
        conf_stack = conf_stack_sum / np.clip(conf_weight[None, ...], 1e-6, None)
    if save_hdec_embedding:
        hdec_embedding = None
        if hdec_embedding_parts:
            hdec_embedding = np.stack(hdec_embedding_parts, axis=0).mean(axis=0).astype(np.float32)
        return conf_map, conf_stack, z_gen_all.detach().cpu().float().numpy(), hdec_embedding
    return conf_map, conf_stack, z_gen_all.detach().cpu().float().numpy()
