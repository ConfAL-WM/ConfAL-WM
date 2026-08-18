"""
C3 Confidence Probe - v2 (Unified Version)
====================================
Unified implementation merging c3_probe_pixel.py and c3_probe_decoder.py.

Core Parameters:
  feat_mode      : "dec_only" | "mid_only" | "mid_dec"
                   Controls UNet feature source and fusion mode
                     dec_only → Use only h_dec from specified output_block
                     mid_only → Use only bottleneck h_mid (pass h_dec parameter)
                     mid_dec  → Concatenate upsampled h_mid + h_dec (original pixel version approach)

  target_space   : "pixel" | "latent"
                   Flag for trainer reading only; probe itself is agnostic.
                   compute_target_confidence accepts pred/target tensors from any space.

  threshold_mode : "ema_warmup" | "fixed"
    ema_warmup:
      If threshold_warmup_steps is not None:for the first threshold_warmup_steps steps, use
        EMA (momentum 0.98) to rapidly update thresh_low/thresh_high, then freeze permanently.
      If threshold_warmup_steps is None (not specified):fine-tune throughout, no freezing.
        EMA rate coupled with LR warmup:
          ema_step < lr_warmup_steps  → alpha=0.20(fast convergence to true quantile)
          ema_step >= lr_warmup_steps → alpha=0.002(small steps for continuous fine-tuning)
        lr_warmup_steps passed via constructor (usually same as --warmup_steps).
      Important: initialize threshold_low/high to the expected MAE distribution center
        (not extremely small values) to avoid cold-start issues where all labels become 0!
          pixel space -> suggested low=0.15, high=0.35
          latent space -> suggested low=0.20, high=0.70
    fixed:      Directly use the provided threshold_low/threshold_high without EMA updates.
                Suitable for situations with known data distributions.

In both modes: threshold uniformly samples for each (B,T) sample within [thresh_low, thresh_high]
independent uniform sampling -> preserves continuous CDF learning characteristics (label_mean automatically stabilizes around 0.5).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


# =============================================================================
# Transformer Layers
# =============================================================================

class TransformerLayer(nn.Module):
    """Pre-Norm Transformer layer (no timestep condition)."""

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 1024,dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,dropout=dropout,
                                               batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                _c: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                     attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int, n_modulations: int = 2):
        super().__init__()
        self.n_modulations = n_modulations
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, n_modulations * dim, bias=True),
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.modulation(c).chunk(self.n_modulations, dim=-1)
        return self.norm(x) * (1 + params[0]) + params[1]


class AdaLNTransformerLayer(nn.Module):
    """AdaLN-conditioned Transformer layer (used when use_emb_cond=True)."""

    def __init__(self, d_model: int, nhead: int, cond_dim: int,
                 dim_feedforward: int = 1024,dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,dropout=dropout,
                                               batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.adaLN_attn = AdaLN(d_model, cond_dim)
        self.adaLN_ffn  = AdaLN(d_model, cond_dim)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        if c is None:
            raise ValueError("AdaLNTransformerLayer requires condition c")
        if c.dim() == 2:
            c = c.unsqueeze(1).expand(-1, T, -1)
        elif c.dim() == 3 and c.shape[1] == 1:
            c = c.expand(-1, T, -1)
        x_flat = x.reshape(B * T, D)
        c_flat = c.reshape(B * T, -1)
        x = x + self.self_attn(
            self.adaLN_attn(x_flat, c_flat).view(B, T, D),
            self.adaLN_attn(x_flat, c_flat).view(B, T, D),
            self.adaLN_attn(x_flat, c_flat).view(B, T, D),
            attn_mask=mask,
        )[0]
        x_flat = x.reshape(B * T, D)
        x = x + self.ffn(self.adaLN_ffn(x_flat, c_flat).view(B, T, D))
        return x


# =============================================================================
# Main Probe class
# =============================================================================

class C3ConfidenceProbe(nn.Module):
    """
    C3 Confidence Probe v2 -- unified version.

    Usage example:
        # dec_only + latent space + fixed threshold
        probe = C3ConfidenceProbe(
            feat_mode="dec_only", dec_channels=1280,
            target_space="latent",
            threshold_mode="fixed", threshold_low=0.20, threshold_high=0.70)

        # mid_dec + pixel space + EMA warmup (freeze after 2000 steps)
        probe = C3ConfidenceProbe(
            feat_mode="mid_dec", mid_channels=1280, dec_channels=1280,
            target_space="pixel",
            threshold_mode="ema_warmup", threshold_warmup_steps=2000,
            threshold_low=0.15, threshold_high=0.35)
    """

    def __init__(
        self,
        # -- Feature input --------------------------------------------------
        feat_mode:     str   = "dec_only",  # "dec_only" | "mid_only" | "mid_dec"
        mid_channels:  int   = 1280,        # h_mid channel count(5×8 bottleneck)
        dec_channels:  int   = 1280,        # h_dec channel count(depends on probe_block_idx)
        # -- Architecture ----------------------------------------------------
        emb_dim:           int   = 1280,
        probe_dim:         int   = 256,
        n_heads:           int   = 4,
        n_spatial_layers:  int   = 3,
        n_temporal_layers: int   = 3,
        max_h:             int   = 64,
        max_w:             int   = 64,
        max_T:             int   = 32,
       dropout:           float = 0.1,
        use_emb_cond:      bool  = False,
        use_tau_cond:      bool  = False,
        # -- Target space (flag for the trainer only) -----------------------------
        target_space:  str   = "pixel",   # "pixel" | "latent"
        # -- Threshold configuration --------------------------------------
        threshold_mode:         str            = "ema_warmup",  # "ema_warmup" | "fixed"
        threshold_warmup_steps: Optional[int]  = None,
        # None -> full fine-tuning (no freezing); EMA rate couples with lr_warmup_steps
        # int  -> fast EMA for the first N steps, then freeze
        lr_warmup_steps:        Optional[int]  = None,
        # Only used when threshold_warmup_steps=None: LR warmup step count,
        # determines the fast/slow EMA switch point
        threshold_low:          float = 0.15,  # initial/frozen lower bound: should sit near data MAE p10
        threshold_high:         float = 0.35,  # initial/frozen upper bound: should sit near data MAE p90
        ema_p_low:              float = 0.10,
        ema_p_high:             float = 0.90,
    ):
        super().__init__()

        assert feat_mode in ("dec_only", "mid_only", "mid_dec"), \
            f"feat_mode must be one of: dec_only, mid_only, mid_dec. Got: {feat_mode}"
        assert target_space  in ("pixel", "latent"),  f"Unknown target_space: {target_space}"
        assert threshold_mode in ("ema_warmup", "fixed"), \
            f"threshold_mode must be 'ema_warmup' or 'fixed'. Got: {threshold_mode}"

        # -- Store config --------------------------------------------------
        self.feat_mode               = feat_mode
        self.target_space            = target_space
        self.probe_dim               = probe_dim
        self.use_emb_cond            = use_emb_cond
        self.use_tau_cond            = use_tau_cond
        self.use_any_cond            = use_emb_cond or use_tau_cond
        self.threshold_mode          = threshold_mode
        self.threshold_warmup_steps  = threshold_warmup_steps   # None = never freeze
        self.ema_p_low               = float(ema_p_low)
        self.ema_p_high              = float(ema_p_high)
        assert 0.0 < self.ema_p_low < self.ema_p_high < 1.0, \
            f"Require 0 < ema_p_low < ema_p_high < 1, got ({self.ema_p_low}, {self.ema_p_high})"

        # -- Threshold buffers ---------------------------------------------
        # thresh_low / thresh_high: the active threshold range (persisted in checkpoints)
        # ema_step: completed training-step counter (used to decide the freeze point / LR switch)
        # lr_warmup_steps_buf: LR warmup step count (-1 means unset)
        self.register_buffer('thresh_low',         torch.tensor(threshold_low,  dtype=torch.float32))
        self.register_buffer('thresh_high',        torch.tensor(threshold_high, dtype=torch.float32))
        self.register_buffer('ema_step',           torch.tensor(0,              dtype=torch.long))
        self.register_buffer('lr_warmup_steps_buf',
                             torch.tensor(lr_warmup_steps if lr_warmup_steps is not None else -1,
                                          dtype=torch.long))

        # -- Feature input channels ---------------------------------------
        if feat_mode == "dec_only":
            in_channels = dec_channels
        elif feat_mode == "mid_only":
            in_channels = mid_channels
        else:  # mid_dec
            in_channels = mid_channels + dec_channels

        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, probe_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, probe_dim),
            nn.SiLU(),
        )

        # -- Positional encoding ------------------------------------------
        self.spatial_pe  = nn.Parameter(torch.zeros(1, probe_dim, max_h, max_w))
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_T, probe_dim))
        nn.init.trunc_normal_(self.spatial_pe,  std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        # -- Condition projection (optional) -----------------------------
        if self.use_any_cond:
            if use_emb_cond:
                self.cond_proj = nn.Sequential(
                    nn.Linear(emb_dim, probe_dim), nn.SiLU(),
                    nn.Linear(probe_dim, probe_dim),
                )
            else:
                self.cond_proj = None
            if use_tau_cond:
                self.tau_proj = nn.Sequential(
                    nn.Linear(1, probe_dim), nn.SiLU(),
                    nn.Linear(probe_dim, probe_dim),
                )
            else:
                self.tau_proj = None
            cond_dim = probe_dim
            make_sp  = lambda: AdaLNTransformerLayer(probe_dim, n_heads, cond_dim, probe_dim * 4,dropout)
            make_tmp = lambda: AdaLNTransformerLayer(probe_dim, n_heads, cond_dim, probe_dim * 4,dropout)
        else:
            self.cond_proj = None
            self.tau_proj = None
            make_sp  = lambda: TransformerLayer(probe_dim, n_heads, probe_dim * 4,dropout)
            make_tmp = lambda: TransformerLayer(probe_dim, n_heads, probe_dim * 4,dropout)

        # -- Transformers -------------------------------------------------
        self.spatial_layers  = nn.ModuleList([make_sp()  for _ in range(n_spatial_layers)])
        self.temporal_layers = nn.ModuleList([make_tmp() for _ in range(n_temporal_layers)])

        # -- Output head --------------------------------------------------
        self.out_proj = nn.Sequential(
            nn.LayerNorm(probe_dim),
            nn.Linear(probe_dim, 1),
        )
        nn.init.normal_(self.out_proj[-1].weight, std=0.02)
        nn.init.constant_(self.out_proj[-1].bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h_dec: torch.Tensor,              # [(B×T), dec_channels, h_f, w_f]  -- main feature(required)
        T:     int,
        h_mid: Optional[torch.Tensor] = None,  # [(B×T), mid_channels, 5, 8]  -- only needed by mid_dec
        emb:   Optional[torch.Tensor] = None,  # [(B×T), emb_dim]  -- only needed when use_emb_cond=True
        tau:   Optional[Union[float, torch.Tensor]] = None,  # scalar / [B] / [B,T]
    ) -> torch.Tensor:
        """
        return logits [B, T, h_f, w_f](unprocessed sigmoid).

        feat_mode call convention:
          dec_only : forward(h_dec=h_dec, T=T)
          mid_only : forward(h_dec=h_mid, T=T)          <- h_mid passed as the main feature
          mid_dec  : forward(h_dec=h_dec, T=T, h_mid=h_mid)
        """
        BT, _, h_f, w_f = h_dec.shape
        B = BT // T

        # 1. feature fusion
        if self.feat_mode == "mid_dec":
            assert h_mid is not None, "h_mid is required when feat_mode='mid_dec'"
            h_mid_up = F.interpolate(h_mid, size=(h_f, w_f),
                                     mode="bilinear", align_corners=False)
            fused = torch.cat([h_mid_up, h_dec], dim=1)
        else:
            fused = h_dec   # dec_only / mid_only (the caller passes the appropriate feature)

        x = self.feat_proj(fused)                          # [BT, probe_dim, h_f, w_f]

        # 2. spatial positional encoding
        x = x + self.spatial_pe[:, :, :h_f, :w_f]

        # 3. condition vectors
        cond_sp = cond_tmp = None
        cond = None
        if self.use_emb_cond:
            if emb is None:
                raise ValueError("use_emb_cond=True requires passing emb")
            cond = self.cond_proj(emb).view(B, T, self.probe_dim)
        if self.use_tau_cond:
            tau_bt = self._broadcast_tau(tau, B=B, T=T, device=h_dec.device, dtype=h_dec.dtype)
            tau_cond = self.tau_proj(tau_bt.unsqueeze(-1))
            cond = tau_cond if cond is None else (cond + tau_cond)
        if self.use_any_cond:
            if cond is None:
                raise ValueError("probe condition branch enabled but no emb/tau condition was provided")
            cond_sp  = cond.reshape(BT, self.probe_dim).unsqueeze(1)
            cond_tmp = (cond.unsqueeze(1).expand(-1, h_f * w_f, -1, -1)
                        .reshape(B * h_f * w_f, T, self.probe_dim))

        # 4. spatial Transformer
        x = x.permute(0, 2, 3, 1).reshape(BT, h_f * w_f, self.probe_dim)
        for layer in self.spatial_layers:
            x = layer(x, cond_sp)

        # 5. temporal Transformer
        x = x.reshape(B, T, h_f * w_f, self.probe_dim)
        x = x.permute(0, 2, 1, 3).reshape(B * h_f * w_f, T, self.probe_dim)
        x = x + self.temporal_pe[:, :T, :]
        for layer in self.temporal_layers:
            x = layer(x, cond_tmp)

        # 6. output
        x = x.reshape(B, h_f * w_f, T, self.probe_dim)
        x = x.permute(0, 2, 1, 3).reshape(BT, h_f * w_f, self.probe_dim)
        logits = self.out_proj(x).squeeze(-1).reshape(B, T, h_f, w_f)
        return logits

    # ------------------------------------------------------------------
    # Confidence target generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_target_confidence(
        self,
        pred:    torch.Tensor,  # [B, C, T, H, W]  -- predicted values (pixels or latents)
        target:  torch.Tensor,  # [B, C, T, H, W]  -- ground truth
        probe_h: int,
        probe_w: int,
        return_threshold: bool = False,
    ) -> Union[Tuple[torch.Tensor, dict], Tuple[torch.Tensor, dict, torch.Tensor]]:
        """
        Compute the downsampled MAE and derive binary labels from a random threshold drawn inside [thresh_low, thresh_high].

        Threshold update policy (threshold_mode):
          "ema_warmup": update with momentum 0.98 before threshold_warmup_steps
                        (fast convergence), then freeze permanently. Initializing
                        to the center of the expected MAE distribution completely
                        avoids the cold-start (all-zero labels) problem.
          "fixed":      thresh_low / thresh_high stay frozen; no EMA.

        Continuous labels (CDF learning):
          each (B,T) sample draws an independent threshold from
          U(thresh_low, thresh_high), so
          E[label=1 | mae] = (thresh_high - mae) / (thresh_high - thresh_low):
          low MAE (accurate frames) -> labels biased to 1;
          high MAE (blurry frames) -> labels biased to 0.
        """
        B, C, T, H, W = pred.shape
        mae_map = (pred - target).abs().mean(dim=1)         # [B, T, H, W]
        mae_down = F.adaptive_avg_pool2d(
            mae_map.reshape(B * T, 1, H, W),
            output_size=(probe_h, probe_w),
        ).squeeze(1).reshape(B, T, probe_h, probe_w)
        mae_flat = mae_down.detach().flatten()

        current_p_low = torch.quantile(mae_flat, self.ema_p_low)
        current_p_high = torch.quantile(mae_flat, self.ema_p_high)

        # -- Threshold update ----------------------------------------------------
        if self.threshold_mode == "ema_warmup" and self.training:
            frozen = (self.threshold_warmup_steps is not None and
                      self.ema_step >= self.threshold_warmup_steps)
            if not frozen:
                if self.threshold_warmup_steps is None:
                    # full fine-tuning mode: EMA rate couples with the LR warmup
                    #   during LR warmup (ema_step < lr_warmup_steps) -> alpha=0.20 (fast)
                    #   after LR warmup                                -> alpha=0.002 (slow)
                    lr_ws = self.lr_warmup_steps_buf.item()
                    if lr_ws > 0 and self.ema_step < lr_ws:
                        alpha = 0.20   # fast convergence to the true quantile
                    else:
                        alpha = 0.002  # small steps, continuous fine-tuning
                else:
                    # fixed-warmup-steps mode: fast EMA (momentum 0.98), stops after reaching the freeze point
                    alpha = 0.02
                self.thresh_low.mul_(1.0 - alpha).add_(current_p_low * alpha)
                self.thresh_high.mul_(1.0 - alpha).add_(current_p_high * alpha)
            self.ema_step.add_(1)
        # threshold_mode == "fixed": never updated

        t_low  = self.thresh_low.item()
        t_high = max(self.thresh_high.item(), t_low + 1e-3)

        # -- sample one random threshold per (B,T) sample -------------------------
        threshold = torch.rand(B, T, 1, 1, device=mae_down.device) * (t_high - t_low) + t_low
        target_conf = (mae_down < threshold).float()

        stats = {
            "mae_mean":    mae_down.mean().item(),
            "mae_p_low":   current_p_low.item(),
            "mae_p50":     torch.quantile(mae_flat, 0.50).item(),
            "mae_p_high":  current_p_high.item(),
            "ema_p_low":   self.ema_p_low,
            "ema_p_high":  self.ema_p_high,
            "thresh_low":  t_low,
            "thresh_high": t_high,
            "ema_step":    self.ema_step.item(),
            "label_mean":  target_conf.mean().item(),
            "label_std":   target_conf.std().item(),
        }
        if return_threshold:
            return target_conf, stats, threshold.squeeze(-1).squeeze(-1)
        return target_conf, stats

    def compute_loss(
        self,
        logits:        torch.Tensor,
        target_conf:   torch.Tensor,
        n_cond_frames: int = 0,
    ) -> torch.Tensor:
        if n_cond_frames > 0:
            logits      = logits[:, n_cond_frames:]
            target_conf = target_conf[:, n_cond_frames:]
        return F.binary_cross_entropy_with_logits(logits, target_conf)

    def _broadcast_tau(
        self,
        tau: Optional[Union[float, torch.Tensor]],
        B: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tau is None:
            raise ValueError("use_tau_cond=True requires passing tau explicitly")
        if not torch.is_tensor(tau):
            tau = torch.tensor(float(tau), device=device, dtype=dtype)
        else:
            tau = tau.to(device=device, dtype=dtype)

        if tau.dim() == 0:
            return tau.view(1, 1).expand(B, T)
        if tau.dim() == 1:
            if tau.shape[0] == 1:
                return tau.view(1, 1).expand(B, T)
            if tau.shape[0] == B:
                return tau.view(B, 1).expand(B, T)
            if tau.shape[0] == T:
                return tau.view(1, T).expand(B, T)
            if tau.shape[0] == B * T:
                return tau.view(B, T)
            raise ValueError(f"Cannot broadcast tau with shape={tuple(tau.shape)}, expected size 1 / B / T / B*T")
        if tau.dim() == 2:
            if tau.shape == (B, T):
                return tau
            if tau.shape == (B, 1):
                return tau.expand(B, T)
            if tau.shape == (1, T):
                return tau.expand(B, T)
            if tau.shape == (1, 1):
                return tau.expand(B, T)
            raise ValueError(f"Cannot broadcast tau with shape={tuple(tau.shape)}, expected [B,T] / [B,1] / [1,T]")
        if tau.dim() == 4 and tau.shape[-2:] == (1, 1):
            return self._broadcast_tau(tau.squeeze(-1).squeeze(-1), B=B, T=T, device=device, dtype=dtype)
        raise ValueError(f"Unsupported tau dimension: shape={tuple(tau.shape)}")

    # ------------------------------------------------------------------
    # Helper: recover x0_pred from v_pred + z_t (latent space)
    # ------------------------------------------------------------------

    @staticmethod
    def v_pred_to_x0(
        z_t:                  torch.Tensor,
        v_pred:               torch.Tensor,
        sqrt_alpha:           torch.Tensor,
        sqrt_one_minus_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """x0_pred = sqrt_alpha_t * z_t - sqrt_one_minus_alpha_t * v_pred"""
        def _expand(coef, target):
            while coef.dim() < target.dim():
                coef = coef.unsqueeze(-1)
            return coef
        return _expand(sqrt_alpha, z_t) * z_t - _expand(sqrt_one_minus_alpha, z_t) * v_pred


__all__ = [
    "C3ConfidenceProbe",
    "TransformerLayer",
    "AdaLN",
    "AdaLNTransformerLayer",
]
