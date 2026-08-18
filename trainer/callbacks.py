import json
import os
import subprocess
import sys
import time
import csv
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
import numpy as np
from pathlib import Path

from save_video import log_local, prepare_to_log


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images=8, clamp=True, rescale=True, save_dir=None, \
                to_local=False, log_images_kwargs=None, val_batch_frequency=6, point_cloud_scale=1.0, save_point_cloud=False, point_cloud_save_dir=None, cat_all_to_save=False, save_fps=10):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.to_local = to_local
        self.clamp = clamp
        self.val_batch_frequency = val_batch_frequency
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.point_cloud_scale = point_cloud_scale
        self.save_point_cloud = save_point_cloud
        if self.save_point_cloud:
            assert (point_cloud_save_dir is not None)
            self.point_cloud_save_dir = point_cloud_save_dir
            if not os.path.exists(self.point_cloud_save_dir):
                os.mkdir(self.point_cloud_save_dir)

        if self.to_local:
            ## default save dir
            self.save_dir = os.path.join(save_dir, "images")
            os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "val"), exist_ok=True)
            self.cat_all_to_save = cat_all_to_save
            self.save_fps = save_fps

    def log_to_tensorboard(self, pl_module, batch_logs, filename, split, save_fps=2):
        """ log images and videos to tensorboard """
        global_step = pl_module.global_step
        for key in batch_logs:
            value = batch_logs[key]
            tag = "gs%d-%s/%s-%s"%(global_step, split, filename, key)
            if isinstance(value, list) and isinstance(value[0], str):
                captions = ' |------| '.join(value)
                pl_module.logger.experiment.add_text(tag, captions, global_step=global_step)

            elif isinstance(value, torch.Tensor) and value.dim() == 5:
                video = value
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, n*h, 1*w]
                grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                pl_module.logger.experiment.add_video(tag, grid, fps=save_fps, global_step=global_step)

            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                img = value
                grid = torchvision.utils.make_grid(img, nrow=1, padding=0)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                pl_module.logger.experiment.add_image(tag, grid, global_step=global_step)

            elif isinstance(value, torch.Tensor) and value.dim() == 3:
                assert (value.shape[1]%2==0)
                timestep = value.shape[1]//2
                colors = torch.zeros((value.shape[0],value.shape[1],3))
                colors[0, :timestep, 1] = torch.linspace(0, 1, timestep)
                colors[0, timestep:, 0] = torch.linspace(0, 1, timestep)

                if self.save_point_cloud:
                    np.savetxt(os.path.join(self.point_cloud_save_dir, tag.replace("/", "_")+'point_cloud_pred.txt'), value.cpu().numpy().reshape(-1, 3)[:timestep], fmt='%f')
                    np.savetxt(os.path.join(self.point_cloud_save_dir, tag.replace("/", "_")+'point_cloud_gt.txt'), value.cpu().numpy().reshape(-1, 3)[timestep:], fmt='%f')

                pl_module.logger.experiment.add_mesh(tag, vertices=value*self.point_cloud_scale, colors=(colors*255).to(torch.int8), global_step=global_step)
            else:
                pass

    @rank_zero_only
    def log_batch_imgs(self, pl_module, batch, batch_idx, split="train"):
        """ generate images, then save and log to tensorboard """
        skip_freq = self.batch_freq if split == "train" else self.val_batch_frequency
        criterion = pl_module.global_step if split == "train" else batch_idx
        # TODO: here directly modified to global step
        if (criterion+1) % skip_freq == 0:
            is_train = pl_module.training
            # if is_train:
            #     pl_module.eval()
            pl_module.eval()
            # torch.cuda.empty_cache()
            with torch.no_grad():
                log_func = pl_module.log_images
                batch_logs = log_func(
                    batch, split=split,
                    cat_v_to_w=not (self.to_local and self.cat_all_to_save),
                    **self.log_images_kwargs
                )

            ## process: move to CPU and clamp
            batch_logs = prepare_to_log(batch_logs, self.max_images, self.clamp)
            # torch.cuda.empty_cache()

            filename = "ep{}_idx{}_rank{}".format(
                pl_module.current_epoch,
                batch_idx,
                pl_module.global_rank)

            if self.to_local:
                mainlogger.info("Log [%s] batch <%s> to local ..."%(split, filename))
                filename = "gs{}_".format(pl_module.global_step) + filename

                if self.cat_all_to_save:
                    all_videos = []
                    batch_list = list(batch_logs.keys())

                    t_max = 0
                    for k in batch_list:
                        if isinstance(batch_logs[k], torch.Tensor):
                            if batch_logs[k].dim() == 5 and batch_logs[k].shape[2]>1:
                                t = batch_logs[k].shape[2]
                                t_max = max(t, t_max)
                    for k in batch_list:
                        if isinstance(batch_logs[k], torch.Tensor):
                            ### v, c, t, h, w
                            if batch_logs[k].dim() == 5 and batch_logs[k].shape[2]>1:
                                v = batch_logs.pop(k)
                                t = v.shape[2]
                                if t < t_max:
                                    v = torch.cat((v, torch.zeros(v.shape[0], v.shape[1], t_max-t, v.shape[3], v.shape[4])), dim=2)
                                all_videos.append(v)
                    all_videos = torch.cat(all_videos, dim=-1)
                    batch_logs.update({"ALL": all_videos})
                log_local(batch_logs, os.path.join(self.save_dir, split), filename, save_fps=self.save_fps)

            else:
                mainlogger.info("Log [%s] batch <%s> to tensorboard ..."%(split, filename))
                self.log_to_tensorboard(pl_module, batch_logs, filename, split, save_fps=self.save_fps)

            mainlogger.info('Finish!')

            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class PeriodicCheckpointCallback(Callback):
    """Save periodic checkpoints during training for later post-training eval.

    Rank 0 saves an epoch-step checkpoint on a schedule (starting at
    ``start_fraction`` of ``max_steps``, every ``interval_fraction``) directly
    into ``{checkpoint_dir}/`` (``logs/checkpoints/``), unified with the rest of
    the run's checkpoints — no ``every_eval_with_ewmbench/`` subfolder. When a
    ``loss_best_monitor`` is configured, the best-loss checkpoint is marked by
    renaming its file to ``epoch=...-step=...(best_loss).ckpt``. The exact
    final-step checkpoint is guaranteed at train end.

    Evaluation is intentionally NOT launched here. The previous design ran
    EWMBench concurrently with training, but those subprocesses inherited
    torchrun's ``WORLD_SIZE`` and deadlocked at distributed init (600s timeout).
    ``train_evac_with_al.py`` now launches
    ``eval/al_results/eval_periodic_checkpoints.py`` automatically after training
    (when ``--save_every_with_ewmbench`` is set); it reuses the checkpoints saved here.
    """

    def __init__(
        self,
        output_dir,
        checkpoint_dir,
        include_ewmbench=False,
        start_fraction=0.75,
        interval_fraction=0.025,
        max_eval_episodes=50,
        metrics="pixel_mae,latent_loss,risk_reduction",
        loss_best_monitor=None,
        loss_best_mode="min",
        # Accepted for backward config-compat; no longer used for in-training eval.
        project_root=None,
        score_method=None,
        select_method=None,
        weighting=None,
        gpus=None,
        use_ewmbench_evaluate_py=True,
        wait_on_train_end=True,
        pipeline_config=None,
        val_manifest=None,
        python_executable=None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.include_ewmbench = bool(include_ewmbench)
        self.start_fraction = float(start_fraction)
        self.interval_fraction = float(interval_fraction)
        self.max_eval_episodes = int(max_eval_episodes)
        self.metrics = str(metrics)
        self.loss_best_monitor = str(loss_best_monitor).strip() if loss_best_monitor else None
        self.loss_best_mode = str(loss_best_mode or "min").lower()
        if self.loss_best_mode not in {"min", "max"}:
            raise ValueError("loss_best_mode must be 'min' or 'max'")

        # Periodic checkpoints (incl. the final step) are saved directly into the
        # base checkpoint_dir (logs/checkpoints/), unified with the rest of the
        # run — no every_eval_with_ewmbench/ subfolder. Post-training EWMBench
        # eval is launched by train_evac_with_al.py, not from this callback.
        self.eval_ckpt_dir = self.checkpoint_dir
        self.best_loss_meta_path = self.checkpoint_dir / "best_loss_checkpoint.json"

        self.queued_steps = set()
        self.next_eval_step = None
        self.step_to_ckpt_path = {}
        self.best_loss_step = None
        self.best_loss_score = None
        self.deferred_unmark_steps = set()

    def _rank_zero(self, trainer):
        return bool(getattr(trainer, "is_global_zero", False))

    def _schedule(self, trainer):
        max_steps = int(getattr(trainer, "max_steps", -1) or -1)
        if max_steps <= 0:
            max_steps = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        if max_steps <= 0:
            return None
        start = max(1, int(round(max_steps * self.start_fraction)))
        interval = max(1, int(round(max_steps * self.interval_fraction)))
        return max_steps, start, interval

    def _plain_ckpt_path(self, epoch, step):
        return self.eval_ckpt_dir / f"epoch={int(epoch)}-step={int(step)}.ckpt"

    def _marked_ckpt_path(self, path):
        path = Path(path)
        if path.name.endswith("(best_loss).ckpt"):
            return path
        return path.with_name(path.name[:-5] + "(best_loss).ckpt") if path.name.endswith(".ckpt") else path

    def _unmarked_ckpt_path(self, path):
        path = Path(path)
        return path.with_name(path.name.replace("(best_loss)", ""))

    def _save_eval_checkpoint(self, trainer, step, epoch):
        self.eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._plain_ckpt_path(epoch, step)
        marked_path = self._marked_ckpt_path(ckpt_path)
        if marked_path.exists():
            self.step_to_ckpt_path[int(step)] = str(marked_path)
            return marked_path
        if ckpt_path.exists():
            self.step_to_ckpt_path[int(step)] = str(ckpt_path)
            return ckpt_path
        mainlogger.info(f"[every_eval] saving checkpoint at step={step}: {ckpt_path}")
        try:
            trainer.save_checkpoint(str(ckpt_path), weights_only=True)
        except TypeError:
            trainer.save_checkpoint(str(ckpt_path))
        self.step_to_ckpt_path[int(step)] = str(ckpt_path)
        return ckpt_path

    def _metric_candidates(self):
        if not self.loss_best_monitor:
            return []
        keys = [self.loss_best_monitor]
        if self.loss_best_monitor.endswith("_epoch"):
            keys.append(self.loss_best_monitor[: -len("_epoch")] + "_step")
        return keys

    def _metric_value(self, trainer):
        for source_name in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
            source = getattr(trainer, source_name, None)
            if not source:
                continue
            for key in self._metric_candidates():
                if key not in source:
                    continue
                value = source[key]
                try:
                    if hasattr(value, "detach"):
                        value = value.detach()
                    if hasattr(value, "item"):
                        value = value.item()
                    return float(value), key
                except Exception:
                    continue
        return None, None

    def _is_better_loss(self, value):
        if value is None:
            return False
        if self.best_loss_score is None:
            return True
        if self.loss_best_mode == "max":
            return float(value) > float(self.best_loss_score)
        return float(value) < float(self.best_loss_score)

    def _step_in_flight(self, step):
        return int(step) in self.queued_steps or int(step) == self.best_loss_step

    def _record_checkpoint_rename(self, step, path):
        step = int(step)
        self.step_to_ckpt_path[step] = str(path)

    def _rename_checkpoint(self, step, src, dst):
        src = Path(src)
        dst = Path(dst)
        if src == dst:
            self._record_checkpoint_rename(step, dst)
            return dst
        if not src.exists():
            if dst.exists():
                self._record_checkpoint_rename(step, dst)
                return dst
            return src
        if dst.exists():
            src.unlink()
        else:
            src.rename(dst)
        self._record_checkpoint_rename(step, dst)
        return dst

    def _unmark_loss_step(self, step):
        if step is None:
            return
        step = int(step)
        path = Path(self.step_to_ckpt_path.get(step, ""))
        if not path.name.endswith("(best_loss).ckpt"):
            return
        self._rename_checkpoint(step, path, self._unmarked_ckpt_path(path))

    def _apply_deferred_unmarks(self):
        for step in list(self.deferred_unmark_steps):
            if step == self.best_loss_step or self._step_in_flight(step):
                continue
            self._unmark_loss_step(step)
            self.deferred_unmark_steps.discard(step)

    def _write_best_loss_meta(self, metric_key, value, step, path):
        self.eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "monitor": self.loss_best_monitor,
            "resolved_metric": metric_key,
            "mode": self.loss_best_mode,
            "step": int(step),
            "score": float(value),
            "checkpoint": str(path),
        }
        with self.best_loss_meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _maybe_update_loss_best(self, trainer, step, ckpt_path):
        if not self.loss_best_monitor:
            return Path(ckpt_path)
        value, metric_key = self._metric_value(trainer)
        if not self._is_better_loss(value):
            return Path(ckpt_path)
        old_step = self.best_loss_step
        if old_step is not None and int(old_step) != int(step):
            if self._step_in_flight(old_step):
                self.deferred_unmark_steps.add(int(old_step))
            else:
                self._unmark_loss_step(old_step)
        marked = self._rename_checkpoint(step, ckpt_path, self._marked_ckpt_path(ckpt_path))
        self.best_loss_step = int(step)
        self.best_loss_score = float(value)
        self._write_best_loss_meta(metric_key, value, step, marked)
        mainlogger.info(
            f"[every_eval] new loss best {metric_key}={value:.6f} "
            f"at step={step}: {marked}"
        )
        self._apply_deferred_unmarks()
        return marked

    def _enqueue_eval(self, trainer, step):
        step = int(step)
        if step in self.queued_steps:
            return
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        ckpt_path = self._save_eval_checkpoint(trainer, step, epoch)
        ckpt_path = self._maybe_update_loss_best(trainer, step, ckpt_path)
        self.queued_steps.add(step)
        mainlogger.info(f"[every_eval] saved checkpoint for step={step}: {ckpt_path}")

    def _maybe_queue_due_eval(self, trainer, step):
        schedule = self._schedule(trainer)
        if schedule is None:
            return
        max_steps, start, interval = schedule
        if self.next_eval_step is None:
            self.next_eval_step = start
        if step < self.next_eval_step:
            return
        if step > max_steps:
            return
        self._enqueue_eval(trainer, step)
        while self.next_eval_step <= step:
            self.next_eval_step += interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self._rank_zero(trainer):
            return
        step = int(getattr(trainer, "global_step", getattr(pl_module, "global_step", 0)) or 0)
        self._maybe_queue_due_eval(trainer, step)

    def on_train_end(self, trainer, pl_module):
        if not self._rank_zero(trainer):
            return
        schedule = self._schedule(trainer)
        if schedule is None:
            return
        max_steps, _, _ = schedule
        # Guarantee the exact final-step checkpoint exists even when the interval
        # cadence did not land on max_steps (e.g. non-multiple nproc scaling).
        if max_steps not in self.queued_steps:
            epoch = int(getattr(trainer, "current_epoch", 0) or 0)
            final_plain = self._plain_ckpt_path(epoch, max_steps)
            final_marked = self._marked_ckpt_path(final_plain)
            if not final_plain.exists() and not final_marked.exists():
                ckpt_path = self._save_eval_checkpoint(trainer, max_steps, epoch)
                self._maybe_update_loss_best(trainer, max_steps, ckpt_path)
                self.queued_steps.add(max_steps)
                mainlogger.info(f"[every_eval] saved final-step checkpoint: step={max_steps}")
        self._apply_deferred_unmarks()


class FinalStepCheckpoint(Callback):
    """Guarantee a final-step ``epoch={epoch}-step={step}.ckpt`` at train end.

    Registered when periodic every-eval saving is OFF (the default retrain
    path), so a usable final-step checkpoint always exists without relying on
    Lightning's ``last.ckpt``. If the base ``model_checkpoint`` already wrote a
    checkpoint at exactly the final step (same name), this is a no-op.
    """

    def __init__(self, checkpoint_dir, filename="epoch={epoch}-step={step}"):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename = filename

    def _rank_zero(self, trainer):
        return bool(getattr(trainer, "is_global_zero", False))

    def on_train_end(self, trainer, pl_module):
        if not self._rank_zero(trainer):
            return
        step = int(getattr(trainer, "global_step", 0) or 0)
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        name = self.filename.format(epoch=epoch, step=step) + ".ckpt"
        path = self.checkpoint_dir / name
        if path.exists():
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mainlogger.info(f"[final_step] saving final-step checkpoint: {path}")
        try:
            trainer.save_checkpoint(str(path), weights_only=True)
        except TypeError:
            trainer.save_checkpoint(str(path))
