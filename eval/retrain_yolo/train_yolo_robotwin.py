#!/usr/bin/env python3
"""Fine-tune a YOLO detector on RoboTwin gripper pseudo-labels."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from al_pipeline.utils import save_json  # noqa: E402


# ---------------------------------------------------------------------------
# Per-step training metrics logger (ultralytics callback)
# ---------------------------------------------------------------------------

class _PerStepCSVLogger:
    """Ultralytics callback that saves per‑batch loss & LR to CSV (rank 0 only).

    The CSV is opened once in *on_train_start* and flushed every *flush_every*
    steps.  Validation metrics are captured from the most recent validator pass.
    """

    def __init__(self, save_path: Path, flush_every: int = 50):
        self.save_path = Path(save_path).resolve()
        self.flush_every = flush_every
        self._fh: Any = None
        self._n = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rank0() -> bool:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _ensure_header(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.save_path.exists()
        self._fh = self.save_path.open("a", encoding="utf-8")
        if not exists:
            self._fh.write(
                "epoch,global_step,box_loss,cls_loss,dfl_loss,lr_pg0,"
                "val_box_loss,val_cls_loss,val_dfl_loss\n"
            )

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: Any = None) -> None:
        if not self._is_rank0():
            return
        self._ensure_header()

    def on_train_batch_end(self, trainer: Any) -> None:
        if not self._is_rank0():
            return
        loss = trainer.loss_items
        lr = trainer.scheduler.get_last_lr()
        box = float(loss[0]) if len(loss) > 0 else float("nan")
        cls = float(loss[1]) if len(loss) > 1 else float("nan")
        dfl = float(loss[2]) if len(loss) > 2 else float("nan")
        lr0 = float(lr[0]) if lr else float("nan")

        val_box = val_cls = val_dfl = ""
        if hasattr(trainer, "validator") and getattr(trainer, "metrics", None):
            m = trainer.metrics
            if hasattr(m, "box_loss") and m.box_loss is not None:
                val_box = f"{float(m.box_loss):.6f}"
            if hasattr(m, "cls_loss") and m.cls_loss is not None:
                val_cls = f"{float(m.cls_loss):.6f}"
            if hasattr(m, "dfl_loss") and m.dfl_loss is not None:
                val_dfl = f"{float(m.dfl_loss):.6f}"

        self._fh.write(
            f"{trainer.epoch:.4f},{trainer.global_step},{box:.6f},{cls:.6f},{dfl:.6f},{lr0:.8f},"
            f"{val_box},{val_cls},{val_dfl}\n"
        )
        self._n += 1
        if self._n % self.flush_every == 0:
            self._fh.flush()

    def on_train_end(self, trainer: Any = None) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

HF_MODEL_REPOS = {
    "yolo26n.pt": "Ultralytics/YOLO26",
    "yolo26s.pt": "Ultralytics/YOLO26",
    "yolo26m.pt": "Ultralytics/YOLO26",
    "yolo26l.pt": "Ultralytics/YOLO26",
    "yolo26x.pt": "Ultralytics/YOLO26",
}

GITHUB_MODEL_URLS = {
    "yolo26n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolo26s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
    "yolo26m.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
    "yolo26l.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
    "yolo26x.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
    "yolov8s-world.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt",
    "yolov8m-world.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-world.pt",
    "yolov8l-world.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt",
    "yolov8x-world.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt",
    "yolov8s-worldv2.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_REPO / p)


def _download_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"[yolo train] downloading base model: {url} -> {dst}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f)
        tmp.replace(dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dst


def _default_model_path(model_name: str) -> Path:
    return _REPO / "eval" / "retrain_yolo" / "base_models" / model_name


def _known_model_urls(model_name: str) -> list[str]:
    urls: list[str] = []
    hf_repo = HF_MODEL_REPOS.get(model_name)
    if hf_repo:
        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
        urls.append(f"{endpoint}/{hf_repo}/resolve/main/{model_name}")
        if endpoint != "https://huggingface.co":
            urls.append(f"https://huggingface.co/{hf_repo}/resolve/main/{model_name}")
    github_url = GITHUB_MODEL_URLS.get(model_name)
    if github_url:
        urls.append(github_url)
    return urls


def _resolve_base_model(base_model: str) -> Path:
    """Return an existing local model path, downloading known YOLO assets if needed."""
    if base_model.startswith(("http://", "https://")):
        dst = _default_model_path(Path(base_model).name)
        return dst if dst.exists() else _download_file(base_model, dst)

    p = Path(base_model)
    if p.is_absolute() or p.parent != Path("."):
        local_path = p if p.is_absolute() else _REPO / p
    else:
        local_path = _default_model_path(p.name)

    if local_path.exists():
        return local_path

    urls = _known_model_urls(p.name)
    if urls:
        errors: list[str] = []
        for url in urls:
            try:
                return _download_file(url, local_path)
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                print(f"[yolo train WARNING] download failed from {url}: {exc}")
        raise RuntimeError(
            f"Could not download {p.name}. Tried:\n  " + "\n  ".join(errors)
        )

    raise FileNotFoundError(
        f"Base model not found: {local_path}. Provide an existing --base_model path, "
        "a URL, or one of: " + ", ".join(sorted(set(HF_MODEL_REPOS) | set(GITHUB_MODEL_URLS)))
    )


def _device_from_gpus(gpus: str | None) -> str:
    if gpus is None or str(gpus).strip() == "":
        return "0"
    value = str(gpus).strip()
    if value.lower() in {"cpu", "mps"}:
        return value.lower()
    return ",".join(part.strip() for part in value.split(",") if part.strip())


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required in the YOLO training environment. "
            "Install it or run this script inside the external YOLO/EWMBench conda env."
        ) from exc

    dataset_dir = _resolve_path(args.dataset_dir)
    data_yaml = _resolve_path(args.data_yaml) if args.data_yaml else dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")

    project = _resolve_path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    base_model = _resolve_base_model(args.base_model)
    device = _device_from_gpus(args.gpus)
    cache = False if args.cache == "none" else args.cache

    model = YOLO(str(base_model))

    # ---- per-step CSV logger (registered globally so DDP workers inherit it) ----
    from ultralytics.utils.callbacks import base as _cb_base  # noqa: E402

    save_dir = (project / args.name).resolve()
    per_step_csv = save_dir / "per_step_metrics.csv"
    logger_cb = _PerStepCSVLogger(per_step_csv)

    _events = ("on_train_start", "on_train_batch_end", "on_train_end")
    _hooks = {
        "on_train_start": logger_cb.on_train_start,
        "on_train_batch_end": logger_cb.on_train_batch_end,
        "on_train_end": logger_cb.on_train_end,
    }
    for _ev, _fn in _hooks.items():
        _cb_base.default_callbacks.setdefault(_ev, []).append(_fn)

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            device=device,
            workers=int(args.workers),
            project=str(project),
            name=args.name,
            exist_ok=bool(args.exist_ok),
            pretrained=True,
            patience=int(args.patience),
            amp=bool(args.amp),
            cache=cache,
            plots=bool(args.plots),
        )
    finally:
        # Remove hooks to avoid side effects if called again in the same process
        for _ev, _fn in _hooks.items():
            _lst = _cb_base.default_callbacks.get(_ev, [])
            if _fn in _lst:
                _lst.remove(_fn)

    # resolve final save_dir (ultralytics may append an integer suffix)
    save_dir = Path(getattr(results, "save_dir", project / args.name))
    summary = {
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "base_model": str(base_model),
        "gpus": args.gpus,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": device,
        "amp": bool(args.amp),
        "cache": args.cache,
        "plots": bool(args.plots),
        "run_dir": str(save_dir),
        "per_step_csv": str(per_step_csv),
        "best_weight": str(save_dir / "weights" / "best.pt"),
        "last_weight": str(save_dir / "weights" / "last.pt"),
    }
    save_json(summary, project / f"{args.name}_summary.json")
    print(f"[yolo train] run_dir: {save_dir}")
    print(f"[yolo train] best: {summary['best_weight']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RoboTwin gripper YOLO detector")
    parser.add_argument("--dataset_dir", "--dataset-dir", dest="dataset_dir", default="eval/retrain_yolo/robotwin_gripper_yolo")
    parser.add_argument("--data_yaml", "--data-yaml", dest="data_yaml", default=None)
    parser.add_argument(
        "--base_model",
        "--base-model",
        dest="base_model",
        default="eval/retrain_yolo/base_models/yolo26s.pt",
        help="Base YOLO checkpoint path, known model name, or URL. Missing known weights are downloaded to this path.",
    )
    parser.add_argument("--project", default="eval/retrain_yolo/runs")
    parser.add_argument("--name", default="robotwin_gripper_yolo26s")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--gpus", default="0", help="GPU id list for Ultralytics, e.g. 0 or 0,1; use cpu for CPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--cache",
        choices=["none", "ram", "disk"],
        default="none",
        help="Ultralytics image cache mode. Use ram for fastest training if CPU RAM is enough; disk can help repeated runs.",
    )
    parser.add_argument("--plots", action="store_true", help="Enable Ultralytics training plots. Disabled by default for speed.")
    parser.add_argument("--amp", action="store_true", help="Enable Ultralytics AMP. Disabled by default to avoid offline AMP-check downloads.")
    parser.add_argument("--exist_ok", "--exist-ok", dest="exist_ok", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
