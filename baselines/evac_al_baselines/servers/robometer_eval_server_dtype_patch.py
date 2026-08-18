#!/usr/bin/env python3
"""Launch Robometer eval_server with an ConfAL-WM-local dtype compatibility patch.

This leaves baselines/robometer untouched.  It only patches the imported
Robometer server module in the current process and starts the server without
invoking Robometer's Hydra entrypoint.
"""

from __future__ import annotations

import sys
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def _robometer_dir() -> Path:
    baselines_dir = Path(__file__).resolve().parents[2]
    return baselines_dir / "robometer"


def _ensure_robometer_on_path() -> None:
    robometer_dir = _robometer_dir()
    if robometer_dir.exists():
        sys.path.insert(0, str(robometer_dir))


_ensure_robometer_on_path()

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from robometer.evals import eval_server as robometer_eval_server  # noqa: E402


_ORIGINAL_FORWARD_MODEL = robometer_eval_server.forward_model
_DTYPE_CACHE: dict[int, torch.dtype | None] = {}
_DTYPE_COUNT_CACHE: dict[int, dict[str, int]] = {}
_PREPARED_MODULES: set[int] = set()


def _dtype_from_name(name: Any) -> torch.dtype | None:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return None
    normalized = str(name).lower().replace("torch.", "").replace("-", "_")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    return aliases.get(normalized)


def _override_visual_dtype() -> torch.dtype | None:
    return _dtype_from_name(os.environ.get("EVAC_ROBOMETER_VISUAL_DTYPE"))


def _module_float_dtype(module: Any) -> torch.dtype | None:
    if module is None:
        return None

    dtype = getattr(module, "dtype", None)
    dtype_candidates = [dtype] if isinstance(dtype, torch.dtype) else []

    if not hasattr(module, "parameters"):
        return dtype if isinstance(dtype, torch.dtype) else None

    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            dtype_candidates.append(param.dtype)

    if not dtype_candidates:
        return None

    counts = Counter(dtype_candidates)
    _DTYPE_COUNT_CACHE[id(module)] = {str(key): int(value) for key, value in counts.items()}
    for preferred in (torch.bfloat16, torch.float16):
        if counts.get(preferred, 0) > 0:
            return preferred
    return counts.most_common(1)[0][0]


def _model_visual_float_dtype(model: Any) -> torch.dtype | None:
    override = _override_visual_dtype()
    if override is not None:
        return override

    cache_key = id(model)
    if cache_key in _DTYPE_CACHE:
        return _DTYPE_CACHE[cache_key]

    inner_model = getattr(model, "model", None)
    candidates = [
        getattr(inner_model, "visual", None),
        getattr(getattr(inner_model, "model", None), "visual", None),
    ]

    for candidate in candidates:
        dtype = _module_float_dtype(candidate)
        if dtype is not None:
            _DTYPE_CACHE[cache_key] = dtype
            return dtype

    if hasattr(inner_model, "named_modules"):
        for name, module in inner_model.named_modules():
            if name == "visual" or name.endswith(".visual"):
                dtype = _module_float_dtype(module)
                if dtype is not None:
                    _DTYPE_CACHE[cache_key] = dtype
                    return dtype

    config_dtype = _dtype_from_name(getattr(getattr(model, "model_config", None), "torch_dtype", None))
    if config_dtype is not None:
        _DTYPE_CACHE[cache_key] = config_dtype
        return config_dtype

    dtype = _module_float_dtype(model)
    _DTYPE_CACHE[cache_key] = dtype
    return dtype


def _find_visual_module(model: Any) -> Any:
    inner_model = getattr(model, "model", None)
    candidates = [
        getattr(inner_model, "visual", None),
        getattr(getattr(inner_model, "model", None), "visual", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate

    if hasattr(inner_model, "named_modules"):
        for name, module in inner_model.named_modules():
            if name == "visual" or name.endswith(".visual"):
                return module
    return None


def _prepare_model_module(model: Any, dtype: torch.dtype | None) -> None:
    if dtype is None:
        return

    override = _override_visual_dtype()
    cast_scope = os.environ.get("EVAC_ROBOMETER_CAST_SCOPE", "").lower()
    cast_full_model = override is not None or cast_scope == "full"
    module = model if cast_full_model else _find_visual_module(model)
    module_name = "model" if cast_full_model else "visual"
    if module is None:
        return

    cache_key = id(module)
    if cache_key in _PREPARED_MODULES:
        return

    before_dtype = getattr(module, "dtype", None)
    module.to(dtype=dtype)
    _PREPARED_MODULES.add(cache_key)
    _DTYPE_COUNT_CACHE.pop(cache_key, None)
    _module_float_dtype(module)
    after_dtype = getattr(module, "dtype", None)
    robometer_eval_server.logger.info(
        f"[ConfAL-WM dtype patch] {module_name}.to(dtype={dtype}) before={before_dtype} after={after_dtype}"
    )


def _to_float32(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
        return value.float()
    if isinstance(value, dict):
        return {key: _to_float32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_float32(item) for item in value)
    return value


def _cast_model_output_to_float32(model_output: Any) -> Any:
    for attr in ("pref_logits", "progress_logits", "success_logits"):
        if hasattr(model_output, attr):
            setattr(model_output, attr, _to_float32(getattr(model_output, attr)))
    return model_output


def _debug_visual_dtypes(model: Any, batch_inputs: Dict[str, Any], dtype: torch.dtype | None) -> None:
    input_dtypes = {}
    for key in ("pixel_values", "pixel_values_videos"):
        value = batch_inputs.get(key)
        if isinstance(value, torch.Tensor):
            input_dtypes[key] = str(value.dtype)
    if getattr(model, "_evac_dtype_patch_logged", False):
        return
    setattr(model, "_evac_dtype_patch_logged", True)

    inner_model = getattr(model, "model", None)
    visual = getattr(inner_model, "visual", None)
    visual_counts = _DTYPE_COUNT_CACHE.get(id(visual), {})
    robometer_eval_server.logger.info(
        f"[ConfAL-WM dtype patch] visual_dtype={dtype} "
        f"visual_dtype_counts={visual_counts} input_dtypes_before={input_dtypes}"
    )
    return None


def _cast_visual_inputs(batch_inputs: Dict[str, Any], dtype: torch.dtype | None) -> None:
    if dtype is None:
        return
    for key in ("pixel_values", "pixel_values_videos"):
        value = batch_inputs.get(key)
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != dtype:
            batch_inputs[key] = value.to(dtype=dtype)
            robometer_eval_server.logger.debug(
                f"[ConfAL-WM dtype patch] cast {key}: {value.dtype} -> {batch_inputs[key].dtype}"
            )


def _forward_model_dtype_patch(
    model: Any,
    batch_inputs: Dict[str, Any],
    sample_type: str = "progress",
):
    visual_dtype = _model_visual_float_dtype(model)
    _prepare_model_module(model, visual_dtype)
    _debug_visual_dtypes(model, batch_inputs, visual_dtype)
    _cast_visual_inputs(batch_inputs, visual_dtype)
    model_output, extra = _ORIGINAL_FORWARD_MODEL(model, batch_inputs, sample_type=sample_type)
    return _cast_model_output_to_float32(model_output), extra


def _load_eval_config(argv: list[str]):
    cfg_path = _robometer_dir() / "robometer" / "configs" / "eval_config_server.yaml"
    cfg = OmegaConf.load(cfg_path)
    if argv:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(argv))
    return robometer_eval_server.convert_hydra_to_dataclass(
        cfg,
        robometer_eval_server.EvalServerConfig,
    )


def main() -> None:
    robometer_eval_server.forward_model = _forward_model_dtype_patch

    eval_cfg = _load_eval_config(sys.argv[1:])
    robometer_eval_server.display_config(eval_cfg)

    if not eval_cfg.model_path:
        raise ValueError("Eval config must set model_path to a pretrained checkpoint.")

    multi_gpu_server = robometer_eval_server.MultiGPUEvalServer(
        model_path=eval_cfg.model_path,
        num_gpus=eval_cfg.num_gpus,
        max_workers=eval_cfg.max_workers,
    )
    robometer_eval_server.display_config(multi_gpu_server.exp_config)

    app = robometer_eval_server.create_app(eval_cfg, multi_gpu_server)
    print(f"Running multi-GPU eval server on {eval_cfg.server_url}:{eval_cfg.server_port}")
    print(f"Using {eval_cfg.num_gpus or torch.cuda.device_count()} GPUs")
    robometer_eval_server.uvicorn.run(app, host=eval_cfg.server_url, port=eval_cfg.server_port)


if __name__ == "__main__":
    main()
