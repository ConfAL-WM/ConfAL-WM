from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from al_pipeline.utils import load_json


def load_metrics_json(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dict into dotted keys.

    e.g. {"latent_loss": {"mean": 0.1}} -> {"latent_loss.mean": 0.1}
    """
    out: dict[str, Any] = {}
    for key, val in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and not any(
            isinstance(v, (dict, list)) for v in val.values()
        ):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, dict):
                    out.update(flatten_metrics(sub_val, full_key))
                else:
                    out[f"{full_key}.{sub_key}"] = sub_val
        elif isinstance(val, dict):
            out.update(flatten_metrics(val, full_key))
        else:
            out[full_key] = val
    return out


def write_summary_csv(
    rows: list[dict[str, Any]], output_csv: str | Path
) -> None:
    """Write a list of flat metric dicts to a CSV."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    all_keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    # Keep key columns first
    key_cols = ["method", "round_id"]
    ordered = [c for c in key_cols if c in all_keys]
    ordered += [c for c in all_keys if c not in ordered]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[aggregate] wrote {len(rows)} rows to {path}")
