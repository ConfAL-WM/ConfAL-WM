"""Shared summary-table writer for periodic checkpoint evaluation.

Used by both the (save-only) ``PeriodicCheckpointCallback`` flow and the
post-training ``eval_periodic_checkpoints.py`` orchestrator so the
``checkpoint_metrics[_with_ewmbench].{json,csv,png}`` tables are rendered by a
single implementation.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

mainlogger = logging.getLogger("mainlogger")


def _summary_paths(report_dir: Path, include_ewmbench: bool) -> tuple[Path, Path, Path]:
    suffix = "_with_ewmbench" if include_ewmbench else ""
    return (
        report_dir / f"checkpoint_metrics{suffix}.json",
        report_dir / f"checkpoint_metrics{suffix}.csv",
        report_dir / f"checkpoint_metrics{suffix}.png",
    )


def nested(payload: Any, dotted: str) -> Any:
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def metric_columns(include_ewmbench: bool) -> list[tuple[str, str, str, str]]:
    columns = [
        ("step", "Step", "meta", "lower"),
        ("checkpoint_name", "Checkpoint", "meta", "lower"),
    ]
    if include_ewmbench:
        columns.extend(
            [
                ("ewmbench.mean.psnr", "PSNR ↑", "ewmbench", "higher"),
                ("ewmbench.mean.ssim", "SSIM ↑", "ewmbench", "higher"),
                ("ewmbench.mean.scene_consistency", "Scene Cons. ↑", "ewmbench", "higher"),
                ("ewmbench.mean.logics", "Logics ↑", "ewmbench", "higher"),
                ("ewmbench.mean.semantics_CLIPScore", "Sem.-CLIP ↑", "ewmbench", "higher"),
                ("ewmbench.mean.semantics_BLEUScore", "Sem.-BLEU ↑", "ewmbench", "higher"),
                ("ewmbench.mean.traj_hsd", "Traj-HSD ↓", "ewmbench", "lower"),
                ("ewmbench.mean.traj_dyn", "Traj-Dyn ↓", "ewmbench", "lower"),
                ("ewmbench.mean.traj_ndtw", "Traj-nDTW ↓", "ewmbench", "lower"),
            ]
        )
    columns.extend(
        [
            ("pixel_mae.mean", "Pixel-MAE ↓", "internal", "lower"),
            ("latent_loss.mean", "Latent Loss ↓", "internal", "lower"),
            ("risk.risk_cvar90", "Risk-CVaR90 ↓", "internal", "lower"),
        ]
    )
    return columns


def rows_from_results(
    results: list[dict[str, Any]],
    step_to_ckpt_path: dict[int, str],
    include_ewmbench: bool,
) -> list[dict[str, Any]]:
    columns = metric_columns(include_ewmbench)
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda x: int(x.get("step", -1))):
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        step = int(result.get("step", -1))
        checkpoint = step_to_ckpt_path.get(step, result.get("checkpoint"))
        row: dict[str, Any] = {
            "step": step,
            "checkpoint": checkpoint,
            "checkpoint_name": Path(str(checkpoint or "")).name,
            "status": result.get("status"),
            "gpu": result.get("gpu"),
            "metrics_path": result.get("metrics_path"),
            "log_path": result.get("log_path"),
        }
        for key, _, _, _ in columns:
            if key in row:
                continue
            row[key] = nested(metrics, key)
        rows.append(row)
    return rows


def format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def rank_colors(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, str, str]],
) -> dict[tuple[int, int], str]:
    colors: dict[tuple[int, int], str] = {}
    palette = {0: "#d7f7d7", 1: "#fff4bf", 2: "#f8d6d6"}
    for col_idx, (key, _, group, direction) in enumerate(columns):
        if group == "meta":
            continue
        vals: list[tuple[float, int]] = []
        for row_idx, row in enumerate(rows):
            value = row.get(key)
            try:
                vals.append((float(value), row_idx))
            except Exception:
                continue
        vals.sort(key=lambda x: x[0], reverse=(direction == "higher"))
        for rank, (_, row_idx) in enumerate(vals[:3]):
            colors[(row_idx, col_idx)] = palette[rank]
    return colors


def write_table_png(png_path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str, str, str]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        mainlogger.warning(f"[every_eval] matplotlib unavailable; cannot write PNG table: {exc}")
        return

    if not rows:
        rows = [{"step": "-", "checkpoint_name": "no finished eval jobs"}]
    display_columns: list[tuple[str, str]] = []
    last_group = None
    source_col_indices: list[int | None] = []
    for src_idx, (_, label, group, _) in enumerate(columns):
        if last_group is not None and group != last_group:
            display_columns.append(("sep", ""))
            source_col_indices.append(None)
        display_columns.append(("data", label))
        source_col_indices.append(src_idx)
        last_group = group

    cell_text = []
    for row in rows:
        rendered = []
        for kind, src_idx in zip([x[0] for x in display_columns], source_col_indices):
            if kind == "sep":
                rendered.append("")
            else:
                key = columns[src_idx][0]
                rendered.append(format_cell(row.get(key)))
        cell_text.append(rendered)

    colors = rank_colors(rows, columns)
    cell_colours = []
    for row_idx, _ in enumerate(rows):
        row_colors = []
        for kind, src_idx in zip([x[0] for x in display_columns], source_col_indices):
            if kind == "sep":
                row_colors.append("#eeeeee")
            else:
                row_colors.append(colors.get((row_idx, src_idx), "white"))
        cell_colours.append(row_colors)

    labels = [label for _, label in display_columns]
    width = max(12.0, min(42.0, 1.25 * len(labels)))
    height = max(3.0, 0.42 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=labels,
        cellColours=cell_colours,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    font_size = 7 if len(labels) <= 14 else 6
    table.set_fontsize(font_size)
    table.scale(1.0, 1.25)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row_idx == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        if source_col_indices[col_idx] is None:
            cell.set_width(0.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    report_dir: Path,
    results: list[dict[str, Any]],
    step_to_ckpt_path: dict[int, str],
    include_ewmbench: bool,
    metrics: str,
) -> tuple[Path, Path, Path]:
    """Write the checkpoint metric table as JSON/CSV/PNG under ``report_dir``.

    Returns the (json, csv, png) paths. ``results`` is the list of per-step
    eval result dicts (each carrying ``metrics``, ``step``, ``checkpoint`` ...).
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_from_results(results, step_to_ckpt_path, include_ewmbench)
    columns = metric_columns(include_ewmbench)
    payload = {
        "include_ewmbench": include_ewmbench,
        "metrics": metrics,
        "num_checkpoints": len(rows),
        "columns": [key for key, _, _, _ in columns],
        "items": rows,
    }
    json_path, csv_path, png_path = _summary_paths(report_dir, include_ewmbench)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key for key, _, _, _ in columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key, _, _, _ in columns})
    write_table_png(png_path, rows, columns)
    mainlogger.info(
        f"[every_eval] wrote checkpoint metric table: {json_path}, {csv_path}, {png_path}"
    )
    return json_path, csv_path, png_path
