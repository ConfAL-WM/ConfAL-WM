from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import sample_points


def _save_figure(fig: plt.Figure, path_stem: str, save_png: bool = True, save_pdf: bool = True) -> None:
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if save_png:
        fig.savefig(str(stem.with_suffix(".png")), dpi=180, bbox_inches="tight")
    if save_pdf:
        fig.savefig(str(stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def _draw_reliability(ax: plt.Axes, calib_stats: dict, title: str, add_density: bool = True) -> None:
    edges = np.asarray(calib_stats["bin_edges"], dtype=np.float64)
    bin_conf = np.asarray(calib_stats["bin_confidence"], dtype=np.float64)
    bin_acc = np.asarray(calib_stats["bin_accuracy"], dtype=np.float64)
    counts = np.asarray(calib_stats["bin_counts"], dtype=np.float64)
    valid = np.isfinite(bin_conf) & np.isfinite(bin_acc)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.0)
    ax.bar(
        centers[valid],
        bin_acc[valid],
        width=widths[valid],
        alpha=0.55,
        color="#4C78A8",
        edgecolor="black",
    )
    ax.scatter(bin_conf[valid], bin_acc[valid], s=18, color="#E45756")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Correctness")

    if add_density:
        density = counts / max(np.sum(counts), 1.0)
        ax2 = ax.twinx()
        ax2.plot(centers, density, color="#72B7B2", linewidth=1.2)
        ymax = float(np.nanmax(density)) if density.size else 0.0
        ax2.set_ylim(0.0, max(0.05, ymax * 1.25))
        ax2.set_ylabel("Density")


def plot_reliability_diagram(calib_stats: dict, title: str, path_stem: str, save_png: bool = True, save_pdf: bool = True) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 6))
    _draw_reliability(ax1, calib_stats=calib_stats, title=title, add_density=True)
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)


def plot_confidence_error_scatter(
    confidence: np.ndarray,
    error: np.ndarray,
    path_stem: str,
    title: str,
    max_points: int = 30000,
    seed: int = 0,
    save_png: bool = True,
    save_pdf: bool = True,
) -> None:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    err = np.asarray(error, dtype=np.float64).reshape(-1)
    conf, err = sample_points(conf, err, max_points=max_points, seed=seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(conf, err, s=6, alpha=0.12, color="#E45756", edgecolors="none")
    ax.set_xlabel("confidence")
    ax.set_ylabel("oracle error")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)


def plot_mean_error_by_conf_bin(
    confidence: np.ndarray,
    error: np.ndarray,
    num_bins: int,
    path_stem: str,
    title: str,
    save_png: bool = True,
    save_pdf: bool = True,
) -> None:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    err = np.asarray(error, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ids = np.digitize(conf, edges[1:-1], right=False)
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(num_bins, np.nan, dtype=np.float64)
    for idx in range(num_bins):
        mask = ids == idx
        if np.any(mask):
            means[idx] = np.mean(err[mask])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(centers, means, marker="o", color="#4C78A8")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Mean oracle error")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)


def plot_ood_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    path_stem: str,
    title: str,
    save_png: bool = True,
    save_pdf: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(id_scores, bins=30, alpha=0.55, color="#4C78A8", label="ID", density=True)
    ax.hist(ood_scores, bins=30, alpha=0.55, color="#E45756", label="OOD", density=True)
    ax.set_xlabel("OOD score / aggregated risk")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)


def plot_process_lag_curve(
    lags: list[int],
    mean_curve: list[float],
    std_curve: list[float],
    path_stem: str,
    title: str,
    save_png: bool = True,
    save_pdf: bool = True,
) -> None:
    lag_arr = np.asarray(lags, dtype=np.int32)
    mean_arr = np.asarray(mean_curve, dtype=np.float64)
    std_arr = np.asarray(std_curve, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lag_arr, mean_arr, color="#4C78A8", marker="o")
    ax.fill_between(lag_arr, mean_arr - std_arr, mean_arr + std_arr, color="#4C78A8", alpha=0.2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("lag")
    ax.set_ylabel("Corr(frame risk, frame error)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)


def plot_qualitative_panel(
    episode_id: str,
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    oracle_error: np.ndarray,
    risk_map: np.ndarray,
    path_stem: str,
    selection_note: str | None = None,
    save_png: bool = True,
    save_pdf: bool = True,
) -> None:
    t = min(len(gt_frames), len(pred_frames), oracle_error.shape[0], risk_map.shape[0])
    if t <= 0:
        return
    sample_ids = np.linspace(0, t - 1, num=min(4, t), dtype=int)
    n_rows = len(sample_ids)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, idx in enumerate(sample_ids):
        axes[row, 0].imshow(np.clip(gt_frames[idx], 0.0, 1.0))
        axes[row, 0].set_title(f"GT t={idx}")
        axes[row, 1].imshow(np.clip(pred_frames[idx], 0.0, 1.0))
        axes[row, 1].set_title("Prediction")
        im_err = axes[row, 2].imshow(oracle_error[idx], cmap="magma")
        axes[row, 2].set_title("Oracle Error")
        im_risk = axes[row, 3].imshow(risk_map[idx], cmap="inferno")
        axes[row, 3].set_title("Risk Map")
        for col in range(4):
            axes[row, col].axis("off")
    if selection_note:
        title = f"{episode_id} qualitative panel\n{selection_note}"
    else:
        title = f"{episode_id} qualitative panel"
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.colorbar(im_err, ax=axes[:, 2], fraction=0.02, pad=0.02)
    fig.colorbar(im_risk, ax=axes[:, 3], fraction=0.02, pad=0.02)
    fig.subplots_adjust(top=0.86, wspace=0.08, hspace=0.18)
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)


def plot_main_dashboard(
    operational_calib: dict,
    pooled_conf: np.ndarray,
    pooled_err: np.ndarray,
    process_summary: dict | None,
    ood_payload: dict | None,
    summary_payload: dict,
    path_stem: str,
    seed: int = 0,
    max_points: int = 30000,
    save_png: bool = True,
    save_pdf: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10.8))
    axes = axes.reshape(2, 3)

    _draw_reliability(axes[0, 0], operational_calib, title="Operational reliability", add_density=True)

    conf = np.asarray(pooled_conf, dtype=np.float64).reshape(-1)
    err = np.asarray(pooled_err, dtype=np.float64).reshape(-1)
    conf_s, err_s = sample_points(conf, err, max_points=max_points, seed=seed)
    axes[0, 1].scatter(conf_s, err_s, s=5, alpha=0.10, color="#E45756", edgecolors="none")
    axes[0, 1].set_xlabel("Confidence")
    axes[0, 1].set_ylabel("Oracle error")
    axes[0, 1].set_title("Confidence vs oracle error")
    axes[0, 1].grid(alpha=0.2)

    edges = np.asarray(operational_calib["bin_edges"], dtype=np.float64)
    ids = np.digitize(conf, edges[1:-1], right=False)
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(centers.shape, np.nan, dtype=np.float64)
    for idx in range(len(centers)):
        mask = ids == idx
        if np.any(mask):
            means[idx] = np.mean(err[mask])
    axes[0, 2].plot(centers, means, marker="o", color="#4C78A8")
    axes[0, 2].set_xlabel("Confidence bin")
    axes[0, 2].set_ylabel("Mean oracle error")
    axes[0, 2].set_title("Mean error by confidence bin")
    axes[0, 2].grid(alpha=0.2)

    if process_summary is not None and "lag_curve_lags" in process_summary:
        lag_arr = np.asarray(process_summary["lag_curve_lags"], dtype=np.int32)
        mean_arr = np.asarray(process_summary["lag_curve_mean"], dtype=np.float64)
        std_arr = np.asarray(process_summary["lag_curve_std"], dtype=np.float64)
        axes[1, 0].plot(lag_arr, mean_arr, color="#4C78A8", marker="o")
        axes[1, 0].fill_between(lag_arr, mean_arr - std_arr, mean_arr + std_arr, color="#4C78A8", alpha=0.2)
        axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1.0)
        axes[1, 0].set_xlabel("Lag")
        axes[1, 0].set_ylabel("Corr")
        axes[1, 0].set_title("Process validity lag curve")
        axes[1, 0].grid(alpha=0.2)
    else:
        axes[1, 0].axis("off")

    if ood_payload is not None and ood_payload.get("trajectory_scores") is not None:
        scores = np.asarray(ood_payload["trajectory_scores"], dtype=np.float64)
        labels = np.asarray(ood_payload["trajectory_labels"], dtype=np.int32)
        if np.any(labels == 0) and np.any(labels == 1):
            axes[1, 1].hist(scores[labels == 0], bins=25, alpha=0.55, color="#4C78A8", label="ID", density=True)
            axes[1, 1].hist(scores[labels == 1], bins=25, alpha=0.55, color="#E45756", label="OOD", density=True)
            axes[1, 1].legend()
            axes[1, 1].set_xlabel("Trajectory OOD score")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].set_title("OOD histogram")
            axes[1, 1].grid(alpha=0.2)
        else:
            axes[1, 1].axis("off")
            axes[1, 1].text(0.5, 0.5, "No OOD labels\nin this evaluation", ha="center", va="center", fontsize=13)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "No OOD payload", ha="center", va="center", fontsize=13)

    axes[1, 2].axis("off")
    text_lines = [
        f"Episodes: {summary_payload.get('num_episodes', 'NA')}",
        f"Frames: {summary_payload.get('num_frames', 'NA')}",
        f"Pooled patch Spearman: {summary_payload.get('pooled_patch_spearman', float('nan')):.4f}",
        f"Pooled patch Pearson: {summary_payload.get('pooled_patch_pearson', float('nan')):.4f}",
        f"Operational tau: {summary_payload.get('operational_tau', float('nan')):.4f}",
        f"Operational ECE: {summary_payload.get('operational_ece', float('nan')):.4f}",
        f"Operational Brier: {summary_payload.get('operational_brier', float('nan')):.4f}",
        f"Top-1 AUROC: {summary_payload.get('high_error_top1_auroc', float('nan')):.4f}",
        f"Top-5 AUROC: {summary_payload.get('high_error_top5_auroc', float('nan')):.4f}",
        f"Top-10 AUROC: {summary_payload.get('high_error_top10_auroc', float('nan')):.4f}",
        f"Frame corr: {summary_payload.get('pooled_frame_spearman', float('nan')):.4f}",
        f"Trajectory corr: {summary_payload.get('trajectory_spearman', float('nan')):.4f}",
    ]
    axes[1, 2].text(0.0, 1.0, "\n".join(text_lines), ha="left", va="top", fontsize=12, family="monospace")
    fig.suptitle("Why Confidence? overview", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.08, top=0.88, wspace=0.22, hspace=0.38)
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)


def plot_calibration_sweep_dashboard(
    sweep_results: list[dict],
    path_stem: str,
    ncols: int = 3,
    save_png: bool = True,
    save_pdf: bool = False,
) -> None:
    if not sweep_results:
        return
    n = len(sweep_results)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 5.8 * nrows))
    axes = np.asarray(axes).reshape(nrows, ncols)
    for idx, item in enumerate(sweep_results):
        ax = axes[idx // ncols, idx % ncols]
        title = (
            f"tau={item['tau']:.3f}\n"
            f"ECE={item['ece']:.3f}, Brier={item['brier']:.3f}"
        )
        _draw_reliability(ax, item, title=title, add_density=False)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    fig.suptitle("Calibration sweep overview", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.08, top=0.88, wspace=0.20, hspace=0.40)
    _save_figure(fig, path_stem, save_png=save_png, save_pdf=save_pdf)
