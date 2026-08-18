from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from tqdm import tqdm


COMPOSITE_TERMS = ("tail_risk_top5", "persistent_risk", "risk_area")


def normalize_values(values: list[float], method: str) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    if method == "rank":
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty_like(arr, dtype=np.float32)
        if arr.size == 1:
            ranks[order] = 1.0
        else:
            ranks[order] = np.linspace(0.0, 1.0, arr.size, dtype=np.float32)
        return ranks.tolist()
    if method == "minmax":
        lo = float(arr.min())
        hi = float(arr.max())
        if hi - lo < 1e-8:
            return [0.5] * int(arr.size)
        return ((arr - lo) / (hi - lo)).tolist()
    if method == "zscore":
        std = float(arr.std())
        if std < 1e-8:
            return [0.0] * int(arr.size)
        return ((arr - float(arr.mean())) / std).tolist()
    raise ValueError(f"score_normalization must be rank, minmax, or zscore; got {method!r}")


def add_normalized_composite_scores(items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    method = params.get("score_normalization", "rank")
    norm_by_term: dict[str, list[float]] = {}
    for term in COMPOSITE_TERMS:
        norm_by_term[term] = normalize_values([float(item.get(term, 0.0)) for item in items], method)

    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 0.5))
    gamma = float(params.get("gamma", 0.25))
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        row = dict(item)
        row["raw_tail_risk_top5"] = float(item.get("tail_risk_top5", 0.0))
        row["raw_persistent_risk"] = float(item.get("persistent_risk", 0.0))
        row["raw_risk_area"] = float(item.get("risk_area", 0.0))
        row["norm_tail_risk_top5"] = float(norm_by_term["tail_risk_top5"][idx])
        row["norm_persistent_risk"] = float(norm_by_term["persistent_risk"][idx])
        row["norm_risk_area"] = float(norm_by_term["risk_area"][idx])
        row["score_normalization"] = method
        row["composite_score"] = (
            alpha * row["norm_tail_risk_top5"]
            + beta * row["norm_persistent_risk"]
            + gamma * row["norm_risk_area"]
        )
        rows.append(row)
    return rows


def composite_score(item: dict[str, Any], params: dict[str, Any]) -> float:
    if "composite_score" in item:
        return float(item["composite_score"])
    return add_normalized_composite_scores([item], params)[0]["composite_score"]


class BaseSelector(ABC):
    name = "base"

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.config = config or {}
        self.seed = seed

    @abstractmethod
    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return selected items and score rows for all candidates."""

    def _with_score(self, item: dict[str, Any], score: float, **extra: Any) -> dict[str, Any]:
        row = dict(item)
        row["selection_score"] = float(score)
        row["selection_method"] = self.name
        row.update(extra)
        return row


class RandomSelector(BaseSelector):
    name = "random"

    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rng = random.Random(self.seed)
        rows = [self._with_score(item, rng.random()) for item in items]
        selected_ids = {id(item) for item in rng.sample(items, min(budget, len(items)))}
        selected = [row for item, row in zip(items, rows) if id(item) in selected_ids]
        return selected, rows


class MeanRiskSelector(BaseSelector):
    name = "mean_risk"

    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = [self._with_score(item, float(item.get("mean_risk", 0.0))) for item in items]
        rows.sort(key=lambda r: r["selection_score"], reverse=True)
        return rows[:budget], rows


class TailRiskSelector(BaseSelector):
    name = "tail_risk"

    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        key = self.config.get("tail_key", "tail_risk_top5")
        rows = [self._with_score(item, float(item.get(key, 0.0)), score_key=key) for item in items]
        rows.sort(key=lambda r: r["selection_score"], reverse=True)
        return rows[:budget], rows


class PersistentRiskSelector(BaseSelector):
    name = "persistent_risk"

    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = [self._with_score(item, float(item.get("persistent_risk", 0.0))) for item in items]
        rows.sort(key=lambda r: r["selection_score"], reverse=True)
        return rows[:budget], rows


class CompositeRiskSelector(BaseSelector):
    name = "composite"

    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scored_items = add_normalized_composite_scores(items, self.config)
        rows = [self._with_score(item, composite_score(item, self.config)) for item in scored_items]
        rows.sort(key=lambda r: r["selection_score"], reverse=True)
        return rows[:budget], rows


class DiverseCompositeRiskSelector(BaseSelector):
    name = "diverse"

    def _load_embeddings(self, items: list[dict[str, Any]]) -> np.ndarray:
        embeddings = []
        iterator = items
        if bool(self.config.get("show_progress", False)):
            iterator = tqdm(items, desc="[al selection] loading hdec embeddings", unit="ep", dynamic_ncols=True)
        for item in iterator:
            path = item.get("hdec_embedding_path")
            if not path:
                raise ValueError(f"Missing hdec_embedding_path for {item.get('episode_id')}")
            arr = np.load(path).astype(np.float32).reshape(-1)
            embeddings.append(arr)
        max_dim = max(arr.size for arr in embeddings)
        out = np.zeros((len(embeddings), max_dim), dtype=np.float32)
        for idx, arr in enumerate(embeddings):
            out[idx, : arr.size] = arr
        mean = out.mean(axis=0, keepdims=True)
        std = out.std(axis=0, keepdims=True) + 1e-6
        return (out - mean) / std

    def _cluster(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        if n_clusters <= 1 or embeddings.shape[0] <= 1:
            return np.zeros((embeddings.shape[0],), dtype=np.int64)
        n_clusters = min(n_clusters, embeddings.shape[0])
        try:
            from sklearn.cluster import KMeans

            km = KMeans(
                n_clusters=n_clusters,
                random_state=self.seed,
                n_init=int(self.config.get("kmeans_n_init", 10)),
            )
            return km.fit_predict(embeddings).astype(np.int64)
        except Exception:
            return self._greedy_diversity_clusters(embeddings, n_clusters)

    def _greedy_diversity_clusters(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = embeddings.shape[0]
        centers = [int(rng.integers(0, n))]
        dists = np.linalg.norm(embeddings - embeddings[centers[0]], axis=1)
        while len(centers) < n_clusters:
            next_idx = int(np.argmax(dists))
            centers.append(next_idx)
            dists = np.minimum(dists, np.linalg.norm(embeddings - embeddings[next_idx], axis=1))
        center_arr = embeddings[centers]
        all_d = ((embeddings[:, None, :] - center_arr[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(all_d, axis=1).astype(np.int64)

    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not items:
            return [], []
        requested_clusters = int(self.config.get("n_clusters", min(8, max(1, int(math.sqrt(len(items)))))))
        n_clusters = max(1, min(requested_clusters, len(items), max(1, budget)))
        embeddings = self._load_embeddings(items)
        if bool(self.config.get("show_progress", False)):
            print(f"[al selection] clustering {len(items)} embeddings into {n_clusters} cluster(s)")
        labels = self._cluster(embeddings, n_clusters)
        if bool(self.config.get("show_progress", False)):
            print("[al selection] clustering complete")

        rows = []
        cluster_to_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        scored_items = add_normalized_composite_scores(items, self.config)
        for item, cluster_id in zip(scored_items, labels):
            score = composite_score(item, self.config)
            row = self._with_score(
                item,
                score,
                cluster_id=int(cluster_id),
                requested_n_clusters=requested_clusters,
                effective_n_clusters=n_clusters,
            )
            rows.append(row)
            cluster_to_rows[int(cluster_id)].append(row)
        for cluster_rows in cluster_to_rows.values():
            cluster_rows.sort(key=lambda r: r["selection_score"], reverse=True)

        # Two-stage cluster-balanced selection:
        # 1. pick a representative top sample from each non-empty cluster, sorted
        #    by cluster best score when there are more clusters than budget.
        # 2. fill remaining budget globally by normalized composite score.
        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        cluster_ids = sorted(
            cluster_to_rows,
            key=lambda cid: cluster_to_rows[cid][0]["selection_score"],
            reverse=True,
        )
        for cid in cluster_ids:
            if len(selected) >= budget:
                break
            row = dict(cluster_to_rows[cid][0])
            row["selection_stage"] = "cluster_representative"
            selected.append(row)
            selected_keys.add(str(row.get("episode_id", id(row))))

        rows_sorted = sorted(rows, key=lambda r: r["selection_score"], reverse=True)
        for row in rows_sorted:
            if len(selected) >= budget:
                break
            key = str(row.get("episode_id", id(row)))
            if key in selected_keys:
                continue
            row = dict(row)
            row["selection_stage"] = "global_score_fill"
            selected.append(row)
            selected_keys.add(key)
        rows = rows_sorted
        return selected, rows


class NotImplementedSelector(BaseSelector):
    def select(self, items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raise NotImplementedError(f"{self.name} is reserved for future active-learning baselines.")


class LossProxySelector(NotImplementedSelector):
    name = "loss_proxy"


class OracleLossSelector(NotImplementedSelector):
    name = "oracle_loss"


class DisagreementSelector(NotImplementedSelector):
    name = "disagreement"


class RewardModelSelector(NotImplementedSelector):
    name = "reward_model_selector"


SELECTOR_REGISTRY: dict[str, type[BaseSelector]] = {
    "random": RandomSelector,
    "mean_risk": MeanRiskSelector,
    "tail_risk": TailRiskSelector,
    "c3_tail_risk": TailRiskSelector,  # backward-compatible alias
    "persistent_risk": PersistentRiskSelector,
    "c3_persistent_risk": PersistentRiskSelector,  # backward-compatible alias
    "composite": CompositeRiskSelector,
    "c3_composite": CompositeRiskSelector,  # backward-compatible alias
    "diverse": DiverseCompositeRiskSelector,
    "c3_diverse": DiverseCompositeRiskSelector,  # backward-compatible alias
    "loss_proxy": LossProxySelector,
    "oracle_loss": OracleLossSelector,
    "disagreement": DisagreementSelector,
    "reward_model_selector": RewardModelSelector,
}


def selected_task_counts(selected: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("task_id", "unknown")) for row in selected).items()))


def selected_cluster_counts(selected: list[dict[str, Any]]) -> dict[str, int]:
    vals = [str(row.get("cluster_id", "none")) for row in selected if "cluster_id" in row]
    return dict(sorted(Counter(vals).items()))
