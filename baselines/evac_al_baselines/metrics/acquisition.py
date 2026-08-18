from typing import Any, Dict, Iterable, List, Optional


def _scores(frame_scores: Optional[Iterable[float]], episode_score: Optional[float] = None) -> List[float]:
    vals = [float(x) for x in (frame_scores or [])]
    if not vals and episode_score is not None:
        vals = [float(episode_score)]
    return vals


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def random_score(seed_value: float) -> float:
    return _clip01(seed_value)


def progress_low_score(frame_scores=None, episode_score=None, **_: Any) -> float:
    vals = _scores(frame_scores, episode_score)
    if not vals:
        return 0.0
    return _clip01(1.0 - vals[-1])


def near_miss_score(frame_scores=None, episode_score=None, threshold: float = 0.75, **_: Any) -> float:
    vals = _scores(frame_scores, episode_score)
    if not vals:
        return 0.0
    best = max(vals)
    final = vals[-1]
    closeness = 1.0 - min(abs(best - threshold) / max(threshold, 1e-6), 1.0)
    return _clip01(closeness * (1.0 - final))


def stagnation_score(frame_scores=None, episode_score=None, **_: Any) -> float:
    vals = _scores(frame_scores, episode_score)
    if len(vals) < 2:
        return progress_low_score(vals)
    deltas = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    movement = sum(deltas) / len(deltas)
    return _clip01((1.0 - vals[-1]) * (1.0 - movement))


def regret_score(frame_scores=None, episode_score=None, **_: Any) -> float:
    vals = _scores(frame_scores, episode_score)
    if not vals:
        return 0.0
    return _clip01(max(vals) - vals[-1])


def c3_uncertainty_score(extra: Optional[Dict[str, Any]] = None, episode_score=None, **_: Any) -> float:
    extra = extra or {}
    p = extra.get("success_probability", extra.get("confidence", episode_score))
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.0
    return _clip01(1.0 - abs(p - 0.5) * 2.0)


def combined_score(frame_scores=None, episode_score=None, extra=None, weights: Optional[Dict[str, float]] = None, **kwargs: Any) -> float:
    weights = weights or {
        "progress_low": 0.25,
        "near_miss": 0.25,
        "stagnation": 0.20,
        "regret": 0.15,
        "c3_uncertainty": 0.15,
    }
    parts = {
        "progress_low": progress_low_score(frame_scores, episode_score),
        "near_miss": near_miss_score(frame_scores, episode_score, **kwargs),
        "stagnation": stagnation_score(frame_scores, episode_score),
        "regret": regret_score(frame_scores, episode_score),
        "c3_uncertainty": c3_uncertainty_score(extra, episode_score),
    }
    denom = sum(max(0.0, float(w)) for w in weights.values())
    if denom <= 0:
        return 0.0
    return _clip01(sum(parts[k] * max(0.0, float(weights.get(k, 0.0))) for k in parts) / denom)


def compute_acquisition(strategy: str, frame_scores=None, episode_score=None, extra=None, random_value=None, **kwargs: Any) -> float:
    if strategy == "random":
        return random_score(float(random_value or 0.0))
    if strategy == "progress_low":
        return progress_low_score(frame_scores, episode_score)
    if strategy == "near_miss":
        return near_miss_score(frame_scores, episode_score, **kwargs)
    if strategy == "stagnation":
        return stagnation_score(frame_scores, episode_score)
    if strategy == "regret":
        return regret_score(frame_scores, episode_score)
    if strategy == "c3_uncertainty":
        return c3_uncertainty_score(extra, episode_score)
    if strategy == "combined":
        return combined_score(frame_scores, episode_score, extra, **kwargs)
    return episode_score if episode_score is not None else 0.0

