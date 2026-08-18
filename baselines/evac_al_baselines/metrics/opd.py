from typing import Any, Dict, Iterable, List

from utils import read_json_or_csv


OPD_KEYS = ("MC", "MP", "PPL", "CRA", "STR")


def _scores(frame_scores: Iterable[float]) -> List[float]:
    return [float(x) for x in frame_scores or []]


def compute_opd_metrics(frame_scores: Iterable[float], success_threshold: float = 0.5) -> Dict[str, float]:
    scores = _scores(frame_scores)
    if not scores:
        return {key: 0.0 for key in OPD_KEYS}
    positives = [max(0.0, scores[i + 1] - scores[i]) for i in range(len(scores) - 1)]
    total_abs = sum(abs(scores[i + 1] - scores[i]) for i in range(len(scores) - 1))
    monotonic_pairs = 0
    total_pairs = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            total_pairs += 1
            if scores[j] >= scores[i]:
                monotonic_pairs += 1
    crossings = sum(
        1
        for i in range(len(scores) - 1)
        if scores[i] < success_threshold <= scores[i + 1]
    )
    return {
        "MC": max(scores),
        "MP": sum(positives) / max(1, len(positives)),
        "PPL": sum(positives),
        "CRA": 0.0 if total_abs == 0 else sum(positives) / total_abs,
        "STR": crossings / max(1, len(scores) - 1),
    }


def read_opd_metrics(path: str) -> Dict[str, float]:
    rows = read_json_or_csv(path)
    for row in rows:
        found: Dict[str, float] = {}
        for key in OPD_KEYS:
            if key in row:
                try:
                    found[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
            elif key.lower() in row:
                try:
                    found[key] = float(row[key.lower()])
                except (TypeError, ValueError):
                    pass
        if found:
            return {key: found.get(key, 0.0) for key in OPD_KEYS}
    return {key: 0.0 for key in OPD_KEYS}

