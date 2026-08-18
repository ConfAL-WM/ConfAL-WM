from typing import Dict, Iterable, List, Optional


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    denom = (sum(x * x for x in da) * sum(x * x for x in db)) ** 0.5
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / denom


def spearman(frame_scores: Iterable[float], frame_order: Optional[Iterable[int]] = None) -> float:
    scores = [float(x) for x in frame_scores]
    order = list(frame_order) if frame_order is not None else list(range(len(scores)))
    if len(scores) != len(order) or len(scores) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr

        result = spearmanr(order, scores)
        return 0.0 if result.correlation != result.correlation else float(result.correlation)
    except Exception:
        return _pearson(_rank([float(x) for x in order]), _rank(scores))


def kendall(frame_scores: Iterable[float], frame_order: Optional[Iterable[int]] = None) -> float:
    scores = [float(x) for x in frame_scores]
    order = list(frame_order) if frame_order is not None else list(range(len(scores)))
    if len(scores) != len(order) or len(scores) < 2:
        return 0.0
    try:
        from scipy.stats import kendalltau

        result = kendalltau(order, scores)
        return 0.0 if result.correlation != result.correlation else float(result.correlation)
    except Exception:
        concordant = discordant = 0
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                dx = order[j] - order[i]
                dy = scores[j] - scores[i]
                prod = dx * dy
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
        denom = concordant + discordant
        return 0.0 if denom == 0 else (concordant - discordant) / denom


def voc_auc(frame_scores: Iterable[float]) -> float:
    scores = [max(0.0, min(1.0, float(x))) for x in frame_scores]
    if not scores:
        return 0.0
    if len(scores) == 1:
        return scores[0]
    return sum((scores[i] + scores[i + 1]) * 0.5 for i in range(len(scores) - 1)) / (len(scores) - 1)


def compute_voc(frame_scores: Iterable[float], frame_order: Optional[Iterable[int]] = None) -> Dict[str, float]:
    scores = [float(x) for x in frame_scores]
    return {
        "spearman": spearman(scores, frame_order),
        "kendall": kendall(scores, frame_order),
        "voc_auc": voc_auc(scores),
        "voc_delta": (scores[-1] - scores[0]) if len(scores) >= 2 else 0.0,
    }

