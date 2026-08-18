import argparse
import random
from typing import Any, Dict, List

from utils import iter_jsonl, write_jsonl


def parse_budget(value: str, n: int) -> int:
    raw = value.strip()
    if raw.endswith("%"):
        k = int(round(n * float(raw[:-1]) / 100.0))
    else:
        numeric = float(raw)
        k = int(round(n * numeric)) if 0 < numeric < 1 else int(numeric)
    return max(0, min(n, k))


def select_rows(rows: List[Dict[str, Any]], budget: str, strategy: str) -> List[Dict[str, Any]]:
    k = parse_budget(budget, len(rows))
    if k == 0:
        return []
    if strategy in {"top", "topk", "top-k"}:
        ranked = sorted(rows, key=lambda r: float(r.get("acquisition_score", 0.0)), reverse=True)
    elif strategy in {"bottom", "bottomk", "bottom-k"}:
        ranked = sorted(rows, key=lambda r: float(r.get("acquisition_score", 0.0)))
    elif strategy == "random":
        ranked = rows[:]
        random.Random(0).shuffle(ranked)
    else:
        raise ValueError("strategy must be one of: top, bottom, random")
    selected = []
    for rank, row in enumerate(ranked[:k], start=1):
        out = dict(row)
        out["selection_rank"] = rank
        selected.append(out)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top budget episodes from baseline scores.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--budget", required=True, help="Integer count, ratio in (0,1), or percentage like 10%.")
    parser.add_argument("--strategy", default="top", choices=["top", "topk", "top-k", "bottom", "bottomk", "bottom-k", "random"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.scores))
    selected = select_rows(rows, args.budget, args.strategy)
    write_jsonl(args.output, selected)
    print(f"[evac_al_baselines] selected {len(selected)} / {len(rows)} episodes -> {args.output}")


if __name__ == "__main__":
    main()

