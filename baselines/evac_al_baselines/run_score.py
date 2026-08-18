import argparse
from pathlib import Path
from typing import Any, Dict

from adapters.base import C3ManifestScorer, RandomScorer
from adapters.gvl_qwen_adapter import GVLQwenScorer
from adapters.gvl_robometer_adapter import GVLRobometerScorer
from adapters.lrm_adapter import LRMScorer
from adapters.prm_judge_adapter import PRMJudgeScorer
from adapters.robometer_adapter import RobometerScorer
from adapters.roboreward_lrm_adapter import RoboRewardLRMScorer
from utils import load_config


def build_scorer(method: str, config: Dict[str, Any]):
    methods = config.get("methods", {})
    if method not in methods:
        raise KeyError(f"Unknown method '{method}'. Available: {', '.join(sorted(methods))}")
    method_config = methods[method]
    typ = method_config.get("type")
    if typ == "random":
        return RandomScorer(method, config, method_config)
    if typ == "robometer":
        return RobometerScorer(method, config, method_config)
    if typ == "robometer_gvl":
        return GVLRobometerScorer(method, config, method_config)
    if typ == "gvl_qwen":
        return GVLQwenScorer(method, config, method_config)
    if typ == "lrm":
        if method_config.get("mode") == "roboreward":
            return RoboRewardLRMScorer(method, config, method_config)
        return LRMScorer(method, config, method_config)
    if typ == "prm_judge":
        return PRMJudgeScorer(method, config, method_config)
    if typ == "c3":
        return C3ManifestScorer(method, config, method_config)
    raise ValueError(f"Unsupported method type '{typ}' for method '{method}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an EVAC candidate pool with an AL baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "baselines.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    scorer = build_scorer(args.method, config)
    scorer.score_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()
