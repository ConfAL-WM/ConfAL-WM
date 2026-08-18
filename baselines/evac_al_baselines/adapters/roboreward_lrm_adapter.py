from typing import Any, Dict

from adapters.lrm_adapter import LRMScorer


class RoboRewardLRMScorer(LRMScorer):
    """RoboReward baseline routed through LRMs roboreward mode."""

    def __init__(self, method: str, config: Dict[str, Any], method_config: Dict[str, Any]):
        method_config = dict(method_config)
        method_config["mode"] = "roboreward"
        method_config.setdefault("endpoint", "/compute_roboreward")
        super().__init__(method, config, method_config)

