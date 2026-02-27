"""
scenarios.py

Defines structured macro stress scenarios for the lending model.
Each scenario returns a modified parameter dictionary.
"""

from typing import Dict
from copy import deepcopy
from src.config import base_case


def apply_recession_scenario(params: Dict) -> Dict:
    """
    Recession scenario:
    - Higher defaults
    - Lower origination volume
    - Higher CAC
    - Slightly higher funding cost
    """
    shocked = deepcopy(params)

    shocked["default_rate_mean"] *= 1.75
    shocked["origination_volume_mean"] *= 0.70
    shocked["cac_mean"] *= 1.25
    shocked["funding_cost_mean"] += 0.015

    return shocked


def apply_rate_spike_scenario(params: Dict) -> Dict:
    """
    Rate spike scenario:
    - Funding costs increase significantly
    - Volume declines modestly
    """
    shocked = deepcopy(params)

    shocked["funding_cost_mean"] += 0.03
    shocked["origination_volume_mean"] *= 0.85

    return shocked


def apply_competition_scenario(params: Dict) -> Dict:
    """
    Competitive pressure scenario:
    - Lower take rate
    - Higher CAC
    """
    shocked = deepcopy(params)

    shocked["take_rate_mean"] *= 0.80
    shocked["cac_mean"] *= 1.30

    return shocked