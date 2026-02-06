from dataclasses import dataclass
from typing import Dict

from .config import RuleConfig


@dataclass(frozen=True)
class RuleResult:
    voltage: str
    confidence: float
    matched_rules: int
    total_rules: int
    details: Dict[str, str]


def classify_voltage(level_count: int, wire_density: int, config: RuleConfig) -> RuleResult:
    matched = 0
    total = 3
    details: Dict[str, str] = {}

    if level_count >= config.high_voltage_levels:
        matched += 1
        details["levels"] = "high"
        voltage = config.high_voltage_label
    elif level_count == config.mid_voltage_levels:
        matched += 1
        details["levels"] = "mid"
        voltage = config.mid_voltage_label
    else:
        details["levels"] = "low"
        voltage = config.low_voltage_label

    if wire_density >= 4:
        matched += 1
        details["density"] = "high"
    else:
        details["density"] = "low"

    if level_count >= 1:
        matched += 1
        details["structure"] = "valid"
    else:
        details["structure"] = "missing"

    confidence = matched / total if total else 0.0
    return RuleResult(
        voltage=voltage,
        confidence=confidence,
        matched_rules=matched,
        total_rules=total,
        details=details,
    )
