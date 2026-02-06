from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class PreprocessConfig:
    blur_kernel: int
    canny_low: int
    canny_high: int


@dataclass(frozen=True)
class LineConfig:
    hough_rho: float
    hough_theta: float
    hough_threshold: int
    min_line_length: int
    max_line_gap: int
    horizontal_angle_deg: float
    vertical_angle_deg: float


@dataclass(frozen=True)
class LevelConfig:
    y_cluster_px: int


@dataclass(frozen=True)
class InsulatorConfig:
    min_area: int
    roundness_threshold: float


@dataclass(frozen=True)
class RuleConfig:
    high_voltage_levels: int
    mid_voltage_levels: int
    high_voltage_label: str
    mid_voltage_label: str
    low_voltage_label: str


@dataclass(frozen=True)
class AppConfig:
    preprocess: PreprocessConfig
    lines: LineConfig
    levels: LevelConfig
    insulators: InsulatorConfig
    rules: RuleConfig


def _section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in data:
        raise KeyError(f"Missing '{key}' config section")
    return data[key]


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    preprocess = _section(data, "preprocess")
    lines = _section(data, "lines")
    levels = _section(data, "levels")
    insulators = _section(data, "insulators")
    rules = _section(data, "rules")

    return AppConfig(
        preprocess=PreprocessConfig(**preprocess),
        lines=LineConfig(**lines),
        levels=LevelConfig(**levels),
        insulators=InsulatorConfig(**insulators),
        rules=RuleConfig(**rules),
    )
