from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import AppConfig
from .features import (
    FeatureSet,
    cluster_lines_by_y,
    count_insulators,
    estimate_wire_density,
    lines_from_hough,
    split_lines,
)
from .rules import RuleResult, classify_voltage


@dataclass(frozen=True)
class DetectionResult:
    image_path: str
    image_size: Tuple[int, int]
    features: Dict[str, object]
    rule_result: RuleResult


def preprocess_image(image: np.ndarray, config: AppConfig) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = config.preprocess.blur_kernel
    if kernel > 1:
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return gray


def detect_lines(gray: np.ndarray, config: AppConfig) -> List[np.ndarray]:
    edges = cv2.Canny(gray, config.preprocess.canny_low, config.preprocess.canny_high)
    return cv2.HoughLinesP(
        edges,
        rho=config.lines.hough_rho,
        theta=config.lines.hough_theta,
        threshold=config.lines.hough_threshold,
        minLineLength=config.lines.min_line_length,
        maxLineGap=config.lines.max_line_gap,
    )


def extract_features(gray: np.ndarray, config: AppConfig) -> FeatureSet:
    raw_lines = detect_lines(gray, config)
    lines = lines_from_hough(raw_lines)
    horizontal, vertical = split_lines(
        lines,
        horizontal_tol=config.lines.horizontal_angle_deg,
        vertical_tol=config.lines.vertical_angle_deg,
    )
    level_centers = cluster_lines_by_y(horizontal, config.levels.y_cluster_px)
    insulators = count_insulators(
        gray,
        min_area=config.insulators.min_area,
        roundness_threshold=config.insulators.roundness_threshold,
    )
    wire_density = estimate_wire_density(horizontal)
    return FeatureSet(
        horizontal_lines=horizontal,
        vertical_lines=vertical,
        level_centers=level_centers,
        insulator_count=insulators,
        wire_density=wire_density,
    )


def detect_from_image(image_path: Path, config: AppConfig) -> DetectionResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    gray = preprocess_image(image, config)
    features = extract_features(gray, config)
    rule_result = classify_voltage(
        level_count=len(features.level_centers),
        wire_density=features.wire_density,
        config=config.rules,
    )
    return DetectionResult(
        image_path=str(image_path),
        image_size=(width, height),
        features={
            "horizontal_lines": len(features.horizontal_lines),
            "vertical_lines": len(features.vertical_lines),
            "level_centers": features.level_centers,
            "insulator_count": features.insulator_count,
            "wire_density": features.wire_density,
        },
        rule_result=rule_result,
    )


def result_to_dict(result: DetectionResult) -> Dict[str, object]:
    data = asdict(result)
    data["rule_result"] = asdict(result.rule_result)
    return data
