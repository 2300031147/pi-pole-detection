from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LineFeature:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def angle_deg(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return float(np.degrees(np.arctan2(dy, dx)))

    @property
    def y_center(self) -> float:
        return float((self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class FeatureSet:
    horizontal_lines: List[LineFeature]
    vertical_lines: List[LineFeature]
    level_centers: List[float]
    insulator_count: int
    wire_density: int


def lines_from_hough(raw_lines: np.ndarray | None) -> List[LineFeature]:
    if raw_lines is None:
        return []
    lines = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        lines.append(LineFeature(int(x1), int(y1), int(x2), int(y2)))
    return lines


def split_lines(
    lines: Iterable[LineFeature], horizontal_tol: float, vertical_tol: float
) -> Tuple[List[LineFeature], List[LineFeature]]:
    horizontal: List[LineFeature] = []
    vertical: List[LineFeature] = []
    for line in lines:
        angle = abs(line.angle_deg)
        if angle <= horizontal_tol or abs(angle - 180) <= horizontal_tol:
            horizontal.append(line)
        elif abs(angle - 90) <= vertical_tol:
            vertical.append(line)
    return horizontal, vertical


def cluster_lines_by_y(lines: Iterable[LineFeature], cluster_px: float) -> List[float]:
    centers = sorted(line.y_center for line in lines)
    if not centers:
        return []
    clusters = [[centers[0]]]
    for center in centers[1:]:
        if abs(center - clusters[-1][-1]) <= cluster_px:
            clusters[-1].append(center)
        else:
            clusters.append([center])
    return [float(np.mean(cluster)) for cluster in clusters]


def count_insulators(
    gray: np.ndarray,
    min_area: int,
    roundness_threshold: float,
) -> int:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        roundness = 4 * np.pi * (area / (perimeter * perimeter))
        if roundness >= roundness_threshold:
            count += 1
    return count


def estimate_wire_density(horizontal_lines: Iterable[LineFeature]) -> int:
    return len(list(horizontal_lines))
