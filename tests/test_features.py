import numpy as np

from pi_pole_detection.features import LineFeature, cluster_lines_by_y, split_lines


def test_cluster_lines_by_y_groups_close_centers():
    lines = [
        LineFeature(0, 10, 10, 10),
        LineFeature(0, 12, 10, 12),
        LineFeature(0, 40, 10, 40),
    ]
    centers = cluster_lines_by_y(lines, cluster_px=5)
    assert len(centers) == 2
    assert np.isclose(centers[0], 11.0)
    assert np.isclose(centers[1], 40.0)


def test_split_lines_separates_horizontal_and_vertical():
    lines = [
        LineFeature(0, 0, 10, 0),
        LineFeature(0, 0, 0, 10),
    ]
    horizontal, vertical = split_lines(lines, horizontal_tol=10, vertical_tol=10)
    assert len(horizontal) == 1
    assert len(vertical) == 1
