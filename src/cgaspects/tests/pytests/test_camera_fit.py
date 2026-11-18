"""Tests for fitting the camera to a set of points."""

import numpy as np
import pytest
from PySide6.QtGui import QVector3D

from cgaspects.gui.visualisation.camera import Camera


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_fit_keeps_a_valid_view_for_any_long_axis(axis):
    rng = np.random.default_rng(0)
    points = rng.normal(scale=0.5, size=(200, 3))
    points[:, axis] *= 20

    camera = Camera()
    camera.fitToObject(points)

    view = (camera.target - camera.position).normalized()
    assert camera.up.length() > 1e-6
    assert abs(QVector3D.dotProduct(camera.right.normalized(), view)) < 0.99
