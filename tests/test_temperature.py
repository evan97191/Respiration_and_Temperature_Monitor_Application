import os
import sys

import numpy as np

# Add project root to path so we can import analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.temperature import calculate_average_pixel_value


def test_calculate_average_pixel_value_valid():
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    box = {"x1": 0, "y1": 0, "x2": 2, "y2": 2}
    # (10+20+30+40)/4 = 25
    assert calculate_average_pixel_value(image, box) == 25.0


def test_calculate_average_pixel_value_none():
    assert calculate_average_pixel_value(None, {"x1": 0, "y1": 0, "x2": 1, "y2": 1}) is None
    assert calculate_average_pixel_value(np.zeros((10, 10)), None) is None


def test_calculate_average_pixel_value_out_of_bounds():
    image = np.ones((10, 10)) * 5
    box = {"x1": -5, "y1": -5, "x2": 15, "y2": 15}
    # Should clip to [0, 10] and return 5.0
    assert calculate_average_pixel_value(image, box) == 5.0


def test_calculate_average_pixel_value_invalid_format():
    image = np.zeros((10, 10))
    box = "invalid"
    assert calculate_average_pixel_value(image, box) is None
