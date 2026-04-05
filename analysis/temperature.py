# analysis/temperature.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_average_pixel_value(image, box):
    """
    Calculates the average pixel value within a bounding box.
    """
    if image is None or box is None:
        # logger.warning("Cannot calculate avg temp, image or box is None.")
        return None

    try:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    except (KeyError, TypeError, ValueError):
        logger.error("Invalid box format for avg temp calculation.")
        return None

    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        # logger.warning("Invalid ROI for avg temp calculation.")
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        # logger.warning("ROI size is zero for avg temp calculation.")
        return None

    try:
        avg_value = np.mean(roi)
        return float(avg_value)
    except Exception as e:
        logger.error(f"Error calculating average pixel value: {e}")
        return None
