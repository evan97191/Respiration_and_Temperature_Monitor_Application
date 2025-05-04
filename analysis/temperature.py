# analysis/temperature.py

import numpy as np

def calculate_average_pixel_value(image, box):
    """ Calculates the average pixel value within a bounding box in an image. """
    if image is None or box is None:
        # print("Warning: Cannot calculate avg temp, image or box is None.")
        return None

    # Use basic_ops.cut_roi if refactored there, or keep logic here
    # For simplicity, keeping logic here but using the roi cutting concept:
    try:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    except (KeyError, TypeError):
        print("Error: Invalid box format for avg temp calculation.")
        return None

    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x1 >= x2 or y1 >= y2:
        # print("Warning: Invalid ROI for avg temp calculation.")
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        # print("Warning: ROI size is zero for avg temp calculation.")
        return None

    try:
        # Works for grayscale (thermal) or calculates mean across all channels if color
        avg_pixel_value = np.mean(roi)
        return avg_pixel_value
    except Exception as e:
        print(f"Error calculating average pixel value: {e}")
        return None


def calculate_maximum_pixel_value(image, box):
    """ Calculates the maximum pixel value within a bounding box in an image. """
    if image is None or box is None:
        # print("Warning: Cannot calculate max temp, image or box is None.")
        return None

    try:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    except (KeyError, TypeError):
        print("Error: Invalid box format for max temp calculation.")
        return None

    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x1 >= x2 or y1 >= y2:
        # print("Warning: Invalid ROI for max temp calculation.")
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        # print("Warning: ROI size is zero for max temp calculation.")
        return None

    try:
        max_pixel_value = np.max(roi)
        return max_pixel_value
    except Exception as e:
        print(f"Error calculating maximum pixel value: {e}")
        return None