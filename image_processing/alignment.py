# image_processing/alignment.py

import cv2
import numpy as np
import config # For points
import logging

logger = logging.getLogger(__name__)

# Note: select_points UI logic is commented out as per original code
# If manual selection is needed again, uncomment and potentially refactor
# def select_points(event, x, y, flags, param):
#     """ Mouse callback for selecting points """
#     points = param
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         print(f"Selected point: {x}, {y}")

def calculate_perspective_matrix(points_vis=config.POINTS_VIS, points_ir=config.POINTS_IR):
    """
    Calculates the perspective transform matrix from visible to IR coordinates.
    Uses points defined in config.py by default.
    """
    logger.info("Calculating perspective transform matrix...")
    if len(points_vis) < 4 or len(points_ir) < 4:
        raise ValueError("At least 4 corresponding points are required for perspective transform.")
    if len(points_vis) != len(points_ir):
         raise ValueError("Visible and IR point lists must have the same length.")

    # Convert to NumPy arrays
    np_points_vis = np.array(points_vis, dtype=np.float32)
    np_points_ir = np.array(points_ir, dtype=np.float32)

    # Calculate the transformation matrix (Visible -> IR)
    matrix, status = cv2.findHomography(np_points_vis, np_points_ir)
    # Or use getPerspectiveTransform if sure about 4 points and no outliers:
    # matrix = cv2.getPerspectiveTransform(np_points_vis, np_points_ir)

    if matrix is None:
         raise RuntimeError("Failed to compute perspective transform matrix.")

    logger.info("Perspective transform matrix calculated successfully.")
    return matrix


def transform_bbox(bbox, matrix):
    """
    Transforms a bounding box dictionary {x1, y1, x2, y2} using the perspective matrix.
    Reduces the overhead of warping the entire image.
    """
    if bbox is None or matrix is None:
        return None
    
    # Original points of the bounding box
    points = np.array([
        [[bbox['x1'], bbox['y1']]],
        [[bbox['x2'], bbox['y1']]],
        [[bbox['x2'], bbox['y2']]],
        [[bbox['x1'], bbox['y2']]]
    ], dtype=np.float32)
    
    # Apply perspective transformation to the 4 corners
    transformed_points = cv2.perspectiveTransform(points, matrix)
    
    # Get the bounding box of the transformed points
    x_coords = transformed_points[:, 0, 0]
    y_coords = transformed_points[:, 0, 1]
    
    new_bbox = bbox.copy()
    new_bbox['x1'] = max(0, float(np.min(x_coords)))
    new_bbox['y1'] = max(0, float(np.min(y_coords)))
    new_bbox['x2'] = max(0, float(np.max(x_coords)))
    new_bbox['y2'] = max(0, float(np.max(y_coords)))
    
    return new_bbox