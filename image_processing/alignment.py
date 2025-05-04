# image_processing/alignment.py

import cv2
import numpy as np
import config # For points

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
    print("Calculating perspective transform matrix...")
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

    print("Perspective transform matrix calculated successfully.")
    return matrix

def apply_perspective(frame, matrix, target_size=(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)):
    """
    Applies the perspective transformation matrix to align the frame.
    Target size should match the dimensions of the target coordinate system (e.g., the IR frame size).
    """
    if frame is None or matrix is None:
        print("Warning: Cannot apply perspective, frame or matrix is None.")
        return None
    # Note: target_size should be (width, height) for cv2.warpPerspective
    aligned_frame = cv2.warpPerspective(frame, matrix, target_size)
    return aligned_frame