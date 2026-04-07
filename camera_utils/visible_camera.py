# camera_utils/visible_camera.py

import logging

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class VisibleCamera:
    """Handles Visible Light Camera interaction using GStreamer."""

    def __init__(self, pipeline=None):
        logger.info("Initializing visible camera with GStreamer pipeline...")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            logger.error(f"Error: Could not open camera using pipeline: {pipeline}")
            raise RuntimeError("Visible camera failed to initialize.")
        logger.info("Visible camera initialized.")
        # Get default FPS (might not be accurate for GStreamer)
        self.default_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.default_fps <= 0 or np.isnan(self.default_fps):
            self.default_fps = config.DEFAULT_FPS  # Fallback
        logger.info(
            f"Camera default FPS reported: {self.default_fps} (using {config.DEFAULT_FPS} as fallback if needed)"
        )
        self.error_count = 0

    def get_frame(self):
        """Reads a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            self.error_count += 1
            if self.error_count > 5:
                logger.error(f"Visible camera failed {self.error_count} times in a row.")
            else:
                logger.warning(f"Failed to capture frame from visible camera. (Attempt {self.error_count})")
            return False, None

        self.error_count = 0

        return True, frame

    def get_default_fps(self):
        """Returns the default FPS reported by the camera (or fallback)."""
        return self.default_fps

    def release(self):
        """Releases the camera capture resource."""
        if hasattr(self, "cap") and self.cap.isOpened():
            logger.info("Releasing visible camera...")
            self.cap.release()
            logger.info("Visible camera released.")
