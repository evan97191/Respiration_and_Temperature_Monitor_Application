# camera_utils/visible_camera.py

import cv2
import config
import logging
import numpy as np

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
             self.default_fps = config.DEFAULT_FPS # Fallback
        logger.info(f"Camera default FPS reported: {self.default_fps} (using {config.DEFAULT_FPS} as fallback if needed)")


    def get_frame(self):
        """Reads a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to capture frame from visible camera.")
            return False, None
            
        # frame = cv2.resize(frame, (960,616))
        return True, frame

    def get_default_fps(self):
        """Returns the default FPS reported by the camera (or fallback)."""
        return self.default_fps

    def release(self):
        """Releases the camera capture resource."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            logger.info("Releasing visible camera...")
            self.cap.release()
            logger.info("Visible camera released.")