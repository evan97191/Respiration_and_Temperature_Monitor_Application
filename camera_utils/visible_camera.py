# camera_utils/visible_camera.py

import cv2
import config

class VisibleCamera:
    """Handles Visible Light Camera interaction using GStreamer."""

    def __init__(self, pipeline=config.GST_PIPELINE):
        print("Initializing visible camera with GStreamer pipeline...")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera using pipeline: {pipeline}")
            raise ConnectionError("Failed to open GStreamer pipeline")
        print("Visible camera initialized.")
        # Get default FPS (might not be accurate for GStreamer)
        self.default_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.default_fps <= 0:
             self.default_fps = config.DEFAULT_FPS # Fallback from config
        print(f"Camera default FPS reported: {self.default_fps} (using {config.DEFAULT_FPS} as fallback if needed)")


    def get_frame(self):
        """Reads a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Warning: Failed to capture frame from visible camera.")
            return False, None
            
        # frame = cv2.resize(frame, (960,616))
        return True, frame

    def is_opened(self):
        """Checks if the camera capture is open."""
        return self.cap.isOpened()

    def get_default_fps(self):
        """Returns the default FPS reported by the camera (or fallback)."""
        return self.default_fps

    def release(self):
        """Releases the camera capture resource."""
        if self.cap.isOpened():
            print("Releasing visible camera...")
            self.cap.release()
            print("Visible camera released.")