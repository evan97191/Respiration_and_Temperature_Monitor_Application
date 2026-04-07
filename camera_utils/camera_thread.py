import logging
logger = logging.getLogger(__name__)

import cv2
import threading
import time

class CameraThread:
    def __init__(self, camera_obj, name="CameraThread"):
        self.camera = camera_obj
        self.name = name
        self.frame = None
        self.ret = False
        self.timestamp = 0.0
        
        self.stopped = False
        self.lock = threading.Lock()
        
        # Start immediately upon initialization
        self.thread = threading.Thread(target=self.update, name=self.name, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        # If camera has start_streaming (like thermal), call it here if not active
        if hasattr(self.camera, 'start_streaming') and not getattr(self.camera, 'is_streaming', True):
             try:
                 self.camera.start_streaming()
             except Exception as e:
                 logger.info(f"[{self.name}] Failed to start stream: {e}")

        while not self.stopped:
            # Get the frame from the underlying camera object
            result = self.camera.get_frame()
            
            with self.lock:
                if isinstance(result, tuple):
                    # Visible camera returns (ret, frame)
                    self.ret = result[0]
                    frame = result[1]
                else:
                    # Thermal camera returns just frame (or None)
                    frame = result
                    self.ret = frame is not None
                    
                if self.ret and frame is not None:
                    self.frame = frame.copy()
                    self.timestamp = time.time()
                elif not self.ret:
                    self.frame = None
                    
            # Small sleep to prevent 100% CPU usage on thread, 
            # while still reading faster than any camera FPS (e.g., 200Hz loop)
            time.sleep(0.005) 

    def read(self):
        # Return the most recently read frame
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy(), self.timestamp
            else:
                return self.ret, None, self.timestamp

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"[{self.name}] Thread join timed out after 2 seconds.")
        
        # Only attempt to release if the object has a release method
        if hasattr(self.camera, 'release'):
            logger.info(f"[{self.name}] Releasing underlying camera...")
            self.camera.release()
