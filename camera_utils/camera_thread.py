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
        while not self.stopped:
            # Get the frame from the underlying camera object
            ret, frame = self.camera.get_frame()
            
            with self.lock:
                self.ret = ret
                if ret and frame is not None:
                    self.frame = frame.copy()
                    self.timestamp = time.time()
                elif not ret:
                    self.frame = None
                    
            # Small sleep to prevent 100% CPU usage on thread, 
            # while still reading faster than any camera FPS (e.g., 200Hz loop)
            # time.sleep(0.005) 

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
            self.thread.join()
        
        # Only attempt to release if the object has a release method
        if hasattr(self.camera, 'release'):
            print(f"[{self.name}] Releasing underlying camera...")
            self.camera.release()
