# camera_utils/thermal_camera.py

import time
# from queue import Queue # Remove this line if only using the try/except below
from ctypes import *
import numpy as np
import cv2

# Import UVC types and constants - assumes uvctypes.py is in project root
import sys
import os
# Add project root to path to find uvctypes
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Comment out if uvctypes is now found via standard path
try:
    from uvctypes import *
except ImportError:
    print("ERROR: uvctypes.py not found. Make sure it's in the project root or PYTHONPATH.")
    sys.exit(1)

# Import config for constants
try:
    import config
except ModuleNotFoundError:
    print("ERROR: config.py not found. Make sure it's in the project root.")
    sys.exit(1)


# --- Corrected Queue and Empty Exception Import ---
try:
  from queue import Queue, Empty # Python 3: Import both Queue and Empty
except ImportError:
  from Queue import Queue, Empty # Python 2: Import both Queue and Empty
# --------------------------------------------------


# --- Frame Callback ---
# Needs access to the queue, make it global or pass via userptr
frame_queue = Queue(config.THERMAL_BUFFER_SIZE)

def py_frame_callback(frame, userptr):
    """Internal callback function to put frames into the queue."""
    global frame_queue
    try:
        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        data = np.copy(np.frombuffer(
            array_pointer.contents, dtype=np.dtype(np.uint16)
        )).reshape(
            frame.contents.height, frame.contents.width
        )

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            print("Warning: Thermal frame data bytes mismatch.")
            return

        if not frame_queue.full():
            frame_queue.put(data)
        # else:
        #    print("Warning: Thermal frame queue is full. Frame dropped.") # Optional warning
    except Exception as e:
        print(f"Error in thermal frame callback: {e}")

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

class ThermalCameraUVC:
    """Handles UVC Thermal Camera interaction."""

    def __init__(self, vid=config.THERMAL_VID, pid=config.THERMAL_PID):
        self.ctx = POINTER(uvc_context)()
        self.dev = POINTER(uvc_device)()
        self.devh = POINTER(uvc_device_handle)()
        self.ctrl = uvc_stream_ctrl()
        self.is_streaming = False
        self.vid = vid
        self.pid = pid

        print("Initializing UVC context...")
        res = libuvc.uvc_init(byref(self.ctx), 0)
        if res < 0:
            print("uvc_init error")
            raise RuntimeError("Could not initialize UVC context")

        print(f"Finding UVC device (VID={self.vid:#0x}, PID={self.pid:#0x})...")
        res = libuvc.uvc_find_device(self.ctx, byref(self.dev), self.vid, self.pid, 0)
        if res < 0:
            print("uvc_find_device error")
            libuvc.uvc_exit(self.ctx)
            raise RuntimeError("Could not find UVC device")

        print("Opening UVC device...")
        res = libuvc.uvc_open(self.dev, byref(self.devh))
        if res < 0:
            print(f"uvc_open error {res}")
            libuvc.uvc_unref_device(self.dev)
            libuvc.uvc_exit(self.ctx)
            raise RuntimeError("Could not open UVC device")

        print("Device opened successfully.")
        # print_device_info(self.devh) # Optional: Print device info

        print("Getting frame formats...")
        frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_Y16)
        if not frame_formats:
             print("Device does not support Y16 format.")
             self.release() # Clean up before raising error
             raise RuntimeError("Device does not support Y16 format")

        print("Getting stream control format size...")
        # Use the first available Y16 format
        target_width = frame_formats[0].wWidth
        target_height = frame_formats[0].wHeight
        target_fps = int(1e7 / frame_formats[0].dwDefaultFrameInterval)
        print(f"Requesting format: {target_width}x{target_height} @ {target_fps}fps")

        res = libuvc.uvc_get_stream_ctrl_format_size(
            self.devh, byref(self.ctrl), UVC_FRAME_FORMAT_Y16,
            target_width, target_height, target_fps
        )
        if res < 0:
            print(f"uvc_get_stream_ctrl_format_size error {res}")
            self.release()
            raise RuntimeError("Could not get stream control format size")

        # Clear the queue initially
        while not frame_queue.empty():
            frame_queue.get()

        print("Thermal camera initialized.")

    def start_streaming(self):
        """Starts the UVC stream."""
        if self.is_streaming:
            print("Streaming is already active.")
            return
        print("Starting UVC stream...")
        res = libuvc.uvc_start_streaming(self.devh, byref(self.ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        if res < 0:
            print(f"uvc_start_streaming error {res}")
            self.release()
            raise RuntimeError("Could not start UVC stream")
        self.is_streaming = True
        print("UVC stream started.")

    def stop_streaming(self):
        """Stops the UVC stream."""
        if self.is_streaming and self.devh:
            print("Stopping UVC stream...")
            libuvc.uvc_stop_streaming(self.devh)
            self.is_streaming = False
            print("UVC stream stopped.")
        else:
             print("Stream not active or device handle invalid.")

    def get_frame(self, timeout=0.5):
        """Gets a frame from the queue."""
        try:
            data = frame_queue.get(True, timeout) # Wait with timeout
            data_resized = cv2.resize(data, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                                      interpolation=cv2.INTER_NEAREST) # Nearest for temp data
            return data_resized
        except Empty: # <<< --- Catch the imported 'Empty' exception directly ---
            # print("Warning: Thermal frame queue timed out.") # Can be noisy
            return None
        except Exception as e:
            print(f"Error getting thermal frame: {e}")
            return None

    def release(self):
        """Releases UVC resources."""
        print("Releasing UVC resources...")
        self.stop_streaming() # Ensure stream is stopped first
        if self.devh:
            libuvc.uvc_close(self.devh)
            self.devh = None # Mark as closed
            print("UVC device closed.")
        if self.dev:
            libuvc.uvc_unref_device(self.dev)
            self.dev = None # Mark as unreferenced
            print("UVC device unreferenced.")
        if self.ctx:
            libuvc.uvc_exit(self.ctx)
            self.ctx = None # Mark as exited
            print("UVC context exited.")
        print("UVC resources released.")