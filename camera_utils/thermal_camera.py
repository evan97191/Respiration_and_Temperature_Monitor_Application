import logging
import sys
from ctypes import *
from queue import Empty, Queue

import numpy as np

import config

try:
    from uvctypes import *
except ImportError:
    # We delay the logger.error until after imports are settled
    pass

logger = logging.getLogger(__name__)

# Re-check uvctypes with logger available
try:
    from uvctypes import *
except ImportError:
    logger.error("ERROR : uvctypes.py not found. Make sure it's in the project root or PYTHONPATH.")
    sys.exit(1)



# --- Frame Callback ---
def py_frame_callback(frame, userptr):
    """Internal callback function to put frames into the queue."""
    try:
        frame_queue = cast(userptr, py_object).value
        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        data = np.copy(np.frombuffer(
            array_pointer.contents, dtype=np.dtype(np.uint16)
        )).reshape(
            frame.contents.height, frame.contents.width
        )

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            logger.warning("Warning: Thermal frame data bytes mismatch.")
            return

        # Always keep the latest frame. If the queue is full, remove the oldest one.
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put(data)
    except Exception as e:
        logger.error(f"Error in thermal frame callback: {e}")

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

        self.frame_queue = Queue(config.THERMAL_BUFFER_SIZE)

        logger.info("Initializing UVC context...")
        res = libuvc.uvc_init(byref(self.ctx), 0)
        if res < 0:
            logger.error("uvc_init error ")
            raise RuntimeError("Could not initialize UVC context")

        logger.info(f"Finding UVC device (VID={self.vid:#0x}, PID={self.pid:#0x})...")
        res = libuvc.uvc_find_device(self.ctx, byref(self.dev), self.vid, self.pid, 0)
        if res < 0:
            logger.error("uvc_find_device error ")
            libuvc.uvc_exit(self.ctx)
            raise RuntimeError("Could not find UVC device")

        logger.info("Opening UVC device...")
        res = libuvc.uvc_open(self.dev, byref(self.devh))
        if res < 0:
            logger.error(f"uvc_open error {res}")
            libuvc.uvc_unref_device(self.dev)
            libuvc.uvc_exit(self.ctx)
            raise RuntimeError("Could not open UVC device")

        logger.info("Device opened successfully.")
        # print_device_info(self.devh) # Optional: Print device info

        logger.info("Getting frame formats...")
        frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_Y16)
        if not frame_formats:
             logger.info("Device does not support Y16 format.")
             self.release() # Clean up before raising error
             raise RuntimeError("Device does not support Y16 format")

        logger.info("Getting stream control format size...")
        # Use the first available Y16 format
        target_width = frame_formats[0].wWidth
        target_height = frame_formats[0].wHeight
        target_fps = int(1e7 / frame_formats[0].dwDefaultFrameInterval)
        logger.info(f"Requesting format: {target_width}x{target_height} @ {target_fps}fps")

        res = libuvc.uvc_get_stream_ctrl_format_size(
            self.devh, byref(self.ctrl), UVC_FRAME_FORMAT_Y16,
            target_width, target_height, target_fps
        )
        if res < 0:
            logger.error(f"uvc_get_stream_ctrl_format_size error {res}")
            self.release()
            raise RuntimeError("Could not get stream control format size")

        # Clear the queue initially
        while not self.frame_queue.empty():
            self.frame_queue.get()

        logger.info("Thermal camera initialized.")

    def start_streaming(self):
        """Starts the UVC stream."""
        if self.is_streaming:
            logger.info("Streaming is already active.")
            return
        logger.info("Starting UVC stream...")
        self._userptr_q = py_object(self.frame_queue)
        res = libuvc.uvc_start_streaming(self.devh, byref(self.ctrl), PTR_PY_FRAME_CALLBACK, self._userptr_q, 0)
        if res < 0:
            logger.error(f"uvc_start_streaming error {res}")
            self.release()
            raise RuntimeError("Could not start UVC stream")
        self.is_streaming = True
        logger.info("UVC stream started.")

    def stop_streaming(self):
        """Stops the UVC stream."""
        if self.is_streaming and self.devh:
            logger.info("Stopping UVC stream...")
            libuvc.uvc_stop_streaming(self.devh)
            self.is_streaming = False
            logger.info("UVC stream stopped.")
        else:
             logger.info("Stream not active or device handle invalid.")

    def get_frame(self, timeout=0.5):
        """Gets a frame from the queue."""
        try:
            data = self.frame_queue.get(True, timeout) # Wait with timeout
            return data
        except Empty: # <<< --- Catch the imported 'Empty' exception directly ---
            # logger.warning(f"Warning: Thermal frame queue timed out.") # Can be noisy
            return None
        except Exception as e:
            logger.error(f"Error getting thermal frame: {e}")
            return None

    def release(self):
        """Releases UVC resources."""
        logger.info("Releasing UVC resources...")
        self.stop_streaming() # Ensure stream is stopped first
        if self.devh:
            libuvc.uvc_close(self.devh)
            self.devh = None # Mark as closed
            logger.info("UVC device closed.")
        if self.dev:
            libuvc.uvc_unref_device(self.dev)
            self.dev = None # Mark as unreferenced
            logger.info("UVC device unreferenced.")
        if self.ctx:
            libuvc.uvc_exit(self.ctx)
            self.ctx = None # Mark as exited
            logger.info("UVC context exited.")
        logger.info("UVC resources released.")
