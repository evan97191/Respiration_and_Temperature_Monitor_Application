import cv2
import numpy as np
import time
from ctypes import *
try:
    from queue import Queue, Empty # Python 3
except ImportError:
    from Queue import Queue, Empty # Python 2

# --- Import UVC specifics ---
try:
    # Assuming uvctypes.py is in the same directory or PYTHONPATH
    from uvctypes import *
except ImportError:
    print("ERROR: uvctypes.py not found. Make sure it's accessible.")
    exit(1)

# --- Configuration ---
THERMAL_VID = PT_USB_VID # Use VID from uvctypes
THERMAL_PID = PT_USB_PID # Use PID from uvctypes
THERMAL_BUFFER_SIZE = 2
DISPLAY_WIDTH = 640 # Desired display width
DISPLAY_HEIGHT = 480 # Desired display height
ROI_COLOR = (0, 255, 0) # Green for ROI box
TEMP_TEXT_COLOR = (0, 0, 255) # Red for temperature text

# --- Global variables for ROI selection ---
roi_selected = False
roi_coords = None # Will store (x1, y1, x2, y2)
drawing = False
start_point = (-1, -1)
current_roi_frame = None # To display drawing rectangle

# --- Thermal Data Queue ---
frame_queue = Queue(THERMAL_BUFFER_SIZE)

# --- UVC Frame Callback ---
def py_frame_callback(frame, userptr):
    global frame_queue
    try:
        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        # Make a copy to own the data before putting in queue
        data = np.copy(np.frombuffer(
            array_pointer.contents, dtype=np.dtype(np.uint16)
        )).reshape(frame.contents.height, frame.contents.width)

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            print("Warning: Thermal frame data bytes mismatch.")
            return

        if not frame_queue.full():
            frame_queue.put(data)
    except Exception as e:
        print(f"Error in thermal frame callback: {e}")

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

# --- Temperature Conversion Functions (copied from previous example) ---
def temp_correction(temp):
    """ Applies polynomial temperature correction. """
    Tx = np.array([30760, 30850, 30950, 31040, 31120, 31260, 31360, 31420, 31520, 31570])
    Tc = np.array([3200, 3300, 3400, 3500, 3600, 3800, 3900, 4000, 4100, 4200])
    try:
        # Check if polyfit can be done (needs at least degree+1 points)
        if len(Tx) > 2:
            z = np.polyfit(Tx, Tc, 2)
            pp = np.poly1d(z)
            return pp(temp)
        else: # Fallback linear interpolation or return raw if too few points
             print("Warning: Not enough points for polynomial temp correction, using linear.")
             if len(Tx) == 2:
                 return np.interp(temp, Tx, Tc)
             else:
                 return temp # Return raw K*100 if only one point or less
    except Exception as e:
        print(f"Error during temp_correction: {e}")
        return temp # Return raw K*100 on error


def ktoc(val):
    """ Converts Kelvin * 100 to Celsius using correction. """
    if val is None:
        return None
    try:
        corrected_val = temp_correction(val)
        return corrected_val / 100.0
    except Exception as e:
        print(f"Error in ktoc conversion: {e}")
        return None

def raw_to_8bit(data):
    """ Converts raw 16-bit thermal data to an 8-bit BGR image. """
    if data is None:
        return None
    try:
        data_copy = data.copy()
        cv2.normalize(data_copy, data_copy, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(data_copy, 8, data_copy)
        img_bgr = cv2.cvtColor(np.uint8(data_copy), cv2.COLOR_GRAY2BGR)
        return img_bgr
    except Exception as e:
        print(f"Error in raw_to_8bit conversion: {e}")
        return None

# --- Mouse Callback for ROI Selection ---
def select_roi(event, x, y, flags, param):
    global start_point, roi_coords, roi_selected, drawing, current_roi_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True
        # Clear previous roi_coords if re-selecting
        roi_coords = None
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw rectangle on a copy for preview
            preview_frame = current_roi_frame.copy()
            cv2.rectangle(preview_frame, start_point, (x, y), ROI_COLOR, 1)
            cv2.imshow("Select ROI", preview_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Ensure x1 < x2 and y1 < y2
        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        # Check if the ROI has valid dimensions
        if x2 > x1 and y2 > y1:
            roi_coords = (x1, y1, x2, y2)
            roi_selected = True
            print(f"ROI Selected: {roi_coords}")
            # Draw final ROI on the frame used for selection
            cv2.rectangle(current_roi_frame, (x1, y1), (x2, y2), (0,0,255), 2) # Red final box
            cv2.imshow("Select ROI", current_roi_frame)
            time.sleep(0.5) # Briefly show the final selection
            cv2.destroyWindow("Select ROI") # Close selection window
        else:
            print("Invalid ROI selected (zero width or height). Please try again.")
            # Reset ROI display if selection was invalid
            cv2.imshow("Select ROI", current_roi_frame)


if __name__ == "__main__":
    # --- Initialize UVC Camera ---
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()
    devh_ptr = None # To check if device was opened

    try:
        print("Initializing UVC context...")
        res = libuvc.uvc_init(byref(ctx), 0)
        if res < 0: raise RuntimeError("uvc_init error")

        print(f"Finding UVC device (VID={THERMAL_VID:#0x}, PID={THERMAL_PID:#0x})...")
        res = libuvc.uvc_find_device(ctx, byref(dev), THERMAL_VID, THERMAL_PID, 0)
        if res < 0: raise RuntimeError("uvc_find_device error")

        print("Opening UVC device...")
        res = libuvc.uvc_open(dev, byref(devh))
        if res < 0: raise RuntimeError(f"uvc_open error {res}")
        devh_ptr = devh # Store handle pointer

        print("Getting frame formats...")
        frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
        if not frame_formats: raise RuntimeError("Device does not support Y16 format")

        print("Getting stream control format size...")
        target_width = frame_formats[0].wWidth
        target_height = frame_formats[0].wHeight
        target_fps = int(1e7 / frame_formats[0].dwDefaultFrameInterval)

        res = libuvc.uvc_get_stream_ctrl_format_size(
            devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
            target_width, target_height, target_fps
        )
        if res < 0: raise RuntimeError(f"uvc_get_stream_ctrl_format_size error {res}")

        print("Device initialized successfully.")

        # --- Start Streaming Briefly for ROI Selection Frame ---
        print("Starting stream to get first frame for ROI selection...")
        res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        if res < 0: raise RuntimeError("Could not start UVC stream")

        print("Waiting for the first valid frame...")
        first_raw_frame = None
        start_wait_time = time.time()
        while first_raw_frame is None:
             try:
                  # Use a slightly longer timeout for the first frame
                  first_raw_frame = frame_queue.get(True, 2.0)
             except Empty:
                  print(".")
                  if time.time() - start_wait_time > 10: # Timeout after 10 seconds
                      raise TimeoutError("Timeout waiting for the first thermal frame.")
                  time.sleep(0.1)
        print("First frame received.")

        # --- Stop Streaming Temporarily (optional, but safer for blocking UI) ---
        # libuvc.uvc_stop_streaming(devh)
        # print("Stream stopped for ROI selection.")

        # --- ROI Selection Phase ---
        print("\nPlease select the Region of Interest (ROI) by clicking and dragging.")
        print("Press any key in the 'Select ROI' window after selection.")

        # Resize the raw frame for consistent display size before converting
        first_raw_frame_resized = cv2.resize(first_raw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        current_roi_frame = raw_to_8bit(first_raw_frame_resized) # Convert the resized frame to 8-bit for display

        if current_roi_frame is None:
             raise RuntimeError("Failed to convert first frame to 8-bit for ROI selection.")

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", select_roi)

        while not roi_selected:
            # Display the frame where selection happens
            display_frame_roi = current_roi_frame.copy() # Work on a copy
            if drawing and start_point != (-1,-1): # Draw temporary rectangle if needed (handled in callback now)
                 pass # Callback handles preview drawing
            elif roi_coords: # Draw final ROI if already selected but loop is waiting
                 cv2.rectangle(display_frame_roi, (roi_coords[0],roi_coords[1]), (roi_coords[2],roi_coords[3]), (0,0,255), 2)

            cv2.putText(display_frame_roi, "Click and drag to select ROI, then press any key", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("Select ROI", display_frame_roi)

            key = cv2.waitKey(20) & 0xFF
            if roi_selected and key != 255: # Check if key pressed *after* selection
                break # Exit loop once ROI is selected and a key is pressed
            if key == ord('q'): # Allow quitting during selection
                 print("Quit during ROI selection.")
                 roi_selected = False # Ensure loop breaks below
                 break

        if not roi_selected:
            raise InterruptedError("ROI selection was cancelled or failed.")

        # --- Continuous Processing Phase ---
        print("ROI selected. Starting continuous temperature monitoring...")
        # Restart streaming if it was stopped
        # res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        # if res < 0: raise RuntimeError("Could not restart UVC stream")

        cv2.namedWindow("Thermal Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Thermal Feed", DISPLAY_WIDTH + 1, DISPLAY_HEIGHT + 1)

        while True:
            time.sleep(0.25)
            raw_frame = None
            try:
                raw_frame = frame_queue.get(True, 0.5) # Get subsequent frames
            except Empty:
                # print("Waiting for frame...") # Can be noisy
                continue # Skip if no frame available

            if raw_frame is None:
                continue

            # Resize raw frame to match the display and ROI coordinates system
            raw_frame_resized = cv2.resize(raw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

            max_temp_celsius = None
            # Calculate max temperature within ROI
            if roi_coords:
                x1, y1, x2, y2 = roi_coords
                try:
                    # Crop from the *resized* raw frame
                    roi_raw_data = raw_frame_resized[y1:y2, x1:x2]

                    if roi_raw_data.size > 0:
                        max_val_raw = np.max(roi_raw_data)
                        max_temp_celsius = ktoc(max_val_raw)
                    else:
                        # print("Warning: ROI crop resulted in empty array.") # Can be noisy
                        pass
                except Exception as e:
                    print(f"Error processing ROI: {e}")

            # Prepare display frame (convert resized raw frame to 8-bit BGR)
            display_frame = raw_to_8bit(raw_frame_resized)
            if display_frame is None:
                 print("Warning: Failed to convert frame to 8-bit for display.")
                 # Show a black screen on error
                 display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                 cv2.putText(display_frame, "Display Error", (50, DISPLAY_HEIGHT // 2),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)


            # Draw ROI box on display frame
            if roi_coords:
                cv2.rectangle(display_frame, (roi_coords[0], roi_coords[1]),
                              (roi_coords[2], roi_coords[3]), ROI_COLOR, 1)

            # Display calculated temperature
            if max_temp_celsius is not None:
                temp_text = f"Max ROI: {max_temp_celsius:.1f} C"
                # --- 修改這裡 ---
                font_scale = 2.0  # 增加字體大小 (例如從 0.6 增加到 1.0 或更大)
                thickness = 2     # 增加線條粗細 (例如從 1 增加到 2)
                # 計算文字位置，可以稍微向上調整以容納更大的字體
                text_y_offset = 15 # 向上偏移量
                text_pos = (roi_coords[0]//2, max(15, roi_coords[1] - text_y_offset)) # 確保文字不會超出頂部
                
                cv2.putText(display_frame, temp_text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEMP_TEXT_COLOR, thickness)

            # Show the frame
            cv2.resizeWindow("Thermal Feed", 1200+1, 900+1)
            cv2.imshow("Thermal Feed", display_frame)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed.")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- Cleanup ---
        print("Cleaning up...")
        if devh_ptr: # Check if device was successfully opened
            print("Stopping UVC stream...")
            libuvc.uvc_stop_streaming(devh_ptr)
            print("Closing UVC device...")
            libuvc.uvc_close(devh_ptr)
        if dev: # Check if device was found
            print("Unreferencing UVC device...")
            libuvc.uvc_unref_device(dev)
        if ctx: # Check if context was initialized
            print("Exiting UVC context...")
            libuvc.uvc_exit(ctx)
        cv2.destroyAllWindows()
        print("Resources released.")