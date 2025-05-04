# main_app.py

import time
import cv2
import numpy as np
import torch # Keep torch import for device check maybe

# Import configuration
import config

# Import utilities
from utils.timing import FPSTracker
from utils.visualization import DisplayManager, draw_bounding_box, display_value

# Import camera handlers
from camera_utils.thermal_camera import ThermalCameraUVC
from camera_utils.visible_camera import VisibleCamera

# Import image processing
from image_processing.alignment import calculate_perspective_matrix, apply_perspective
from image_processing.basic_ops import raw_to_8bit, cut_roi

# Import models
from models.detector import YoloDetector
from models.segmenter import UNetSegmenter

# Import analysis
from analysis.temperature import calculate_average_pixel_value, calculate_maximum_pixel_value
from analysis.respiration import update_temperature_queue, calculate_respiration_fft

def main():
    print(f"Using device: {config.DEVICE}")

    # --- Initialization ---
    fps_tracker = FPSTracker(buffer_size=10) # Use class based tracker
    display_manager = DisplayManager([
        config.WINDOW_CAMERA,
        config.WINDOW_THERMAL,
        config.WINDOW_HEAD_OVERLAY,
        config.WINDOW_HEAD_SEGMENTED
    ], default_width=config.DISPLAY_WIDTH, default_height=config.DISPLAY_HEIGHT)

    try:
        thermal_cam = ThermalCameraUVC(vid=config.THERMAL_VID, pid=config.THERMAL_PID)
        visible_cam = VisibleCamera(pipeline=config.GST_PIPELINE)
        detector = YoloDetector(model_path=config.YOLO_MODEL_PATH)
        segmenter = UNetSegmenter(model_path=config.UNET_MODEL_PATH, device=config.DEVICE)
    except Exception as e:
        print(f"FATAL: Initialization failed: {e}")
        return # Exit if essential components fail

    # --- Initial Setup ---
    # Get initial frames for perspective calculation
    print("Getting initial frames for alignment...")
    initial_thermal_frame = None
    while initial_thermal_frame is None:
         thermal_cam.start_streaming() # Ensure stream is running
         initial_thermal_frame = thermal_cam.get_frame()
         if initial_thermal_frame is None:
              print("Waiting for initial thermal frame...")
              time.sleep(0.5)
         thermal_cam.stop_streaming() # Stop after getting one frame for now

    ret_vis, initial_visible_frame = visible_cam.get_frame()
    if not ret_vis or initial_visible_frame is None:
        print("FATAL: Failed to get initial visible frame.")
        thermal_cam.release()
        visible_cam.release()
        return

    # Convert initial thermal frame for display during potential point selection
    initial_thermal_display = raw_to_8bit(initial_thermal_frame)
    if initial_thermal_display is None: initial_thermal_display = np.zeros((config.DISPLAY_HEIGHT,config.DISPLAY_WIDTH,3), dtype=np.uint8)


    # Calculate perspective matrix (using points from config)
    try:
        # Resize initial visible frame to match display size before calculating matrix?
        # Or assume points_vis correspond to the native visible camera output size used in pipeline?
        # Assuming points_vis match the output of the pipeline's first resize (960x616) - check this!
        # For now, assuming points match the size of initial_visible_frame directly
        matrix = calculate_perspective_matrix() # Uses config points
    except Exception as e:
        print(f"FATAL: Failed to calculate perspective matrix: {e}")
        thermal_cam.release()
        visible_cam.release()
        display_manager.destroy_windows()
        return

    # --- Main Loop ---
    thermal_cam.start_streaming() # Start streaming continuously now
    print("Starting main loop...")
    temp_data_list = []
    max_temp = None # Initialize max_temp outside loop

    while True:
        frame_interval = fps_tracker.tick()
        current_avg_fps = fps_tracker.get_average_fps(default_fps=visible_cam.get_default_fps())
        # print(f"FPS: {current_avg_fps:.2f}") # Optional FPS print

        # 1. Get Frames
        thermal_data = thermal_cam.get_frame() # Already resized to display size
        ret_vis, visible_frame = visible_cam.get_frame()

        if not ret_vis or visible_frame is None:
            print("Warning: Skipping loop iteration, failed to get visible frame.")
            time.sleep(0.01) # Avoid busy-waiting
            continue
        if thermal_data is None:
             print("Warning: Skipping loop iteration, failed to get thermal frame.")
             time.sleep(0.01)
             continue

        # 2. Align Visible Frame
        aligned_visible_frame = apply_perspective(visible_frame, matrix,
                                                 (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        if aligned_visible_frame is None:
            print("Warning: Skipping loop iteration, alignment failed.")
            time.sleep(0.01)
            continue

        # 3. Object Detection (YOLO)
        yolo_results = detector.predict(aligned_visible_frame, conf_threshold=config.YOLO_CONF_THRESHOLD)
        largest_box = detector.find_largest_box(yolo_results)

        # 4. Segmentation and Analysis (if box found)
        head_overlay_display = None
        head_segmented_display = None
        avg_temp = None # Reset values for this frame

        if largest_box is not None:
            # --- Cut ROI from Aligned Visible Frame ---
            yolo_head_roi = cut_roi(aligned_visible_frame, largest_box)

            if yolo_head_roi is not None and yolo_head_roi.size > 0:
                # --- UNet Segmentation ---
                try:
                    input_tensor = segmenter.preprocess(yolo_head_roi, target_size=config.UNET_INPUT_SIZE)
                    pred_mask_np = segmenter.predict(input_tensor, threshold=config.UNET_CONF_THRESHOLD)

                    if pred_mask_np is not None:
                        # --- Visualization of Segmentation ---
                        head_overlay_display = segmenter.overlay_mask(yolo_head_roi, pred_mask_np,
                                                                      color=config.MASK_OVERLAY_COLOR,
                                                                      alpha=config.MASK_OVERLAY_ALPHA)
                        head_segmented_display = segmenter.extract_foreground(yolo_head_roi, pred_mask_np)
                    else:
                        print("Warning: UNet prediction mask was None.")
                        head_overlay_display = yolo_head_roi # Show original ROI if mask fails
                        head_segmented_display = np.zeros_like(yolo_head_roi)

                except Exception as e:
                    print(f"Error during segmentation processing: {e}")
                    head_overlay_display = yolo_head_roi # Fallback display
                    head_segmented_display = np.zeros_like(yolo_head_roi) if yolo_head_roi is not None else None
            else:
                # print("Warning: YOLO head ROI is empty or None.") # Can be noisy
                pass # No head ROI, no segmentation display needed

            # --- Temperature Calculation (using Thermal Data and Box) ---
            avg_temp = calculate_average_pixel_value(thermal_data, largest_box)
            max_temp = calculate_maximum_pixel_value(thermal_data, largest_box) # Update max_temp here

            # --- Update Temperature Queue ---
            temp_data_list = update_temperature_queue(avg_temp, temp_data_list, config.TEMPERATURE_QUEUE_MAX_SIZE)

        else: # No largest box found
             max_temp = None # Reset max_temp if no box


        # 5. Respiration Calculation
        breathing_rate_bpm = None
        valid_temps = [t for t in temp_data_list if t is not None] # Filter None again just in case
        if len(valid_temps) >= config.RESPIRATION_MIN_DATA_POINTS:
            breathing_rate_bpm = calculate_respiration_fft(valid_temps, current_avg_fps)


        # 6. Prepare Frames for Display
        # --- Visible Frame ---
        frame_to_show = aligned_visible_frame.copy() # Start with aligned frame
        if largest_box:
             frame_to_show = draw_bounding_box(frame_to_show, largest_box, color=config.BBOX_COLOR) # Draw YOLO box
        if breathing_rate_bpm is not None:
             frame_to_show = display_value(frame_to_show, breathing_rate_bpm, value_type="Respiration", is_thermal=False)

        # --- Thermal Frame ---
        thermal_frame_display = raw_to_8bit(thermal_data)
        if thermal_frame_display is None: # Handle potential conversion error
            thermal_frame_display = np.zeros((config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(thermal_frame_display, "Thermal Error", (50, config.DISPLAY_HEIGHT // 2),
                        config.LABEL_FONT, 1.0, (0, 0, 255), 1)
        else:
            if largest_box:
                 thermal_frame_display = draw_bounding_box(thermal_frame_display, largest_box, color=config.BBOX_COLOR)
            if max_temp is not None:
                 thermal_frame_display = display_value(thermal_frame_display, max_temp, value_type="Temperature", is_thermal=True)


        # 7. Update Displays
        display_manager.show(config.WINDOW_CAMERA, frame_to_show)
        display_manager.show(config.WINDOW_THERMAL, thermal_frame_display)
        # Only show head windows if they were generated
        if head_overlay_display is not None:
             display_manager.show(config.WINDOW_HEAD_OVERLAY, head_overlay_display)
        else:
             # Optionally clear the window if no head ROI
             display_manager.show(config.WINDOW_HEAD_OVERLAY, None) # Show black screen

        if head_segmented_display is not None:
             display_manager.show(config.WINDOW_HEAD_SEGMENTED, head_segmented_display)
        else:
             display_manager.show(config.WINDOW_HEAD_SEGMENTED, None) # Show black screen


        # 8. Handle User Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exit key pressed.")
            break
        elif key == ord('t'): # Test pause
            print("Pausing for 100 seconds...")
            time.sleep(100)
            print("Resuming...")
            fps_tracker.last_time = time.time() # Reset timer after pause


    # --- Cleanup ---
    print("Exiting main loop.")
    thermal_cam.release()
    visible_cam.release()
    display_manager.destroy_windows()
    print("Application finished.")

if __name__ == "__main__":
    main()