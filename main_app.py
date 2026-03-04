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
from image_processing.alignment import calculate_perspective_matrix, apply_perspective, transform_bbox
from image_processing.basic_ops import raw_to_8bit, cut_roi, ktoc, create_skin_mask

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
        config.WINDOW_THERMAL
        # config.WINDOW_MASK_OVERLAY,
        # config.WINDOW_MASK_SEGMENTED,
        # config.WINDOW_THERMAL_MASK_SEGMENTED,
        # config.WINDOW_THERMAL_SKIN_MASK_SEGMENTED
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
    ret_vis, initial_visible_frame = visible_cam.get_frame()
    if not ret_vis or initial_visible_frame is None:
        print("FATAL: Failed to get initial visible frame.")
        thermal_cam.release()
        visible_cam.release()
        return 0.0, 0.0
    
    print("Waiting for initial YOLO...")
    detector.predict(initial_visible_frame, conf_threshold=config.YOLO_CONF_THRESHOLD) #initial yolo

    # Get initial frames for perspective calculation
    print("Getting initial frames for alignment...")
    initial_thermal_frame = None
    if initial_thermal_frame is None:
         thermal_cam.start_streaming() # Ensure stream is running
         initial_thermal_frame = thermal_cam.get_frame()
         if initial_thermal_frame is None:
              print("Waiting for initial thermal frame...")
              time.sleep(3)
        #thermal_cam.stop_streaming() # Stop after getting one frame for now  
    
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
        return 0.0, 0.0

    # --- Main Loop ---

    
    # thermal_cam.start_streaming() # Start streaming continuously now
    from collections import deque
    temp_data_list = deque(maxlen=config.TEMPERATURE_QUEUE_MAX_SIZE)
    timestamp_list = deque(maxlen=config.TEMPERATURE_QUEUE_MAX_SIZE)
    temp_data_list_no_unet = deque(maxlen=config.TEMPERATURE_QUEUE_MAX_SIZE)
    max_temp_list = deque(maxlen=config.TEMPERATURE_QUEUE_MAX_SIZE)
    max_temp = None # Initialize max_temp outside loop
    breathing_rate_bpm_list = []
    output_file_name = "temp_with_Unet.txt" # 定義輸出檔案名
    output_file_name_no_Unet = "temp_with_no_Unet.txt" # 定義輸出檔案名
    output_file_name_max_temp = "max_temp.txt" # 定義輸出檔案名
    output_file_resp = "resp_list.txt"
    start_time = time.time()
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

        # 2. Skip alignment of entire visible frame!
        # We will detect on the raw visible frame, and transform the BBox later
        
        # 3. Object Detection (YOLO) on visible frame
        yolo_results = detector.predict(visible_frame, conf_threshold=config.YOLO_CONF_THRESHOLD)
        largest_box = detector.find_largest_box(yolo_results)
        
        # 4. Segmentation and Analysis (if box found)
        head_overlay_display = None
        head_segmented_display = None
        mask_segmented_thermal_data = None
        avg_temp = None # Reset values for this frame

        if largest_box is not None:
            # Transform visible bbox to thermal bbox using perspective matrix
            thermal_box = transform_bbox(largest_box, matrix)
            
            # --- Cut ROI ---
            # Cut visible ROI directly for UNet
            yolo_head_roi = cut_roi(visible_frame, largest_box)
            # Cut thermal ROI using the transformed box
            yolo_head_thermal_roi = cut_roi(raw_to_8bit(thermal_data), thermal_box)
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
                        mask_segmented_thermal_data = segmenter.extract_foreground(yolo_head_thermal_roi, pred_mask_np)
                    else:
                        print("Warning: UNet prediction mask was None.")
                        head_overlay_display = yolo_head_roi # Show original ROI if mask fails
                        head_segmented_display = np.zeros_like(yolo_head_roi)
                        mask_segmented_thermal_data = np.zeros_like(yolo_head_thermal_roi)

                except Exception as e:
                    print(f"Error during segmentation processing: {e}")
                    head_overlay_display = yolo_head_roi # Fallback display
                    head_segmented_display = np.zeros_like(yolo_head_roi) if yolo_head_roi is not None else None
                    mask_segmented_thermal_data = np.zeros_like(yolo_head_thermal_roi) if yolo_head_thermal_roi is not None else None
            else:
                # print("Warning: YOLO head ROI is empty or None.") # Can be noisy
                pass # No head ROI, no segmentation display needed

            avg_temp_no_unet = calculate_average_pixel_value(raw_to_8bit(thermal_data), thermal_box)
            
            if mask_segmented_thermal_data is not None and np.any(mask_segmented_thermal_data != 0):
                avg_temp = np.mean(mask_segmented_thermal_data, where=mask_segmented_thermal_data != 0)
            else:
                avg_temp = None
            # print(avg_temp)
            # print(f"no unet:{round(avg_temp_no_unet, 2)}, unet:{round(avg_temp, 2)}")

            # avg_temp = calculate_average_pixel_value(yolo_head_roi, largest_box)
            
            # --- Temperature Calculation (using skin mask) ---
            if config.SKIN_COLOR_FILTER:
                thermal_roi = cut_roi(thermal_data, thermal_box)

                skin_mask = create_skin_mask(yolo_head_roi)
                skin_temperatures = thermal_roi[skin_mask == 255]

                if skin_temperatures.size > 0:
                    max_temp = np.max(skin_temperatures)
                    alpha = 0.4
                    thermal_roi_8bit = raw_to_8bit(thermal_roi)
                    color_overlay = np.zeros_like(thermal_roi_8bit)
                    color_overlay[skin_mask == 255] = (0, 0, 255)
                    
                    skin_masked_thermal_data = cv2.addWeighted(
                        thermal_roi_8bit,
                        1 - alpha,
                        color_overlay,
                        alpha,
                        0
                    )
                    
                    display_manager.show(config.WINDOW_THERMAL_SKIN_MASK_SEGMENTED, skin_masked_thermal_data)
                else:
                    print("在此區域中沒有偵測到皮膚，將使用原始YOLO的ROI計算。")
                    if thermal_roi.size > 0:
                        max_temp = np.max(thermal_roi)
                    else:
                        max_temp = None
            else:
                # ... (原始邏輯) ...
                thermal_roi = cut_roi(thermal_data, thermal_box)
                if thermal_roi.size > 0:
                    max_temp = np.max(thermal_roi)
                else:
                    max_temp = None
                
            temp_val = ktoc(max_temp)
            max_temp_list = update_temperature_queue(temp_val, max_temp_list, config.TEMPERATURE_QUEUE_MAX_SIZE)
            # --- Update Temperature Queue ---
            temp_data_list_no_unet = update_temperature_queue(avg_temp_no_unet,  temp_data_list_no_unet, config.TEMPERATURE_QUEUE_MAX_SIZE)
            temp_data_list = update_temperature_queue(avg_temp, temp_data_list, config.TEMPERATURE_QUEUE_MAX_SIZE)
            
            # --- Update Timestamp Queue ---
            if avg_temp is not None:
                timestamp_list = update_temperature_queue(time.time(), timestamp_list, config.TEMPERATURE_QUEUE_MAX_SIZE)

            '''
            # <<< --- 新增：檢查隊列是否已滿，如果滿了就儲存並退出 --- >>>
            if len(temp_data_list) >= config.TEMPERATURE_QUEUE_MAX_SIZE:
                print(f"Temperature data queue reached target size ({config.TEMPERATURE_QUEUE_MAX_SIZE}). Saving data and exiting...")
                try:
                    with open(output_file_name, "w") as f:
                        for temp_value in temp_data_list:
                            if temp_value is not None:
                                f.write(f"{temp_value}\n") # 寫入數值
                            else:
                                f.write("None\n") # 或者寫入 NaN 或空行，取決於你的需求
                    print(f"Temperature data saved successfully to {output_file_name}.")
                except IOError as e:
                    print(f"Error saving temperature data to {output_file_name}: {e}")

                try:
                    with open(output_file_name_no_Unet, "w") as f:
                        for temp_value in temp_data_list_no_unet:
                            if temp_value is not None:
                                f.write(f"{temp_value}\n") # 寫入數值
                            else:
                                f.write("None\n") # 或者寫入 NaN 或空行，取決於你的需求
                    print(f"Temperature data saved successfully to {output_file_name_no_Unet}.")
                except IOError as e:
                    print(f"Error saving temperature data to {output_file_name_no_Unet}: {e}")

                try:
                    with open(output_file_name_max_temp, "w") as f:
                        for temp_value in max_temp_list:
                            if temp_value is not None:
                                f.write(f"{temp_value}\n") # 寫入數值
                            else:
                                f.write("None\n") # 或者寫入 NaN 或空行，取決於你的需求
                    print(f"Temperature data saved successfully to {output_file_name_max_temp}.")
                except IOError as e:
                    print(f"Error saving temperature data to {output_file_name_max_temp}: {e}")

                # 不論儲存是否成功，都退出主迴圈
                break # <--- 退出 while True 迴圈
            '''
            
            # <<< --- 檢查結束 --- >>>

        else: # No largest box found
            # temp_data_list = [] #Reset if no box
            # temp_data_list_no_unet = []
            # max_temp_list = []
            # max_temp = None # Reset max_temp if no box
            pass
        

        # 5. Respiration Calculation
        breathing_rate_bpm = None
        valid_temps = [t for t in temp_data_list if t is not None] # Filter None again just in case
        valid_timestamps = list(timestamp_list)[:len(valid_temps)] # Match lengths exactly in case of misalignments
        
        if len(valid_temps) >= config.RESPIRATION_MIN_DATA_POINTS and len(valid_timestamps) == len(valid_temps):
            breathing_rate_bpm = calculate_respiration_fft(valid_temps, valid_timestamps, current_avg_fps)
            if breathing_rate_bpm is not None:
                breathing_rate_bpm_list.append(breathing_rate_bpm)

        # 6. Prepare Frames for Display
        # --- Visible Frame ---
        frame_to_show = visible_frame.copy() 
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
                 thermal_frame_display = draw_bounding_box(thermal_frame_display, thermal_box, color=config.BBOX_COLOR)
            if max_temp is not None:
                 thermal_frame_display = display_value(thermal_frame_display, max_temp, value_type="Temperature", is_thermal=True)


        # 7. Update Displays
        display_manager.show(config.WINDOW_CAMERA, frame_to_show)
        display_manager.show(config.WINDOW_THERMAL, thermal_frame_display)

        

        
        # # Only show head windows if they were generated
        # if mask_segmented_thermal_data is not None:
        #     display_manager.show(config.WINDOW_THERMAL_MASK_SEGMENTED, mask_segmented_thermal_data)
        # else:
        #      # Optionally clear the window if no head ROI
        #      display_manager.show(config.WINDOW_THERMAL_MASK_SEGMENTED, None) # Show black screen

        # if head_overlay_display is not None:
        #      display_manager.show(config.WINDOW_MASK_OVERLAY, head_overlay_display)
        # else:
        #      # Optionally clear the window if no head ROI
        #      display_manager.show(config.WINDOW_MASK_OVERLAY, None) # Show black screen

        # if head_segmented_display is not None:
        #      display_manager.show(config.WINDOW_MASK_SEGMENTED, head_segmented_display)
        # else:
        #      display_manager.show(config.WINDOW_MASK_SEGMENTED, None) # Show black screen
        

        print(f"{round(time.time() - start_time, 2)}")  #執行時間

        if time.time() - start_time > config.DURATION :
            # try:
            #     with open(output_file_name_max_temp, "w") as f:
            #         for temp_value in max_temp_list:
            #             if temp_value is not None:
            #                 f.write(f"{temp_value}\n") # 寫入數值
            #             else:
            #                 f.write("None\n") # 或者寫入 NaN 或空行，取決於你的需求
            #     print(f"Temperature data saved successfully to {output_file_name_max_temp}.")
            # except IOError as e:
            #     print(f"Error saving temperature data to {output_file_name_max_temp}: {e}")
            
            # try:
            #     with open(output_file_resp, "w") as f:
            #         for resp in breathing_rate_bpm_list:
            #             if resp is not None:
            #                 f.write(f"{resp}\n") # 寫入數值
            #             else:
            #                 f.write("None\n") # 或者寫入 NaN 或空行，取決於你的需求
            #     print(f"Temperature data saved successfully to {output_file_resp}.")
            # except IOError as e:
            #     print(f"Error saving temperature data to {output_file_resp}: {e}")

            temperature = np.mean(max_temp_list) if max_temp_list else 0.0
            resp = np.mean([r for r in breathing_rate_bpm_list if r is not None]) if breathing_rate_bpm_list and any(r is not None for r in breathing_rate_bpm_list) else 0.0
            break

        # 8. Handle User Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            temperature = np.mean(max_temp_list) if max_temp_list else 0.0
            resp = np.mean([r for r in breathing_rate_bpm_list if r is not None]) if breathing_rate_bpm_list and any(r is not None for r in breathing_rate_bpm_list) else 0.0
            print("Exit key pressed.")
            break
        elif key == ord('t'): # Test pause
            print("Pausing for 10 seconds...")
            time.sleep(10)
            print("Resuming...")
            fps_tracker.last_time = time.time() # Reset timer after pause

    # --- Cleanup ---
    print(f"FPS : {current_avg_fps}")
    print("Exiting main loop.")
    thermal_cam.release()
    visible_cam.release()
    display_manager.destroy_windows()
    print("Application finished.")

    return temperature, resp
    

if __name__ == "__main__":
    result = main()
    if result is not None:
        temperature, Resp = result
        print(f"temperature : {round(temperature, 2)}")
        print(f"BPM : {round(Resp, 2)}")
    else:
        print("Application exited with an error.")