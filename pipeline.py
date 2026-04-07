import logging
import time

import cv2
import numpy as np

import config
from analysis.respiration import calculate_fft_raw, calculate_respiration_fft
from analysis.temperature import calculate_average_pixel_value
from camera_utils.camera_thread import CameraThread
from camera_utils.thermal_camera import ThermalCameraUVC
from camera_utils.visible_camera import VisibleCamera
from image_processing.alignment import calculate_perspective_matrix, transform_bbox
from image_processing.basic_ops import create_skin_mask, cut_roi, ktoc, raw_to_8bit
from models.detector import YoloDetector
from models.segmenter import UNetSegmenter
from utils.hardware_monitor import HardwareMonitor
from utils.profiler import Profiler, TimeIt
from utils.session_state import SessionContext
from utils.timing import FPSTracker
from utils.visualization import DisplayManager, Renderer

logger = logging.getLogger(__name__)


class MonitorPipeline:
    def __init__(self):
        logger.info(f"Using device: {config.DEVICE}")

        self.fps_tracker = FPSTracker(buffer_size=10)
        self.session_context = SessionContext()

        # Initialize display windows
        active_windows = []
        if getattr(config, "SHOW_VISIBLE_CAMERA_UI", True):
            active_windows.append(config.WINDOW_CAMERA)
        if getattr(config, "SHOW_THERMAL_UI", True):
            active_windows.append(config.WINDOW_THERMAL)
        if getattr(config, "SHOW_MASK_OVERLAY_UI", True):
            active_windows.append(config.WINDOW_MASK_OVERLAY)
        if getattr(config, "SHOW_MASK_SEGMENTED_UI", True):
            active_windows.append(config.WINDOW_MASK_SEGMENTED)
        if getattr(config, "SHOW_THERMAL_MASK_SEGMENTED_UI", True):
            active_windows.append(config.WINDOW_THERMAL_MASK_SEGMENTED)
        if getattr(config, "SHOW_THERMAL_SKIN_MASK_SEGMENTED_UI", True):
            active_windows.append(config.WINDOW_THERMAL_SKIN_MASK_SEGMENTED)
        if getattr(config, "SHOW_ANALYSIS_UI", True):
            active_windows.append(config.WINDOW_ANALYSIS)

        self.display_manager = DisplayManager(
            active_windows, default_width=config.DISPLAY_WIDTH, default_height=config.DISPLAY_HEIGHT
        )
        self.renderer = Renderer(self.display_manager)

        # Initialize core components
        try:
            if getattr(config, "IS_TESTING", False):
                logger.info("--- RUNNING IN TESTING MODE (Mock Cameras) ---")
                from camera_utils.mock_camera import MockCamera

                self.thermal_cam = MockCamera(config.TEST_THERMAL_VIDEO, target_fps=9)
                self.visible_cam = MockCamera(config.TEST_VISIBLE_VIDEO, target_fps=config.DEFAULT_FPS)
            else:
                self.thermal_cam = ThermalCameraUVC(vid=config.THERMAL_VID, pid=config.THERMAL_PID)
                self.visible_cam = VisibleCamera(pipeline=config.GST_PIPELINE)

            self.detector = YoloDetector(model_path=config.YOLO_MODEL_PATH)
            self.segmenter = UNetSegmenter(model_path=config.UNET_MODEL_PATH, device=config.DEVICE)
        except Exception as e:
            logger.error(f"FATAL: Initialization failed: {e}")
            raise RuntimeError(f"Initialization failed: {e}") from e

        self.monitor = HardwareMonitor(output_csv="hardware_stats.csv")
        self.monitor.start()

        self.thermal_thread = CameraThread(self.thermal_cam, name="ThermalThread")
        self.visible_thread = CameraThread(self.visible_cam, name="VisibleThread")

        logger.info("Waiting for cameras to initialize...")
        time.sleep(2.0)

        ret_vis, initial_visible_frame, _ = self.visible_thread.read()
        if not ret_vis or initial_visible_frame is None:
            logger.error("FATAL: Failed to get initial visible frame from thread.")
            self.cleanup()
            raise RuntimeError("Camera connection failed")

        logger.info("Waiting for initial YOLO...")
        self.detector.predict(initial_visible_frame, conf_threshold=config.YOLO_CONF_THRESHOLD)

        logger.info("Getting initial frames for alignment...")
        ret_therm, initial_thermal_frame, _ = self.thermal_thread.read()
        if initial_thermal_frame is None:
            logger.info("Waiting for initial thermal frame...")
            time.sleep(3)
            ret_therm, initial_thermal_frame, _ = self.thermal_thread.read()

        try:
            self.matrix = calculate_perspective_matrix()
        except Exception as e:
            logger.error(f"FATAL: Failed to calculate perspective matrix: {e}")
            self.cleanup()
            raise RuntimeError(f"Alignment failed: {e}") from e

        self.erode_kernel = np.ones((config.KERNEL_SIZE, config.KERNEL_SIZE), np.uint8)

        # Optimization counters
        self.detection_skip_count = 0
        self.last_face_box = None
        self.fft_skip_count = 0

    def process_frame(self, visible_frame, thermal_data, therm_time, current_avg_fps):
        thermal_8bit = raw_to_8bit(thermal_data)

        temperature_offset_c = 0.0
        temperature_offset_raw = 0.0
        if getattr(config, "ENABLE_BLACKBODY_CALIBRATION", False):
            x1, y1, x2, y2 = config.BLACKBODY_ROI
            scale_x = thermal_data.shape[1] / 640.0
            scale_y = thermal_data.shape[0] / 480.0
            bx, by = int(x1 * scale_x), int(y1 * scale_y)
            bw, bh = int((x2 - x1) * scale_x), int((y2 - y1) * scale_y)

            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, thermal_data.shape[1] - bx), min(bh, thermal_data.shape[0] - by)
            if bw > 0 and bh > 0:
                bb_roi = thermal_data[by : by + bh, bx : bx + bw]
                if bb_roi.size > 0:
                    bb_temp_raw = np.mean(bb_roi)
                    bb_temp_c = ktoc(bb_temp_raw)
                    temperature_offset_c = config.BLACKBODY_TEMP_C - bb_temp_c
                    temperature_offset_raw = temperature_offset_c * 100.0

        yolo_results = None
        # Optimization: Skip YOLO detection if we already have a box and haven't reached the interval
        if self.last_face_box is None or self.detection_skip_count == 0:
            with TimeIt("YOLO_Inference"):
                yolo_results = self.detector.predict(visible_frame, conf_threshold=config.YOLO_CONF_THRESHOLD)
            largest_box = self.detector.find_largest_box(yolo_results)
            self.last_face_box = largest_box
        else:
            # Reuse the last known box
            largest_box = self.last_face_box

        # Increment skip counter
        if config.DETECTION_SKIP_INTERVAL > 0:
            self.detection_skip_count = (self.detection_skip_count + 1) % (config.DETECTION_SKIP_INTERVAL + 1)

        head_overlay_display = None
        head_segmented_display = None
        mask_segmented_thermal_data = None
        avg_temp = None
        avg_temp_no_unet = None
        max_temp = None
        thermal_box = None

        if largest_box is not None:
            thermal_box = transform_bbox(largest_box, self.matrix)
            yolo_head_roi = cut_roi(visible_frame, largest_box)
            yolo_head_thermal_roi = cut_roi(thermal_8bit, thermal_box)

            if yolo_head_roi is not None and yolo_head_roi.size > 0:
                try:
                    with TimeIt("UNet_Preprocess"):
                        input_tensor = self.segmenter.preprocess(yolo_head_roi, target_size=config.UNET_INPUT_SIZE)
                    with TimeIt("UNet_Inference"):
                        pred_mask_np = self.segmenter.predict(input_tensor, threshold=config.UNET_CONF_THRESHOLD)

                    if pred_mask_np is not None:
                        pred_mask_np = pred_mask_np.astype(np.uint8, copy=False)
                        cv2.erode(pred_mask_np, self.erode_kernel, dst=pred_mask_np, iterations=1)

                        with TimeIt("Mask_Visualization"):
                            # Only perform these if UI is enabled to save CPU
                            if getattr(config, "SHOW_MASK_OVERLAY_UI", False):
                                head_overlay_display = self.segmenter.overlay_mask(
                                    yolo_head_roi,
                                    pred_mask_np,
                                    color=config.MASK_OVERLAY_COLOR,
                                    alpha=config.MASK_OVERLAY_ALPHA,
                                )

                            if getattr(config, "SHOW_MASK_SEGMENTED_UI", False):
                                head_segmented_display = self.segmenter.extract_foreground(yolo_head_roi, pred_mask_np)

                            # Always extract thermal mask if UNET is used for temperature?
                            # Or only if the window is showing.
                            # Actually mask_segmented_thermal_data is used below for temperature if available.
                            mask_segmented_thermal_data = self.segmenter.extract_foreground(
                                yolo_head_thermal_roi, pred_mask_np
                            )
                    else:
                        logger.warning("UNet prediction mask was None.")
                        if getattr(config, "SHOW_MASK_OVERLAY_UI", False):
                            head_overlay_display = yolo_head_roi
                        head_segmented_display = np.zeros_like(yolo_head_roi) if yolo_head_roi is not None else None
                        mask_segmented_thermal_data = (
                            np.zeros_like(yolo_head_thermal_roi) if yolo_head_thermal_roi is not None else None
                        )

                except Exception as e:
                    logger.error(f"Error during segmentation processing: {e}")
                    head_overlay_display = yolo_head_roi if getattr(config, "SHOW_MASK_OVERLAY_UI", False) else None
                    head_segmented_display = np.zeros_like(yolo_head_roi) if yolo_head_roi is not None else None
                    mask_segmented_thermal_data = (
                        np.zeros_like(yolo_head_thermal_roi) if yolo_head_thermal_roi is not None else None
                    )

            with TimeIt("Temperature_Processing"):
                avg_temp_no_unet = calculate_average_pixel_value(thermal_8bit, thermal_box)

                if mask_segmented_thermal_data is not None and np.any(mask_segmented_thermal_data != 0):
                    avg_temp = np.mean(mask_segmented_thermal_data, where=mask_segmented_thermal_data != 0)
                else:
                    avg_temp = None

            if config.SKIN_COLOR_FILTER:
                thermal_roi = cut_roi(thermal_data, thermal_box)
                skin_mask = create_skin_mask(yolo_head_roi)
                skin_temperatures = thermal_roi[skin_mask == 255]

                if skin_temperatures.size > 0:
                    if getattr(config, "TEMP_EXTRACTION_METHOD", "percentile") == "max":
                        max_temp = np.max(skin_temperatures)
                    else:
                        threshold = np.percentile(skin_temperatures, 95)
                        max_temp = np.mean(skin_temperatures[skin_temperatures >= threshold])

                    alpha = 0.4
                    thermal_roi_8bit = raw_to_8bit(thermal_roi)
                    color_overlay = np.zeros_like(thermal_roi_8bit)
                    color_overlay[skin_mask == 255] = (0, 0, 255)

                    skin_masked_thermal_data = cv2.addWeighted(thermal_roi_8bit, 1 - alpha, color_overlay, alpha, 0)

                    if getattr(config, "SHOW_THERMAL_SKIN_MASK_SEGMENTED_UI", True):
                        self.display_manager.show(config.WINDOW_THERMAL_SKIN_MASK_SEGMENTED, skin_masked_thermal_data)
                else:
                    logger.info("在此區域中沒有偵測到皮膚，將使用原始YOLO的ROI計算。")
                    if thermal_roi is not None and thermal_roi.size > 0:
                        if getattr(config, "TEMP_EXTRACTION_METHOD", "percentile") == "max":
                            max_temp = np.max(thermal_roi)
                        else:
                            threshold = np.percentile(thermal_roi, 95)
                            max_temp = np.mean(thermal_roi[thermal_roi >= threshold])
                    else:
                        max_temp = None
            else:
                thermal_roi = cut_roi(thermal_data, thermal_box)
                if thermal_roi is not None and thermal_roi.size > 0:
                    if getattr(config, "TEMP_EXTRACTION_METHOD", "percentile") == "max":
                        max_temp = np.max(thermal_roi)
                    else:
                        threshold = np.percentile(thermal_roi, 95)
                        max_temp = np.mean(thermal_roi[thermal_roi >= threshold])
                else:
                    max_temp = None

            if max_temp is not None:
                max_temp += temperature_offset_raw

            temp_val_c = ktoc(max_temp) if max_temp is not None else None

        else:
            max_temp = None
            temp_val_c = None

        # Update session context with current temp data (bpm later)
        current_time_for_temp = time.time()

        valid_temps, valid_timestamps = self.session_context.get_respiration_data()

        breathing_rate_bpm = None
        debug_data_interp = None
        debug_data_raw = None

        # We need to temporarily hold the updated context internally or append to the valid temps to process the FFT?
        # Actually in main_app it updates the deque first, then processes FFT.
        self.session_context.update(avg_temp, temp_val_c, current_time_for_temp, None, avg_temp_no_unet)

        valid_temps, valid_timestamps = self.session_context.get_respiration_data()

        if len(valid_temps) >= config.RESPIRATION_MIN_DATA_POINTS and len(valid_timestamps) == len(valid_temps):
            # Optimization: Run FFT only at specified interval
            if self.fft_skip_count == 0:
                with TimeIt("Signal_Processing_FFT"):
                    breathing_rate_bpm, debug_data_interp = calculate_respiration_fft(
                        valid_temps, valid_timestamps, current_avg_fps
                    )
                    _, debug_data_raw = calculate_fft_raw(valid_temps, current_avg_fps)

            # Keep the count going
            if config.FFT_SKIP_INTERVAL > 0:
                self.fft_skip_count = (self.fft_skip_count + 1) % (config.FFT_SKIP_INTERVAL + 1)

        if breathing_rate_bpm is not None:
            self.session_context.update(None, None, None, breathing_rate_bpm, None)
        else:
            # If skipped, try to get the last known BPM from context to maintain display
            _, breathing_rate_bpm = self.session_context.get_summary()

        if getattr(config, "SHOW_ANALYSIS_UI", True):
            self.renderer.render_analysis_ui(debug_data_raw, debug_data_interp)

        self.renderer.render_visible_frame(visible_frame, largest_box, breathing_rate_bpm)
        self.renderer.render_thermal_frame(thermal_8bit, largest_box, thermal_box, max_temp, temperature_offset_c)
        self.renderer.render_masks(head_overlay_display, head_segmented_display, mask_segmented_thermal_data)

    def run(self):
        start_time = time.time()
        logger.info("Pipeline started.")

        while True:
            with TimeIt("WholeSystem"):
                self.fps_tracker.tick()
                current_avg_fps = self.fps_tracker.get_average_fps(default_fps=self.visible_cam.get_default_fps())

                with TimeIt("Camera_Read"):
                    ret_therm, thermal_data, therm_time = self.thermal_thread.read()
                    ret_vis, visible_frame, _ = self.visible_thread.read()

                if not ret_vis or visible_frame is None:
                    logger.warning("Skipping loop iteration, failed to get visible frame.")
                    time.sleep(0.01)
                    continue
                if not ret_therm or thermal_data is None:
                    logger.warning("Skipping loop iteration, failed to get thermal frame.")
                    time.sleep(0.01)
                    continue

                with TimeIt("Thermal_Resize"):
                    # Resize thermal data in processing pipeline instead of read thread to offload ingest bottleneck
                    thermal_data = cv2.resize(
                        thermal_data, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST
                    )

                self.process_frame(visible_frame, thermal_data, therm_time, current_avg_fps)

            # --- Loop Control & Diagnostics ---
            elapsed_time = time.time() - start_time
            # logger.info(f"Elapsed: {round(elapsed_time, 2)}")

            if elapsed_time > getattr(config, "DURATION", float("inf")):
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Exit key pressed.")
                break
            elif key == ord("t"):
                logger.info("Pausing for 10 seconds...")
                time.sleep(10)
                logger.info("Resuming...")
                self.fps_tracker.last_time = time.time()

        self.cleanup()
        avg_temp, avg_resp = self.session_context.get_summary()
        return avg_temp, avg_resp

    def cleanup(self):
        logger.info("Cleaning up pipeline resources...")
        self.thermal_thread.stop()
        self.visible_thread.stop()
        self.display_manager.destroy_windows()
        self.monitor.stop()
        Profiler().export_json("benchmark_result.json")
