# utils/visualization.py

import logging

import cv2
import numpy as np

import config  # For colors, fonts etc.
from utils.plot import draw_graph_cv2

logger = logging.getLogger(__name__)


def draw_bounding_box(image, box, color=config.BBOX_COLOR, thickness=config.BBOX_THICKNESS):
    """Draws the largest bounding box and its label on the image."""
    if image is None or box is None:
        return image

    try:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        conf = box.get("confidence", 0.0)  # Use get for safety
        cls_id = box.get("class_id", -1)
    except (KeyError, TypeError, ValueError):
        logger.warning("Invalid box format for drawing.")
        return image

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare and draw label
    label = f"Class {cls_id}: {conf:.2f}"
    # Calculate text size to prevent text going off-screen (optional but good)
    (w, h), _ = cv2.getTextSize(label, config.LABEL_FONT, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS)
    text_y = y1 - 10 if y1 - 10 > h else y1 + h + 10  # Adjust position if near top edge
    cv2.putText(image, label, (x1, text_y), config.LABEL_FONT, config.LABEL_FONT_SCALE, color, config.LABEL_THICKNESS)

    return image


def display_value(frame, value, value_type="Temperature", is_thermal=False):
    """Displays a formatted value (Temperature or Respiration Rate) on the frame."""
    if frame is None or value is None:
        return frame

    h, w = frame.shape[:2]

    if is_thermal:
        color = config.TEMP_DISPLAY_COLOR
        # Convert K*100 value to Celsius for display if it's thermal
        # Assumes the input 'value' is raw K*100 from max_pixel_value
        from image_processing.basic_ops import ktoc  # Import here or pass ktoc function

        display_val = ktoc(value)
        text = f"{display_val:.1f} C"  # Display Celsius with 1 decimal
        font_scale = config.TEMP_FONT_SCALE
        thickness = config.TEMP_THICKNESS
        start_y_factor = 1 / 7  # Position for temp
    else:
        color = config.RESP_DISPLAY_COLOR
        display_val = value  # Assume value is already BPM
        text = f"{display_val:.1f} BPM"  # Display BPM with 1 decimal
        font_scale = config.RESP_FONT_SCALE
        thickness = config.RESP_THICKNESS
        start_y_factor = 1 / 7  # Position for resp rate (same position here, adjust if needed)

    # Position calculation (top-right corner)
    text_w, _ = cv2.getTextSize(text, config.LABEL_FONT, font_scale, thickness)[0]
    start_x = w - text_w - 30  # Position from right edge
    start_y = int(h * start_y_factor)  # Position from top

    cv2.putText(frame, text, (start_x, start_y), config.LABEL_FONT, font_scale, color, thickness)

    return frame


class DisplayManager:
    """Manages OpenCV windows."""

    def __init__(self, window_names, default_width=config.DISPLAY_WIDTH, default_height=config.DISPLAY_HEIGHT):
        self.windows = window_names
        self.width = default_width
        self.height = default_height
        logger.info("Initializing display windows...")
        for name in self.windows:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, self.width + 1, self.height + 1)  # +1 as in original code
        logger.info("Display windows initialized.")

    def show(self, window_name, frame):
        """Shows a frame in the specified window."""
        if window_name not in self.windows:
            # Silently return if window is turned off in config
            return
        if frame is None:
            # Optional: Show a black screen or a "No Data" message
            black_screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(black_screen, "No Data", (50, self.height // 2), config.LABEL_FONT, 1.0, (255, 255, 255), 1)
            cv2.imshow(window_name, black_screen)
            # logger.warning(f"Frame for window '{window_name}' is None.")
            return
        try:
            cv2.imshow(window_name, frame)
        except cv2.error as e:
            logger.error(f"Error showing frame in window '{window_name}': {e}")

    def destroy_windows(self):
        """Destroys all managed OpenCV windows."""
        logger.info("Destroying display windows...")
        cv2.destroyAllWindows()
        self.windows = []  # Clear managed windows
        logger.info("Display windows destroyed.")


class Renderer:
    """Handles all drawing and rendering operations, decoupling logic from visualization."""

    def __init__(self, display_manager):
        self.display_manager = display_manager

    def render_visible_frame(self, visible_frame, largest_box, breathing_rate_bpm):
        if not getattr(config, "SHOW_VISIBLE_CAMERA_UI", True):
            return

        frame_to_show = visible_frame.copy()
        if largest_box:
            frame_to_show = draw_bounding_box(frame_to_show, largest_box, color=config.BBOX_COLOR)
        if breathing_rate_bpm is not None:
            frame_to_show = display_value(frame_to_show, breathing_rate_bpm, value_type="Respiration", is_thermal=False)

        self.display_manager.show(config.WINDOW_CAMERA, frame_to_show)

    def render_thermal_frame(self, thermal_8bit, largest_box, thermal_box, max_temp, temperature_offset_c):
        if not getattr(config, "SHOW_THERMAL_UI", True):
            return

        thermal_frame_display = thermal_8bit.copy() if thermal_8bit is not None else None
        if thermal_frame_display is None:
            thermal_frame_display = np.zeros((config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                thermal_frame_display,
                "Thermal Error",
                (50, config.DISPLAY_HEIGHT // 2),
                config.LABEL_FONT,
                1.0,
                (0, 0, 255),
                1,
            )
        else:
            if getattr(config, "ENABLE_BLACKBODY_CALIBRATION", False):
                x1, y1, x2, y2 = config.BLACKBODY_ROI
                scale_x = thermal_frame_display.shape[1] / 640.0
                scale_y = thermal_frame_display.shape[0] / 480.0
                bx, by = int(x1 * scale_x), int(y1 * scale_y)
                bw, bh = int((x2 - x1) * scale_x), int((y2 - y1) * scale_y)

                cv2.rectangle(thermal_frame_display, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                cv2.putText(
                    thermal_frame_display,
                    f"BB: {temperature_offset_c:+.2f}C",
                    (bx, max(15, by - 5)),
                    config.LABEL_FONT,
                    0.4,
                    (255, 0, 0),
                    1,
                )

            if largest_box:
                thermal_frame_display = draw_bounding_box(thermal_frame_display, thermal_box, color=config.BBOX_COLOR)
            if max_temp is not None:
                thermal_frame_display = display_value(
                    thermal_frame_display, max_temp, value_type="Temperature", is_thermal=True
                )

        self.display_manager.show(config.WINDOW_THERMAL, thermal_frame_display)

    def render_analysis_ui(self, debug_data_raw, debug_data_interp):
        if not getattr(config, "SHOW_ANALYSIS_UI", True):
            return

        plots_canvas = np.zeros((config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH, 3), dtype=np.uint8)
        h, w = config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH
        half_h, half_w = h // 2, w // 2

        if debug_data_raw is not None and debug_data_interp is not None:
            # Top Left: Raw Temperature
            draw_graph_cv2(
                plots_canvas,
                data_x=None,
                data_y=debug_data_raw["raw_temp"],
                color=(0, 255, 255),
                rect=(0, 0, half_w, half_h),
                title="Raw Temp (temp_data_list)",
            )

            # Bottom Left: Raw FFT
            draw_graph_cv2(
                plots_canvas,
                data_x=debug_data_raw["positive_freqs"] * 60,  # x is BPM
                data_y=debug_data_raw["positive_magnitude"],
                color=(255, 100, 100),
                rect=(0, half_h, half_w, half_h),
                title="Raw FFT",
                y_min_fixed=0.0,
            )

            # Top Right: Interpolated Temperature
            draw_graph_cv2(
                plots_canvas,
                data_x=debug_data_interp["uniform_time"],
                data_y=debug_data_interp["resampled_temp"],
                color=(0, 255, 0),
                rect=(half_w, 0, half_w, half_h),
                title="Interp Temp (with Timestamps)",
            )

            # Bottom Right: Interpolated FFT
            draw_graph_cv2(
                plots_canvas,
                data_x=debug_data_interp["positive_freqs"] * 60,  # x is BPM
                data_y=debug_data_interp["positive_magnitude"],
                color=(100, 255, 100),
                rect=(half_w, half_h, half_w, half_h),
                title="Interp FFT",
                y_min_fixed=0.0,
            )
        else:
            cv2.putText(plots_canvas, "Gathering data...", (50, half_h), config.LABEL_FONT, 1.0, (255, 255, 255), 1)

        self.display_manager.show(config.WINDOW_ANALYSIS, plots_canvas)

    def render_masks(self, head_overlay_display, head_segmented_display, mask_segmented_thermal_data):
        if getattr(config, "SHOW_MASK_OVERLAY_UI", True):
            self.display_manager.show(config.WINDOW_MASK_OVERLAY, head_overlay_display)

        if getattr(config, "SHOW_MASK_SEGMENTED_UI", True):
            self.display_manager.show(config.WINDOW_MASK_SEGMENTED, head_segmented_display)

        if getattr(config, "SHOW_THERMAL_MASK_SEGMENTED_UI", True):
            self.display_manager.show(config.WINDOW_THERMAL_MASK_SEGMENTED, mask_segmented_thermal_data)
