# utils/visualization.py

import cv2
import config # For colors, fonts etc.

def draw_bounding_box(image, box, color=config.BBOX_COLOR, thickness=config.BBOX_THICKNESS):
    """ Draws the largest bounding box and its label on the image. """
    if image is None or box is None:
        return image

    try:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        conf = box.get("confidence", 0.0) # Use get for safety
        cls_id = box.get("class_id", -1)
    except (KeyError, TypeError, ValueError):
         print("Warning: Invalid box format for drawing.")
         return image

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare and draw label
    label = f'Class {cls_id}: {conf:.2f}'
    # Calculate text size to prevent text going off-screen (optional but good)
    (w, h), _ = cv2.getTextSize(label, config.LABEL_FONT, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS)
    text_y = y1 - 10 if y1 - 10 > h else y1 + h + 10 # Adjust position if near top edge
    cv2.putText(image, label, (x1, text_y), config.LABEL_FONT, config.LABEL_FONT_SCALE, color, config.LABEL_THICKNESS)

    return image

def display_value(frame, value, value_type="Temperature", is_thermal=False):
    """ Displays a formatted value (Temperature or Respiration Rate) on the frame. """
    if frame is None or value is None:
        return frame

    h, w = frame.shape[:2]

    if is_thermal:
        color = config.TEMP_DISPLAY_COLOR
        # Convert K*100 value to Celsius for display if it's thermal
        # Assumes the input 'value' is raw K*100 from max_pixel_value
        from image_processing.basic_ops import ktoc # Import here or pass ktoc function
        display_val = ktoc(value)
        text = f"{display_val:.1f} C" # Display Celsius with 1 decimal
        font_scale = config.TEMP_FONT_SCALE
        thickness = config.TEMP_THICKNESS
        start_y_factor = 1/7 # Position for temp
    else:
        color = config.RESP_DISPLAY_COLOR
        display_val = value # Assume value is already BPM
        text = f"{display_val:.1f} BPM" # Display BPM with 1 decimal
        font_scale = config.RESP_FONT_SCALE
        thickness = config.RESP_THICKNESS
        start_y_factor = 1/7 # Position for resp rate (same position here, adjust if needed)


    # Position calculation (top-right corner)
    text_w, _ = cv2.getTextSize(text, config.LABEL_FONT, font_scale, thickness)[0]
    start_x = w - text_w - 30 # Position from right edge
    start_y = int(h * start_y_factor) # Position from top

    cv2.putText(frame, text, (start_x, start_y), config.LABEL_FONT, font_scale, color, thickness)

    return frame


class DisplayManager:
    """ Manages OpenCV windows. """
    def __init__(self, window_names, default_width=config.DISPLAY_WIDTH, default_height=config.DISPLAY_HEIGHT):
        self.windows = window_names
        self.width = default_width
        self.height = default_height
        print("Initializing display windows...")
        for name in self.windows:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, self.width + 1, self.height + 1) # +1 as in original code
        print("Display windows initialized.")

    def show(self, window_name, frame):
        """ Shows a frame in the specified window. """
        if window_name not in self.windows:
            print(f"Warning: Window '{window_name}' not managed.")
            return
        if frame is None:
            # Optional: Show a black screen or a "No Data" message
            black_screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(black_screen, "No Data", (50, self.height // 2),
                        config.LABEL_FONT, 1.0, (255, 255, 255), 1)
            cv2.imshow(window_name, black_screen)
            # print(f"Warning: Frame for window '{window_name}' is None.")
            return
        try:
             cv2.imshow(window_name, frame)
        except cv2.error as e:
             print(f"Error showing frame in window '{window_name}': {e}")


    def destroy_windows(self):
        """ Destroys all managed OpenCV windows. """
        print("Destroying display windows...")
        cv2.destroyAllWindows()
        self.windows = [] # Clear managed windows
        print("Display windows destroyed.")

    def destroy_window(self, window_name):
        """ Destroys a specific window. """
        if window_name in self.windows:
            try:
                cv2.destroyWindow(window_name)
                self.windows.remove(window_name)
                print(f"Window '{window_name}' destroyed.")
            except cv2.error as e:
                print(f"Error destroying window '{window_name}': {e}")
        else:
             print(f"Warning: Cannot destroy window '{window_name}', it is not managed or already closed.")