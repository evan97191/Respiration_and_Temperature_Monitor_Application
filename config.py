# config.py

# -- Model Paths --
YOLO_MODEL_PATH = "yolo11n_headmask.pt"
UNET_MODEL_PATH = "unet_msfd_model_best.pth"

# Excution time
DURATION = 600 # second

# Filter out skin color (Optional feature)
SKIN_COLOR_FILTER = False

# -- Camera Parameters --
# UVC Thermal Camera VID/PID (from uvctypes.py, maybe keep there or centralize here)
THERMAL_VID = 0x1e4e
THERMAL_PID = 0x0100
THERMAL_BUFFER_SIZE = 2

# Visible Camera GStreamer Pipeline
GST_PIPELINE = (
    "nvarguscamerasrc sensor_mode=0 ! "
    "video/x-raw(memory:NVMM), width=3820, height=2464, framerate=21/1, format=NV12 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=960, height=616 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true sync=false"
)

# -- Processing Parameters --
# Target display/processing resolution (after potential camera resize)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Perspective Transform Points (Hardcoded or from calibration file)
# Format: [[x_ir, y_ir], ...] and [[x_vis, y_vis], ...]
POINTS_IR = [[95, 71], [117, 578], [703, 565], [715, 69]]
POINTS_VIS = [[197, 122], [226, 520], [780, 510], [781, 107]]
# POINTS_IR = [[135, 155], [622, 123], [159, 434], [635, 412]]
# POINTS_VIS = [[195, 206], [666, 175], [204, 436], [675, 417]]
# YOLO Detection Confidence Threshold
YOLO_CONF_THRESHOLD = 0.5

# UNet Segmentation Threshold
UNET_CONF_THRESHOLD = 0.5 # As per original code, adjust if needed (0.5 is common)
# UNet Model Input Size (Height, Width) - MUST MATCH TRAINING
UNET_INPUT_SIZE = (256, 256)

# -- Analysis Parameters --
TEMPERATURE_QUEUE_MAX_SIZE = 10 * 9
RESPIRATION_MIN_DATA_POINTS = 9 # Minimum points needed for FFT
# Default FPS if calculation fails
DEFAULT_FPS = 21 # Adjust based on expected performance

# -- Visualization Parameters --
# BBox Color (BGR)
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 2
LABEL_FONT = 0 # cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 2

# Temperature Display
TEMP_DISPLAY_COLOR = (0, 255, 0) # BGR for Thermal Image (Green)
TEMP_FONT_SCALE = 3
TEMP_THICKNESS = 3

# Respiration Display
RESP_DISPLAY_COLOR = (0, 0, 255) # BGR for Visible Image (Red)
RESP_FONT_SCALE = 3
RESP_THICKNESS = 3

# Respiration Analysis Range
RESP_MIN_BPM = 1.0
RESP_MAX_BPM = 100.0

# Mask Overlay
MASK_OVERLAY_COLOR = [255, 0, 0] # RGB
MASK_OVERLAY_ALPHA = 0.5

# Window Names
WINDOW_CAMERA = 'Camera'
WINDOW_THERMAL = 'Thermal Camera'
WINDOW_MASK_OVERLAY = 'MASK Overlay'
WINDOW_MASK_SEGMENTED = 'MASK Segmented'
WINDOW_THERMAL_MASK_SEGMENTED = 'THERMAL MASK Segmented'
WINDOW_THERMAL_SKIN_MASK_SEGMENTED = 'THERMAL SKIN MASK Segmented'
# -- Device --
# Auto-detect CUDA or use CPU
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
