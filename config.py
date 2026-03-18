# config.py

# -- Model Paths --
YOLO_MODEL_PATH = "yolo11n_headmask.pt"
UNET_MODEL_PATH = "unet_msfd_model_best.pth"

# Excution time
DURATION = 300 # second

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
# YOLO configuration

# UNet Segmentation Threshold
UNET_CONF_THRESHOLD = 0.5 # As per original code, adjust if needed (0.5 is common)
# UNet Model Input Size (Height, Width) - MUST MATCH TRAINING
UNET_INPUT_SIZE = (256, 256)

# -- Analysis Parameters --
TEMPERATURE_QUEUE_MAX_SIZE = 15 * 9
RESPIRATION_MIN_DATA_POINTS = 30 # Minimum points needed for FFT
# Default FPS if calculation fails
DEFAULT_FPS = 21 # Adjust based on expected performance
# FFT zero-padding factor (improves spectral peak detection precision, 4 = 4x interpolation)
FFT_ZERO_PAD_FACTOR = 4
# FFT Minimum target_length (improves spectral peak detection precision)
TARGET_FFT_LEN = 1024
# Minimum samples needed before applying Butterworth bandpass filter
BANDPASS_FILTER_MIN_SAMPLES = 30

# -- Temperature Extraction Parameters --
# Method to extract head temperature from ROI: 
# 'percentile' (recommended, average of top 5% hottest pixels, robust to noise)
# 'max' (legacy, absolute maximum single pixel, prone to hot-pixel noise)
TEMP_EXTRACTION_METHOD = 'percentile'

# -- Blackbody Calibration Parameters --
# Set to True to enable real-time temperature offsetting based on a fixed blackbody source
ENABLE_BLACKBODY_CALIBRATION = True
# The preset actual temperature of the blackbody (°C)
BLACKBODY_TEMP_C = 37.0
# Fixed ROI for the blackbody in the *thermal* coordinate space (x, y, width, height)
# You can use get_temp.py to find the exact coordinates of your blackbody in the thermal frame.
BLACKBODY_ROI = (548, 457, 566, 476)

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
RESP_MIN_BPM = 6.0
RESP_MAX_BPM = 30.0

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
WINDOW_ANALYSIS = 'Analysis Graphs'

SHOW_VISIBLE_CAMERA_UI = True # Toggle to turn off the visible camera popup
SHOW_THERMAL_UI = True # Toggle to turn off the individual Thermal popup
SHOW_MASK_OVERLAY_UI = False # Toggle for MASK Overlay window
SHOW_MASK_SEGMENTED_UI = False # Toggle for MASK Segmented window
SHOW_THERMAL_MASK_SEGMENTED_UI = True # Toggle for THERMAL MASK Segmented window
SHOW_THERMAL_SKIN_MASK_SEGMENTED_UI = False # Toggle for THERMAL SKIN MASK Segmented window
SHOW_ANALYSIS_UI = True # Toggle for real-time Analysis Graphs
# -- Device --
# Auto-detect CUDA or use CPU
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
