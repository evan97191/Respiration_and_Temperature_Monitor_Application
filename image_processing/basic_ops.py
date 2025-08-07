# image_processing/basic_ops.py

import numpy as np
import cv2

def ktof(val):
  """Converts Kelvin * 100 to Fahrenheit."""
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  """Converts Kelvin * 100 to Celsius using correction."""
  return temp_correction(val) / 100.0

def temp_correction(temp):
    """ Applies polynomial temperature correction. """
    # Coefficients derived from the original script's polyfit
    # It's better to calculate this once and store coefficients if possible
    Tx = np.array([30760, 30850, 30950, 31040, 31120, 31260, 31360, 31420, 31520, 31570])
    Tc = np.array([3200, 3300, 3400, 3500, 3600, 3800, 3900, 4000, 4100, 4200])
    z = np.polyfit(Tx, Tc, 2)
    pp = np.poly1d(z)
    return pp(temp)

def raw_to_8bit(data):
  """Converts raw 16-bit thermal data to an 8-bit BGR image."""
  if data is None:
      return None
  # Create a copy to avoid modifying the original thermal data
  data_copy = data.copy()
  cv2.normalize(data_copy, data_copy, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data_copy, 8, data_copy)
  # Convert the normalized 8-bit grayscale to BGR
  img_bgr = cv2.cvtColor(np.uint8(data_copy), cv2.COLOR_GRAY2BGR)
  return img_bgr

def cut_roi(image, box):
    """Cuts a Region of Interest (ROI) from an image based on a bounding box."""
    if image is None or box is None:
        return None

    # Get box coordinates, ensure integer type
    try:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    except (KeyError, TypeError):
        print("Error: Invalid box format for cutting ROI.")
        return None

    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Check if the resulting ROI is valid (has non-zero width and height)
    if x1 >= x2 or y1 >= y2:
        # print(f"Warning: Invalid ROI coordinates after clipping: [{y1}:{y2}, {x1}:{x2}]")
        return None # Return None for empty ROI

    # Extract the ROI
    roi = image[y1:y2, x1:x2]

    # Double-check if slicing resulted in an empty array (should be covered above)
    if roi.size == 0:
        return None

    return roi

def create_skin_mask(bgr_image):
    """Create a mask of skin."""
    # Convert an image from BGR to YCrCb color space
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
    
    # Separate Y, Cr, Cb channels
    Y, Cr, Cb = cv2.split(ycrcb_image)

    # Define skin color conditions
    # That is, we want to find all pixels that *meet* the skin condition
    skin_in_condition = (
        (Y > 80) &                # Luminance Conditions
        (Cr > 135) & (Cr < 180) & # Red Chroma Range
        (Cb > 85) & (Cb < 135)    # Blue Chroma Range
    )

    # Create a completely black mask
    mask = np.zeros(bgr_image.shape[:2], dtype="uint8")
    
    # Set the pixel locations that meet the skin criteria to white (255)
    mask[skin_in_condition] = 255
    
    return mask