# models/segmenter.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Assumes unet_model.py and unet_parts.py are in the same directory
try:
    from .unet_model import UNet
except ImportError:
     # Fallback if running directly or structure issue
    try:
        from models.unet_model import UNet
    except ImportError:
        print("ERROR: Cannot import UNet model. Make sure unet_model.py/unet_parts.py are in the models directory.")
        import sys
        sys.exit(1)


class UNetSegmenter:
    """Handles UNet image segmentation."""

    def __init__(self, model_path, device, n_channels=3, n_classes=1, bilinear=True):
        print(f"Loading UNet model from: {model_path}")
        self.device = device
        self.model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle potential 'module.' prefix if saved with DataParallel
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            print("UNet model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: UNet model file not found at {model_path}")
            raise
        except Exception as e:
            print(f"Error loading UNet state dict: {e}")
            raise

    def preprocess(self, img_bgr, target_size=(256, 256)):
        """Preprocesses BGR image (NumPy) for UNet inference (No PIL)."""
        if img_bgr is None:
            return None

        # 1. Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize
        dsize = (target_size[1], target_size[0]) # (width, height) for cv2.resize
        img_rgb_resized = cv2.resize(img_rgb, dsize, interpolation=cv2.INTER_LINEAR)

        # --- Manual ToTensor ---
        # 3. Type to float32
        img_float = img_rgb_resized.astype(np.float32)
        # 4. Scale to [0.0, 1.0]
        img_normalized = img_float / 255.0
        # 5. HWC -> CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        # 6. To PyTorch Tensor
        img_tensor = torch.from_numpy(img_chw)

        # --- Optional Normalize ---
        # if config.UNET_USE_NORMALIZATION: # Add a flag in config if needed
        #    mean = torch.tensor(config.UNET_MEAN).view(3, 1, 1)
        #    std = torch.tensor(config.UNET_STD).view(3, 1, 1)
        #    img_tensor = (img_tensor - mean) / std

        # 7. Add Batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def predict(self, input_tensor, threshold=0.5):
        """Performs segmentation prediction and returns the mask (NumPy)."""
        if input_tensor is None:
            return None
        input_tensor = input_tensor.to(self.device)
        try:
            with torch.no_grad():
                logits = self.model(input_tensor)
                # Assuming binary classification (n_classes=1)
                probs = torch.sigmoid(logits)
                pred_mask = (probs > threshold).float()

            # Move mask to CPU and convert to NumPy
            pred_mask_np = pred_mask.squeeze().cpu().numpy()
            return pred_mask_np
        except Exception as e:
            print(f"Error during UNet prediction: {e}")
            return None

    @staticmethod
    def overlay_mask(image_np, mask_np, color=[255, 0, 0], alpha=0.5):
        """Applies a mask overlay onto an image (NumPy inputs)."""
        if image_np is None or mask_np is None:
            print("Warning: Cannot overlay mask, image or mask is None.")
            return image_np # Return original image if overlay fails

        # Resize mask to match image if necessary
        if image_np.shape[:2] != mask_np.shape:
            #print(f"Warning: Resizing mask ({mask_np.shape}) to match image ({image_np.shape[:2]}) for overlay.")
            try:
                mask_np_resized = cv2.resize(mask_np.astype(np.uint8),
                                           (image_np.shape[1], image_np.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                mask_np = mask_np_resized
            except Exception as e:
                 print(f"Error resizing mask for overlay: {e}. Cannot apply overlay.")
                 return image_np # Return original if resize fails

        overlay = image_np.copy()
        color_np = np.array(color, dtype=np.uint8) # Use RGB color here

        # Ensure mask_np is boolean for indexing
        foreground = mask_np > 0

        # Apply weighted add for overlay effect
        try:
            # Check if image_np is BGR or RGB (assume RGB based on preprocess)
            # cv2.addWeighted works correctly with multi-channel images
            overlay[foreground] = cv2.addWeighted(image_np[foreground], 1 - alpha,
                                                  np.full_like(image_np[foreground], color_np), alpha, 0)
        except Exception as e:
            print(f"Error applying addWeighted for overlay: {e}")
            # Fallback: Just color the foreground pixels (less visually appealing)
            # overlay[foreground] = color_np
            return image_np # Return original on error

        return overlay

    @staticmethod
    def extract_foreground(image_np, mask_np):
        """Extracts the foreground based on the mask (background becomes black)."""
        if image_np is None or mask_np is None:
            print("Warning: Cannot extract foreground, image or mask is None.")
            return image_np

        # Resize mask to match image if necessary
        if image_np.shape[:2] != mask_np.shape:
            #print(f"Warning: Resizing mask ({mask_np.shape}) to match image ({image_np.shape[:2]}) for extraction.")
            try:
                mask_np_resized = cv2.resize(mask_np.astype(np.uint16),
                                           (image_np.shape[1], image_np.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                mask_np = mask_np_resized
            except Exception as e:
                 print(f"Error resizing mask for extraction: {e}. Cannot extract.")
                 return image_np

        # Create black background
        segmented_img = np.zeros_like(image_np)
        foreground = mask_np > 0
        try:
            segmented_img[foreground] = image_np[foreground]
        except IndexError as e:
             print(f"Error during foreground extraction (likely mask/image mismatch): {e}")
             return image_np # Return original on error
        except Exception as e:
             print(f"Unexpected error during foreground extraction: {e}")
             return image_np

        return segmented_img