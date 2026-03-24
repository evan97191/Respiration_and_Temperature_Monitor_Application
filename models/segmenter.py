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
    """Handles UNet image segmentation (PyTorch or TensorRT)."""

    def __init__(self, model_path, device, n_channels=3, n_classes=1, bilinear=True):
        print(f"Loading UNet model from: {model_path}")
        self.device = device
        self.is_trt = str(model_path).endswith('.engine') or str(model_path).endswith('.trt')
        
        if self.is_trt:
            self._init_trt(model_path)
            return

        self.model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle potential 'module.' prefix if saved with DataParallel
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            
            # Removed self.model.half() as per user request to maintain stability for export
            
            self.model.eval() # Set to evaluation mode
            print("UNet model loaded successfully (PyTorch FP32).")
        except FileNotFoundError:
            print(f"Error: UNet model file not found at {model_path}")
            raise
        except Exception as e:
            print(f"Error loading UNet state dict: {e}")
            raise

    def _init_trt(self, model_path):
        import tensorrt as trt
        print("Initializing TensorRT engine for UNet...")
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.stream = torch.cuda.Stream() # Use non-default stream to satisfy TRT 10+
        
        # Parse bindings using the TRT 10+ V3 API
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            shape = self.engine.get_tensor_shape(name)
            if shape[0] == -1:
                shape = (1,) + shape[1:]
            
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
            # Pre-allocate PyTorch tensor mapped to GPU
            tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            
            if is_input:
                self.inputs.append(tensor)
            else:
                self.outputs.append(tensor)
                
        print("TensorRT UNet engine loaded successfully.")

    def preprocess(self, img_bgr, target_size=(256, 256)):
        """Preprocesses BGR image (NumPy) for inference."""
        if img_bgr is None:
            return None

        # 1. Convert to RGB natively in NumPy
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. To PyTorch Tensor and to Device
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 3. Resize on GPU using torch functional interpolation
        img_tensor = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        # 4. If using TRT, it may require FP16 input natively depending on engine creation
        if self.is_trt and self.inputs[0].dtype == torch.float16:
            img_tensor = img_tensor.half()
        elif not self.is_trt:
            pass # Keep FP32 for PyTorch

        return img_tensor

    def predict(self, input_tensor, threshold=0.5):
        """Performs segmentation prediction and returns the mask (NumPy)."""
        if input_tensor is None:
            return None
            
        try:
            if self.is_trt:
                # Use custom stream for memory copy and execution
                with torch.cuda.stream(self.stream):
                    # Copy dynamic input to the static binding memory address
                    self.inputs[0].copy_(input_tensor.contiguous(), non_blocking=True)
                    # Fire the asynchronous TRT execution (V3 API) with the non-default stream
                    self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                
                # Synchronize stream before fetching output
                self.stream.synchronize()
                logits = self.outputs[0]
            else:
                with torch.no_grad():
                    logits = self.model(input_tensor)
            
            probs = torch.sigmoid(logits.float()) # calculate Sigmoid on FP32 stability
            pred_mask = (probs > threshold).float()

            return pred_mask.squeeze().cpu().numpy()
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