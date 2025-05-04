# models/detector.py

from ultralytics import YOLO

class YoloDetector:
    """Handles YOLO object detection."""

    def __init__(self, model_path):
        print(f"Loading YOLO model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def predict(self, frame, conf_threshold=0.5, verbose=False):
        """Performs object detection on a frame."""
        if frame is None:
            print("Warning: Cannot run YOLO prediction, frame is None.")
            return []
        try:
            results = self.model(frame, conf=conf_threshold, verbose=verbose)
            return results
        except Exception as e:
            print(f"Error during YOLO prediction: {e}")
            return [] # Return empty list on error

    @staticmethod
    def find_largest_box(results):
        """Finds the bounding box with the largest area from YOLO results."""
        largest_box = None
        max_area = 0

        if not results:
            return None

        try:
            # Assuming results is a list (Ultralytics format)
            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                for box in r.boxes:
                    if box.xyxy.numel() == 0: # Check if tensor is empty
                        continue
                    coords = box.xyxy[0]
                    if len(coords) < 4: # Ensure coords has at least 4 elements
                        continue

                    x1, y1, x2, y2 = coords
                    area = (x2 - x1) * (y2 - y1)

                    if area > max_area:
                        max_area = area
                        largest_box = {
                            "x1": x1.item(),
                            "y1": y1.item(),
                            "x2": x2.item(),
                            "y2": y2.item(),
                            "confidence": box.conf[0].item() if box.conf is not None and box.conf.numel() > 0 else 0.0,
                            "class_id": int(box.cls[0].item()) if box.cls is not None and box.cls.numel() > 0 else -1
                        }
        except Exception as e:
            print(f"Error processing YOLO results: {e}")
            return None # Return None if processing fails

        return largest_box