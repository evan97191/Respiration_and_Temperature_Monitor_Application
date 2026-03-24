from ultralytics import YOLO
import sys

def main():
    model_path = 'yolo11n_headmask.pt'
    print(f"Exporting {model_path} to TensorRT...")
    try:
        model = YOLO(model_path)
        # Using format='engine' automatically invokes TensorRT exporter, half=True uses FP16
        model.export(format='engine', half=True)
        print("Success! File saved as yolo11n_headmask.engine")
    except Exception as e:
        print(f"Failed to export YOLO to TRT: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
