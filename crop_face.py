import time
import cv2
import numpy as np
import os
import uuid # 用於生成唯一檔案名
from ultralytics import YOLO

# --- 配置參數 ---
MODEL_PATH = "yolo11n_headmask.pt"  # 你的 YOLO 模型路徑
# CAMERA_SOURCE = 0                 # 使用預設攝像頭 (索引 0)
# 或者使用你的 GStreamer pipeline (如果需要)
CAMERA_SOURCE = (
    "nvarguscamerasrc sensor_mode=0 ! "
    "video/x-raw(memory:NVMM), width=3820, height=2464, framerate=21/1, format=NV12 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=960, height=616 ! " # 降低解析度以提高速度
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true sync=false"
)

OUTPUT_DIR = "face"               # 儲存裁剪圖片的資料夾名稱
TARGET_CLASS_ID = 2               # !!! 重要：戴口罩頭部的 Class ID (根據你的模型修改) !!!
CONF_THRESHOLD = 0.5              # YOLO 偵測的置信度閾值
CONF_THRESHOLD = 0.5              # YOLO 偵測的置信度閾值
# --- 配置結束 ---

def crop_roi(image, box_coords):
    """根據 YOLO 的 box 座標裁剪 ROI"""
    if image is None or box_coords is None:
        return None
    try:
        x1, y1, x2, y2 = map(int, box_coords) # 確保是整數
    except (ValueError, TypeError):
        print("Error: Invalid box coordinates format.")
        return None

    h, w = image.shape[:2]
    # 邊界檢查和修正
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x1 >= x2 or y1 >= y2:
        # print("Warning: Invalid ROI dimensions after clipping.")
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi

if __name__ == "__main__":
    # 1. 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"圖片將儲存到 '{OUTPUT_DIR}' 資料夾。")

    # 2. 載入 YOLO 模型
    try:
        model = YOLO(MODEL_PATH)
        print(f"成功載入 YOLO 模型: {MODEL_PATH}")
    except Exception as e:
        print(f"錯誤：無法載入 YOLO 模型 '{MODEL_PATH}'. {e}")
        exit()

    # 3. 開啟攝影機
    print("開啟攝影機...")
    if isinstance(CAMERA_SOURCE, str): # GStreamer pipeline
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_GSTREAMER)
    else: # Camera index
        cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print(f"錯誤：無法開啟攝影機源 '{CAMERA_SOURCE}'.")
        exit()
    print("攝影機成功開啟。")
    ret, frame = cap.read()
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    # 4. 初始化計時器和計數器
    start_time = time.time()
    saved_image_count = 0
    saved_image_count = 0

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 800+1, 600+1)
    
    while True:
        print(f"{time.time() - start_time}")
        if saved_image_count  >= 20:
            print(f"以儲存{saved_image_count}張圖片")
            break
        ret, frame = cap.read()
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        for box in results[0].boxes:
            if box.cls is not None and box.cls.numel() > 0 and int(box.cls[0].item()) != TARGET_CLASS_ID:
                # 裁剪 ROI
                cropped_face = crop_roi(frame, box.xyxy[0].cpu().numpy())
                if cropped_face is not None:
                    # 生成唯一檔案名
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    unique_id = uuid.uuid4().hex[:6] # 取 UUID 的前 6 位
                    filename = f"face_{timestamp_str}_{unique_id}.png"
                    save_path = os.path.join(OUTPUT_DIR, filename)

                    # 儲存圖片
                    try:
                        cv2.imwrite(save_path, cropped_face)
                        saved_image_count += 1
                        # print(f"Saved: {filename}") # 可以取消註解來看詳細儲存訊息
                    except Exception as e:
                        print(f"儲存圖片時發生錯誤 '{save_path}': {e}")
        cv2.imshow("test", results[0].plot())
        time.sleep(0.5)