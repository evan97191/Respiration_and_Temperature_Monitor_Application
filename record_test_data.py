import os
import time

import cv2
import numpy as np

import config
from camera_utils.camera_thread import CameraThread
from camera_utils.thermal_camera import ThermalCameraUVC
from camera_utils.visible_camera import VisibleCamera


def record_data(duration=20):
    os.makedirs("test_data", exist_ok=True)

    print("初始化實體攝影機...")
    thermal_cam = ThermalCameraUVC(vid=config.THERMAL_VID, pid=config.THERMAL_PID)
    visible_cam = VisibleCamera(pipeline=config.GST_PIPELINE)

    thermal_thread = CameraThread(thermal_cam, name="ThermalThread")
    visible_thread = CameraThread(visible_cam, name="VisibleThread")

    time.sleep(2.0)
    print("等待攝影機暖機完畢...")

    # -- 準備可見光鏡頭錄影 (MP4, 8-bit) --
    ret_vis, sample_frame, _ = visible_thread.read()
    if not ret_vis or sample_frame is None:
        print("無法取得可見光畫面，錄影失敗。")
        thermal_thread.stop()
        visible_thread.stop()
        return

    height, width = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vis_writer = cv2.VideoWriter("test_data/visible_test.mp4", fourcc, 21.0, (width, height))

    # -- 準備熱像儀錄影 (NPY, 16-bit raw data) --
    # 因為熱像儀資料包含絕對溫度的 16-bit 原始輻射值，如果存成 .mp4 會被壓縮成 8-bit，導致溫度無法計算！
    # 所以我們必須把原始矩陣存成 .npy 檔。
    thermal_frames = []

    print(f"開始錄影... (預計 {duration} 秒)")
    start_time = time.time()
    last_therm_time = 0.0

    try:
        while time.time() - start_time < duration:
            ret_vis, vis_frame, _ = visible_thread.read()
            ret_therm, therm_frame, therm_time = thermal_thread.read()

            if ret_vis and vis_frame is not None:
                vis_writer.write(vis_frame)

            if ret_therm and therm_frame is not None:
                if therm_time != last_therm_time:
                    thermal_frames.append(therm_frame)
                    last_therm_time = therm_time

            time.sleep(1 / 25.0)  # 稍微 Sleep 控制讀取迴圈
    except KeyboardInterrupt:
        print("錄影被使用者手動中斷。")

    print("錄影結束！正在儲存檔案...")

    # 儲存與釋放資源
    vis_writer.release()
    thermal_thread.stop()
    visible_thread.stop()

    # 將熱像儀資料轉換為 numpy array (N, H, W) 並存檔
    thermal_array = np.array(thermal_frames, dtype=np.uint16)
    np.save("test_data/thermal_test.npy", thermal_array)

    print(f"✅ 成功儲存：test_data/visible_test.mp4 (共 {int((time.time() - start_time) * 21)} 幀)")
    print(f"✅ 成功儲存：test_data/thermal_test.npy (共 {len(thermal_frames)} 幀)")


if __name__ == "__main__":
    record_data(duration=30)
