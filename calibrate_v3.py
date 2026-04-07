import time

import cv2
import numpy as np

from uvctypes import *

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

# 導入設定檔
import config

# --- 從舊程式碼中借用的熱像儀相關函式 (無變動) ---
BUF_SIZE = config.THERMAL_BUFFER_SIZE
q = Queue(BUF_SIZE)


def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(
        frame.contents.height, frame.contents.width
    )
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return
    if not q.full():
        q.put(data)


PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)


def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)


# --- 函式結束 ---


def select_points_callback(event, x, y, flags, param):
    point_list = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(point_list) < 4:
            point_list.append((x, y))
            print(f"新增點: {(x, y)}. 目前共 {len(point_list)}/4 點。")
        else:
            print("此影像的 4 個點已選取完畢。")


def draw_points(image, points):
    for i, point in enumerate(points):
        cv2.circle(image, point, 5, (0, 255, 0), -1)
        cv2.putText(image, str(i + 1), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def update_config_file(config_path, points_vis, points_ir):
    try:
        with open(config_path, "r") as f:
            lines = f.readlines()
        points_vis_list = [list(pt) for pt in points_vis]
        points_ir_list = [list(pt) for pt in points_ir]
        new_vis_line = f"POINTS_VIS = {points_vis_list}\n"
        new_ir_line = f"POINTS_IR = {points_ir_list}\n"
        vis_updated = False
        ir_updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith("POINTS_VIS"):
                lines[i] = new_vis_line
                vis_updated = True
                print("正在更新 POINTS_VIS...")
            elif line.strip().startswith("POINTS_IR"):
                lines[i] = new_ir_line
                ir_updated = True
                print("正在更新 POINTS_IR...")
        if not (vis_updated and ir_updated):
            print("錯誤：在 config.py 中找不到 'POINTS_IR' 或 'POINTS_VIS' 變數。正在檔案末尾追加...")
            lines.append("\n# Camera Calibration Points\n")
            lines.append(new_vis_line)
            lines.append(new_ir_line)
        with open(config_path, "w") as f:
            f.writelines(lines)
        print(f"\n成功將校準點寫入 '{config_path}'！")
        print("POINTS_VIS =", points_vis_list)
        print("POINTS_IR =", points_ir_list)
    except FileNotFoundError:
        print(f"錯誤：找不到設定檔 '{config_path}'。")
    except Exception as e:
        print(f"寫入設定檔時發生錯誤: {e}")


if __name__ == "__main__":
    # --- 定義與主程式 resp_new_v1.py 完全相同的尺寸 ---
    CALIB_VIS_WIDTH = 960  # 可見光影像寬度 (來自GStreamer)
    CALIB_VIS_HEIGHT = 616  # 可見光影像高度 (來自GStreamer)
    CALIB_IR_WIDTH = 800  # 紅外光影像寬度
    CALIB_IR_HEIGHT = 600  # 紅外光影像高度

    # --- 初始化攝影機 (使用與主程式相同的 GStreamer pipeline) ---
    gst_pipeline = (
        "nvarguscamerasrc sensor_mode=0 ! "
        "video/x-raw(memory:NVMM), width=3820, height=2464, framerate=21/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width={CALIB_VIS_WIDTH}, height={CALIB_VIS_HEIGHT} ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true sync=false"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("錯誤: 無法開啟可見光攝影機。請檢查 GStreamer pipeline。")
        exit()

    # --- 熱像儀初始化 (代碼不變) ---
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()
    try:
        res = libuvc.uvc_init(byref(ctx), 0)
        if res < 0:
            raise Exception("uvc_init error")
        res = libuvc.uvc_find_device(ctx, byref(dev), config.THERMAL_VID, config.THERMAL_PID, 0)
        if res < 0:
            raise Exception("uvc_find_device error")
        res = libuvc.uvc_open(dev, byref(devh))
        if res < 0:
            raise Exception("uvc_open error")
        print("熱像儀已連接。")
        frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
        libuvc.uvc_get_stream_ctrl_format_size(
            devh,
            byref(ctrl),
            UVC_FRAME_FORMAT_Y16,
            frame_formats[0].wWidth,
            frame_formats[0].wHeight,
            int(1e7 / frame_formats[0].dwDefaultFrameInterval),
        )
        res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        if res < 0:
            raise Exception("uvc_start_streaming error")
    except Exception as e:
        print(f"錯誤: 無法初始化熱像儀。錯誤訊息: {e}")
        cap.release()
        exit()

    # --- 準備校準 ---
    points_vis = []
    points_ir = []

    window_name_vis = "Visible Camera - Click 4 points (960x616)"
    window_name_ir = "Thermal Camera - Click 4 points (800x600)"
    cv2.namedWindow(window_name_vis)
    cv2.namedWindow(window_name_ir)

    cv2.setMouseCallback(window_name_vis, select_points_callback, points_vis)
    cv2.setMouseCallback(window_name_ir, select_points_callback, points_ir)

    print("\n" + "=" * 50)
    print("校準開始：請分別在兩個視窗上點擊 4 個對應的特徵點。")
    print("完成後程式會自動關閉。按 'q' 可提早結束。")
    print("=" * 50 + "\n")

    ### 主要變動：將影像獲取和處理移入主迴圈中 ###
    while True:
        # 獲取可見光影像
        ret, frame_vis = cap.read()
        if not ret:
            print("無法讀取可見光影像，跳過此幀。")
            time.sleep(0.1)
            continue

        # 獲取熱影像
        try:
            thermal_data = q.get(True, 0.5)  # 等待最多 0.5 秒
        except Exception:
            print("無法讀取熱影像，跳過此幀。")
            continue

        # 處理熱影像：縮放至目標尺寸並轉為 8-bit RGB
        thermal_data_resized = cv2.resize(thermal_data, (CALIB_IR_WIDTH, CALIB_IR_HEIGHT))
        frame_ir = raw_to_8bit(thermal_data_resized.copy())

        # 創建顯示用的影像副本，避免在原始影像上繪圖
        display_vis = frame_vis.copy()
        display_ir = frame_ir.copy()

        # 在影像上繪製已選取的點
        draw_points(display_vis, points_vis)
        draw_points(display_ir, points_ir)

        # 顯示即時影像
        cv2.imshow(window_name_vis, display_vis)
        cv2.imshow(window_name_ir, display_ir)

        # 檢查按鍵，或是否已點滿 4 對點
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print("使用者手動中斷。")
            break

        if len(points_vis) == 4 and len(points_ir) == 4:
            print("\n已成功選取 4 組對應點！")
            time.sleep(1)  # 暫停一秒，讓使用者看到最終結果
            break

    # --- 更新設定檔 ---
    if len(points_vis) == 4 and len(points_ir) == 4:
        update_config_file("config.py", points_vis, points_ir)
    else:
        print("\n未完成校準，點數不足，設定檔未更新。")

    # --- 清理資源 ---
    cap.release()
    libuvc.uvc_stop_streaming(devh)
    libuvc.uvc_unref_device(dev)
    libuvc.uvc_exit(ctx)
    cv2.destroyAllWindows()
    print("\n校準腳本執行完畢。")
