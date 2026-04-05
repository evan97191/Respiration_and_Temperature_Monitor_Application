# Respiration and Temperature Monitor Application

## 描述 (Description)

本專案旨在結合可見光攝影機和熱影像攝影機，打造一個即時的生理資訊監測系統。系統利用 YOLO 物件偵測模型來定位畫面中的人臉/頭部區域，並利用 Unet 分割模型來分割出臉上的口罩區域，透過計算兩個區域在熱影像上的數據來估算即時體溫與呼吸速率。此外，整個應用程式的設定參數皆可透過 `config.py` 檔案進行集中管理。

除了核心的監測功能，專案還包含了相機校準、臉部圖像擷取 等輔助程式。

## 功能特色 (Features)

  * **極致效能最佳化 (Edge Performance Optimizations)**:
      * **TensorRT FP16 引擎加速**: YOLOv11n 與 U-Net 均已轉換為 TensorRT `.engine`，並完全遷移至 **TensorRT V3 API** (Pointer-based) 以達成零拷貝 (Zero-copy) 傳輸，Orin NX 上的推論延遲分別壓低至 ~27ms 與 ~25ms。
      * **背景非同步與硬體監控**: 除了非同步影像擷取，系統還實作了基於 `jtop` (`jetson-stats`) 的背景硬體監控，能即時錄製 CPU/GPU 使用率與系統功耗。
      * **自動化基準測試 (Benchmarking)**: 導入 `MockCamera` 框架，可讀取預錄的 16-bit 原始數據 (`.npy`)，排除了實體環境變因，確保效能測試 100% 可復現。
      * **局部座標轉換 (BBox Warp)**: 僅對由 YOLO 產生的邊界框進行透視變換，大幅減少 CPU 運算開銷。
      * **UI 渲染降頻 (Decimation)**: 針對 OpenCV 繪圖瓶頸，支援 `SHOW_ANALYSIS_UI` 開關，在極限效能測試時可停用 UI 以釋放算力。

  * **雙攝影機影像整合**:
      * 可從 UVC 熱影像攝影機（如 PureThermal）讀取 16 位元原始熱數據流。
      * 透過 GStreamer pipeline 從可見光攝影機（如 CSI 攝影機）高效讀取影像流。
  * **影像對齊與處理**:
      * 透過預先校準的特徵點，計算透視變換矩陣，以精確對齊可見光與熱影像畫面。
      * 提供獨立的 `calibrate_v3.py` 程式，讓使用者能以圖形化介面手動選點，自動產生並更新校準參數。
  * **AI 驅動的偵測與分割**:
      * 使用 YOLOv11n 模型即時偵測畫面中最大的人臉/頭部區域，作為分析目標。
      * 使用 UNet 模型對偵測到的頭部 ROI (Region of Interest) 進行口罩分割，並能以多種方式（如遮罩疊加、前景提取）視覺化結果。
  * **生理數據分析**:
      * **體溫偵測**: 計算頭部偵測框內熱影像的最高溫度，並轉換為攝氏度顯示。
      * **呼吸率估算**: 維護一個溫度變化的時間序列佇列，並透過快速傅立葉變換 (FFT) 分析此序列，從中找出主要頻率，進而估算出每分鐘的呼吸次數 (BPM)。
  * **豐富的視覺化介面**:
      * 在多個獨立視窗中即時顯示：對齊後的可見光影像、8 位元熱影像、YOLO 偵測框、計算出的體溫和呼吸率、以及 UNet 分割結果。
      * 所有視覺化元件（如邊界框顏色、文字大小與顏色）皆可透過 `config.py` 進行客製化。
  * **輔助工具**:
      * 包含 `crop_face.py` 程式，可自動偵測頭部並將其裁剪儲存為獨立圖片檔案，用於建立訓練資料集。
      * 包含 `get_temp.py` 程式，提供一個簡易的獨立功能，讓使用者能手動框選區域並即時查看該區域的最高溫度。

## 專案結構 (Project Structure)

```
respiration-monitor-app/
├── main_app.py                 # 主程式入口，協調所有模組
├── config.py                   # 配置文件 (模型路徑, 相機參數, 閾值等)
├── README.md                   # 本檔案 - 專案說明
├── uvctypes.py                 # libuvc 的 Python ctypes 接口定義
├── Dockerfile                  # 容器建置腳本
├── docker-compose.yml          # 容器服務編排檔案
├── docker_command.txt          # Docker 執行指令參考
|
├── *.pth / *.pt                # 各式 YOLO 與 UNet 模型權重檔
|
├── calibrate_v3.py             # 獨立程式：用於相機畫面校準
├── record_test_data.py         # 獨立程式：用於錄製 16-bit 原始輻射數據供基準測試使用
├── crop_face.py                # 獨立程式：用於自動裁剪並儲存臉部圖片
├── get_temp.py                 # 獨立程式：用於手動 ROI 溫度量測
|
├── camera_utils/               # 相機相關工具模組
│   ├── camera_thread.py        # 背景非同步相機擷取執行緒
│   ├── thermal_camera.py       # 處理 UVC 熱影像相機
│   └── visible_camera.py       # 處理可見光相機
|
├── image_processing/           # 影像處理模組
│   ├── alignment.py            # 透視變換 (對齊)
│   └── basic_ops.py            # 基本圖像操作 (格式轉換, 溫度校正, 裁剪 ROI)
|
├── models/                     # 機器學習模型相關模組
│   ├── detector.py             # YOLO 物件偵測器 (支援 TensorRT 推論)
│   ├── segmenter.py            # UNet 圖像分割器 (支援 TensorRT 推論)
│   ├── unet_model.py           # UNet 模型架構定義
│   └── unet_parts.py           # UNet 模型組件定義
|
├── analysis/                   # 數據分析模組
│   ├── temperature.py          # 溫度計算 (平均值, 最大值)
│   ├── respiration.py          # 呼吸率計算 (FFT)
│   └── signal_utils.py         # (可選) 信號處理工具 (移動平均濾波等)
|
└── utils/                      # 通用工具模組
    ├── visualization.py        # 視覺化 (繪製框, 顯示文字, 管理窗口)
    ├── profiler.py             # 效能探針 (Profiler, TimeIt)
    ├── hardware_monitor.py     # 基於 jtop 的硬體監控器
    └── timing.py               # 時間/FPS 計算
```

## 安裝設定 (Setup / Installation)

1.  **複製專案:**

    ```bash
    git clone <your-repository-url>
    cd respiration-monitor-app
    ```

2.  **建立 Python 虛擬環境:** (建議)

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **安裝系統依賴 `libuvc`:**
    這是操作 UVC 熱像儀所必需的函式庫。

      * **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update
        sudo apt install build-essential libusb-1.0-0-dev cmake
        git clone https://github.com/libuvc/libuvc.git
        cd libuvc && mkdir build && cd build
        cmake .. && make && sudo make install
        sudo ldconfig -v
        cd ../..
        ```
      * **macOS:**
        ```bash
        brew install libuvc
        ```

5.  **USB 設備權限設定 (USB Device Permissions):**
    在 Linux 主機（包含 Jetson）上，預設情況下一般使用者無法直接存取 UVC 熱像儀的原始數據流。若在運行時遇到 `uvc_open error: -3` (Access Denied)，請在 **Host 主機端** 執行我們提供的設定腳本：

    ```bash
    chmod +x scripts/setup_udev.sh
    ./scripts/setup_udev.sh
    ```
    **設定完成後，請務必重新插拔熱像儀。** 這樣無論是直接執行還是透過 Docker 執行，程式都能獲得正確的存取權限。

4.  **安裝 Python 依賴:**
    專案中的 `crop_face.py` 和 `models/detector.py` 都使用了 `ultralytics` 函式庫，`main_app.py` 和 `models/segmenter.py` 使用了 `torch`。根據程式碼中的 `import` 語句，建議的 `requirements.txt` 如下：

    ```txt
    numpy
    opencv-python
    torch
    torchvision
    ultralytics
    scipy
    jetson-stats   # 用於硬體監控 (jtop)
    pycuda         # TensorRT 推論所需 (若使用 .engine)
    ```

    執行安裝：

    ```bash
    pip install -r requirements.txt
    ```

5.  **模型檔案:**
    確保 `unet_model_best.pth` 和 `yolo11n_headmask.pt` 檔案已放置在專案根目錄，或已在 `config.py` 中指定了它們的正確路徑。

6.  **硬體連接:**
    連接好您的 UVC 熱影像攝影機和可見光攝影機。

## 配置 (Configuration)

所有重要的配置參數都集中在 `config.py` 檔案中。在運行前，務必檢查並修改：

  * **`POINTS_IR` & `POINTS_VIS`**: **極其重要**。這兩組點定義了熱影像與可見光影像之間的對應關係，直接影響畫面是否對齊。
      * **首次使用時，強烈建議執行 `python calibrate_v3.py`**。此程式會引導您在兩個即時視窗中手動點擊對應的特徵點，然後自動將校準好的點寫入 `config.py`。
  * **模型與路徑**:
      * `YOLO_MODEL_PATH`, `UNET_MODEL_PATH`: 確認模型檔案路徑正確。
  * **相機設定**:
      * `GST_PIPELINE`: 如果您的可見光相機或硬體設定不同，需要修改此 GStreamer pipeline 字串。
      * `THERMAL_VID`, `THERMAL_PID`: 如果您的熱像儀 VID/PID 不同，請在此修改。
  * **演算法閾值**:
      * `YOLO_CONF_THRESHOLD`, `UNET_CONF_THRESHOLD`: 可根據實際效果調整偵測和分割的置信度閾值。
  * **執行時間**:
      * `DURATION`: 控制 `main_app.py` 自動執行的總時長（秒）。

## Docker 部署 (Docker Deployment - 推薦)

為了確保套件版本與系統環境一致，強烈建議透過 Docker 來執行本專案：

1. **確認檔案**: 請確保專案目錄下已包含 `Dockerfile` 和 `docker-compose.yml`。
2. **啟動容器**: 參考 `docker_command.txt` 的指令啟動並進入包含視訊裝置權限的容器環境。
    ```bash
    # 範例
    sudo docker-compose up -d
    sudo docker exec -it <container_name> bash
    ```
3. 進入容器後，直接執行主程式 `python main_app.py` 即可。

## 使用方法 (Usage)

1.  **（首次）校準相機:**

    ```bash
    python calibrate_v3.py
    ```

    按照提示在兩個視窗中分別點擊 4 個對應的點。完成後，`config.py` 中的 `POINTS_IR` 和 `POINTS_VIS` 將會自動更新。

2.  **運行主程式:**

    ```bash
    python main_app.py
    ```

      * 程式將會開啟多個視窗顯示結果。
      * 程式會運行 `config.py` 中 `DURATION` 所設定的時間，然後自動計算平均體溫和呼吸率並退出。
      * 在運行期間，按下鍵盤上的 `q` 鍵可提前結束程式。

3.  **效能基準測試 (Automated Benchmarking):**

    若要在固定環境下測試效能，請在 `config.py` 中設置：
    ```python
    IS_TESTING = True
    MOCK_THERMAL_PATH = "test_data/input_thermal.npy" # 預錄數據路徑
    SHOW_ANALYSIS_UI = False # 關閉 UI 以獲得純粹效能數據
    ```
    然後執行 `python main_app.py`。系統會自動在結束時生成 `benchmark_result.json` 與 `hardware_stats.csv`。

4.  **錄製測試數據:**

    ```bash
    python record_test_data.py
    ```
    此程式會擷取熱像儀的 16-bit 原始輻射數據並存成 `.npy` 檔案，保留完整的溫度精準度。

5.  **（可選）使用輔助工具:**

      * **擷取臉部資料:**
        ```bash
        python crop_face.py
        ```
        此程式會運行並自動將偵測到的頭部儲存到 `face/` 資料夾下，可用於訓練您自己的模型。
      * **快速溫度量測:**
        ```bash
        python get_temp.py
        ```
        此程式會開啟熱像儀畫面，讓您用滑鼠拖曳一個矩形區域，並即時顯示該區域的最高溫度。

## 注意事項 (Notes)

  * **對齊精度**: 影像對齊的準確性完全依賴於 `config.py` 中校準點的精度。如果對齊效果不佳，請務必重新運行 `calibrate_v3.py`。
  * **呼吸率計算穩定性**: 呼吸率的計算基於溫度變化的 FFT 分析，其穩定性會受到多種因素影響，例如偵測框是否穩定、頭部是否遠離畫面、實際呼吸模式的規律性等。
  * **效能**: 在嵌入式裝置（如 Jetson Nano）上同時運行 YOLO 和 UNet 對計算資源要求較高，可能會導致 FPS (每秒幀數) 下降。

## 使用到的資料庫 (Datasets)

  * **MSFD**: https://github.com/sadjadrz/MFSD
  * **YOLO helmet/head**: https://www.kaggle.com/datasets/vodan37/yolo-helmethead
  * **Mask-Detection-Dataset**: https://github.com/archie9211/Mask-Detection-Dataset

## Copyright & License

© 2026 Evan. All Rights Reserved.

This project is developed for academic and research purposes. All source code and materials in this repository are the property of the author. You may view and read the code for reference, but you may NOT copy, distribute, modify, or use it for commercial or non-commercial purposes without explicit written permission.

**Disclaimer:** This system is for academic research and experimental purposes only. It is NOT intended for medical diagnosis, physiological monitoring, or treatment. The author provides no warranty and assumes no liability for any use of this software.
