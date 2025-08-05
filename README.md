# Respiration and Temperature Monitor Application

## 描述 (Description)

本專案旨在結合可見光攝影機和熱影像攝影機，打造一個即時的生理資訊監測系統。系統利用 YOLO 物件偵測模型來定位畫面中的人臉/頭部區域，並透過分析該區域在熱影像上的數據來估算即時體溫與呼吸速率。此外，專案還整合了 UNet 模型對頭部進行影像分割，其結果主要用於視覺化展示，以輔助判斷偵測區域的精確性。整個應用程式的設定參數皆可透過 `config.py` 檔案進行集中管理。

除了核心的監測功能，專案還包含了相機校準、臉部圖像擷取 等輔助腳本。

## 功能特色 (Features)

  * **雙攝影機影像整合**:
      * 可從 UVC 熱影像攝影機（如 PureThermal）讀取 16 位元原始熱數據流。
      * 透過 GStreamer pipeline 從可見光攝影機（如 CSI 攝影機）高效讀取影像流。
  * **影像對齊與處理**:
      * 透過預先校準的特徵點，計算透視變換矩陣，以精確對齊可見光與熱影像畫面。
      * 提供獨立的 `calibrate_v3.py` 腳本，讓使用者能以圖形化介面手動選點，自動產生並更新校準參數。
  * **AI 驅動的偵測與分割**:
      * 使用 YOLOv11n 模型即時偵測畫面中最大的人臉/頭部區域，作為分析目標。
      * 使用 UNet 模型對偵測到的頭部 ROI (Region of Interest) 進行影像分割，並能以多種方式（如遮罩疊加、前景提取）視覺化結果。
  * **生理數據分析**:
      * **體溫偵測**: 計算頭部偵測框內熱影像的最高溫度，並轉換為攝氏度顯示。
      * **呼吸率估算**: 維護一個溫度變化的時間序列佇列，並透過快速傅立葉變換 (FFT) 分析此序列，從中找出主要頻率，進而估算出每分鐘的呼吸次數 (BPM)。
  * **豐富的視覺化介面**:
      * 在多個獨立視窗中即時顯示：對齊後的可見光影像、8 位元熱影像、YOLO 偵測框、計算出的體溫和呼吸率、以及 UNet 分割結果。
      * 所有視覺化元件（如邊界框顏色、文字大小與顏色）皆可透過 `config.py` 進行客製化。
  * **輔助工具**:
      * 包含 `crop_face.py` 腳本，可自動偵測頭部並將其裁剪儲存為獨立圖片檔案，用於建立訓練資料集。
      * 包含 `get_temp.py` 腳本，提供一個簡易的獨立功能，讓使用者能手動框選區域並即時查看該區域的最高溫度。

## 專案結構 (Project Structure)

```
respiration-monitor-app/
├── main_app.py                 # 主程式入口，協調所有模組
├── config.py                   # 配置文件 (模型路徑, 相機參數, 閾值等)
├── README.md                   # 本檔案 - 專案說明
├── uvctypes.py                 # libuvc 的 Python ctypes 接口定義
|
├── calibrate_v3.py             # 獨立腳本：用於相機畫面校準
├── crop_face.py                # 獨立腳本：用於自動裁剪並儲存臉部圖片
├── get_temp.py                 # 獨立腳本：用於手動 ROI 溫度量測
|
├── camera_utils/               # 相機相關工具模組
│   ├── thermal_camera.py       # 處理 UVC 熱影像相機
│   └── visible_camera.py       # 處理可見光相機
|
├── image_processing/           # 影像處理模組
│   ├── alignment.py            # 透視變換 (對齊)
│   └── basic_ops.py            # 基本圖像操作 (格式轉換, 溫度校正, 裁剪 ROI)
|
├── models/                     # 機器學習模型相關模組
│   ├── detector.py             # YOLO 物件偵測器
│   ├── segmenter.py            # UNet 圖像分割器
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

4.  **安裝 Python 依賴:**
    專案中的 `crop_face.py` 和 `models/detector.py` 都使用了 `ultralytics` 函式庫，`main_app.py` 和 `models/segmenter.py` 使用了 `torch`。根據程式碼中的 `import` 語句，建議的 `requirements.txt` 如下：

    ```txt
    numpy
    opencv-python
    torch
    torchvision
    ultralytics
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
      * **首次使用時，強烈建議執行 `python calibrate_v3.py`**。此腳本會引導您在兩個即時視窗中手動點擊對應的特徵點，然後自動將校準好的點寫入 `config.py`。
  * **模型與路徑**:
      * `YOLO_MODEL_PATH`, `UNET_MODEL_PATH`: 確認模型檔案路徑正確。
  * **相機設定**:
      * `GST_PIPELINE`: 如果您的可見光相機或硬體設定不同，需要修改此 GStreamer pipeline 字串。
      * `THERMAL_VID`, `THERMAL_PID`: 如果您的熱像儀 VID/PID 不同，請在此修改。
  * **演算法閾值**:
      * `YOLO_CONF_THRESHOLD`, `UNET_CONF_THRESHOLD`: 可根據實際效果調整偵測和分割的置信度閾值。
  * **執行時間**:
      * `DURATION`: 控制 `main_app.py` 自動執行的總時長（秒）。

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

3.  **（可選）使用輔助工具:**

      * **擷取臉部資料:**
        ```bash
        python crop_face.py
        ```
        此腳本會運行並自動將偵測到的頭部儲存到 `face/` 資料夾下，可用於訓練您自己的模型。
      * **快速溫度量測:**
        ```bash
        python get_temp.py
        ```
        此腳本會開啟熱像儀畫面，讓您用滑鼠拖曳一個矩形區域，並即時顯示該區域的最高溫度。

## 注意事項 (Notes)

  * **對齊精度**: 影像對齊的準確性完全依賴於 `config.py` 中校準點的精度。如果對齊效果不佳，請務必重新運行 `calibrate_v3.py`。
  * **呼吸率計算穩定性**: 呼吸率的計算基於溫度變化的 FFT 分析，其穩定性會受到多種因素影響，例如偵測框是否穩定、頭部是否遠離畫面、實際呼吸模式的規律性等。
  * **UNet 分割模型的應用**: 目前 UNet 的分割結果主要用於視覺化展示。一個潛在的優化方向是，利用 `segmenter.extract_foreground` 產生的精確 Mask 來取代矩形的 YOLO 偵測框，作為計算溫度的區域，這可能可以排除背景雜訊的干擾，得到更精準的溫度數據。
  * **效能**: 在嵌入式裝置（如 Jetson Nano）上同時運行 YOLO 和 UNet 對計算資源要求較高，可能會導致 FPS (每秒幀數) 下降。可以考慮在 `config.py` 中提供開關，以選擇性地停用 UNet 分割來提升效能。