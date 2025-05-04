```markdown
# Respiration and Temperature Monitor Application

## 描述 (Description)

本專案旨在結合可見光攝影機和熱影像攝影機，利用 YOLO 物件偵測定位人臉區域，並透過分析熱影像數據來估算呼吸速率和偵測體溫。專案使用了 UNet 模型來輔助分割頭部區域（目前分割結果主要用於視覺化）。

## 功能特色 (Features)

*   從 UVC 熱影像攝影機和 GStreamer 可見光攝影機讀取影像流。
*   透過透視變換對齊可見光與熱影像畫面。
*   使用 YOLOv8n 模型偵測畫面中的人臉/頭部區域。
*   (視覺化) 使用 UNet 模型分割偵測到的頭部區域。
*   計算偵測框內熱影像的平均溫度和最高溫度。
*   透過分析溫度時間序列數據，使用 FFT 計算呼吸速率 (BPM)。
*   視覺化顯示：對齊後的影像、熱影像、偵測框、最高溫度、呼吸速率、分割結果（疊加和提取）。

```
## 專案結構 (Project Structure)

``` # <--- 在這裡加上三個反引號
your_project_directory/
├── main_app.py                 # 主程式入口，協調所有模組
├── config.py                   # 配置文件 (模型路徑, 相機參數, 閾值等)
├── uvctypes.py                 # libuvc 的 Python ctypes 接口定義
├── camera_utils/               # 相機相關工具模組
│   ├── __init__.py
│   ├── thermal_camera.py       # 處理 UVC 熱影像相機 (初始化, 回調, 數據獲取)
│   └── visible_camera.py       # 處理可見光相機 (GStreamer 初始化, 幀獲取)
├── image_processing/           # 影像處理模組
│   ├── __init__.py
│   ├── alignment.py            # 透視變換相關 (矩陣計算, 應用變換)
│   └── basic_ops.py            # 基本圖像操作 (raw_to_8bit, 溫度校正, 裁剪 ROI)
├── models/                     # 機器學習模型相關模組
│   ├── __init__.py
│   ├── detector.py             # YOLO 物件偵測 (模型加載, 推理, 結果處理)
│   └── segmenter.py            # UNet 圖像分割 (模型加載, 預處理, 推理, 後處理)
│   ├── unet_model.py           # UNet 模型架構定義
│   └── unet_parts.py           # UNet 模型組件定義
├── analysis/                   # 數據分析模組
│   ├── __init__.py
│   ├── temperature.py          # 溫度計算相關 (平均值, 最大值)
│   ├── respiration.py          # 呼吸率計算相關 (數據隊列, FFT)
│   └── signal_utils.py         # (可選) 信號處理工具 (移動平均濾波等)
└── utils/                      # 通用工具模組
    ├── __init__.py
    ├── visualization.py        # 視覺化相關 (繪製框, 顯示文字, 管理窗口)
    └── timing.py               # 時間/FPS 計算相關
├── unet_model_best.pth         # (模型權重) UNet 預訓練權重檔案 (*通常在 .gitignore 中*)
└── yolo11n_headmask.pt         # (模型權重) YOLO 預訓練權重檔案 (*通常在 .gitignore 中*)
├── .gitignore                  # 指定 Git 應忽略的檔案和資料夾
└── README.md                   # 本檔案 - 專案說明
```
```

### 檔案/模組說明

*   **`main_app.py`**: 應用程式的主執行檔案。負責初始化各個組件（相機、模型）、進入主迴圈處理影像幀、協調偵測、分割、分析和視覺化，並處理結束時的資源釋放。
*   **`config.py`**: 集中管理專案的所有配置參數，如模型檔案路徑、相機設定、影像處理參數（如對齊點）、偵測/分割閾值、分析參數和視覺化設定。 **修改參數請主要編輯此檔案。**
*   **`uvctypes.py`**: 定義了與 `libuvc` 庫交互所需的 C 語言結構體和常數的 Python `ctypes` 接口。
*   **`camera_utils/`**:
    *   `thermal_camera.py`: 封裝了與 PureThermal (或其他 UVC) 熱影像攝影機的交互邏輯，包括設備初始化、啟動流、設置 C 回調函數以及從隊列中安全地獲取幀數據。
    *   `visible_camera.py`: 封裝了使用 GStreamer pipeline 初始化和讀取可見光攝影機幀的邏輯。
*   **`image_processing/`**:
    *   `alignment.py`: 負責計算和應用透視變換矩陣，以對齊可見光和熱影像視角。
    *   `basic_ops.py`: 提供基礎的影像操作函數，例如將 16 位原始熱數據轉換為 8 位可顯示影像、進行溫度單位轉換與校正、以及根據邊界框裁剪 ROI。
*   **`models/`**:
    *   `detector.py`: 封裝 YOLO 物件偵測模型。包括載入模型、執行預測以及從預測結果中找出最大面積的邊界框。
    *   `segmenter.py`: 封裝 UNet 圖像分割模型。包括載入模型、針對 UNet 的圖像預處理、執行分割預測，以及將預測的 Mask 疊加到原圖或提取分割區域的後處理。
    *   `unet_model.py` / `unet_parts.py`: 定義了 UNet 神經網路的架構和其組成部分。
*   **`analysis/`**:
    *   `temperature.py`: 提供計算指定區域（ROI）內像素平均值和最大值的函數，用於溫度分析。
    *   `respiration.py`: 負責管理溫度的時間序列數據（使用隊列），並應用快速傅立葉變換（FFT）來估算呼吸頻率。
    *   `signal_utils.py`: (目前未使用，但可加入) 包含用於平滑信號（如溫度序列）的函數，例如移動平均濾波。
*   **`utils/`**:
    *   `visualization.py`: 包含在影像上繪製邊界框、顯示計算出的溫度或呼吸率數值、以及管理 OpenCV 顯示窗口的函數/類。
    *   `timing.py`: 提供計算幀處理時間間隔和估算平均 FPS 的工具。
*   **`.pth` / `.pt` 檔案**: 預先訓練好的 UNet 和 YOLO 模型權重檔案。**注意：** 這些檔案通常比較大，建議將它們加入 `.gitignore` 中，不納入 Git 版本控制，並透過其他方式（如手動複製、下載連結、Git LFS）來管理。

## 安裝設定 (Setup / Installation)

1.  **Clone Repository:**
    ```bash
    # 如果你已經按照 GitHub 步驟 clone 了
    # git clone <your-private-repository-url>
    cd your_project_directory
    ```
2.  **Python 環境:** 建議使用 Python 3.7 或更高版本。推薦建立虛擬環境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **安裝 `libuvc`:**
    *   **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update
        sudo apt install build-essential libusb-1.0-0-dev cmake
        git clone https://github.com/libuvc/libuvc.git
        cd libuvc
        mkdir build
        cd build
        cmake ..
        make && sudo make install
        sudo ldconfig -v # 更新 shared library cache
        cd ../.. # 返回專案目錄
        ```
    *   **macOS:**
        ```bash
        brew install libuvc
        ```
    *   **Windows:** 可能需要從源碼編譯或尋找預編譯的庫。
4.  **安裝 Python 依賴:**
    *   建議創建一個 `requirements.txt` 檔案，包含以下內容：
        ```txt
        torch
        torchvision
        torchaudio # Often needed with torch
        numpy
        opencv-python
        ultralytics
        matplotlib # For potential future plotting
        # Add other direct dependencies if any
        ```
    *   然後執行安裝：
        ```bash
        pip install -r requirements.txt
        ```
5.  **模型檔案:** 確保 `unet_model_best.pth` 和 `yolo11n_headmask.pt` 檔案位於專案根目錄，或者你已在 `config.py` 中指定了它們的正確路徑。
6.  **硬體連接:** 連接好你的 UVC 熱影像攝影機和可見光攝影機。

## 配置 (Configuration)

主要的配置參數都集中在 `config.py` 檔案中。在運行前，你可能需要檢查或修改：

*   `YOLO_MODEL_PATH`, `UNET_MODEL_PATH`: 確認模型檔案路徑正確。
*   `GST_PIPELINE`: 如果你的可見光相機或設定不同，需要修改 GStreamer pipeline 字串。
*   `POINTS_IR`, `POINTS_VIS`: **非常重要**，這些點用於圖像對齊，需要根據你的相機實際擺放位置進行標定（目前是硬編碼的，可能需要修改 `image_processing/alignment.py` 以支援手動標定 UI 或從檔案讀取）。
*   `YOLO_CONF_THRESHOLD`, `UNET_CONF_THRESHOLD`: 根據需要調整偵測和分割的置信度閾值。
*   `UNET_INPUT_SIZE`: **必須** 與你訓練 UNet 模型時使用的輸入尺寸一致。
*   其他顯示、分析相關參數。

## 使用方法 (Usage)

1.  確保所有設定和依賴都已完成。
2.  啟動虛擬環境（如果使用了）。
3.  在專案根目錄下，執行主程式：
    ```bash
    python main_app.py
    ```
    *   在某些系統上，如果訪問相機需要權限，可能需要使用 `sudo python main_app.py`。
4.  程式會開啟多個 OpenCV 視窗顯示結果：
    *   `Camera`: 對齊後的可見光影像，包含 YOLO 偵測框和計算出的呼吸率。
    *   `Thermal Camera`: 8 位元的熱影像，包含偵測框和計算出的最高溫度（攝氏度）。
    *   `Head Overlay`: 從可見光影像裁剪出的頭部 ROI，並疊加了 UNet 分割的半透明 Mask。
    *   `Head Segmented`: 從可見光影像裁剪出的頭部 ROI，僅顯示 UNet 分割出的前景區域（背景變黑）。
5.  按下鍵盤上的 `q` 鍵退出程式。
6.  按下鍵盤上的 `t` 鍵可以暫停程式 100 秒（用於測試）。

## 注意事項 (Notes)

*   影像對齊的準確性高度依賴於 `config.py` 中 `POINTS_IR` 和 `POINTS_VIS` 的精度。如果對齊效果不佳，需要重新標定這些點。
*   呼吸率的計算基於溫度變化的 FFT 分析，其穩定性和準確性受多種因素影響（如偵測框穩定性、實際呼吸模式、環境干擾等）。
*   目前 UNet 的分割結果主要用於視覺化展示，並未直接用於優化溫度計算區域。可以考慮後續開發，利用 Mask 獲取更精確的溫度測量區域。
*   同時運行 YOLO 和 UNet 對計算資源要求較高，尤其是在嵌入式設備上，可能會導致 FPS 下降。
```