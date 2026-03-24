# 開發日誌 (Development Log)

本文件用於記錄 `Respiration and Temperature Monitor Application` 的演進歷程、重大技術決策 (ADR) 以及效能優化細節。

---

## 📅 開發日誌：[2026-03-24]
**摘要**：導入自動化基準測試框架、實作 jtop 硬體監控，並將推論管線全面遷移至 TensorRT V3 API 以極大化 Orin NX 效能。

### ✨ 新增功能 (Features)
- **自動化基準測試框架 (`MockCamera`)**：支援從預錄的 16-bit `.npy` 溫度檔案讀取數據，確保效能測試在 100% 可復現的環境下進行，不受物理感測器變動干擾。
- **微秒級 Profiler 框架**：實作單例模式的 `Profiler` 與 `TimeIt` 上下文管理器，精準追蹤 YOLO、U-Net、FFT 等核心子模組的推論延遲。
- **背景硬體偵測器 (`HardwareMonitor`)**：基於 `jtop` 實作非同步監測，即時側錄 CPU/GPU 使用率、功耗 (mW) 與運行溫度，適配 Jetpack 6 API。

### 🛠️ 重構與優化 (Refactor & Optimization)
- **TensorRT V3 API 遷移**：汰除舊版 `get_binding_index`，改用 V3 記憶體指標管理，透過 PyTorch `data_ptr()` 達成零拷貝 (Zero-copy) 數據傳輸。
- **FP16 量化推論**：將 YOLO11 與 U-Net 轉換為 TensorRT FP16 引擎，推論延遲分別壓低至 27.7ms 與 25.5ms，滿足實時監測需求。
- **UI 渲染降頻 (Decimation)**：發現 OpenCV 繪圖在高品質模式下佔用 20% 算力，新增 `SHOW_ANALYSIS_UI` 開關，在測試模式下可關閉繪圖以釋放 CPU/GPU 頻寬。

### 🐛 問題修復 (Bug Fixes)
- **熱像儀 redundant frame 雜訊修復**：修復了因熱像儀輸出頻率 (9Hz) 低於推論頻率 (17Hz) 導致的重複幀讀取。透過 `therm_time` 嚴格比對，消除了 FFT 頻譜中的「階梯狀失真」與高頻偽影。

---

## 📅 開發日誌：[2026-03-20]
**摘要**：優化 U-Net 遮罩處理流程與溫度讀取演算法，強化邊緣端運算效能。

### 🛠️ 重構與優化 (Refactor & Optimization)
- **遮罩腐蝕 (Erosion) 效能重構**：優化了遮罩邊緣處理由 OpenCV 運算的邏輯，降低了大尺寸 Mask 處理時的 CPU 負載。
- **更新 U-Net 卷積核與座標映射**：調整模型參數以提升皮膚區域識別率，並精進了熱影像與可見光影像的透視變換座標映射精準度。

---

## 📅 開發日誌：[2026-03-18]
**摘要**：提升呼吸訊號分析的頻率解析度，並精進分割模型遮罩的邊界品質。

### ✨ 新增功能 (Features)
- **FFT 自動零補償 (Zero-padding)**：實作了 1024 點基礎的零補償機制（`TARGET_FFT_LEN`），即便在短取樣視窗下也能提升頻域解析度，改善 BPM 估算品質。

### 🛠️ 重構與優化 (Refactor & Optimization)
- **遮罩邊界強化**：在 `segmentation` 階段強制對 U-Net Mask 進行侵蝕處理，確保溫度提取僅發生在「純皮膚區域」，徹底隔離背景熱源干擾。

---

## 📅 開發日誌：[2026-03-17]
**摘要**：現代化訊號處理管線，並針對 Jetson 平台進行預運算快取優化。

### 🛠️ 重構與優化 (Refactor & Optimization)
- **訊號鏈重構 (Pipeline Modernization)**：重新定義處理順序為「去趨勢 (Detrend) -> SOS 帶通濾波 -> FFT」，並引入 `scipy.signal.sosfilt` 提升濾波器穩定性。
- **預計算與快取優化**：預先計算多項式擬合係數，並快取 `raw_to_8bit` 轉換表。這減少了每幀轉換時的浮點運算次數，降低了 15% 的 pre-processing 耗時。

### 🐛 問題修復 (Bug Fixes)
- **代碼清理**：利用靜態分析工具移除死碼、修復重複聲明變數，並解決了部分函式在極端邊界條件下返回路徑不明確的問題。

---

## 📅 開發日誌：[2026-03-12]
**摘要**：實裝黑體基準校正 (Blackbody Calibration)，解決熱向儀溫漂問題。

### ✨ 新增功能 (Features)
- **動態溫度平移補償**：新增黑體校正區塊 (`BLACKBODY_ROI`)，自動映射 640x480 與 160x120 座標系，提供即時體溫補償數值。

---

## 📅 開發日誌：[2026-03-04]
**摘要**：訊號穩定度提升與 Python 記憶體管理優化。

### 🛠️ 重構與優化 (Refactor & Optimization)
- **時間序列結構重組**：從 `list` 遷移至 `collections.deque`，解決長時監視下的 $O(N)$ 搬移開銷。
- **時間抖動校正 (Jitter Correction)**：基於 `interp1d` 的線性插值重取樣，將非均勻取樣序列映射至均勻格點，消除頻譜洩漏。

---

## 📅 開發日誌：[2025-06-10]
**摘要**：V1.0.0 Milestone - 系統基礎架構與顯示模組標準化。

### 🛠️ 重構與優化 (Refactor & Optimization)
- **DisplayManager 封裝**：統一多重視窗管理與資源回收機制。
- **主程式標準化**：定義清晰的推論接口，為後續效能測試奠定基礎。
