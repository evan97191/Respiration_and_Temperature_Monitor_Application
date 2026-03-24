# 開發日誌 (Development Log)

本文件用於記錄專案的演進歷程、重大技術決策 (ADR, Architecture Decision Records) 以及效能優化紀錄，避免 README.md 過度臃腫，同時協助團隊與未來的自己回顧開發脈絡。

---

## [2026-03-24] - 里程碑：自動化基準測試與 UI 降頻渲染

### ✨ 新增功能 (Added)
- **導入 `MockCamera` 測試框架**：現在支援讀取預錄影片檔 (`.mp4` 與 16-bit `.npy`) 取代實體串流相機，實現 100% 絕對可復現的效能測試環境 (`IS_TESTING = True`)。
- **微秒級效能探針 (`Profiler`)**：完整實作單例模式與 `TimeIt` Context Manager，精準分析 YOLO、U-Net、FFT 及 UI 渲染的耗時並自動匯出 `benchmark_result.json`。
- **背景硬體監控 (`HardwareMonitor`)**：基於 `jtop` (jetson-stats) 實作的守護執行緒，動態側錄 Orin NX 的 CPU、GPU 使用率、記憶體消耗與系統總功耗 (W)。

### ⚡ 效能優化 (Performance)
- **UI 降頻渲染 (UI Decimation)**：利用 Profiler 偵測出最大的非 AI 效能瓶頸為 OpenCV 即時圖表繪製 (`Analysis_Graphs_Update`，高達 16.4ms)。透過實作 `GRAPH_UPDATE_INTERVAL = 3` (每 3 幀刷新一次 UI)，成功將整體推論幀率大幅提升。
- **Edge GPU 睡眠喚醒分析**：釐清了 TensorRT 推論過快 (約 50ms) 導致 Jetson Orin 電源管理強制 GPU 進入 Deep Sleep，從而在 `jtop` 取樣時發生 `GPU: 0%` 的效能幻覺 (Performance Illusion) 現象。

### 🧠 技術決策與筆記 (Decisions & Notes)
- **為何熱像儀的預錄測資必須存成 `.npy`？**
  Lepton 3.5 輸出的是包含「絕對溫度」的 16-bit 原始輻射矩陣 (Raw Radiometric)。如果像可見光一樣壓縮成 8-bit 的 `.mp4` 影片，會導致溫度資訊被破壞性壓縮，`ktoc()` 解析出的溫度將完全失真。

---

## [撰寫模板] - 未來新增內容請複製此區塊並貼到最上方

## [YYYY-MM-DD] - 里程碑標題 (例如：替換為 YOLOv11)

### ✨ 新增 (Added)
- 簡述加入了什麼新功能...

### 🐛 修復 (Fixed)
- 解決了什麼導致 Crash 或邏輯錯誤的 bug...

### ⚡ 效能優化 (Performance)
- 紀錄優化前後的 FPS、延遲或是記憶體消耗差異...

### 🧠 筆記與決策 (Notes)
- 為什麼選擇某個套件？為什麼這樣設計架構？踩到了什麼硬體坑...
