# TensorRT 模型轉換與使用指南

此專案現在支援直接使用 TensorRT Engine 進行推論以達到最高效能。使用步驟如下：

## 1. 執行 Docker 容器
請確保您已進入具有完整依賴環境的 yolo 容器內部：
```bash
sudo docker exec -it yolo bash
```

## 2. 轉換 YOLO 模型
執行以下指令，它會自動讀取 `yolo11n_headmask.pt` 並轉換輸出為 `yolo11n_headmask.engine` (FP16)：
```bash
python3 export_yolo_trt.py
```

## 3. 轉換 UNet 模型
執行以下指令，它會先將 PyTorch 模型打包成 ONNX 格式 (`unet_msfd.onnx`)，接著呼叫 `trtexec` 將其轉換為 TensorRT 格式 (`unet_msfd.engine`)，並且啟用 FP16 最佳化：
```bash
python3 export_unet_trt.py
```

## 4. 更改 Config 設定檔
轉換完成後，您只需編輯 `config.py`，將模型的副檔名改成 `.engine` 即可：
```python
YOLO_MODEL_PATH = "yolo11n_headmask.engine" # 原為 .pt
UNET_MODEL_PATH = "unet_msfd.engine"        # 原為 .pth
```

> **注意**：新的 UNet 推論引擎會自動偵測副檔名是否為 `.engine` 來選擇執行 PyTorch 或是 TensorRT 推論。為確保匯出穩定性，PyTorch 模式下已移除了 `model.half()`，FP16 推論將完全由 TensorRT 接管。
