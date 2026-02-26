# 1. 以 Ultralytics 的 JetPack 6 版本為基底
FROM ultralytics/ultralytics:latest-jetson-jetpack6

# 🌟 解決卡在選擇 Area 的關鍵：設定為非互動模式，並預設好時區
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 2. 更新 apt 並安裝所需的系統依賴與編譯工具
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    libjpeg-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 3. 下載 groupgets 版本的 libuvc 並進行編譯安裝
RUN git clone https://github.com/groupgets/libuvc /tmp/libuvc && \
    cd /tmp/libuvc && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    make install && \
    ldconfig && \
    rm -rf /tmp/libuvc

# 4. 補齊系統依賴、X11/XCB 元件，以及「原生的 python3-opencv」
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# 5. 強制解除安裝 pip 版本的 OpenCV，確保系統讀取到帶有 GStreamer 的原生版本
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python