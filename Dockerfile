# Stage 1: Builder
FROM ultralytics/ultralytics:latest-jetson-jetpack6 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Build libuvc
RUN git clone https://github.com/groupgets/libuvc /tmp/libuvc && \
    cd /tmp/libuvc && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Stage 2: Final Image
FROM ultralytics/ultralytics:latest-jetson-jetpack6

LABEL maintainer="Evan"
LABEL description="Respiration and Temperature Monitor Application"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    python3-opencv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled libuvc from builder
COPY --from=builder /usr/local/lib/libuvc* /usr/local/lib/
COPY --from=builder /usr/local/include/libuvc /usr/local/include/libuvc
RUN ldconfig

# Ensure system OpenCV is used (remove pip versions)
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python

# Create a non-root user for security
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # Add user to video and dialout groups for hardware access
    usermod -aG video,dialout $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Copy application code (will be handled by volumes in docker-compose for dev, but good for production)
# COPY . /workspace

# CMD ["python3", "main_app.py"]