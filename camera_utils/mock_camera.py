import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MockCamera:
    """模擬攝影機，讀取磁碟上的影片檔或影像序列進行效能測試"""
    def __init__(self, media_path, target_fps=None, loop=True):
        self.media_path = media_path
        self.is_npy = media_path.endswith('.npy')

        if self.is_npy:
            self.frames = np.load(media_path)
            self.total_frames = len(self.frames)
            self.frame_idx = 0
            if self.total_frames == 0:
                logger.warning(f"[MockCamera] Warning: .npy file {media_path} is empty.")
        else:
            self.cap = cv2.VideoCapture(media_path)
            if not self.cap.isOpened():
                logger.warning(f"[MockCamera] Warning: Failed to open video file {media_path}")

        self.target_fps = target_fps
        self.loop = loop
        self.last_frame_time = time.time()
        self.is_streaming = True # Mock as always streaming

    def get_frame(self):
        ret = False
        frame = None

        if self.is_npy:
            if self.frame_idx >= self.total_frames:
                if self.loop:
                    self.frame_idx = 0
            if self.frame_idx < self.total_frames:
                frame = self.frames[self.frame_idx].copy()
                ret = True
                self.frame_idx += 1
        else:
            ret, frame = self.cap.read()
            if not ret and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

        # 模擬實際鏡頭的硬體延遲 (FPS 控制)
        if self.target_fps and ret:
            elapsed = time.time() - self.last_frame_time
            sleep_time = (1.0 / self.target_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.last_frame_time = time.time()
        return (ret, frame)

    def get_default_fps(self):
        return self.target_fps if self.target_fps else 21

    def release(self):
        if not self.is_npy:
            self.cap.release()
        self.is_streaming = False
