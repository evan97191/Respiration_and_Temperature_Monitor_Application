# utils/timing.py

import time
from collections import deque
import numpy as np

class FPSTracker:
    """ Tracks time intervals and calculates average FPS. """
    def __init__(self, buffer_size=5):
        self.last_time = time.time()
        self.interval_list = deque(maxlen=buffer_size)

    def tick(self):
        """ Call this once per loop iteration to get the interval and update FPS. """
        current_time = time.time()
        interval = current_time - self.last_time
        self.last_time = current_time

        self.interval_list.append(interval)

        return interval

    def get_average_fps(self, default_fps=10.0):
        """ Calculates the average FPS based on recent intervals. """
        if not self.interval_list:
            return default_fps # Return default if no intervals recorded yet

        mean_interval = np.mean(self.interval_list)
        if mean_interval > 0:
            return round(1.0 / mean_interval, 2)
        else:
            return default_fps # Avoid division by zero

# Original generator function (can also be used)
# def time_interval_tracker():
#     last_time = time.time()
#     while True:
#         current_time = time.time()
#         interval = current_time - last_time
#         last_time = current_time
#         yield interval