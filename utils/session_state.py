from collections import deque

import numpy as np

import config
from analysis.respiration import update_temperature_queue


class SessionContext:
    def __init__(self):
        self.max_size = getattr(config, "TEMPERATURE_QUEUE_MAX_SIZE", 300)
        self.temp_data_list = deque(maxlen=self.max_size)
        self.timestamp_list = deque(maxlen=self.max_size)
        self.temp_data_list_no_unet = deque(maxlen=self.max_size)
        self.max_temp_list = deque(maxlen=self.max_size)
        self.breathing_rate_bpm_list = []

        self.latest_max_temp = None
        self.latest_avg_temp = None
        self.latest_bpm = None

    def update(self, avg_temp, max_temp, timestamp, bpm, avg_temp_no_unet=None):
        """
        Updates session statistics.
        """
        if max_temp is not None:
            self.latest_max_temp = max_temp
            self.max_temp_list = update_temperature_queue(max_temp, self.max_temp_list, self.max_size)

        if avg_temp_no_unet is not None:
            self.temp_data_list_no_unet = update_temperature_queue(
                avg_temp_no_unet, self.temp_data_list_no_unet, self.max_size
            )

        if avg_temp is not None:
            self.latest_avg_temp = avg_temp
            self.temp_data_list = update_temperature_queue(avg_temp, self.temp_data_list, self.max_size)
            self.timestamp_list = update_temperature_queue(timestamp, self.timestamp_list, self.max_size)

        if bpm is not None:
            self.latest_bpm = bpm
            self.breathing_rate_bpm_list.append(bpm)

    def get_summary(self):
        """
        Returns average temperature and average respiration rate of the session
        """
        avg_temp = np.mean(self.max_temp_list) if self.max_temp_list else 0.0

        valid_bpms = [r for r in self.breathing_rate_bpm_list if r is not None]
        avg_resp = np.mean(valid_bpms) if valid_bpms else 0.0

        return float(avg_temp), float(avg_resp)

    def get_respiration_data(self):
        valid_temps = [t for t in self.temp_data_list if t is not None]
        valid_timestamps = list(self.timestamp_list)[: len(valid_temps)]
        return valid_temps, valid_timestamps
