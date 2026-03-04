# analysis/respiration.py

import numpy as np
import config # 引入 config 來獲取參數

from collections import deque

def update_temperature_queue(new_temp, data_list, max_size: int):
    """ Adds a new temperature value to a deque, maintaining max size. """
    if new_temp is None: # Don't add None values
        return data_list

    # Ensure it's a deque
    if not isinstance(data_list, deque):
        data_list = deque(data_list, maxlen=max_size)
    
    # Add the new temperature
    data_list.append(new_temp)
    return data_list

def detrend(signal):
    p = np.polyfit(range(len(signal)), signal, 1)
    trend = np.polyval(p, range(len(signal)))
    signal_detrended = signal - trend
    return signal_detrended

def calculate_respiration_fft(temp_list, timestamp_list, fps=None, min_bpm=config.RESP_MIN_BPM, max_bpm=config.RESP_MAX_BPM):
    """
    Calculates breathing rate in BPM using FFT, searching within a specified BPM range.
    Uses timestamps for accurate re-sampling to handle non-uniform framerates.
    
    Args:
        temp_list (list): List of temperature values.
        timestamp_list (list): List of corresponding time stamps.
        fps (float): (Optional) No longer strictly required if timestamps are used.
        min_bpm (float): Minimum breathing rate to search for (beats per minute).
        max_bpm (float): Maximum breathing rate to search for (beats per minute).

    Returns:
        float or None: Estimated breathing rate in BPM, or None if calculation fails.
    """
    if not temp_list or len(temp_list) < 2 or not timestamp_list or len(timestamp_list) != len(temp_list):
        return None

    # Use Scipy to resample the signal perfectly evenly
    from scipy.interpolate import interp1d
    
    temp_array = np.array(temp_list)
    time_array = np.array(timestamp_list)
    
    # Calculate effective FPS over the window to determine uniform sampling grid
    total_time = time_array[-1] - time_array[0]
    if total_time <= 0:
         return None
    
    num_points = len(temp_array)
    uniform_fps = num_points / total_time
    
    # Create uniform time grid
    uniform_time = np.linspace(time_array[0], time_array[-1], num_points)
    
    # Interpolate temperature onto uniform time grid
    interpolator = interp1d(time_array, temp_array, kind='cubic')
    resampled_temp_array = interpolator(uniform_time)

    detrend_temp_array = detrend(resampled_temp_array)

    N = len(detrend_temp_array)

    hamming_window = np.hamming(N)
    windowed_temp_array = detrend_temp_array * hamming_window

    sampling_rate = float(uniform_fps)

    try:
        # --- FFT Calculation ---
        freqs = np.fft.fftfreq(N, d=1.0 / sampling_rate)
        fft_values = np.fft.fft(windowed_temp_array)
        fft_magnitude = np.abs(fft_values)

        # --- Frequency Filtering ---
        # Convert BPM range to Hz range
        min_hz = min_bpm / 60.0
        max_hz = max_bpm / 60.0

        # Consider only positive frequencies (excluding DC at index 0)
        half_N = N // 2
        positive_freqs = freqs[:half_N]
        positive_magnitude = fft_magnitude[:half_N]

        if len(positive_magnitude) == 0:
            # print("Warning: No positive frequency components found.")
            return None

        # --- Find indices within the desired frequency (Hz) range ---
        # np.where returns a tuple, we need the first element (the array of indices)
        valid_indices_in_positive = np.where((positive_freqs >= min_hz) & (positive_freqs <= max_hz))[0]

        # Check if any frequencies fall within the desired range
        if len(valid_indices_in_positive) == 0:
            # print(f"Warning: No frequency peaks found within the range {min_bpm}-{max_bpm} BPM.")
            return None # Or maybe return the overall max if you prefer fallback?

        # --- Find the peak magnitude *within the valid range* ---
        # Get the magnitudes corresponding to the valid indices
        magnitudes_in_range = positive_magnitude[valid_indices_in_positive]

        # Find the index of the maximum magnitude *within this subset*
        peak_idx_within_subset = np.argmax(magnitudes_in_range)

        # Map this subset index back to the index in the 'positive_freqs' array
        peak_idx_in_positive = valid_indices_in_positive[peak_idx_within_subset]

        # --- Get the corresponding frequency and convert to BPM ---
        # Remember that positive_freqs started from index 1 of the original freqs
        # So, the index in the full freqs array is peak_idx_in_positive + 1
        peak_freq_hz = freqs[peak_idx_in_positive] # Or positive_freqs[peak_idx_in_positive]

        # Double check the selected frequency is indeed within bounds (due to potential floating point issues)
        # assert min_hz <= peak_freq_hz <= max_hz , f"Peak {peak_freq_hz}Hz out of bounds [{min_hz}, {max_hz}]"

        breathing_rate_bpm = peak_freq_hz * 60.0

        return breathing_rate_bpm

    except Exception as e:
        print(f"Error during FFT calculation: {e}")
        return None