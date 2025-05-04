# analysis/respiration.py

import numpy as np

def update_temperature_queue(new_temp, data_list: list, max_size: int) -> list:
    """ Adds a new temperature value to a list, maintaining max size. """
    if new_temp is None: # Don't add None values
        return data_list

    # If list is full, remove the oldest element
    if len(data_list) >= max_size:
        data_list.pop(0)
    # Add the new temperature
    data_list.append(new_temp)
    return data_list

def calculate_respiration_fft(temp_list, fps):
    """ Calculates breathing rate in BPM using FFT. """
    if not temp_list or len(temp_list) < 2: # Need at least 2 points for FFT
        # print("Warning: Not enough temperature data for FFT.")
        return None
    if fps <= 0:
         print(f"Warning: Invalid FPS ({fps}) for FFT calculation.")
         return None

    # Ensure data is a NumPy array for FFT
    temp_array = np.array(temp_list)
    # Optional: Detrend data (remove linear trend)
    # temp_detrended = temp_array - np.polyval(np.polyfit(np.arange(len(temp_array)), temp_array, 1), np.arange(len(temp_array)))
    # temp_array = temp_detrended # Use detrended data

    N = len(temp_array)
    sampling_rate = float(fps)

    try:
        # Calculate FFT
        freqs = np.fft.fftfreq(N, d=1.0 / sampling_rate)
        fft_values = np.fft.fft(temp_array)
        fft_magnitude = np.abs(fft_values)

        # Consider only positive frequencies (excluding DC)
        half_N = N // 2
        positive_freqs = freqs[1:half_N]
        positive_magnitude = fft_magnitude[1:half_N]

        if len(positive_magnitude) == 0:
            # print("Warning: No positive frequency components found.")
            return None

        # Find the frequency with the maximum magnitude in the positive range
        # Optional: Limit frequency range (e.g., 0.1 Hz to 1.0 Hz for breathing)
        # valid_indices = np.where((positive_freqs >= 0.1) & (positive_freqs <= 1.0))[0]
        # if len(valid_indices) == 0: return None
        # peak_idx_in_positive = np.argmax(positive_magnitude[valid_indices])
        # peak_freq_index = valid_indices[peak_idx_in_positive] + 1 # +1 because we skipped DC

        peak_idx_in_positive = np.argmax(positive_magnitude)
        # Need the index relative to the original freqs array (add 1 to skip DC)
        peak_freq_index = peak_idx_in_positive + 1

        breathing_rate_hz = freqs[peak_freq_index]
        breathing_rate_bpm = breathing_rate_hz * 60.0

        return breathing_rate_bpm

    except Exception as e:
        print(f"Error during FFT calculation: {e}")
        return None