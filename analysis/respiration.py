# analysis/respiration.py

import logging
from collections import deque

import numpy as np
import scipy.fftpack
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt

import config  # 引入 config 來獲取參數

logger = logging.getLogger(__name__)


def update_temperature_queue(new_temp, data_list, max_size: int):
    """Adds a new temperature value to a deque, maintaining max size."""
    if new_temp is None:  # Don't add None values
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


def butter_bandpass_sos(lowcut, highcut, fs, order=5):
    """Designs a Butterworth bandpass filter in SOS form (numerically stable)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Ensure 'low' and 'high' are strictly bounded between 0.0 and 1.0
    low = max(1e-5, min(low, 0.999))
    high = max(low + 1e-5, min(high, 0.999))
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter using sosfiltfilt (zero-phase, numerically stable)."""
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def calculate_respiration_fft(temp_list, timestamp_list, min_bpm=config.RESP_MIN_BPM, max_bpm=config.RESP_MAX_BPM):
    """
    Calculates breathing rate in BPM using FFT, searching within a specified BPM range.
    Uses timestamps for accurate re-sampling to handle non-uniform framerates.

    Args:
        temp_list (list): List of temperature values.
        timestamp_list (list): List of corresponding time stamps.
        min_bpm (float): Minimum breathing rate to search for (beats per minute).
        max_bpm (float): Maximum breathing rate to search for (beats per minute).

    Returns:
        float or None: Estimated breathing rate in BPM, or None if calculation fails.
    """
    if not temp_list or len(temp_list) < 2 or not timestamp_list or len(timestamp_list) != len(temp_list):
        return None, None

    temp_array = np.array(temp_list)
    time_array = np.array(timestamp_list)

    # Calculate effective FPS over the window to determine uniform sampling grid
    total_time = time_array[-1] - time_array[0]
    if total_time <= 0:
        return None, None

    num_points = len(temp_array)
    uniform_fps = num_points / total_time

    # Create uniform time grid
    uniform_time = np.linspace(time_array[0], time_array[-1], num_points)

    # Interpolate temperature onto uniform time grid
    interpolator = interp1d(time_array, temp_array, kind="linear")
    resampled_temp_array = interpolator(uniform_time)

    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0

    # --- Signal processing order: detrend FIRST, then filter ---
    # Detrending before filtering reduces edge effects from DC offset/trends
    detrend_temp_array = detrend(resampled_temp_array)

    min_filter_samples = getattr(config, "BANDPASS_FILTER_MIN_SAMPLES", 30)
    try:
        # Apply filter only if we have sufficient samples
        if len(detrend_temp_array) >= min_filter_samples:
            detrend_temp_array = butter_bandpass_filter(detrend_temp_array, min_hz, max_hz, uniform_fps, order=2)
    except Exception as e:
        logger.warning(f"Bandpass filter failed (likely too few samples), skipping filter: {e}")

    N = len(detrend_temp_array)

    hamming_window = np.hamming(N)
    windowed_temp_array = detrend_temp_array * hamming_window

    sampling_rate = float(uniform_fps)

    try:
        # --- FFT Calculation with Zero-Padding ---
        # Zero-padding improves spectral peak detection precision
        zero_pad_factor = getattr(config, "FFT_ZERO_PAD_FACTOR", 4)
        target_length = N * zero_pad_factor
        final_length = max(config.TARGET_FFT_LEN, target_length)
        n_fft = scipy.fftpack.next_fast_len(final_length)
        freqs = np.fft.fftfreq(n_fft, d=1.0 / sampling_rate)
        fft_values = np.fft.fft(windowed_temp_array, n=n_fft)
        fft_magnitude = np.abs(fft_values)

        # --- Frequency Filtering ---
        # Convert BPM range to Hz range
        min_hz = min_bpm / 60.0
        max_hz = max_bpm / 60.0

        # Consider only positive frequencies (excluding DC at index 0)
        half_N = n_fft // 2
        positive_freqs = freqs[:half_N]
        positive_magnitude = fft_magnitude[:half_N]

        debug_data = {
            "uniform_time": uniform_time,
            "resampled_temp": detrend_temp_array,
            "freqs": freqs,
            "fft_magnitude": fft_magnitude,
            "positive_freqs": positive_freqs,
            "positive_magnitude": positive_magnitude,
            "min_hz": min_hz,
            "max_hz": max_hz,
        }

        if len(positive_magnitude) == 0:
            return None, debug_data

        # --- Find indices within the desired frequency (Hz) range ---
        valid_indices_in_positive = np.where((positive_freqs >= min_hz) & (positive_freqs <= max_hz))[0]

        # Check if any frequencies fall within the desired range
        if len(valid_indices_in_positive) == 0:
            return None, debug_data

        # --- Find the peak magnitude *within the valid range* ---
        magnitudes_in_range = positive_magnitude[valid_indices_in_positive]
        peak_idx_within_subset = np.argmax(magnitudes_in_range)
        peak_idx_in_positive = valid_indices_in_positive[peak_idx_within_subset]

        peak_freq_hz = positive_freqs[peak_idx_in_positive]
        breathing_rate_bpm = peak_freq_hz * 60.0

        return breathing_rate_bpm, debug_data

    except Exception as e:
        logger.error(f"Error during FFT calculation: {e}")
        return None, None


def calculate_fft_raw(temp_list, fps, min_bpm=config.RESP_MIN_BPM, max_bpm=config.RESP_MAX_BPM):
    """
    Calculates FFT directly on the pure temperature array without timestamp-based cubic resampling.
    """
    if not temp_list or len(temp_list) < 2 or fps <= 0:
        return None, None

    temp_array = np.array(temp_list)
    detrend_temp_array = detrend(temp_array)

    N = len(detrend_temp_array)
    hamming_window = np.hamming(N)
    windowed_temp_array = detrend_temp_array * hamming_window

    try:
        freqs = np.fft.fftfreq(N, d=1.0 / fps)
        fft_values = np.fft.fft(windowed_temp_array)
        fft_magnitude = np.abs(fft_values)

        min_hz = min_bpm / 60.0
        max_hz = max_bpm / 60.0

        half_N = N // 2
        positive_freqs = freqs[:half_N]
        positive_magnitude = fft_magnitude[:half_N]

        debug_data = {
            "raw_temp": temp_array,
            "freqs": freqs,
            "fft_magnitude": fft_magnitude,
            "positive_freqs": positive_freqs,
            "positive_magnitude": positive_magnitude,
            "min_hz": min_hz,
            "max_hz": max_hz,
        }

        if len(positive_magnitude) == 0:
            return None, debug_data

        valid_indices = np.where((positive_freqs >= min_hz) & (positive_freqs <= max_hz))[0]
        if len(valid_indices) == 0:
            return None, debug_data

        magnitudes_in_range = positive_magnitude[valid_indices]
        peak_idx_within = np.argmax(magnitudes_in_range)
        peak_idx_in_positive = valid_indices[peak_idx_within]

        peak_freq_hz = positive_freqs[peak_idx_in_positive]
        breathing_rate_bpm = peak_freq_hz * 60.0

        return breathing_rate_bpm, debug_data

    except Exception as e:
        logger.error(f"Error during raw FFT calculation: {e}")
        return None, None
