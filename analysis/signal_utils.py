# analysis/signal_utils.py

import numpy as np

def moving_average_filter(data_list, window_size=3):
    """ Applies a simple moving average filter. """
    if not data_list or len(data_list) < window_size or window_size <= 0:
        # print("Warning: Cannot apply moving average, insufficient data or invalid window size.")
        return data_list # Return original list if filtering is not possible

    # Use convolution for efficient moving average
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(np.array(data_list), weights, mode='valid')

    # Handle edge effect: pad beginning with original values or first smoothed value
    padding_size = window_size - 1
    # Pad with the first few original values to maintain length
    padded_smoothed = np.concatenate((np.array(data_list)[:padding_size], smoothed))

    return padded_smoothed.tolist() # Return as list