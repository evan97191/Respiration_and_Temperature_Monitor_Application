import os
import sys
from collections import deque

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.respiration import detrend, update_temperature_queue


def test_update_temperature_queue():
    q = deque(maxlen=5)
    update_temperature_queue(36.5, q, 5)
    assert len(q) == 1
    assert q[0] == 36.5

    # Test none value
    update_temperature_queue(None, q, 5)
    assert len(q) == 1


def test_detrend():
    # Linear trend: 1, 2, 3, 4, 5
    signal = np.array([1, 2, 3, 4, 5])
    detrended = detrend(signal)
    # After detrending a perfect line, it should be (nearly) zero
    assert np.allclose(detrended, np.zeros(5), atol=1e-10)


# More complex tests for FFT would go here, possibly using pre-recorded signals
