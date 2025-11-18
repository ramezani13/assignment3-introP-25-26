import numpy as np
from moving_average import moving_average

def test_window_size_one_returns_same_signal():
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    window_size = 1
    result = moving_average(signal, window_size)
    np.testing.assert_allclose(result, signal, rtol=1e-7, atol=1e-7)

def test_moving_average_simple_example():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window_size = 3
    expected = np.array([1.5, 2.0, 3.0, 4.0, 4.5])
    result = moving_average(signal, window_size)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

def test_constant_signal():
    signal = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    window_size = 5
    expected = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    result = moving_average(signal, window_size)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

def test_short_signal_large_window():
    signal = np.array([2.0, 4.0])
    window_size = 3
    expected = np.array([3.0, 3.0])
    result = moving_average(signal, window_size)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

