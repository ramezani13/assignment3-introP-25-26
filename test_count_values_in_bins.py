import numpy as np
from count_values_in_bins import count_values_in_bins

def test_basic_bins():
    data = np.array([0.5, 1.0, 1.1, 2.9, 3.0, 4.5])
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Bins:
    # [0,1): 0.5        -> 1
    # [1,2): 1.0, 1.1   -> 2
    # [2,3): 2.9        -> 1
    # [3,4): 3.0        -> 1
    # [4,5]: 4.5        -> 1
    expected = np.array([1, 2, 1, 1, 1])

    counts = count_values_in_bins(data, bin_edges)
    assert counts.shape == (len(bin_edges) - 1,)
    np.testing.assert_array_equal(counts, expected)

def test_values_outside_range_are_ignored():
    data = np.array([-1.0, 0.0, 0.9, 5.0, 6.0])
    bin_edges = np.array([0.0, 1.0, 2.0, 5.0])
    # bins:
    # [0,1): 0.0, 0.9   -> 2
    # [1,2): (none)     -> 0
    # [2,5]: 5.0        -> 1  (last bin inclusive)
    # -1.0 and 6.0 are ignored
    expected = np.array([2, 0, 1])

    counts = count_values_in_bins(data, bin_edges)
    np.testing.assert_array_equal(counts, expected)

def test_empty_data():
    data = np.array([], dtype=float)
    bin_edges = np.array([0.0, 1.0, 2.0])
    # no data -> all counts zero
    expected = np.array([0, 0])

    counts = count_values_in_bins(data, bin_edges)
    np.testing.assert_array_equal(counts, expected)

def test_single_bin():
    data = np.array([0.0, 0.5, 1.0, 1.5])
    bin_edges = np.array([0.0, 2.0])
    # single bin [0,2], inclusve at right =>all values counted
    expected = np.array([4])

    counts = count_values_in_bins(data, bin_edges)
    np.testing.assert_array_equal(counts, expected)
