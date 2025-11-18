import numpy as np
from clean_and_scale_scores import clean_and_scale_scores

def test_simple_1d_clipping_and_scaling():
    scores = np.array([10, 50, 110])
    min_score = 0
    max_score = 100
    expected = np.array([0.1, 0.5, 1.0], dtype=float)

    result = clean_and_scale_scores(scores, min_score, max_score)
    assert result.shape == scores.shape
    assert result.dtype == float
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

def test_2d_array_clipping_and_scaling():
    scores = np.array([
        [-10, 0],
        [50, 200]
    ])
    min_score = 0
    max_score = 100
    expected = np.array([
        [0.0, 0.0],
        [0.5, 1.0]
    ])

    result = clean_and_scale_scores(scores, min_score, max_score)
    assert result.shape == scores.shape
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

def test_scores_already_in_range():
    scores = np.array([20, 40, 60, 80])
    min_score = 0
    max_score = 100
    expected = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)

    result = clean_and_scale_scores(scores, min_score, max_score)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

def test_non_zero_min_score():
    scores = np.array([30, 50, 90])
    min_score = 20
    max_score = 100
    expected = np.array([
        (30 - 20) / 80,
        (50 - 20) / 80,
        (90 - 20) / 80
    ], dtype=float)

    result = clean_and_scale_scores(scores, min_score, max_score)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

