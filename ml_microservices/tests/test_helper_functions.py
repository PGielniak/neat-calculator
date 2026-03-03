"""
Unit tests for shared/helper_functions.py

Tests cover:
  - correlation
  - energy
  - entropy
  - sma
  - mean_freq
  - lowpass_filter
  - angle_between
  - extract_features
"""
import math
import numpy as np
import pytest

from shared.helper_functions import (
    correlation,
    energy,
    entropy,
    sma,
    mean_freq,
    lowpass_filter,
    angle_between,
    extract_features,
    DT,
    FS,
    GRAVITY,
)


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_perfect_positive_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert correlation(x, x) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x
        assert correlation(x, y) == pytest.approx(-1.0)

    def test_uncorrelated_signals(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        # y has std == 0, should return 0
        assert correlation(x, y) == 0

    def test_zero_std_x_returns_zero(self):
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0])
        assert correlation(x, y) == 0

    def test_both_zero_std_returns_zero(self):
        x = np.array([2.0, 2.0, 2.0])
        y = np.array([3.0, 3.0, 3.0])
        assert correlation(x, y) == 0

    def test_result_in_range(self):
        rng = np.random.default_rng(0)
        x = rng.random(100)
        y = rng.random(100)
        r = correlation(x, y)
        assert -1.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_empty_signal_returns_zero(self):
        assert energy(np.array([])) == 0.0

    def test_unit_signal(self):
        sig = np.ones(10)
        assert energy(sig) == pytest.approx(1.0)

    def test_known_value(self):
        sig = np.array([1.0, 2.0, 3.0])
        # sum([1,4,9]) / 3 = 14/3
        assert energy(sig) == pytest.approx(14.0 / 3.0)

    def test_all_zeros(self):
        assert energy(np.zeros(5)) == pytest.approx(0.0)

    def test_normalisation(self):
        sig = np.array([2.0, 2.0])
        assert energy(sig) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_empty_signal_returns_zero(self):
        assert entropy(np.array([])) == 0.0

    def test_constant_signal_returns_zero(self):
        assert entropy(np.ones(20)) == 0.0

    def test_very_small_range_returns_zero(self):
        sig = np.full(10, 1.0) + np.random.default_rng(0).random(10) * 1e-15
        assert entropy(sig) == 0.0

    def test_uniform_signal_positive_entropy(self):
        rng = np.random.default_rng(42)
        sig = rng.uniform(0, 1, 1000)
        ent = entropy(sig, bins=10)
        assert ent > 0.0

    def test_entropy_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            sig = rng.standard_normal(200)
            assert entropy(sig) >= 0.0

    def test_entropy_clipped_to_max_ten(self):
        rng = np.random.default_rng(7)
        sig = rng.standard_normal(500)
        assert entropy(sig) <= 10.0

    def test_custom_bins(self):
        rng = np.random.default_rng(1)
        sig = rng.uniform(0, 1, 500)
        ent5 = entropy(sig, bins=5)
        assert ent5 >= 0.0

    def test_bins_larger_than_unique_values(self):
        # Triggers the "Too many bins" fallback path
        sig = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        result = entropy(sig, bins=1000)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# sma
# ---------------------------------------------------------------------------

class TestSma:
    def test_known_values(self):
        x = np.array([1.0, -1.0])
        y = np.array([2.0, -2.0])
        z = np.array([3.0, -3.0])
        # sum(|x| + |y| + |z|) / len(x) = (1+2+3 + 1+2+3) / 2 = 12/2 = 6
        assert sma(x, y, z) == pytest.approx(6.0)

    def test_all_zeros(self):
        x = y = z = np.zeros(5)
        assert sma(x, y, z) == pytest.approx(0.0)

    def test_single_element(self):
        assert sma([3.0], [4.0], [5.0]) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# mean_freq
# ---------------------------------------------------------------------------

class TestMeanFreq:
    def test_zero_spectrum_returns_zero(self):
        spec = np.zeros(10)
        freqs = np.linspace(0, 25, 10)
        assert mean_freq(spec, freqs) == 0.0

    def test_impulse_at_known_freq(self):
        freqs = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        spec = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        assert mean_freq(spec, freqs) == pytest.approx(10.0)

    def test_uniform_spectrum(self):
        freqs = np.array([0.0, 10.0, 20.0, 30.0])
        spec = np.ones(4)
        expected = np.sum(freqs * spec) / np.sum(spec)
        assert mean_freq(spec, freqs) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# lowpass_filter
# ---------------------------------------------------------------------------

class TestLowpassFilter:
    def test_output_same_length_as_input(self):
        sig = np.sin(2 * np.pi * 1.0 * np.arange(256) / FS)
        out = lowpass_filter(sig)
        assert len(out) == len(sig)

    def test_dc_component_unchanged(self):
        # A DC signal (constant) should pass a low-pass filter unchanged
        sig = np.ones(256)
        out = lowpass_filter(sig)
        assert np.allclose(out, sig, atol=1e-6)

    def test_high_freq_attenuated(self):
        # High-frequency component (24 Hz) should be strongly attenuated
        t = np.arange(512) / FS
        high_freq = np.sin(2 * np.pi * 24.0 * t)
        out = lowpass_filter(high_freq, cutoff=0.3, fs=FS)
        assert np.std(out) < np.std(high_freq) * 0.1

    def test_custom_parameters(self):
        sig = np.random.default_rng(0).standard_normal(256)
        out = lowpass_filter(sig, cutoff=1.0, fs=FS, order=2)
        assert len(out) == len(sig)


# ---------------------------------------------------------------------------
# angle_between
# ---------------------------------------------------------------------------

class TestAngleBetween:
    def test_zero_vector_first_returns_zero(self):
        assert angle_between([0, 0, 0], [1, 0, 0]) == 0.0

    def test_zero_vector_second_returns_zero(self):
        assert angle_between([1, 0, 0], [0, 0, 0]) == 0.0

    def test_parallel_vectors_zero_angle(self):
        v = [1.0, 2.0, 3.0]
        assert angle_between(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_anti_parallel_vectors_pi_angle(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert angle_between(v1, v2) == pytest.approx(math.pi, abs=1e-9)

    def test_perpendicular_vectors_pi_over_2(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert angle_between(v1, v2) == pytest.approx(math.pi / 2, abs=1e-9)

    def test_result_within_range(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            v1 = rng.standard_normal(3)
            v2 = rng.standard_normal(3)
            a = angle_between(v1, v2)
            assert 0.0 <= a <= math.pi


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    @pytest.fixture
    def synthetic_window(self):
        """128-sample window of synthetic sensor data (accX..Z, gyroX..Z)."""
        rng = np.random.default_rng(42)
        window = np.zeros((128, 6))
        # Accelerometer in m/s² (~1g on Z axis)
        window[:, 0] = rng.normal(0.1, 0.5, 128)
        window[:, 1] = rng.normal(0.1, 0.5, 128)
        window[:, 2] = rng.normal(GRAVITY, 0.5, 128)
        # Gyroscope in rad/s
        window[:, 3] = rng.normal(0.0, 0.1, 128)
        window[:, 4] = rng.normal(0.0, 0.1, 128)
        window[:, 5] = rng.normal(0.0, 0.1, 128)
        return window

    def test_returns_dict(self, synthetic_window):
        features = extract_features(synthetic_window)
        assert isinstance(features, dict)

    def test_contains_expected_keys(self, synthetic_window):
        features = extract_features(synthetic_window)
        required_keys = [
            "tBodyAcc-mean()-X",
            "tBodyAcc-std()-X",
            "tBodyAcc-energy()-X",
            "tBodyAcc-entropy()-X",
            "tGravityAcc-mean()-X",
            "tBodyAccJerk-mean()-X",
            "tBodyGyro-mean()-X",
            "tBodyAccMag-mean()",
            "fBodyAcc-mean()-X",
            "fBodyAcc-meanFreq()-X",
            "angle(tBodyAccMean,gravityMean)",
            "angle(X,gravityMean)",
        ]
        for key in required_keys:
            assert key in features, f"Missing key: {key}"

    def test_all_values_are_finite(self, synthetic_window):
        features = extract_features(synthetic_window)
        for k, v in features.items():
            assert math.isfinite(float(v)), f"Non-finite value for key: {k}"

    def test_non_negative_energy(self, synthetic_window):
        features = extract_features(synthetic_window)
        energy_keys = [k for k in features if "energy" in k]
        for k in energy_keys:
            assert features[k] >= 0.0, f"Negative energy for {k}"

    def test_non_negative_entropy(self, synthetic_window):
        features = extract_features(synthetic_window)
        entropy_keys = [k for k in features if "entropy" in k]
        for k in entropy_keys:
            assert features[k] >= 0.0, f"Negative entropy for {k}"

    def test_angle_features_in_range(self, synthetic_window):
        features = extract_features(synthetic_window)
        angle_keys = [k for k in features if k.startswith("angle")]
        for k in angle_keys:
            assert 0.0 <= features[k] <= math.pi, f"Angle out of [0, pi] range: {k}"

    def test_feature_count_reasonable(self, synthetic_window):
        features = extract_features(synthetic_window)
        # The UCI HAR feature set has hundreds of features
        assert len(features) > 50

    def test_custom_dt_fs(self, synthetic_window):
        features = extract_features(synthetic_window, dt=DT, fs=FS)
        assert isinstance(features, dict)
        assert len(features) > 0
