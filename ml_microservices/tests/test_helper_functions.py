"""
Unit tests for shared/helper_functions.py
"""
import math
import numpy as np
import pytest

from shared.helper_functions import (
    GRAVITY,
    DT,
    FS,
    correlation,
    energy,
    entropy,
    sma,
    mean_freq,
    lowpass_filter,
    angle_between,
    extract_features,
)


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_identical_signals_return_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert correlation(x, x) == pytest.approx(1.0)

    def test_opposite_signals_return_minus_one(self):
        x = np.array([1.0, 2.0, 3.0])
        y = -x
        assert correlation(x, y) == pytest.approx(-1.0)

    def test_uncorrelated_orthogonal_signals(self):
        x = np.array([1.0, -1.0, 1.0, -1.0])
        y = np.array([1.0, 1.0, -1.0, -1.0])
        result = correlation(x, y)
        assert -1.0 <= result <= 1.0

    def test_constant_signal_returns_zero(self):
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0])
        assert correlation(x, y) == 0

    def test_both_constant_returns_zero(self):
        x = np.array([2.0, 2.0, 2.0])
        assert correlation(x, x) == 0


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_known_values(self):
        sig = np.array([1.0, 2.0, 3.0])
        # energy = (1 + 4 + 9) / 3
        assert energy(sig) == pytest.approx(14.0 / 3.0)

    def test_zeros_returns_zero(self):
        assert energy(np.zeros(10)) == pytest.approx(0.0)

    def test_empty_signal_returns_zero(self):
        assert energy(np.array([])) == pytest.approx(0.0)

    def test_unit_signal(self):
        sig = np.ones(100)
        assert energy(sig) == pytest.approx(1.0)

    def test_normalized_energy_scales_with_amplitude(self):
        sig = np.array([2.0, 2.0])
        assert energy(sig) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_distribution_has_max_entropy(self):
        # A signal with many distinct evenly-spaced values has near-max entropy.
        sig = np.linspace(0, 1, 1000)
        result = entropy(sig, bins=10)
        assert result > 2.0  # should be close to log2(10) ≈ 3.32

    def test_constant_signal_returns_zero(self):
        sig = np.ones(100) * 3.14
        assert entropy(sig) == pytest.approx(0.0)

    def test_empty_signal_returns_zero(self):
        assert entropy(np.array([])) == pytest.approx(0.0)

    def test_all_mass_on_one_bin_returns_zero(self):
        sig = np.ones(50)  # std == 0
        assert entropy(sig) == pytest.approx(0.0)

    def test_result_is_non_negative(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(200)
        assert entropy(sig) >= 0.0

    def test_result_within_expected_range(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(500)
        result = entropy(sig, bins=10)
        assert 0.0 <= result <= 10.0


# ---------------------------------------------------------------------------
# sma
# ---------------------------------------------------------------------------

class TestSma:
    def test_unit_axes(self):
        x = np.ones(4)
        y = np.ones(4)
        z = np.ones(4)
        # sum(|1|+|1|+|1|) / 4 = 12/4 = 3
        assert sma(x, y, z) == pytest.approx(3.0)

    def test_zeros(self):
        n = np.zeros(5)
        assert sma(n, n, n) == pytest.approx(0.0)

    def test_single_sample(self):
        assert sma(np.array([2.0]), np.array([3.0]), np.array([4.0])) == pytest.approx(9.0)

    def test_negative_values_counted_absolutely(self):
        x = np.array([-1.0])
        y = np.array([-1.0])
        z = np.array([-1.0])
        assert sma(x, y, z) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# mean_freq
# ---------------------------------------------------------------------------

class TestMeanFreq:
    def test_single_frequency_spike(self):
        freqs = np.array([0.0, 1.0, 2.0, 3.0])
        spec = np.array([0.0, 0.0, 10.0, 0.0])  # all power at 2 Hz
        assert mean_freq(spec, freqs) == pytest.approx(2.0)

    def test_zero_spectrum_returns_zero(self):
        freqs = np.array([1.0, 2.0, 3.0])
        spec = np.zeros(3)
        assert mean_freq(spec, freqs) == pytest.approx(0.0)

    def test_equal_power_returns_mean_frequency(self):
        freqs = np.array([0.0, 2.0, 4.0])
        spec = np.ones(3)
        assert mean_freq(spec, freqs) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# lowpass_filter
# ---------------------------------------------------------------------------

class TestLowpassFilter:
    def test_output_shape_matches_input(self):
        sig = np.random.default_rng(1).standard_normal(256)
        out = lowpass_filter(sig)
        assert out.shape == sig.shape

    def test_low_frequency_signal_passes_through(self):
        """A signal well below the cutoff should not be heavily attenuated."""
        t = np.linspace(0, 10, 500)
        # 0.1 Hz sinusoid — well below the default 0.3 Hz cutoff
        sig = np.sin(2 * np.pi * 0.05 * t)
        out = lowpass_filter(sig, cutoff=0.3, fs=50.0)
        # Energy should be preserved (within 10%)
        assert energy(out) == pytest.approx(energy(sig), rel=0.1)

    def test_high_frequency_signal_is_attenuated(self):
        """A signal far above the cutoff should be significantly attenuated."""
        t = np.linspace(0, 10, 500)
        # 10 Hz sinusoid — far above the 0.3 Hz cutoff
        sig = np.sin(2 * np.pi * 10 * t)
        out = lowpass_filter(sig, cutoff=0.3, fs=50.0)
        assert energy(out) < 0.05 * energy(sig)


# ---------------------------------------------------------------------------
# angle_between
# ---------------------------------------------------------------------------

class TestAngleBetween:
    def test_parallel_vectors_zero_angle(self):
        v = np.array([1.0, 0.0, 0.0])
        assert angle_between(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_anti_parallel_vectors_pi(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert angle_between(v1, v2) == pytest.approx(math.pi, abs=1e-9)

    def test_orthogonal_vectors_ninety_degrees(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert angle_between(v1, v2) == pytest.approx(math.pi / 2)

    def test_zero_vector_returns_zero(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        assert angle_between(v1, v2) == pytest.approx(0.0)

    def test_result_in_range_zero_to_pi(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            v1 = rng.standard_normal(3)
            v2 = rng.standard_normal(3)
            angle = angle_between(v1, v2)
            assert 0.0 <= angle <= math.pi


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    """Integration test: build a (128, 6) window and verify the feature dict."""

    @pytest.fixture()
    def flat_window(self):
        """A constant (non-zero) window to exercise all code paths."""
        # [accX, accY, accZ, gyroX, gyroY, gyroZ] — 128 rows
        return np.ones((128, 6)) * 0.5 * GRAVITY  # 0.5 g on all axes

    @pytest.fixture()
    def random_window(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((128, 6))

    def test_returns_dict(self, random_window):
        result = extract_features(random_window)
        assert isinstance(result, dict)

    def test_non_empty(self, random_window):
        result = extract_features(random_window)
        assert len(result) > 0

    def test_all_values_are_floats(self, random_window):
        result = extract_features(random_window)
        for key, val in result.items():
            assert isinstance(val, float), f"Feature '{key}' is not a float: {type(val)}"

    def test_expected_time_domain_keys_present(self, random_window):
        result = extract_features(random_window)
        for axis in ("X", "Y", "Z"):
            assert f"tBodyAcc-mean()-{axis}" in result
            assert f"tBodyAcc-std()-{axis}" in result
            assert f"tBodyGyro-mean()-{axis}" in result

    def test_expected_frequency_domain_keys_present(self, random_window):
        result = extract_features(random_window)
        for axis in ("X", "Y", "Z"):
            assert f"fBodyAcc-mean()-{axis}" in result
            assert f"fBodyAcc-energy()-{axis}" in result

    def test_angle_keys_present(self, random_window):
        result = extract_features(random_window)
        assert "angle(tBodyAccMean,gravityMean)" in result
        assert "angle(X,gravityMean)" in result

    def test_constant_window_does_not_raise(self, flat_window):
        result = extract_features(flat_window)
        assert isinstance(result, dict)

    def test_custom_dt_and_fs(self, random_window):
        result = extract_features(random_window, dt=0.02, fs=50.0)
        assert isinstance(result, dict)
        assert len(result) > 0
