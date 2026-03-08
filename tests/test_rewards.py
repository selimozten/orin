"""Tests for reward functions."""

from __future__ import annotations

from orin.rewards.calibration import CalibrationReward
from orin.rewards.composite import AdaptiveCompositeReward, CompositeReward
from orin.rewards.directional import DirectionalReward
from orin.rewards.sharpe import SharpeReward


def test_directional_correct_up():
    r = DirectionalReward(scale_by_magnitude=False)
    assert r.compute(predicted_direction=2, actual_return=0.05) == 1.0


def test_directional_correct_down():
    r = DirectionalReward(scale_by_magnitude=False)
    assert r.compute(predicted_direction=0, actual_return=-0.05) == 1.0


def test_directional_wrong():
    r = DirectionalReward(scale_by_magnitude=False)
    assert r.compute(predicted_direction=2, actual_return=-0.05) == -1.0


def test_directional_flat_correct():
    r = DirectionalReward(scale_by_magnitude=False)
    reward = r.compute(predicted_direction=1, actual_return=0.001)
    assert reward == 1.0


def test_directional_scaled():
    r = DirectionalReward(scale_by_magnitude=True)
    reward = r.compute(predicted_direction=2, actual_return=0.10)
    assert reward > 1.0


def test_calibration_correct_high_confidence():
    r = CalibrationReward(mode="linear")
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.9)
    assert reward == 0.9


def test_calibration_wrong_high_confidence():
    r = CalibrationReward(mode="linear")
    reward = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.9)
    assert reward < 0


def test_calibration_wrong_low_confidence():
    r = CalibrationReward(mode="linear")
    r_high = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.9)
    r_low = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.1)
    assert r_low > r_high  # lower confidence = less penalty


def test_composite_combines():
    r = CompositeReward(directional_weight=0.5, calibration_weight=0.5)
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    assert reward > 0


def test_composite_custom_weights():
    r1 = CompositeReward(directional_weight=1.0, calibration_weight=0.0)
    r2 = CompositeReward(directional_weight=0.0, calibration_weight=1.0)
    reward1 = r1.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    reward2 = r2.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    # Pure directional vs pure calibration should differ
    assert reward1 != reward2


# ── Brier-mode calibration tests ───────────────────────────────────────


def test_brier_correct_high_confidence():
    r = CalibrationReward(mode="brier")
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.9)
    assert abs(reward - 0.99) < 1e-6


def test_brier_wrong_high_confidence():
    r = CalibrationReward(mode="brier")
    reward = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.9)
    assert abs(reward - 0.19) < 1e-6


def test_brier_penalizes_overconfidence():
    r = CalibrationReward(mode="brier")
    r_overconfident = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.9)
    r_modest = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.5)
    assert r_modest > r_overconfident


def test_brier_optimal_confidence():
    r = CalibrationReward(mode="brier")
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=1.0)
    assert reward == 1.0


def test_linear_mode_backward_compat():
    r = CalibrationReward(mode="linear")
    # correct + high confidence => confidence
    reward_correct = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.9)
    assert reward_correct == 0.9
    # wrong + high confidence => -confidence * penalty_scale
    reward_wrong = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.9)
    assert reward_wrong == -0.9 * 2.0


# ── Sharpe reward tests ───────────────────────────────────────────────


def test_sharpe_reward_basic():
    r = SharpeReward(window_size=10, min_window=3)
    # Feed a few returns; should not error
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    assert isinstance(reward, float)


def test_sharpe_reward_consistent_returns():
    consistent = SharpeReward(window_size=20, min_window=5)
    volatile = SharpeReward(window_size=20, min_window=5)

    # Consistent: slightly varying positive returns
    consistent_returns = [0.019 + 0.002 * (i % 3) for i in range(20)]
    for ret in consistent_returns:
        consistent.compute(predicted_direction=2, actual_return=ret, confidence=0.8)
    # Volatile: same direction but huge swings (including wrong-way losses)
    volatile_returns = [0.10, -0.08, 0.12, -0.09, 0.11, -0.07, 0.10, -0.08,
                        0.12, -0.09, 0.11, -0.07, 0.10, -0.08, 0.12, -0.09,
                        0.11, -0.07, 0.10, -0.08]
    for ret in volatile_returns:
        volatile.compute(predicted_direction=2, actual_return=ret, confidence=0.8)

    r_consistent = consistent.compute(predicted_direction=2, actual_return=0.02, confidence=0.8)
    r_volatile = volatile.compute(predicted_direction=2, actual_return=0.02, confidence=0.8)
    assert r_consistent > r_volatile


def test_sharpe_reward_reset():
    r = SharpeReward(window_size=10, min_window=3)
    for _ in range(10):
        r.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    r.reset()
    assert len(r._returns) == 0


# ── Partial credit tests ──────────────────────────────────────────────


def test_partial_credit():
    r = DirectionalReward(scale_by_magnitude=False, partial_credit=True, flat_threshold=0.005)
    # Predict up, actual is slightly down (within 2*flat_threshold=0.01)
    reward = r.compute(predicted_direction=2, actual_return=-0.008)
    assert abs(reward - (-0.3)) < 1e-6


def test_partial_credit_disabled_default():
    r = DirectionalReward(scale_by_magnitude=False)
    assert r.partial_credit is False
    # Same near-miss should give -1.0 without partial credit
    reward = r.compute(predicted_direction=2, actual_return=-0.008)
    assert reward == -1.0


# ── Adaptive composite tests ─────────────────────────────────────────


def test_adaptive_composite_initial_weights():
    r = AdaptiveCompositeReward(window_size=100)
    # With fewer than window_size samples, should use initial_dir_weight (0.8)
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    assert isinstance(reward, float)
    # Verify it uses initial weights by comparing with manual computation
    d = r.directional.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    c = r.calibration.compute(
        predicted_direction=2, actual_return=0.05, confidence=0.8,
        flat_threshold=r.directional.flat_threshold,
    )
    expected = 0.8 * d + 0.2 * c
    assert abs(reward - expected) < 1e-6


def test_adaptive_composite_shifts():
    r = AdaptiveCompositeReward(window_size=50, accuracy_threshold=0.70, conf_var_threshold=0.05)
    # Feed 50 correct predictions with low confidence variance
    for _ in range(50):
        r.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)

    # Now the window is full with 100% accuracy and near-zero conf variance
    # Next prediction should use final_dir_weight (0.4)
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    d = r.directional.compute(predicted_direction=2, actual_return=0.05, confidence=0.8)
    c = r.calibration.compute(
        predicted_direction=2, actual_return=0.05, confidence=0.8,
        flat_threshold=r.directional.flat_threshold,
    )
    expected = 0.4 * d + 0.6 * c
    assert abs(reward - expected) < 1e-6
