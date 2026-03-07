"""Tests for reward functions."""

from __future__ import annotations

from orin.rewards.calibration import CalibrationReward
from orin.rewards.composite import CompositeReward
from orin.rewards.directional import DirectionalReward


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
    r = CalibrationReward()
    reward = r.compute(predicted_direction=2, actual_return=0.05, confidence=0.9)
    assert reward == 0.9


def test_calibration_wrong_high_confidence():
    r = CalibrationReward()
    reward = r.compute(predicted_direction=2, actual_return=-0.05, confidence=0.9)
    assert reward < 0


def test_calibration_wrong_low_confidence():
    r = CalibrationReward()
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
