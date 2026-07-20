"""Tests for longitudinal change and mutual-information metrics."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.metrics import (
    acquisition_calibrated_longitudinal_metrics,
    all_metrics,
    change_cosine,
    change_gain,
    mutual_information,
    prior_followup_mutual_information,
)


def _longitudinal_example():
    prior = np.zeros((8, 8), dtype=np.float32)
    prior[0, 0] = 1.0  # unchanged intensity anchor keeps robust scales aligned
    ref = prior.copy()
    ref[2, 2] = 0.8
    ref[5, 4] = 0.4
    return prior, ref


def test_change_metrics_distinguish_copy_gain_and_wrong_location():
    prior, ref = _longitudinal_example()
    half = prior + 0.5 * (ref - prior)
    wrong = prior.copy()
    wrong[2, 5] = 0.8
    wrong[5, 1] = 0.4

    assert change_cosine(ref, ref, prior) == pytest.approx(1.0, abs=1e-6)
    assert change_gain(ref, ref, prior) == pytest.approx(1.0, abs=1e-6)
    assert change_cosine(prior, ref, prior) == 0.0
    assert change_gain(prior, ref, prior) == pytest.approx(0.0, abs=1e-7)
    assert change_cosine(half, ref, prior) == pytest.approx(1.0, abs=1e-6)
    assert change_gain(half, ref, prior) == pytest.approx(0.5, abs=2e-2)
    assert abs(change_cosine(wrong, ref, prior)) < 1e-6


def test_change_metrics_are_undefined_without_reference_change():
    prior = np.eye(8, dtype=np.float32)
    assert np.isnan(change_cosine(prior, prior, prior))
    assert np.isnan(change_gain(prior, prior, prior))


def test_mutual_information_is_symmetric_and_detects_correspondence():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(64, 64)).astype(np.float32)
    y = x.copy()
    shuffled = rng.permutation(x.reshape(-1)).reshape(x.shape)

    mi_same = mutual_information(x, y)
    mi_shuffled = mutual_information(x, shuffled)
    assert mi_same > mi_shuffled
    assert mutual_information(x, shuffled) == pytest.approx(
        mutual_information(shuffled, x), abs=1e-12
    )
    assert mutual_information(7.0 * x, 0.2 * y) == pytest.approx(mi_same, abs=1e-12)


def test_binary_mutual_information_has_known_values_in_bits():
    identical = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    independent_x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    independent_y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    assert mutual_information(
        identical, identical, bins=2, x_scale=1.0, y_scale=1.0
    ) == pytest.approx(1.0, abs=1e-12)
    assert mutual_information(
        independent_x, independent_y, bins=2, x_scale=1.0, y_scale=1.0
    ) == pytest.approx(0.0, abs=1e-12)


def test_all_metrics_reports_requested_prior_followup_mi_and_target_delta():
    prior, ref = _longitudinal_example()
    with np.errstate(divide="ignore"):
        out = all_metrics(ref, ref, prior)

    assert out["mi_prior_ref"] == pytest.approx(
        prior_followup_mutual_information(prior, ref), abs=1e-12
    )
    assert out["mi_prior_recon"] == pytest.approx(out["mi_prior_ref"], abs=1e-12)
    assert out["mi_prior_delta"] == pytest.approx(0.0, abs=1e-12)
    assert "cpe" not in out and "pbs" not in out


def test_acquisition_calibrated_change_does_not_oracle_align_reconstruction():
    prior, ref = _longitudinal_example()
    reference_scale = 1.7
    prior_scale = 2.2
    calibrated_prior = prior_scale * prior
    calibrated_reference = reference_scale * ref
    true_change = calibrated_reference - calibrated_prior
    half = calibrated_prior + 0.5 * true_change

    copied = acquisition_calibrated_longitudinal_metrics(
        calibrated_prior,
        ref,
        prior,
        reference_to_acquisition=reference_scale,
        prior_to_acquisition=prior_scale,
    )
    perfect = acquisition_calibrated_longitudinal_metrics(
        calibrated_reference,
        ref,
        prior,
        reference_to_acquisition=reference_scale,
        prior_to_acquisition=prior_scale,
    )
    partial = acquisition_calibrated_longitudinal_metrics(
        half,
        ref,
        prior,
        reference_to_acquisition=reference_scale,
        prior_to_acquisition=prior_scale,
    )

    assert copied["change_cosine"] == pytest.approx(0.0, abs=1e-7)
    assert copied["change_gain"] == pytest.approx(0.0, abs=1e-7)
    assert perfect["change_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["change_gain"] == pytest.approx(1.0, abs=1e-6)
    assert partial["change_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert partial["change_gain"] == pytest.approx(0.5, abs=2e-2)


@pytest.mark.parametrize("factor", [0.0, -1.0, np.inf, np.nan])
def test_acquisition_calibrated_metrics_reject_invalid_scale(factor):
    prior, ref = _longitudinal_example()
    with pytest.raises(ValueError, match="reference_to_acquisition"):
        acquisition_calibrated_longitudinal_metrics(
            ref,
            ref,
            prior,
            reference_to_acquisition=factor,
            prior_to_acquisition=1.0,
        )
