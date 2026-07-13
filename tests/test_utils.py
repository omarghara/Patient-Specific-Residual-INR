"""Tests for exact spatial alignment helpers."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.utils import center_crop_to, center_pad_to


def test_center_crop_undoes_slam_style_padding():
    native = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    padded = center_pad_to(native, (5, 9))

    assert padded.shape == (5, 9)
    assert torch.equal(center_crop_to(padded, native.shape), native)
    assert torch.count_nonzero(padded).item() == torch.count_nonzero(native).item()


def test_slam_256_by_206_padding_has_25_columns_on_each_side():
    native = torch.ones(256, 206)
    padded = center_pad_to(native, (256, 256))

    assert torch.count_nonzero(padded[:, :25]).item() == 0
    assert torch.count_nonzero(padded[:, -25:]).item() == 0
    assert torch.equal(padded[:, 25:231], native)


def test_center_crop_and_pad_preserve_leading_dims_complex_dtype_and_gradients():
    real = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    x = torch.complex(real, -real).requires_grad_()

    cropped = center_crop_to(x, (2, 3))
    padded = center_pad_to(cropped, (4, 6))
    padded.abs().sum().backward()

    assert cropped.shape == (2, 2, 3)
    assert padded.shape == x.shape
    assert padded.is_complex()
    assert x.grad is not None


def test_center_crop_and_pad_reject_the_wrong_direction():
    x = torch.zeros(4, 5)
    with pytest.raises(ValueError, match="cannot crop"):
        center_crop_to(x, (5, 5))
    with pytest.raises(ValueError, match="cannot pad"):
        center_pad_to(x, (3, 5))
