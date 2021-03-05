"""This module tests Eval functionality in stereomideval module"""
import pytest
import numpy as np
from stereomideval.eval import Eval, Metric
from stereomideval.exceptions import ImageSizeNotEqual


def test_init_eval():
    """Test initalising Eval class"""
    Eval()

def test_invalid_pixels():
    invalid_signifier = 0
    test_image = np.zeros((5, 6))
    # all pixels should be invalid as image is full of zeros
    assert Metric.calc_invalid_pixels(test_image,invalid_signifier) == test_image.size
    test_image = np.ones((5, 6))
    # all pixels should be valid as image is full of ones
    assert Metric.calc_invalid_pixels(test_image,invalid_signifier) == 0

def test_catch_invalid_image_sizes():
    """Test catching invalid image sizes"""
    gt_image = np.zeros((5, 5))
    test_image = np.zeros((5, 6))
    with pytest.raises(ImageSizeNotEqual):
        Metric.calc_diff(gt_image, test_image)
    with pytest.raises(ImageSizeNotEqual):
        Metric.calc_mse(gt_image, test_image)
    with pytest.raises(ImageSizeNotEqual):
        Metric.calc_rmse(gt_image, test_image)
    with pytest.raises(ImageSizeNotEqual):
        Metric.calc_bad_pix_error(gt_image, test_image)
