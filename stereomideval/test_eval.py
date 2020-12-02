"""This module tests Eval functionality in stereomideval module"""
import pytest
import numpy as np
from stereomideval.eval import Eval, Metric
from stereomideval.exceptions import ImageSizeNotEqual

def test_init_eval():
    """Test initalising Eval class"""
    Eval()


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
