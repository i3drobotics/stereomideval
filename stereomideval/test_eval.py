"""This module tests Eval functionality in stereomideval module"""
import pytest
import numpy as np
from stereomideval.eval import Eval
from stereomideval.exceptions import ImageSizeNotEqual

def test_init_eval():
    """Test initalising Eval class"""
    Eval()

def test_catch_invalid_image_sizes():
    """Test catching invalid image sizes"""
    gt_image = np.zeros((5, 5))
    test_image = np.zeros((5, 6))
    with pytest.raises(ImageSizeNotEqual):
        Eval.diff(gt_image,test_image)
    with pytest.raises(ImageSizeNotEqual):
        Eval.mse(gt_image,test_image)
    with pytest.raises(ImageSizeNotEqual):
        Eval.rmse(gt_image,test_image)
    with pytest.raises(ImageSizeNotEqual):
        Eval.bad_pix_error(gt_image,test_image)
