"""This module tests Eval functionality in stereomideval module"""
import pytest
import numpy as np
from stereomideval import Eval, ImageSizeNotEqual

def test_init_dataset():
    """Test initalising Eval class"""
    Eval()

def test_catch_invalid_image_sizes():
    """Test catching invalid image sizes"""
    stmid_eval = Eval()
    gt_image = np.zeros((5, 5))
    test_image = np.zeros((5, 6))
    with pytest.raises(ImageSizeNotEqual):
        stmid_eval.diff(gt_image,test_image)
    with pytest.raises(ImageSizeNotEqual):
        stmid_eval.mse(gt_image,test_image)
    with pytest.raises(ImageSizeNotEqual):
        stmid_eval.rmse(gt_image,test_image)
    with pytest.raises(ImageSizeNotEqual):
        stmid_eval.bad_pix_error(gt_image,test_image)
