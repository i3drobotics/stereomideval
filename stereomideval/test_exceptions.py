"""This module tests Exceptions functionality in stereomideval module"""
import pytest
import numpy as np
from stereomideval.dataset import Dataset
from stereomideval.exceptions import ImageSizeNotEqual, PathNotFound, InvalidSceneName

def test_catch_invalid_image_sizes():
    """Test catching invalid image sizes"""
    image_a = np.zeros((5, 5))
    image_b = np.zeros((5, 6))
    with pytest.raises(ImageSizeNotEqual):
        ImageSizeNotEqual.validate(image_a, image_b)


def test_catch_path_not_found():
    """Test catching path not found"""
    path = "stereomideval/not_a_path"
    with pytest.raises(PathNotFound):
        PathNotFound.validate(path)


def test_catch_invalid_scene_name():
    """Test catching invalid scene name"""
    scene_name = "Invalid scene name"
    with pytest.raises(InvalidSceneName):
        InvalidSceneName.validate_scene_list(scene_name, Dataset.get_scene_list())
    with pytest.raises(InvalidSceneName):
        InvalidSceneName.validate_scene_info_list(scene_name, Dataset.get_training_scene_list())
    