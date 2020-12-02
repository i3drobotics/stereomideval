"""This module tests Dataset functionality in stereomideval module"""
import pytest
import validators
from stereomideval.dataset import Dataset
from stereomideval.exceptions import InvalidSceneName


def test_init_dataset():
    """Test initalising Dataset class"""
    Dataset()


def test_dataset_valid_scene_urls():
    """Test valid url creation for scenes"""
    for scene_name in Dataset.get_scene_list():
        url = Dataset.get_url_from_scene(scene_name)
        assert validators.url(url)


def test_check_valid_scene_names():
    """Test valid scene names"""
    for scene_name in Dataset.get_scene_list():
        assert InvalidSceneName.validate_scene_list(scene_name, Dataset.get_scene_list()) is None


def test_catch_invalid_scene_name():
    """Test valid scene names"""
    with pytest.raises(InvalidSceneName):
        InvalidSceneName.validate_scene_list("INVALID SCENE NAME", Dataset.get_scene_list())
