"""This module tests Dataset functionality in stereomideval module"""
import validators
from stereomideval import Dataset

def test_init_dataset():
    """Test initalising Dataset class"""
    Dataset()

def test_dataset_list_length():
    """Test returning scene list"""
    scenenames = Dataset.get_scene_list()
    assert len(scenenames) == 23

def test_dataset_valid_scene_urls():
    """Test valid url creation for scenes"""
    for scene_name in Dataset.get_scene_list():
        url = Dataset.get_url_from_scene(scene_name)
        assert validators.url(url)

def test_check_valid_scene_names():
    """Test valid scene names"""
    for scene_name in Dataset.get_scene_list():
        assert Dataset.check_valid_scene_name(scene_name)

def test_catch_invalid_scene_name():
    """Test valid scene names"""
    assert not Dataset.check_valid_scene_name("INVALID SCENE NAME")
