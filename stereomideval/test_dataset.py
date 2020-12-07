"""This module tests Dataset functionality in stereomideval module"""
import pytest
import validators
import os
import ssl
from stereomideval.structures import DatasetType
from stereomideval.dataset import Dataset, SceneInfo
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

def test_2003_disparity():
    """Test 2003 load disparity image"""
    ssl._create_default_https_context = ssl._create_unverified_context
    DATASET_FOLDER = os.path.join(os.getcwd(),"datasets") #Path to download datasets
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    scene_info = SceneInfo(Dataset.Teddy, DatasetType.imperfect, 1.0)
    scene_name = scene_info.scene_name
    dataset_type = scene_info.dataset_type
    Dataset.download_scene_data(scene_name,DATASET_FOLDER,dataset_type)
    Dataset.load_scene_data(
        scene_name=scene_name,dataset_folder=DATASET_FOLDER,
        dataset_type=dataset_type)
    
