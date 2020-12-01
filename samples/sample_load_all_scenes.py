"""
Sample: Load all scenes

This module demonstrates loading data from all scenes using the stereomideval module.
"""
import os
from stereomideval.dataset import Dataset

DATASET_FOLDER = os.path.join(os.getcwd(),"datasets") #Path to download datasets
DISPLAY_IMAGES = False

# Create dataset folder
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

# Get list of scenes in Milddlebury's stereo training dataset and iterate through them
for scene_info in Dataset.get_training_scene_list():
    scene_name=scene_info.scene_name
    dataset_type=scene_info.dataset_type
    # Download dataset from middlebury servers
    # will only download it if it hasn't already been downloaded
    print("Downloading data for scene '"+scene_name+"'...")
    Dataset.download_scene_data(scene_name,DATASET_FOLDER,dataset_type)
    # Load scene data from downloaded folder
    print("Loading data for scene '"+scene_name+"'...")
    scene_data = Dataset.load_scene_data(
        scene_name=scene_name,dataset_folder=DATASET_FOLDER,
        dataset_type=dataset_type,display_images=DISPLAY_IMAGES)
    left_image = scene_data.left_image
    right_image = scene_data.right_image
    ground_truth_disp_image = scene_data.disp_image
    ndisp = scene_data.ndisp