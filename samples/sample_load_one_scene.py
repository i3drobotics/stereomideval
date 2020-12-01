"""
Sample: Load one scene

This module demonstrates loading data from one scene using the stereomideval module.
"""
import os
from stereomideval.dataset import Dataset

# Path to dowmload datasets
DATASET_FOLDER = os.path.join(os.getcwd(),"datasets")
# Scene name (see here for list of scenes: https://vision.middlebury.edu/stereo/data/scenes2014/)
SCENE_NAME = "Adirondack"
# Display images to OpenCV window
DISPLAY_IMAGES = False

# Create dataset folder
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

# Download dataset from middlebury servers
# will only download it if it hasn't already been downloaded
print("Downloading data for scene '"+SCENE_NAME+"'...")
Dataset.download_scene_data(SCENE_NAME,DATASET_FOLDER)
# Load scene data from downloaded folder
print("Loading data for scene '"+SCENE_NAME+"'...")
scene_data = Dataset.load_scene_data(SCENE_NAME,DATASET_FOLDER,DISPLAY_IMAGES)
