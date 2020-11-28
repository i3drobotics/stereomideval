"""This module shows example functionality of stereomideval module"""
import os
from stereomideval import Dataset

# Path to dowmload datasets
dataset_folder = os.path.join(os.getcwd(),"datasets") 

# Create dataset folder
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Initalise stereomideval Dataset object
stmid_dataset = Dataset()

# Get list of scene in dataset (2014) and iterate through them
for scenename in stmid_dataset.get_scene_list():
    # Download dataset from middlebury servers
    # will only download it if it hasn't already been downloaded
    print("Downloading data for scene '"+scenename+"'...")
    stmid_dataset.download_scene_data(scenename,dataset_folder) 
    # Load scene data from downloaded folder
    print("Loading data for scene '"+scenename+"'...")
    scene_data = stmid_dataset.load_scene_data(scenename,dataset_folder,True)