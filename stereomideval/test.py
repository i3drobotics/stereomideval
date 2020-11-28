"""This module tests functionality of stereomideval module"""
import os
from stereomideval import Dataset, Eval

if __name__ == "__main__":
    dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to dowmload datasets

    # Create dataset folder
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Initalise stereomideval objects
    stmid_dataset = Dataset()
    stmid_eval = Eval()

    # Get list of scenes in Milddlebury's stereo dataset (2014) and iterate through them
    for scenename in stmid_dataset.get_scene_list():
        # Download dataset from middlebury servers
        # will only download it if it hasn't already been downloaded
        print("Downloading data for scene '"+scenename+"'...")
        stmid_dataset.download_scene_data(scenename,dataset_folder)
        # Load scene data from downloaded folder
        print("Loading data for scene '"+scenename+"'...")
        scene_data = stmid_dataset.load_scene_data(scenename,dataset_folder,True)
        disp_image = scene_data.disp_image
        # Demonstate evaluation by comparing the ground truth to itself
        mse = stmid_eval.mse(disp_image,disp_image)
        print(mse)
