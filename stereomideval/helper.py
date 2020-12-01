"""This module is used for manually testing functionality while developing stereomideval module"""
import os
import numpy as np
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer

def run():
    dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to dowmload datasets

    # Create dataset folder
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Get list of scenes in Milddlebury's stereo dataset (2014) and iterate through them
    for scene_list in Dataset.get_training_scene_list():
        scene_name=scene_list['name']
        dataset_type=scene_list['dataset_type']
        # Download dataset from middlebury servers
        # will only download it if it hasn't already been downloaded
        print("Downloading data for scene '"+scene_name+"'...")
        Dataset.download_scene_data(scene_name,dataset_folder,dataset_type)
        continue
        # Load scene data from downloaded folder
        print("Loading data for scene '"+scene_name+"'...")
        scene_data = Dataset.load_scene_data(
            scene_name=scene_name,dataset_folder=dataset_folder,
            dataset_type=dataset_type)
        left_image = scene_data.left_image
        ground_truth_disp_image = scene_data.disp_image
        # Start timer
        timer = Timer()
        timer.start()
        # Simluate match result by adding a bit of noise to the ground truth
        noise = np.random.uniform(low=0, high=1.0, size=ground_truth_disp_image.shape)
        test_disp_image = ground_truth_disp_image + noise
        # Record elapsed time for match
        elapsed_time = timer.elapsed()
        # Evaluate test data against all metrics
        evals = Eval.eval_all(
            ground_truth=ground_truth_disp_image,test_data=test_disp_image,
            elapsed_time=elapsed_time,
            scene_name=scene_name,dataset_type=dataset_type,dense=False)
        print(evals)
        Eval.display_results(left_image,ground_truth_disp_image,test_disp_image,evals,wait=0)

if __name__ == "__main__":
    run()
