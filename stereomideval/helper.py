"""This module is used for manually testing functionality while developing stereomideval module"""
import os
import shutil
import numpy as np
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer

def run():
    dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to download datasets
    eval_folder = os.path.join(os.getcwd(),"evaluation") #Path to store evaluation
    get_metric_rank = False
    get_av_metric_rank = True

    # Create dataset folder
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Create eval folder
    if os.path.exists(eval_folder):
        shutil.rmtree(eval_folder)
    os.makedirs(eval_folder)

    match_data_list = []
    # Get list of scenes in Milddlebury's stereo training dataset and iterate through them
    for scene_info in Dataset.get_training_scene_list():
        scene_name=scene_info.scene_name
        dataset_type=scene_info.dataset_type
        # Download dataset from middlebury servers
        # will only download it if it hasn't already been downloaded
        print("Downloading data for scene '"+scene_name+"'...")
        Dataset.download_scene_data(scene_name,dataset_folder,dataset_type)
        # Load scene data from downloaded folder
        print("Loading data for scene '"+scene_name+"'...")
        scene_data = Dataset.load_scene_data(
            scene_name=scene_name,dataset_folder=dataset_folder,
            dataset_type=dataset_type)
        left_image = scene_data.left_image
        right_image = scene_data.right_image
        ground_truth_disp_image = scene_data.disp_image
        ndisp = scene_data.ndisp
        # Start timer
        timer = Timer()
        timer.start()

        # Simluate match result by adding a bit of noise to the ground truth
        noise = np.random.uniform(low=0, high=3.0, size=ground_truth_disp_image.shape)
        test_disp_image = ground_truth_disp_image + noise
        # Record elapsed time for match
        elapsed_time = timer.elapsed()

        match_result = MatchData.MatchResult(
            left_image,right_image,ground_truth_disp_image,test_disp_image,elapsed_time,ndisp)
        match_data = MatchData(scene_info,match_result)
        match_data_list.append(match_data)

    Eval.evaluate_match_data_list(match_data_list,get_metric_rank,get_av_metric_rank)

if __name__ == "__main__":
    run()
