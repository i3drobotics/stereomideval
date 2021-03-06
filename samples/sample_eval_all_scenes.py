"""
Sample: Evaluatuate all scenes

This module demonstrates evaluating data from all scene using the stereomideval module.
For demonstration purposes the test data is generated by adding noise to the ground truth
"""
import os
import numpy as np
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer

DATASET_FOLDER = os.path.join(os.getcwd(),"datasets") #Path to download datasets
GET_METRIC_RANK = False # Compare each match data against online ranking
GET_AV_METRIC_RANK = True # Compare average results across all scenes against online ranking

# Create dataset folder
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

match_data_list = []
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
        dataset_type=dataset_type)
    # Scene data class contains the following data:
    left_image = scene_data.left_image
    right_image = scene_data.right_image
    ground_truth_disp_image = scene_data.disp_image
    ndisp = scene_data.ndisp

    # Start timer
    timer = Timer()
    timer.start()
    # Simluate match result by adding a bit of noise to the ground truth
    # REPLACE THIS WITH THE RESULT FROM YOUR STEREO ALGORITHM
    # e.g. test_disp_image = cv2.imread("disp_result.tif",cv2.IMREAD_UNCHANGED)
    noise = np.random.uniform(low=0, high=3.0, size=ground_truth_disp_image.shape)
    test_disp_image = ground_truth_disp_image + noise
    # Record elapsed time for simulated match
    elapsed_time = timer.elapsed()

    # Store match results
    match_result = MatchData.MatchResult(
        left_image,right_image,ground_truth_disp_image,test_disp_image,elapsed_time,ndisp)
    # Create match data (which includes scene data needed for rank comparision in eval)
    match_data = MatchData(scene_info,match_result)
    # Add match data to list
    match_data_list.append(match_data)

    # Display match result to OpenCV window
    Eval.display_results(match_result)

# Evaluate list of match data against all middlesbury metrics
metric_average_list, eval_data_list = \
    Eval.evaluate_match_data_list(
        match_data_list,GET_METRIC_RANK,GET_AV_METRIC_RANK,display_results=True)
# metric_average_list: list of average metrics across all test data
# eval_data_list: list of evaluation data, seperate list for each test data provided

# Iterate evaluation data for each scene
for eval_data in eval_data_list:
    # Iterate each metric and evaluation result
    for eval_result in eval_data.eval_result_list:
        # Print metric and result for each scene
        print("{}: {}".format(eval_result.metric,eval_result.result))

# Iterate average metrics across all scenes
for metric_average in metric_average_list:
    # Print metric and average result across all scenes
    print("{}: {}".format(metric_average.metric,metric_average.result))
