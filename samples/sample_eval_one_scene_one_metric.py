"""
Sample: Load one scene

This module demonstrates loading data from one scene using the stereomideval module.
"""
import os
import numpy as np
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer, Metric

# Path to dowmload datasets
DATASET_FOLDER = os.path.join(os.getcwd(),"datasets")
# Scene name (see here for list of scenes: https://vision.middlebury.edu/stereo/data/scenes2014/)
SCENE_NAME = "Adirondack"
# Display loaded scene data to OpenCV window
DISPLAY_IMAGES = True

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
# Format match result into expected format for use in evaluation
match_result = MatchData.MatchResult(
    left_image,right_image,ground_truth_disp_image,test_disp_image,elapsed_time,ndisp)
# Evalulate match results against all Middlebury metrics
metric_result = Eval.eval_metric(Metric.bad200,match_result)
# Print metric and result
print("{}: {}".format(metric_result.metric,metric_result.result))
