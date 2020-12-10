# stereo-mideval
Python package for dataset and evaluation tools from the [Middlebury stereo evaulation 2014](https://vision.middlebury.edu/stereo/data/scenes2014/) dataset.
This project is in development by [I3DR](https://i3drobotics.com/) for evaluating stereo matching algorithms for use in stereo cameras. However, this project is fully open-source with no limitations to encorage and support others who may need this. 

## Compatibility
Compatible with python 3.5, 3.6, 3.7, 3.8 on Windows x64
Tested on Windows using Git Actions.

[![Actions Status](https://github.com/i3drobotics/stereomideval/workflows/Test%20Python%20package/badge.svg?event=push)](https://github.com/i3drobotics/stereomideval/actions)  
[![Actions Status](https://github.com/i3drobotics/stereomideval/workflows/Upload%20Python%20Package/badge.svg)](https://github.com/i3drobotics/stereomideval/actions)

## Install
```
pip install stereo-mideval
```

## Features
- Download scene data from Middlebury servers
- Load disparity image and stereo pair from scene data
- Display normalised colormaped disparity image
- Convert disparity image to depth image using calibration file from scene data
- Evaluation metrics of disparity image compared to ground truth disparity (rmse, mse, bad pixel percentage, ...)

## Examples
Find full directory of examples [here](https://github.com/i3drobotics/stereomideval/tree/main/samples)
### Download Adirondack scene and evaluate bad2.0 metric
```python
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
```

### Download and evaluate all Middlebury dataset scenes
```python
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
        print("{}:{}".format(eval_result.metric,eval_result.result))

# Iterate average metrics across all scenes
for metric_average in eval_data_list:
    # Print metric and average result across all scenes
    print("{}:{}".format(metric_average.metric,metric_average.result))
```

## Dataset file structure
When downloading the full stereo Middlebury dataset for evaulating against the online ranking table they are downloaded in the folder structure detailed below. In the ranking table most are from the 2014 dataset apart from 2 (Art and Teddy). The non 2014 scene only download what is required for the evaulation to try and keep things simple. The 2014 datasets have the folder name of '-perfect' or '-imperfect' to denote if they have epipolar aligned images. 'Imperfect' versions will be more challenging. Teddy will be downloaded as full quality ppm version ['teddyF-ppm-2.zip'](https://vision.middlebury.edu/stereo/data/scenes2003/newdata/full/teddyF-ppm-2.zip) as the png version is only available in half resolution. 
```
datasets
    Adirondack-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Art
        disp1.png
        view1.png
        view5.png
    Jadeplant-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png 
    Motorcycle-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Piano-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Pipes-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Playroom-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Playtable-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Playtable-perfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Recycle-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Shelves-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
    Teddy
        disp2.pgm
        disp6.pgm
        im2.ppm
        im6.ppm
    Vintage-imperfect
        calib.txt
        disp0.pfm
        disp0y.pfm
        disp1.pfm
        disp1y.pfm
        im0.png
        im1.png
        im1E.png
        im1L.png
```

## Development
### Upcomming features
 - Offline ranking by caching webpage table data