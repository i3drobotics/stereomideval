# stereo-mideval
Python package for dataset and evaluation tools from the [Middlebury stereo evaulation 2014](https://vision.middlebury.edu/stereo/data/scenes2014/) dataset.
This project is in development by [I3DR](https://i3drobotics.com/) for evaluating stereo matching algorithms for use in stereo cameras. However, this project is fully open-source with no limitations to encorage and support others who may need this. 

## Install
```
pip install stereo-mideval
```

## Features
- Download scene data from Middlebury servers
- Load disparity image and stereo pair from scene data
- Display normalised colormaped disparity image
- Convert disparity image to depth image using calibration file from scene data
- Evaluation metrics of disparity image compared to ground truth disparity (rmse, mse, bad pixel percentage)

## Examples
### Download and display data from all scenes in Middlebury stereo dataset (2014)
```python
import os
from stereomideval import Dataset

# Path to dowmload datasets
DATASET_FOLDER = os.path.join(os.getcwd(),"datasets")

# Create dataset folder
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

# Initalise stereomideval Dataset object
stmid_dataset = Dataset()

# Get list of scene in dataset (2014) and iterate through them
for scene_name in stmid_dataset.get_scene_list():
    # Download dataset from middlebury servers
    # will only download it if it hasn't already been downloaded
    print("Downloading data for scene '"+scene_name+"'...")
    stmid_dataset.download_scene_data(scene_name,DATASET_FOLDER)
    # Load scene data from downloaded folder
    print("Loading data for scene '"+scene_name+"'...")
    scene_data = stmid_dataset.load_scene_data(scene_name,DATASET_FOLDER,True)
```

### Download and display data from a single scene in Middlebury stereo dataset (2014)
```python
import os
from stereomideval import Dataset

# Path to dowmload datasets
DATASET_FOLDER = os.path.join(os.getcwd(),"datasets")
# Scene name (see here for list of scenes: https://vision.middlebury.edu/stereo/data/scenes2014/)
SCENE_NAME = "Adirondack"

# Create dataset folder
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

# Initalise stereomideval Dataset object
stmid_dataset = Dataset()

# Download dataset from middlebury servers
# will only download it if it hasn't already been downloaded
print("Downloading data for scene '"+SCENE_NAME+"'...")
stmid_dataset.download_scene_data(SCENE_NAME,DATASET_FOLDER)
# Load scene data from downloaded folder
print("Loading data for scene '"+SCENE_NAME+"'...")
stmid_dataset.load_scene_data(SCENE_NAME,DATASET_FOLDER,True,0)
```

### Download and evaluatuate all scenes in Middlebury stereo dataset (2014)
```python
import os
import numpy as np
from stereomideval import Dataset, Eval

dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to dowmload datasets

# Create dataset folder
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Initalise stereomideval objects
stmid_dataset = Dataset()
stmid_eval = Eval()

# Get list of scenes in Milddlebury's stereo dataset (2014) and iterate through them
for scene_name in stmid_dataset.get_scene_list():
    # Download dataset from middlebury servers
    # will only download it if it hasn't already been downloaded
    print("Downloading data for scene '"+scene_name+"'...")
    stmid_dataset.download_scene_data(scene_name,dataset_folder)
    # Load scene data from downloaded folder
    print("Loading data for scene '"+scene_name+"'...")
    scene_data = stmid_dataset.load_scene_data(scene_name,dataset_folder,True,1)
    gt_disp_image = scene_data.disp_image
    # Demonstate evaluation by comparing the ground truth to itelf with a bit of noise
    noise = np.random.normal(0, 1.5, gt_disp_image.shape)
    test_disp_image = gt_disp_image + noise
    rmse = stmid_eval.rmse(gt_disp_image,test_disp_image)
    bad_pix_error = stmid_eval.bad_pix_error(gt_disp_image,test_disp_image)
    print("RMSE: {:.2f}".format(rmse))
    print("Bad pixel 2.0: {:.2f}%".format(bad_pix_error))
```