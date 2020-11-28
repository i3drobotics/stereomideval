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
for scenename in stmid_dataset.get_scene_list():
    # Download dataset from middlebury servers
    # will only download it if it hasn't already been downloaded
    print("Downloading data for scene '"+scenename+"'...")
    stmid_dataset.download_scene_data(scenename,DATASET_FOLDER)
    # Load scene data from downloaded folder
    print("Loading data for scene '"+scenename+"'...")
    scene_data = stmid_dataset.load_scene_data(scenename,DATASET_FOLDER,True)
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

## Developement
### Upcomming features
- Evaluation of disparity image compared to ground truth disparity
- Evaulation of depth image compared to ground truth depth for real-world error metrics

### Build
```
python -m pip install --user --upgrade twine wheel && python setup.py clean --all && python setup.py sdist bdist_wheel
```

### Upload to Test Pip
Test pip package is maintained by user: [i3DR](https://pypi.org/user/i3DR/)
```
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

### Upload to Pip
Pip package is maintained by user: [i3DR](https://pypi.org/user/i3DR/)
```
python -m twine upload dist/*
```