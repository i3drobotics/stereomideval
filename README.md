# stereo-mideval
Python package for evaluation dataset and tools from the [Middlebury stereo evaulation 2014](https://vision.middlebury.edu/stereo/data/scenes2014/) dataset.
This project is in development by [I3DR](https://i3drobotics.com/) for evaluation stereo matching algorithms for use in stereo cameras. However this is project is fully open-source with no limitations to encorage and support others who may need access to this tools. 

## Install
```
pip install stereo-mideval
```

# Features
- Download scene data from Middlebury servers
- Load disparity image and stereo pair from scene data
- Display normalised colormaped disparity image
- Convert disparity image to depth image using calibration file from scene data

# Examples
## Download and display scene data
```python
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