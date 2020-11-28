"""
Stereo Middlebury Evaluation package

This module is for loading the stereo Middlebury dataset loading
and includes tools evaluatating stereo matching algorithms
"""
import os
import glob
import re
import zipfile
import sys
import numpy as np
import cv2
import wget
import math

# List of available scenes in the Middlebury stereo dataset (2014)
STEREO_MIDDLEBURY_SCENES = [
    "Adirondack",
    "Backpack",
    "Bicycle1",
    "Cable",
    "Classroom1",
    "Couch",
    "Flowers",
    "Jadeplant",
    "Mask",
    "Motorcycle",
    "Piano",
    "Pipes",
    "Playroom",
    "Playtable",
    "Recycle",
    "Shelves",
    "Shopvac",
    "Sticks",
    "Storage",
    "Sword1",
    "Sword2",
    "Umbrella",
    "Vintage"
]

# Exceptions
class ImageSizeNotEqual(Exception):
    """Image size not equal exception"""
    def __str__(self):
        return "Image sizes must be equal"

class InvalidSceneName(Exception):
    """Invalid scene exception"""
    def __str__(self):
        return "Invalid scene name. Use getSceneList() to get full list of scenes available"

class MalformedPFM(Exception):
    """Malformed PFM file exception"""
    def __init__(self, message="Malformed PFM file"):
        """
        Exception handelling for malformed PFM file
        Parameters:
            message (string): Message to display in exception,
                will use default message if no message is provided
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """Overload of exception message"""
        return self.message

class PathNotFound(Exception):
    """Path not found exception"""
    def __init__(self, filename, message="Path does not exist: "):
        """
        Exception handelling for path not found
        Parameters:
            filename (string): Filepath that was not found.
                Will be displayed in exception message
            message (string): Message to display in exception,
                will use default message if no message is provided
        """
        self.filename = filename
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """Overload of exception message"""
        return self.message+self.filename

class Eval:
    """Evaluate disparity image against ground truth"""
    def __init__(self):
        pass

    def diff(self, ground_truth, test_data):
        """
        Difference between ground truth and test data

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth
        Returns:
            diff (numpy): test data subtracted from ground truth
        """
        # Check images are the same size
        if test_data.shape != ground_truth.shape:
            raise ImageSizeNotEqual()
        # Replace nan and inf values with zero
        test_data_no_nan = np.nan_to_num(test_data, nan=0.0,posinf=0.0,neginf=0.0)
        ground_truth_no_nan = np.nan_to_num(ground_truth, nan=0.0,posinf=0.0,neginf=0.0)
        # Subtract test data from ground truth to find difference
        diff = np.subtract(test_data_no_nan,ground_truth_no_nan)
        return diff

    def bad_pix_error(self,ground_truth,test_data,threshold=2.0):
        """
        Bad pixel error

        Calcuate percentage of bad pixels between test data and ground truth
        A bad pixel is defined as a disparity with an error larger than the threshold

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth
            threshold (float): Threshold below which to classify a bad pixel

        Returns:
            perc_bad (float): Bad pixel percentage error in test data.
        """
        if test_data.shape != ground_truth.shape:
            raise ImageSizeNotEqual()
        # Calculate pixel difference between ground truth and test data
        diff = self.diff(ground_truth,test_data)
        # Get the absolute difference (positive only)
        abs_diff = np.abs(diff)
        # Count number of 'bad' pixels
        bad_count = (~(abs_diff < threshold)).sum()
        # Convert number of 'bad' pixels to percentage
        total_size = ground_truth.shape[0] * ground_truth.shape[1]
        perc_bad = (bad_count/total_size)*100
        return perc_bad

    def rmse(self, ground_truth, test_data):
        """
        Root mean squared error

        Calculate root mean squared error between test data and ground truth

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth

        Returns:
            err (float): Root mean squared error of two images,
                the lower the error, the more "similar" the two images
        """
        if test_data.shape != ground_truth.shape:
            raise ImageSizeNotEqual()
        rmse = math.sqrt(self.mse(ground_truth,test_data))
        return rmse

    def mse(self, ground_truth, test_data):
        """
        Mean squared error

        Calculate the mean squared error between test data and ground truth

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth

        Returns:
            err (float): Mean squared error of two images,
                the lower the error, the more "similar" the two images
        """
        if test_data.shape != ground_truth.shape:
            raise ImageSizeNotEqual()
        # Calculate difference between ground truth and test data (gt-td)
        diff = self.diff(ground_truth,test_data)
        # Calculate MSE
        err = np.square(diff).mean()
        return err

class SceneData:
    """
    Scene data

    Used to make returning and accessing scene data simple.
    """
    def __init__(self,left_image,right_image,disp_image,depth_image):
        """
        Initalisaiton of SceneData structure

        Parameters:
            left_image (numpy): 2D image from left camera in
                stereo pair
            right_image (numpy): 2D image from right camera in
                stereo pair
            disp_image (numpy): 2D disparity image (result of stereo matching)
            depth_image (numpy): 2D depth image (units: meters)
        """
        self.left_image = left_image
        self.right_image = right_image
        self.disp_image = disp_image
        self.depth_image = depth_image

class Dataset:
    """Download and read data from stereo Middlesbury dataset (2014)"""
    def __init__(self):
        pass

    def normalise_pfm_data(self,data,max_val_pct=0.1):
        """
        Normalise disparity pfm image

        Parameters:
            data (numpy): 2D pfm image data to normalise
            max_val_pct (float): TODO what is this?

        Returns:
            norm_pfm_data (numpy): 2D normalised pfm image data
        """
        norm_pfm_data = np.where(data == np.inf, -1, data)
        max_val = np.max(norm_pfm_data)
        max_val += max_val * max_val_pct
        norm_pfm_data = np.where(norm_pfm_data == -1, max_val, norm_pfm_data)
        norm_pfm_data = cv2.normalize(
            norm_pfm_data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        norm_pfm_data = norm_pfm_data.astype(np.uint8)
        return norm_pfm_data

    def disp_to_depth(self,disp,cal_filepath):
        """
        Convert from disparity to depth using calibration file

        Parameters:
            disp (numpy): 2D disparity image (result of stereo matching)
            cal_filepath (string): filepath to calibration file (usually calib.txt)
                Expected format:
                    cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
                    cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
                    doffs=131.111
                    baseline=193.001
                    width=2964
                    height=1988
                    ndisp=280
                    isint=0
                    vmin=31
                    vmax=257
                    dyavg=0.918
                    dymax=1.516

        Returns:
            depth (numpy): 2D depth image (units meters)
        """

        # Check calibration file exists
        if not os.path.exists(cal_filepath):
            print("Calibration file not found")
            print(cal_filepath)
            raise PathNotFound(cal_filepath,"Calibration file not found")

        # Open calibration file
        file = open(cal_filepath, 'rb')
        # Read first line
        # expected format: "cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]"
        cam0_line = file.readline().decode('utf-8').rstrip()
        # Read second line but ignore the data as cam0 and cam1 have the same parameters
        _ = file.readline().decode('utf-8').rstrip()
        # Read third line (expected format: "doffs=131.111")
        doffs_line = file.readline().decode('utf-8').rstrip()
        # Read third line (expected format: "baseline=193.001")
        baseline_line = file.readline().decode('utf-8').rstrip()

        # Read all numbers from cam0 line using regex
        nums = re.findall("\\d+\\.\\d+", cam0_line)
        # Get camera parmeters from file data
        cam0_f = float(nums[0])
        #cam0_cx = float(nums[1])
        #cam0_cy = float(nums[3])

        # Get doffs and baseline from file data
        doffs = float(re.findall("\\d+\\.\\d+", doffs_line)[0])
        baseline = float(re.findall("\\d+\\.\\d+", baseline_line)[0])

        # Calculate depth from disparitiy
        # Z = baseline * f / (disp + doeff)
        z_mm = baseline * cam0_f / (disp + doffs)
        # Z is in mm, convert to meters
        depth = z_mm / 1000
        return depth

    def load_pfm(self,filepath):
        """
        Load pfm data from file

        Parameters:
            filepath (string): filepath to pfm image (e.g. image.pfm)

        Returns:
            pfm_data (numpy): 2D image filled with data from pfm file
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise PathNotFound(filepath,"Pfm file does not exist")
        # Open pfm file
        file = open(filepath, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        # Read header to check pf type
        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise MalformedPFM('Not a PFM file.')

        # Read dimensions from pfm file and check they match expected
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise MalformedPFM('Malformed PFM header')

        # Read data scale from file
        scale = float(file.readline().decode('utf-8').rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        # Read image data from pfm file
        data = np.fromfile(file, endian + 'f')
        # Format image data into expected numpy image
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        # Flip image vertically as image appears upside-down
        data = cv2.flip(data,0)
        return data, scale

    def load_scene_disparity(self,scene_name,dataset_folder,display_images=False,display_time=500):
        """
        Load disparity image from scene folder

        Parameters:
            scene_name (string): Scene name to load data (use 'get_scene_list()'
                to see possible scenes) Must have already been downloaded to the
                dataset foldering using 'download_scene_data()'
            dataset_folder (string): Path to folder where dataset has been downloaded
            display_image (bool): Optional. Should the scene data be displayed when it is loaded.
            display_time (int): Optional. Length of time to display each image once loaded.

        Returns:
            disp_image (numpy): 2D disparity image
                loaded from scene data (result of stereo matching)
        """
        # Get name of disparity image (pfm) in folder
        disp_filename = os.path.join(dataset_folder,scene_name,"disp0.pfm")
        # Check disparity file exists
        if not os.path.exists(disp_filename):
            print("Disparity pfm file does not exist")
            print(disp_filename)
            raise PathNotFound(disp_filename,"Disparity pfm file does not exist")
        # Load disparity file to numpy image
        disp_image, _ = self.load_pfm(disp_filename)
        if display_images:
            # Display disparity image in opencv window
            norm_disp_image = self.normalise_pfm_data(disp_image)
            norm_disp_image_resize = cv2.resize(norm_disp_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', cv2.applyColorMap(norm_disp_image_resize, cv2.COLORMAP_JET))
            cv2.waitKey(display_time)
        return disp_image

    def load_scene_stereo_pair(self,
            scene_name,dataset_folder,display_images=False,display_time=500):
        """
        Load stereo pair images from scene folder

        Parameters:
            scene_name (string): Scene name to load data (use 'get_scene_list()'
                to see possible scenes) Must have already been downloaded to the
                dataset foldering using 'download_scene_data()'
            dataset_folder (string): Path to folder where dataset has been downloaded
            display_image (bool): Optional. Should the scene data be displayed when it is loaded.
            display_time (int): Optional. Length of time to display each image once loaded.

        Returns:
            left_image (numpy): 2D image from left camera, loaded from scene data
            right_image (numpy): 2D image from right camera, loaded from scene data
        """
        # Define left and right image files in scene folder
        left_image_filename = os.path.join(dataset_folder,scene_name,"im0.png")
        right_image_filename = os.path.join(dataset_folder,scene_name,"im1.png")
        # Check left and right image files exist
        if not os.path.exists(left_image_filename) or not os.path.exists(right_image_filename):
            print("Left or right image file does not exist")
            print(left_image_filename)
            print(right_image_filename)
            raise PathNotFound(
                left_image_filename+","+right_image_filename,
                "Left or right image file does not exist")
        # Read left and right image files to numpy image
        left_image = cv2.imread(left_image_filename,cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(right_image_filename,cv2.IMREAD_UNCHANGED)
        if display_images:
            # Display left and right image files to OpenCV window
            left_image_resize = cv2.resize(left_image, dsize=(0, 0), fx=0.2, fy=0.2)
            right_image_resize = cv2.resize(right_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', left_image_resize)
            cv2.waitKey(display_time)
            cv2.imshow('image', right_image_resize)
            cv2.waitKey(display_time)
        return left_image, right_image

    def load_scene_data(self,scene_name,dataset_folder,display_images=False,display_time=500):
        """Load scene data from scene folder

        Parameters:
            scene_name (string): Scene name to load data (use 'get_scene_list()'
                to see possible scenes) Must have already been downloaded to the
                dataset foldering using 'download_scene_data()'
            dataset_folder (string): Path to folder where dataset has been downloaded
            display_image (bool): Optional. Should the scene data be displayed when it is loaded.
            display_time (int): Optional. Length of time to display each image once loaded.

        Returns:
            scene_data (SceneData): Data loaded from scene folder
                (see SceneData for details on this structure)
        """
        if self.check_valid_scene_name(scene_name):
            # Load disparity image from scene folder
            disp_image = self.load_scene_disparity(
                scene_name,dataset_folder,display_images,display_time)

            # Define path to calibration file in scene folder
            cal_file = os.path.join(dataset_folder,scene_name,"calib.txt")
            # Calculate depth image from disparity using calibration file
            depth_image = self.disp_to_depth(disp_image,cal_file)

            # Load stereo pair images from scene folder
            left_image, right_image = self.load_scene_stereo_pair(
                scene_name,dataset_folder,display_images,display_time)

            return SceneData(left_image,right_image,disp_image,depth_image)
        raise InvalidSceneName()

    def get_scene_list(self):
        """Return full list of scenes available"""
        return STEREO_MIDDLEBURY_SCENES

    def get_url_from_scene(self,scene_name):
        """
        Get URL on middlebury servers for 2014 dataset for chosen scene

        Parameters:
            scene_name (string): Scene name to get url of

        Returns:
            url (string): Url for chosen scene name in middlebury stereo dataset (2014)
        """
        if not self.check_valid_scene_name(scene_name):
            raise InvalidSceneName()
        base_url = "http://vision.middlebury.edu/stereo/data/scenes2014/zip/"
        url = base_url+scene_name+"-perfect.zip"
        return url

    def bar_progress(self, current, total, *_):
        """
        Progress bar to display download progress in wget

        Parameters:
            current (int): current byte count
            total (int): total number of bytes
            *_: required by wget but is ignored in this function
        """
        base_progress_msg = "Downloading: %d%% [%d / %d] bytes"
        progress_message = base_progress_msg % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    def check_valid_scene_name(self,scene_name):
        """
        Check scene name is avaiable for download

        Parameters:
            scene_name (string): scene name to check validity

        Returns:
            valid (bool): returns existence of scene name in avaiable scene list
        """
        return scene_name in self.get_scene_list()

    def download_scene_data(self,scene_name,output_folder):
        """
        Download scene data

        Parameters:
            scene_name (string): Scene to download from Middlesbury stereo dataset (2014)
            output_folder (string): Path to download scene data
        """
        if self.check_valid_scene_name(scene_name):
            # Check output folder exists
            if os.path.exists(output_folder):
                # Download dataset from middlebury servers
                # Get url from scene name
                url = self.get_url_from_scene(scene_name)
                # Get name of scene data folder
                scene_output_folder = os.path.join(output_folder,scene_name)
                # Define destination name for zip file
                zip_filepath = os.path.join(output_folder,scene_name+".zip")
                # clean-up tmp files from incomplete downloads
                tmp_files = glob.glob(os.path.join(output_folder,"*tmp"))
                for tmp_file in tmp_files:
                    os.remove(tmp_file)
                # Check scene folder doesn't already exist
                if not os.path.exists(scene_output_folder):
                    # Check zip file doesn't already exist
                    if not os.path.exists(zip_filepath):
                        print("Downloading from: "+url)
                        # download file from middlebury server
                        wget.download(url, zip_filepath, bar=self.bar_progress)
                    else:
                        msg = "Zip file for dataset already exists here,"
                        msg+= "skipping download of file: "+zip_filepath
                        print(msg)
                    print("Extracting zip...")
                    # unzip downloaded file
                    with zipfile.ZipFile(zip_filepath,"r") as zip_ref:
                        zip_ref.extractall(output_folder)
                    # rename scene folder to remove '-perfect'
                    os.rename(os.path.join(output_folder,scene_name+"-perfect"),scene_output_folder)
                    # removing zip file
                    os.remove(zip_filepath)
                else:
                    print("Dataset already exists here, skipping re-download of "+scene_name)
            else:
                raise Exception('Output folder not found for storing datasets')
        else:
            raise InvalidSceneName()
        