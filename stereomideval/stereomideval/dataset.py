"""Dataset module"""
import os
import sys
import zipfile
import re
import glob
import numpy as np
import cv2
import wget
from stereomideval.structures import DatasetType, CalibrationData, TestData, SceneInfo
from stereomideval.exceptions import PathNotFound, MalformedPFM, InvalidSceneName


class Dataset:
    """Download and read data from stereo Middlesbury dataset (2014)"""

    # List of available scenes in the Middlebury stereo dataset (2014)
    Adirondack = "Adirondack"
    Backpack = "Backpack"
    Bicycle1 = "Bicycle1"
    Books = "Books"
    Cable = "Cable"
    Classroom1 = "Classroom1"
    Couch = "Couch"
    Computer = "Computer"
    Cones = "Cones"
    Dolls = "Dolls"
    Drumsticks = "Drumsticks"
    Dwarves = "Dwarves"
    Flowers = "Flowers"
    Jadeplant = "Jadeplant"
    Laundry = "Laundry"
    Mask = "Mask"
    Moebius = "Moebius"
    Motorcycle = "Motorcycle"
    Piano = "Piano"
    Pipes = "Pipes"
    Playroom = "Playroom"
    Playtable = "Playtable"
    Reindeer = "Reindeer"
    Recycle = "Recycle"
    Shelves = "Shelves"
    Shopvac = "Shopvac"
    Sticks = "Sticks"
    Storage = "Storage"
    Sword1 = "Sword1"
    Sword2 = "Sword2"
    Umbrella = "Umbrella"
    Vintage = "Vintage"
    Art = "Art"
    Teddy = "Teddy"
    STEREO_MIDDLEBURY_SCENES_2014 = [
        Adirondack, Backpack, Bicycle1, Cable, Classroom1,
        Couch, Flowers, Jadeplant, Mask, Motorcycle,
        Piano, Pipes, Playroom, Playtable, Recycle,
        Shelves, Shopvac, Sticks, Storage,
        Sword1, Sword2, Umbrella, Vintage
    ]
    STEREO_MIDDLEBURY_SCENES_2003 = [
        Cones, Teddy
    ]
    STEREO_MIDDLEBURY_SCENES_2005 = [
        Art, Books, Computer, Dolls, Drumsticks,
        Dwarves, Laundry, Moebius, Reindeer
    ]

    STEREO_MIDDLEBURY_SCENES = STEREO_MIDDLEBURY_SCENES_2003 + \
        STEREO_MIDDLEBURY_SCENES_2005 + STEREO_MIDDLEBURY_SCENES_2014

    STEREO_MIDDLEBURY_TRAINING_SCENES = [
        SceneInfo(Adirondack, DatasetType.imperfect, 1.0),
        SceneInfo(Art, DatasetType.lighting_changed, 1.0),
        SceneInfo(Jadeplant, DatasetType.imperfect, 1.0),
        SceneInfo(Motorcycle, DatasetType.imperfect, 1.0),
        SceneInfo(Motorcycle, DatasetType.exposure_changed, 1.0),
        SceneInfo(Piano, DatasetType.imperfect, 1.0),
        SceneInfo(Piano, DatasetType.lighting_changed, 0.5),
        SceneInfo(Pipes, DatasetType.imperfect, 1.0),
        SceneInfo(Playroom, DatasetType.imperfect, 0.5),
        SceneInfo(Playtable, DatasetType.imperfect, 0.5),
        SceneInfo(Playtable, DatasetType.perfect, 1.0),
        SceneInfo(Recycle, DatasetType.imperfect, 1.0),
        SceneInfo(Shelves, DatasetType.imperfect, 0.5),
        SceneInfo(Teddy, DatasetType.imperfect, 1.0),
        SceneInfo(Vintage, DatasetType.imperfect, 0.5),
    ]

    # TODO: add 2014 test scenes (not in organised single zips like test data so requires more work)

    @staticmethod
    def normalise_pfm_data(data, max_val_pct=0.1):
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

    @staticmethod
    def load_cal(cal_filepath):
        """
        Load camera parameters from calibration file

        Parameters:
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
            raise PathNotFound(cal_filepath, "Calibration file not found")

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
        # Read 4th line (expected format: "width=2964")
        width_line = file.readline().decode('utf-8').rstrip()
        # Read 5th line (expected format: "height=19881")
        height_line = file.readline().decode('utf-8').rstrip()
        # Read 6th line (expected format: "ndisp=280")
        ndisp_line = file.readline().decode('utf-8').rstrip()

        # Read all numbers from cam0 line using regex
        nums = re.findall("\\d+\\.\\d+", cam0_line)
        # Get camera parmeters from file data
        cam0_f = float(nums[0])
        cam0_cx = float(nums[1])
        cam0_cy = float(nums[3])

        # Get doffs and baseline from file data
        doffs = float(re.findall("\\d+\\.\\d+", doffs_line)[0])
        baseline = float(re.findall("\\d+", baseline_line)[0])
        width = float(re.findall("\\d+", width_line)[0])
        height = float(re.findall("\\d+", height_line)[0])
        ndisp = float(re.findall("\\d+", ndisp_line)[0])

        cal_data = CalibrationData(width, height, cam0_cx, cam0_cy, cam0_f, doffs, baseline, ndisp)
        return cal_data

    @staticmethod
    def disp_to_depth(disp, focal_length, doffs, baseline):
        """
        Convert from disparity to depth using calibration file

        Parameters:
            disp (numpy): 2D disparity image (result of stereo matching)
            focal_length (float): Focal length of left camera (in pixels)
            doffs (float): x-difference of principal points, doffs = cx1 - cx0
            baseline (float): Baseline distance between left and right camera (in mm)

        Returns:
            depth (numpy): 2D depth image (units meters)
        """
        # Calculate depth from disparitiy
        # Z = baseline * f / (disp + doeff)
        z_mm = baseline * focal_length / (disp + doffs)
        # Z is in mm, convert to meters
        depth = z_mm / 1000
        return depth

    @staticmethod
    def load_pfm(filepath):
        """
        Load pfm data from file

        Parameters:
            filepath (string): filepath to pfm image (e.g. image.pfm)

        Returns:
            pfm_data (numpy): 2D image filled with data from pfm file
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise PathNotFound(filepath, "Pfm file does not exist")
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
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        # Read image data from pfm file
        data = np.fromfile(file, endian + 'f')
        # Format image data into expected numpy image
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        # Flip image vertically as image appears upside-down
        data = cv2.flip(data, 0)
        return data, scale

    @staticmethod
    def load_scene_disparity(scene_name, dataset_folder, display_images=False,
                             display_time=500, load_perfect=False):
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
        scene_year = Dataset.get_scene_year(scene_name)
        if scene_year == "2014":
            perfect_suffix = Dataset.get_perfect_suffix(load_perfect)
            # Get name of disparity image (pfm) in folder
            disp_filename = os.path.join(dataset_folder, scene_name+perfect_suffix, "disp0.pfm")
        elif scene_year == "2003":
            disp_filename = os.path.join(dataset_folder, scene_name, "disp2.pgm")
        elif scene_year == "2005":
            disp_filename = os.path.join(dataset_folder, scene_name, "disp1.png")
        # Check disparity file exists
        if not os.path.exists(disp_filename):
            print("Disparity pfm file does not exist")
            print(disp_filename)
            raise PathNotFound(disp_filename, "Disparity pfm file does not exist")
        # Load disparity file to numpy image
        if scene_year == "2014":
            disp_image, _ = Dataset.load_pfm(disp_filename)
        elif scene_year == "2005":
            disp_image = cv2.imread(disp_filename, cv2.IMREAD_UNCHANGED)
        elif scene_year == "2003":
            disp_image = cv2.imread(disp_filename, cv2.IMREAD_UNCHANGED)
            #orig_dtype = disp_image.dtype
            #disp_image = disp_image.astype(np.float32)
            #disp_image /= 4
            #disp_image = disp_image.astype(orig_dtype)
        if display_images:
            # Display disparity image in opencv window
            norm_disp_image = Dataset.normalise_pfm_data(disp_image)
            norm_disp_image_resize = cv2.resize(norm_disp_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', cv2.applyColorMap(norm_disp_image_resize, cv2.COLORMAP_JET))
            cv2.waitKey(display_time)
        return disp_image

    @staticmethod
    def get_perfect_suffix(dataset_type):
        """Get perfect suffix"""
        perfect_suffix = "-imperfect"
        if dataset_type == DatasetType.perfect:
            perfect_suffix = "-perfect"
        return perfect_suffix

    @staticmethod
    def get_image_suffix(dataset_type):
        """Get image suffix"""
        image_suffix = ""
        if dataset_type in DatasetType.SCENE_CHANGED_TYPES:
            image_suffix = dataset_type
        return image_suffix

    @staticmethod
    def load_scene_stereo_pair(scene_name, dataset_folder, display_images=False, display_time=500,
                               dataset_type=DatasetType.imperfect):
        """
        Load stereo pair images from scene folder

        Parameters:
            scene_name (string): Scene name to load data (use 'get_scene_list()'
                to see possible scenes) Must have already been downloaded to the
                dataset foldering using 'download_scene_data()'
            dataset_folder (string): Path to folder where dataset has been downloaded
            display_image (bool): Optional. Should the scene data be displayed when it is loaded.
            display_time (int): Optional. Length of time to display each image once loaded.
            image_suffix (string): Optional. Addition suffix to load alternate left views
                                                'E' means exposure changed between views
                                                'L' means lighting changed between views

        Returns:
            left_image (numpy): 2D image from left camera, loaded from scene data
            right_image (numpy): 2D image from right camera, loaded from scene data
        """
        scene_year = Dataset.get_scene_year(scene_name)
        perfect_suffix = ""
        if scene_year == "2014":
            perfect_suffix = Dataset.get_perfect_suffix(dataset_type)
            left_image_filename = "im0.png"
            right_image_filename = "im1{}.png".format(Dataset.get_image_suffix(dataset_type))
        elif scene_year == "2003":
            left_image_filename = "im2.ppm"
            right_image_filename = "im6.ppm"
        elif scene_year == "2005":
            left_image_filename = "view1.png"
            right_image_filename = "view5.png"
        
        # Define left and right image files in scene folder
        left_image_filepath = os.path.join(dataset_folder,
                                           scene_name+perfect_suffix,
                                           left_image_filename)
        right_image_filepath = os.path.join(dataset_folder,
                                            scene_name+perfect_suffix,
                                            right_image_filename)
        # Check left and right image files exist
        if not os.path.exists(left_image_filepath) or not os.path.exists(right_image_filepath):
            print("Left or right image file does not exist")
            print(left_image_filepath)
            print(right_image_filepath)
            raise PathNotFound(
                left_image_filepath+","+right_image_filepath,
                "Left or right image file does not exist")
        # Read left and right image files to numpy image
        left_image = cv2.imread(left_image_filepath, cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(right_image_filepath, cv2.IMREAD_UNCHANGED)
        if display_images:
            # Display left and right image files to OpenCV window
            left_image_resize = cv2.resize(left_image, dsize=(0, 0), fx=0.2, fy=0.2)
            right_image_resize = cv2.resize(right_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', left_image_resize)
            cv2.waitKey(display_time)
            cv2.imshow('image', right_image_resize)
            cv2.waitKey(display_time)
        return left_image, right_image

    @staticmethod
    def load_scene_data(scene_name, dataset_folder, display_images=False,
                        display_time=500, dataset_type=DatasetType.imperfect):
        """Load scene data from scene folder

        Parameters:
            scene_name (string): Scene name to load data (use 'get_scene_list()'
                to see possible scenes) Must have already been downloaded to the
                dataset foldering using 'download_scene_data()'
            dataset_folder (string): Path to folder where dataset has been downloaded
            display_image (bool): Optional. Should the scene data be displayed when it is loaded.
            display_time (int): Optional. Length of time to display each image once loaded.

        Returns:
            scene_data (TestData): Data loaded from scene folder
                (see TestData for details on this structure)
        """
        InvalidSceneName.validate_scene_list(scene_name, Dataset.get_scene_list())
        # Load stereo pair images from scene folder
        left_image, right_image = Dataset.load_scene_stereo_pair(
            scene_name, dataset_folder, display_images, display_time,
            dataset_type)

        # Load disparity image from scene folder
        disp_image = Dataset.load_scene_disparity(
            scene_name, dataset_folder, display_images, display_time,
            dataset_type)

        scene_year = Dataset.get_scene_year(scene_name)
        if scene_year == "2014":
            # Define path to calibration file in scene folder
            perfect_suffix = Dataset.get_perfect_suffix(dataset_type)
            cal_file = os.path.join(dataset_folder, scene_name+perfect_suffix, "calib.txt")
            # Get calibration data from calibration file
            cal_data = Dataset.load_cal(cal_file)
            ndisp = cal_data.ndisp
            # Calculate depth image from disparity using calibration file
            depth_image = Dataset.disp_to_depth(disp_image, cal_data.focal_length,
                                                cal_data.doffs, cal_data.baseline)
        elif scene_year == "2003":
            depth_image = None
            ndisp = 256 # 256*4 (scaling factor of 4 for 2003 dataset)
        else:
            depth_image = None
            # TODO: replace this with list of files that don't have cal data on website
            # however all images in training dataset without calibration files have a
            # ndisp = 256
            ndisp = 256

        return TestData(left_image, right_image, disp_image, depth_image, ndisp)

    @staticmethod
    def get_scene_list():
        """Return full list of scenes available"""
        return Dataset.STEREO_MIDDLEBURY_SCENES

    @staticmethod
    def get_training_scene_list():
        """Returns list of training scenes aviablable"""
        return Dataset.STEREO_MIDDLEBURY_TRAINING_SCENES

    @staticmethod
    def get_url_from_scene(scene_name, dataset_type=DatasetType.imperfect):
        """
        Get URL on middlebury servers for 2014 dataset for chosen scene

        Parameters:
            scene_name (string): Scene name to get url of

        Returns:
            url (string): Url for chosen scene name in middlebury stereo dataset (2014)
        """
        scene_year = Dataset.get_scene_year(scene_name)
        if scene_year == "2014":
            base_url = "http://vision.middlebury.edu/stereo/data/scenes2014/zip/"
            perfect_suffix = Dataset.get_perfect_suffix(dataset_type)
            url = base_url+scene_name+perfect_suffix+".zip"
        elif scene_year == "2003":
            #scene_name = scene_name.lower()
            url = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/full/"
            #url_suffix = "newdata/{}/{}-png-2.zip"
            #url = base_url+url_suffix.format(scene_name, scene_name)
        elif scene_year == "2005":
            # Images are not stored in zip file for this year.
            # Will return base url for this scene instead
            url = "https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/Art/"
        return url

    @staticmethod
    def bar_progress(current, total, *_):
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

    @staticmethod
    def get_scene_year(scene_name):
        """
        Get dataset year scene is from (2003, 2005, or 2014)

        Parameters:
            scene_name (string): scene name in dataset

        Return:
            scene_year (string): dataset year the scene is from (2003, 2005, or 2014)
        """
        if scene_name in Dataset.STEREO_MIDDLEBURY_SCENES_2014:
            return "2014"
        if scene_name in Dataset.STEREO_MIDDLEBURY_SCENES_2005:
            return "2005"
        if scene_name in Dataset.STEREO_MIDDLEBURY_SCENES_2003:
            return "2003"
        raise InvalidSceneName(scene_name)

    @staticmethod
    def download_scene_2005_data(scene_name, output_folder, url):
        """
        Download scene data for 2005 Middlebury data

        Parameters:
            scene_name (string): Scene to download from Middlesbury stereo dataset (2014)
            output_folder (string): Path to download scene data
            url (string): url to download from
        """
        scene_output_folder = os.path.join(output_folder, scene_name)
        # Check scene folder doesn't already exist
        if not os.path.exists(scene_output_folder):
            # Create scene folder
            os.makedirs(scene_output_folder)
            img0_filepath = os.path.join(scene_output_folder, "view1.png")
            img1_filepath = os.path.join(scene_output_folder, "view5.png")
            disp_filepath = os.path.join(scene_output_folder, "disp1.png")
            # TODO: find cal data
            filepaths = [img0_filepath, img1_filepath, disp_filepath]
            base_url = url
            urls = ["Illum1/Exp1/view1.png", "Illum2/Exp1/view5.png", "disp1.png"]
            urls = [base_url + url for url in urls]
            print("Downloading from: "+base_url)
            for index, url in enumerate(urls):
                filepath = filepaths[index]
                # Check file doesn't already exist
                if not os.path.exists(filepath):
                    # download file from middlebury server
                    wget.download(url, filepath, bar=Dataset.bar_progress)
                else:
                    msg = "Image file for dataset already exists here,"
                    msg += "skipping download of file: " + filepath
                    print(msg)
        else:
            print("Dataset already exists here, skipping re-download of "+scene_name)

    @staticmethod
    def download_scene_2003_data(scene_name, output_folder, url):
        """
        Download scene data for 2003 Middlebury data

        Parameters:
            scene_name (string): Scene to download from Middlesbury stereo dataset (2014)
            output_folder (string): Path to download scene data
            url (string): url to download from
        """
        scene_output_folder = os.path.join(output_folder, scene_name)
        # Check scene folder doesn't already exist
        if not os.path.exists(scene_output_folder):
            # Create scene folder
            #os.makedirs(scene_output_folder)
            #img0_filepath = os.path.join(scene_output_folder, "im2.pgm")
            #img1_filepath = os.path.join(scene_output_folder, "im6.pgm")
            #disp_filepath = os.path.join(scene_output_folder, "disp2.ppm")
            # TODO: find cal data
            base_url = url
            scene_name = scene_name.lower()
            zip_filename = "{}F-ppm-2.zip".format(scene_name)
            zip_filepath = os.path.join(output_folder,zip_filename)
            zip_url = base_url+zip_filename
            # download file from middlebury server
            print("Downloading from: "+zip_url)
            wget.download(zip_url, zip_filepath, bar=Dataset.bar_progress)

            # unzip downloaded file
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(output_folder)
            # removing zip file
            os.remove(zip_filepath)

            os.rename(os.path.join(output_folder,"teddyF"),scene_output_folder)
        else:
            print("Dataset already exists here, skipping re-download of "+scene_name)

    @staticmethod
    def download_scene_2014_dataset(scene_name, output_folder, url, dataset_type):
        """
        Download scene data for 2014 Middlebury data

        Parameters:
            scene_name (string): Scene to download from Middlesbury stereo dataset (2014)
            output_folder (string): Path to download scene data
            url (string): url to download from
            dataset_type (DatasetType): used to get scene suffix folder naming
        """
        # Get perfect suffix from dataset ('-imperfect' or '-perfect')
        scene_suffix = Dataset.get_perfect_suffix(dataset_type)
        # Download dataset from middlebury servers
        # Get name of scene data folder
        scene_output_folder = os.path.join(output_folder, scene_name+scene_suffix)
        # Define destination name for zip file
        zip_filepath = os.path.join(output_folder, scene_name+scene_suffix+".zip")
        # Check scene folder doesn't already exist
        if not os.path.exists(scene_output_folder):
            # Check zip file doesn't already exist
            if not os.path.exists(zip_filepath):
                print("Downloading from: "+url)
                # download file from middlebury server
                wget.download(url, zip_filepath, bar=Dataset.bar_progress)
            else:
                msg = "Zip file for dataset already exists here,"
                msg += "skipping download of file: " + zip_filepath
                print(msg)
            print("Extracting zip...")
            # unzip downloaded file
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(output_folder)
            # removing zip file
            os.remove(zip_filepath)
        else:
            print("Dataset already exists here, skipping re-download of "+scene_name)

    @staticmethod
    def download_scene_data(scene_name, output_folder, dataset_type=DatasetType.imperfect):
        """
        Download scene data

        Parameters:
            scene_name (string): Scene to download from Middlesbury stereo dataset (2014)
            output_folder (string): Path to download scene data
        """
        # Check output folder exists
        if os.path.exists(output_folder):
            # clean-up tmp files from incomplete downloads
            tmp_files = glob.glob(os.path.join(output_folder, "*tmp"))
            for tmp_file in tmp_files:
                os.remove(tmp_file)
            # Get url from scene name
            url = Dataset.get_url_from_scene(scene_name, dataset_type)
            # get scene year (will change how data is downloaded)
            scene_year = Dataset.get_scene_year(scene_name)
            if scene_year == "2014":
                Dataset.download_scene_2014_dataset(scene_name, output_folder,
                                                    url, dataset_type)
            elif scene_year == "2005":
                Dataset.download_scene_2005_data(scene_name, output_folder, url)
            elif scene_year == "2003":
                Dataset.download_scene_2003_data(scene_name, output_folder, url)

        else:
            raise Exception('Output folder not found for storing datasets')
