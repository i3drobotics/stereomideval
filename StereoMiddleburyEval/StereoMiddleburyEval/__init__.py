import os
import glob
import re
import numpy as np
import cv2
import wget
import sys
import zipfile

class StereoMiddleburyEval():
    def __init__(self):
        pass

    def disp_to_depth(self,disp,cal_filepath):
        #Sample calib.txt file for one of the full-size training image pairs:
        # cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
        # cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
        # doffs=131.111
        # baseline=193.001
        # width=2964
        # height=1988
        # ndisp=280
        # isint=0
        # vmin=31
        # vmax=257
        # dyavg=0.918
        # dymax=1.516

        # Check calibration file exists
        if (not os.path.exists(cal_filepath)):
            print("Calibration file not found")
            print(cal_filepath)
            raise Exception("Calibration file not found")
        
        # Open calibration file
        file = open(cal_filepath, 'rb')
        # Read first line (expected format: "cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]")
        cam0_line = file.readline().decode('utf-8').rstrip()
        # Read second line but ignore the data as cam0 and cam1 have the same parameters
        _ = file.readline().decode('utf-8').rstrip()
        # Read third line (expected format: "doffs=131.111")
        doffs_line = file.readline().decode('utf-8').rstrip()
        # Read third line (expected format: "baseline=193.001")
        baseline_line = file.readline().decode('utf-8').rstrip()

        # Read all numbers from cam0 line using regex
        nums = re.findall("\d+\.\d+", cam0_line)
        # Get camera parmeters from file data
        f = float(nums[0])
        cx = float(nums[1])
        cy = float(nums[3])

        # Get doffs and baseline from file data
        doffs = float(re.findall("\d+\.\d+", doffs_line)[0])
        baseline = float(re.findall("\d+\.\d+", baseline_line)[0])

        # Calculate depth from disparitiy
        Z = baseline * f / (disp + doffs)
        # Z is in mm, convert to meters
        depth = Z / 1000
        return depth

    def normalise_pfm_data(self,data,max_val_pct=0.1):
        # Normalise disparity pfm image
        norm_pfm_data = np.where(data == np.inf, -1, data)
        max_val = np.max(norm_pfm_data)
        max_val += max_val * max_val_pct
        norm_pfm_data = np.where(norm_pfm_data == -1, max_val, norm_pfm_data)
        return cv2.normalize(norm_pfm_data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    '''
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    '''
    def load_pfm(self,filepath):
        # Load pfm data from file

        # Check file exists
        if (os.path.exists(filepath)):
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
                raise Exception('Not a PFM file.')
            
            # Read dimensions from pfm file and check they match expected
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception('Malformed PFM header.')
                
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
        else:
            raise Exception('File does not exist')

    def load_scene_data(self,scene_folder,display_images=False):
        # Load scene data from scene folder

        # Get name of disparity image (pfm) in folder
        disp_filename = os.path.join(scene_folder,"disp0.pfm")
        # Check disparity file exists
        if (not os.path.exists(disp_filename)):
            print("Disparity pfm file does not exist")
            print(disp_filename)
            raise Exception("Disparity pfm file does not exist")
        # Load disparity file to numpy image
        disp_image, _ = self.load_pfm(disp_filename)
        if (display_images):
            # Display disparity image in opencv window
            norm_disp_image = self.normalise_pfm_data(disp_image)
            norm_disp_image_resize = cv2.resize(norm_disp_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', cv2.applyColorMap(norm_disp_image_resize, cv2.COLORMAP_JET))
            cv2.waitKey(1000)
        
        # Define calibration file in scene folder
        cal_file = os.path.join(scene_folder,"calib.txt")
        # Calculate depth image from disparity using calibration file
        depth_image = self.disp_to_depth(disp_image,cal_file)

        # Define left and right image files in scene folder
        left_image_filename = os.path.join(scene_folder,"im0.png")
        right_image_filename = os.path.join(scene_folder,"im1.png")
        # Check left and right image files exist
        if (not os.path.exists(left_image_filename) or not os.path.exists(right_image_filename)):
            print("Left or right image file does not exist")
            print(left_image_filename)
            print(right_image_filename)
            raise Exception("Left or right image file does not exist")
        
        # Read left and right image files to numpy image
        left_image = cv2.imread(left_image_filename,cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(right_image_filename,cv2.IMREAD_UNCHANGED)
        if (display_images):
            # Display left and right image files to OpenCV window
            left_image_resize = cv2.resize(left_image, dsize=(0, 0), fx=0.2, fy=0.2)
            right_image_resize = cv2.resize(right_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', left_image_resize)
            cv2.waitKey(1000)
            cv2.imshow('image', right_image_resize)
            cv2.waitKey(1000)

        return left_image,right_image,disp_image,depth_image

    def getSceneList(self):
        # Return full list of scenes available
        return [
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
            "Vintage"]

    def getURLfromScene(self,scene_name):
        # Get URL on middlebury servers for 2014 dataset for chosen scene
        return "http://vision.middlebury.edu/stereo/data/scenes2014/zip/"+scene_name+"-perfect.zip"

    #create this bar_progress method which is invoked automatically from wget
    def bar_progress(self,current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    def downloadDataset(self,scene_name,output_folder):
        # Check output folder exists
        if (os.path.exists(output_folder)):
            # Download dataset from middlebury servers
            # Get url from scene name
            url = self.getURLfromScene(scene_name)
            # Get name of scene data folder
            scene_output_folder = os.path.join(output_folder,scene_name)
            # Define destination name for zip file
            zip_filepath = os.path.join(output_folder,scene_name+".zip")
            # clean-up tmp files from incomplete downloads
            tmp_files = glob.glob(os.path.join(output_folder,"*tmp"))
            for tmp_file in tmp_files:
                os.remove(tmp_file)
            # Check scene folder doesn't already exist
            if (not os.path.exists(scene_output_folder)):
                # Check zip file doesn't already exist
                if (not os.path.exists(zip_filepath)):
                    print("Downloading from: "+url)
                    # download file from middlebury server
                    wget.download(url, zip_filepath, bar=self.bar_progress)
                else:
                    print("Zip file for dataset already exists here, skipping download of file: "+zip_filepath)
                print("Extracting zip...")
                # unzip downloaded file
                with zipfile.ZipFile(zip_filepath,"r") as zip_ref:
                    zip_ref.extractall(output_folder)
                print("Organising scene folder")
                # rename scene folder to remove '-perfect'
                os.rename(os.path.join(output_folder,scene_name+"-perfect"),scene_output_folder)
                # removing zip file
                os.remove(zip_filepath)
            else:
                print("Dataset already exists here, skipping re-download of "+scene_name)
        else:
            raise Exception('Output folder not found for storing datasets')

if __name__ == "__main__":
    dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to dowmload datasets
    display_images = True #Option to display dataset images as they are loaded

    # Create dataset folder
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Initalise StereoMiddleburyEval object
    stme = StereoMiddleburyEval()

    # Get list of scene in dataset (2014) and iterate through them
    for scenename in stme.getSceneList():
        # Download dataset from middlebury servers
        print("Downloading data for scene '"+scenename+"'...")
        stme.downloadDataset(scenename,dataset_folder) # will only download it if it hasn't already been downloaded
        # Load scene data from downloaded folder
        print("Loading data for scene '"+scenename+"'...")
        scene_folder = os.path.join(dataset_folder,scenename)
        left_image,right_image,disp_image,depth_image = stme.load_scene_data(scene_folder,display_images)