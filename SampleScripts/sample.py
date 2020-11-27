import StereoMiddleburyEval
import os

if __name__ == "__main__":
    dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to dowmload datasets
    display_images = True #Option to display dataset images as they are loaded

    # Create dataset folder
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Initalise StereoMiddleburyEval object
    stme = StereoMiddleburyEval.StereoMiddleburyEval()

    # Get list of scene in dataset (2014) and iterate through them
    for scenename in stme.getSceneList():
        # Download dataset from middlebury servers
        print("Downloading data for scene '"+scenename+"'...")
        stme.downloadDataset(scenename,dataset_folder) # will only download it if it hasn't already been downloaded
        # Load scene data from downloaded folder
        print("Loading data for scene '"+scenename+"'...")
        scene_folder = os.path.join(dataset_folder,scenename)
        left_image,right_image,disp_image,depth_image = stme.load_scene_data(scene_folder,display_images)