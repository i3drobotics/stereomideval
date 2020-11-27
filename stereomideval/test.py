import os
from stereomideval import Dataset, Eval

if __name__ == "__main__":
    dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to dowmload datasets

    # Create dataset folder
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Initalise Dataset object
    stmid_dataset = Dataset()

    # Get list of scene in dataset (2014) and iterate through them
    for scenename in stmid_dataset.get_scene_list():
        # Download dataset from middlebury servers
        # will only download it if it hasn't already been downloaded
        print("Downloading data for scene '"+scenename+"'...")
        stmid_dataset.download_dataset(scenename,dataset_folder) 
        # Load scene data from downloaded folder
        print("Loading data for scene '"+scenename+"'...")
        left_image,right_image,disp_image,depth_image = stmid_dataset.load_scene_data(scenename,dataset_folder,True)
        # Demonstate evaluation by comparing the ground truth to itself
        stmid_eval = Eval(disp_image,disp_image)
        stmid_eval.evaluate()