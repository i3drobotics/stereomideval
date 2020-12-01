"""This module is used for manually testing functionality while developing stereomideval module"""
import os
import shutil
import json
import numpy as np
import cv2
from stereomideval.dataset import Dataset, DatasetType
from stereomideval.eval import Eval, Timer

def run():
    dataset_folder = os.path.join(os.getcwd(),"datasets") #Path to download datasets
    eval_folder = os.path.join(os.getcwd(),"evaluation") #Path to store evaluation
    alg_name = "TEST" #Name of your stereo algorithm

    # Create dataset folder
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Create eval folder
    if os.path.exists(eval_folder):
        shutil.rmtree(eval_folder)
    os.makedirs(eval_folder)

    all_evals = {}
    # Get list of scenes in Milddlebury's stereo dataset (2014) and iterate through them
    for scene_list in Dataset.get_training_scene_list():
        scene_name=scene_list['name']
        dataset_type=scene_list['dataset_type']
        # Download dataset from middlebury servers
        # will only download it if it hasn't already been downloaded
        print("Downloading data for scene '"+scene_name+"'...")
        Dataset.download_scene_data(scene_name,dataset_folder,dataset_type)
        # Load scene data from downloaded folder
        print("Loading data for scene '"+scene_name+"'...")
        scene_data = Dataset.load_scene_data(
            scene_name=scene_name,dataset_folder=dataset_folder,
            dataset_type=dataset_type)
        left_image = scene_data.left_image
        ground_truth_disp_image = scene_data.disp_image
        # Start timer
        timer = Timer()
        timer.start()
        # Simluate match result by adding a bit of noise to the ground truth
        noise = np.random.uniform(low=0, high=1.0, size=ground_truth_disp_image.shape)
        test_disp_image = ground_truth_disp_image + noise
        # Record elapsed time for match
        elapsed_time = timer.elapsed()
        # Evaluate test data against all metrics
        evals = Eval.eval_all(
            ground_truth=ground_truth_disp_image,test_data=test_disp_image,
            elapsed_time=elapsed_time,
            scene_name=scene_name,dataset_type=dataset_type,dense=False)
        print(evals)

        scene_eval_name = scene_name
        if dataset_type != DatasetType.I:
            scene_eval_name = scene_name+dataset_type

        # Create scene evaluation folder
        scene_eval_folder = os.path.join(eval_folder,scene_eval_name)
        if not os.path.exists(scene_eval_folder):
            os.makedirs(scene_eval_folder)

        # write scene evaluation to json
        eval_json_filepath = os.path.join(scene_eval_folder,"eval.json")
        with open(eval_json_filepath,"w") as eval_json_file:
            json.dump(evals,eval_json_file)

        # save colormap resulting disparity image to scene evaluation folder
        colormap_test_disp = Eval.colormap_disp(test_disp_image)
        colormap_gt_disp = Eval.colormap_disp(ground_truth_disp_image)
        test_disp_filepath = os.path.join(scene_eval_folder,"disp0{}.png".format(alg_name))
        gt_disp_filepath = os.path.join(scene_eval_folder,"disp0gt.png")
        left_filepath = os.path.join(scene_eval_folder,"im0.png")
        cv2.imwrite(test_disp_filepath,colormap_test_disp)
        cv2.imwrite(gt_disp_filepath,colormap_gt_disp)
        cv2.imwrite(left_filepath,left_image)

        all_evals[scene_eval_name] = evals

        Eval.display_results(left_image,ground_truth_disp_image,test_disp_image,evals,wait=1)

    # write evaluations to json
    all_eval_json_filepath = os.path.join(eval_folder,"evals.json")
    with open(all_eval_json_filepath,"w") as all_eval_json_file:
        json.dump(all_evals,all_eval_json_file)

if __name__ == "__main__":
    run()
