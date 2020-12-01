"""Evaluation module"""
import time
import math
import numpy as np
import cv2
import urllib3
import requests
from bs4 import BeautifulSoup
from stereomideval import image_resize, puttext_multiline
from stereomideval.structures import DatasetType
from stereomideval.exceptions import ImageSizeNotEqual, InvalidSceneName
from stereomideval.dataset import Dataset

class WebscrapeMiddlebury:
    """Get table data from a webpage"""
    def __init__(self):
        """Initalise WebscapeMiddlebury class"""

    @staticmethod
    def get_web_scene_name_suffix(dataset_type):
        """Get web scene name suffix"""
        scene_name_suffix = ""
        if not dataset_type == DatasetType.I:
            scene_name_suffix = dataset_type
        return scene_name_suffix

    @staticmethod
    def get_table_url(test_or_training,sparsity,metric,mask,has_invalid):
        """
        Get url for table on the Middlebury website using parameters

        Parameters:
            test_or_training (string): table should be for 'test' or 'training' set
            sparsity (string): table should be for 'dense' or 'sparse' sparsity
            metric (Eval.Metric): table should be for given metric
            mask (string): table should be for mask 'nonocc' or 'all'
            has_invalid (bool): table should include invalid results

        Returns:
            url (string): url to table based on given parameters
        """
        # define valid inputs for parameters
        sparse_options = ["dense","sparse"]
        test_train_options = ["test","training"]
        metric_options = Metrics.get_metrics_list()
        mask_options = ["nonocc","all"]
        has_invalid_options = ["true","false"]

        # check validitiy of parameters
        if isinstance(has_invalid,bool):
            if has_invalid:
                has_invalid = "true"
            else:
                has_invalid = "false"
        else:
            if has_invalid not in has_invalid_options:
                raise Exception("has_invalid MUST be {}".format(has_invalid_options))
        if test_or_training not in test_train_options:
            raise Exception("test_or_training MUST be {}".format(test_train_options))
        if sparsity not in sparse_options:
            raise Exception("sparsity MUST be {}".format(sparse_options))
        if metric not in metric_options:
            raise Exception("metric MUST be {}".format(metric_options))
        if mask not in mask_options:
            raise Exception("mask MUST be {}".format(mask_options))

        # define url
        base_url = "https://vision.middlebury.edu/stereo/eval3/table.php"
        base_url += "?dbfile=../results3/results.db"
        db_arg_url = "&type={}&sparsity={}&stat={}&mask={}&hasInvalid={}&ids="
        # fill url with parameters
        url = base_url+db_arg_url.format(test_or_training,sparsity,metric,mask,has_invalid)
        return url

    @staticmethod
    def get_metric_vals(metric,scene_name,dataset_type=DatasetType.I,dense=True):
        """
        Get list of metrics values from Middlebury website

        Currently assumes using training dense set with nonocc mask

        Parameters:
            metric (Eval.Metric): metric list to get values for
            scene_name (string): scene name to get values for

        Returns:
            alg_names (list(string)): list of algorithm names
            metric_list (list(string)): list of values for given metric
        """
        # Check scene name is valid
        InvalidSceneName.validate_scene_dict_list(scene_name,Dataset.get_training_scene_list())
        dense_or_sparse = "dense"
        if not dense:
            dense_or_sparse = "sparse"
        # Get url for table on Middlebury website
        url = WebscrapeMiddlebury.get_table_url("training",dense_or_sparse,metric,"nonocc",False)
        # disable warnings to avoid message:
        # 'InsecureRequestWarning: Unverified HTTPS request
        # is being made to host 'vision.middlebury.edu'.
        # Adding certificate verification is strongly advised.'
        urllib3.disable_warnings()
        # Load url
        print("Loading results from Middlebury website...")
        page = requests.get(url, verify=False)
        # Parse webpage
        soup = BeautifulSoup(page.content,'html.parser')
        # Get stereo table from webpage
        table = soup.find('table',{'id':'stereoTable'})
        # Find table body
        table_body = table.find('tbody')
        # Find table body rows
        table_rows = table_body.findAll('tr')
        # Initalise lists for storing result of table
        alg_names_list = []
        metric_list = []
        web_scene_name_suffix = WebscrapeMiddlebury.get_web_scene_name_suffix(dataset_type)
        # Iteration through rows in table
        for table_row in list(table_rows):
            # Find algorithm name in row data
            alg_name = table_row.find('td',{'class':['algName']})
            # Find metric value in row data
            metric_val = table_row.find('td',{
                "class":["{} data datasetCol".format(scene_name+web_scene_name_suffix)]},
                partial=False)
            # Check if metric value was found
            if metric_val is None:
                # Find metric value in with 'firstPlace'
                # (the first row in the data has this extra class definition)
                metric_val = table_row.find('td',{
                    "class":["{} firstPlace data datasetCol".format(
                        scene_name+web_scene_name_suffix)]},
                    partial=False)
            # Check metric value and algorithm name were found
            if metric_val is not None and alg_name is not None:
                # Add metric value and algorithm name to lists
                metric_list.append(float(metric_val.string))
                alg_names_list.append(alg_name.string)
        return alg_names_list,metric_list


class Timer:
    """
    Timer

    Simplify getting elapsed time for tasks
    """
    def __init__(self):
        self.start_time = None

    def start(self):
        """Start timer"""
        self.start_time = time.time()

    def elapsed(self):
        """
        Get elapsed time

        Returns:
            elapsed_time (float): time since start of time (seconds)
        """
        # Check timer has been started
        if self.start_time is None:
            raise Exception("Timer not running")
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        return elapsed_time

class Metrics:
    """
    Metrics used for evaluating data in the Middlebury dataset
    """
    # Define metrics (must match stat value in Middlebury website url)
    bad050 = "bad050"
    bad100 = "bad100"
    bad200 = "bad200"
    bad400 = "bad400"
    avgerr = "avgerr"
    rms = "rms"
    a50 = "A50"
    a90 = "A90"
    a95 = "A95"
    a99 = "A99"
    time = "time"
    time_mp = "time/MP"
    time_gp = "time/Gdisp"

    @staticmethod
    def get_metrics_list():
        """Get metric strings as a list"""
        m = Metrics
        return [
            m.bad050,m.bad100,m.bad200,m.bad400,
            m.avgerr,m.rms,
            m.a50,m.a90,m.a95,m.a99,
            m.time,m.time_mp,m.time_gp
        ]

class Eval:
    """Evaluate disparity image against ground truth"""

    @staticmethod
    def display_results(left_image,gt_disp,test_disp,eval_results=None,
        window_name="Results",wait=1000):
        """Display match results and evaluation to OpenCV window"""
        # remove negative disparities
        test_disp[test_disp<0]=0.0
        gt_disp[gt_disp<0]=0.0
        # Replace nan and inf values with zero
        test_disp = np.nan_to_num(test_disp, nan=0.0,posinf=0.0,neginf=0.0)
        gt_disp = np.nan_to_num(gt_disp, nan=0.0,posinf=0.0,neginf=0.0)
        # normalise image
        test_disp = cv2.normalize(test_disp, None, 0, 255, cv2.NORM_MINMAX)
        gt_disp = cv2.normalize(gt_disp, None, 0, 255, cv2.NORM_MINMAX)
        # convert to uint8 (required by applyColorMap function)
        test_disp = test_disp.astype(np.uint8)
        gt_disp = gt_disp.astype(np.uint8)
        # apply colormap
        test_disp = cv2.applyColorMap(test_disp, cv2.COLORMAP_JET)
        gt_disp = cv2.applyColorMap(gt_disp, cv2.COLORMAP_JET)
        # Resize colormap image for displaying in window
        test_disp = image_resize(test_disp, width=480)
        gt_disp = image_resize(gt_disp, width=480)
        left_image = image_resize(left_image, width=480)
        # Concatinate images
        display_image = cv2.hconcat([left_image, gt_disp, test_disp])

        if eval_results is not None:
            # Print metrics on display image
            eval_time = eval_results[Metrics.time]
            eval_rms = eval_results[Metrics.rms]
            eval_bad200 = eval_results[Metrics.bad200]
            msg = "Match time {:.2f}s ({})\n".format(eval_time['result'],eval_time['rank'])
            msg += "RMSE: {:.2f} ({})\n".format(eval_rms['result'],eval_rms['rank'])
            msg += "Bad pixel 2.0: {:.2f}% ({})\n".format(eval_bad200['result'],eval_bad200['rank'])
            display_image = puttext_multiline(
                img=display_image, text=msg, org=(10, 10), font=cv2.FONT_HERSHEY_TRIPLEX,
                font_scale=0.7, color=(0, 0, 0), thickness=1, outline_color=(255,255,255))
        else:
            print(msg)

        # Display in OpenCV window
        cv2.imshow(window_name, display_image)
        cv2.waitKey(wait)

    @staticmethod
    def get_metric_rank(metric,scene_name,value,dataset_type=DatasetType.I,dense=True):
        """
        Compare new value to middlebury metric and
        calculate it's rank when compared to the current
        table of results from the Middlebury website

        Parameters:
            metric (Eval.Metrics): metric list to search against
            scene_name (string): name of scene from Middlebury stereo dataset (2014)
            value (float): new value to compare against online metric list

        Returns:
            rank (int): rank position of value when compared to metric list
        """
        _,metric_vals = WebscrapeMiddlebury.get_metric_vals(metric,scene_name,dataset_type,dense)
        # add your metric value to the list
        metric_vals.append(value)
        metric_ranks = Eval.rank_vals(metric_vals)
        # get last element in list (this will be the element we added)
        return metric_ranks[-1]

    @staticmethod
    def rank_vals(data):
        """
        Rank values in list from 1 to n based on value (1 is lowest)

        Parameters:
            data (list(float)): list of values to rank

        Returns:
            rank (list(int)): list of ranks for values
        """
        # Get data length
        data_len = len(data)
        ivec=sorted(range(len(data)), key=data.__getitem__)
        svec=[data[rank] for rank in ivec]
        sumranks = 0
        dupcount = 0
        newarray = [0]*data_len
        for data_index in range(data_len):
            sumranks += data_index
            dupcount += 1
            if data_index==data_len-1 or svec[data_index] != svec[data_index+1]:
                averank = sumranks / float(dupcount) + 1
                for j in range(data_index-dupcount+1,data_index+1):
                    newarray[ivec[j]] = int(averank)
                sumranks = 0
                dupcount = 0
        return newarray

    @staticmethod
    def eval_all(ground_truth, test_data,
        scene_name=None, elapsed_time=None, dataset_type=DatasetType.I, dense=True):
        """
        Evaluate test data using all Middlesbury metrics

        This function simplified the running of evaluation for all metrics
        and rank in that metric

        Parameters:
            ground_truth (numpy): 2D ground truth image
            test_data (numpy): 2D test image
            scene_name (string): Scene name in Middlesbury stereo dataset (2014)
            elapsed_time (float): time to complete match (only used in time metrics)

        Returns:
            eval_dict (dict(dict('result','rank'))): Dictionary for each metric which contains
                                                        dictionary of result and rank.
        """
        # Initalise dictionary
        eval_dict = {}
        # Run evaluation for each metric
        for metric in Metrics.get_metrics_list():
            # Ignore time_gp metric as there is no function for this yet
            if metric is not Metrics.time_gp:
                # Evaluate metric and compare rank
                result,rank = Eval.eval_metric(
                    metric,ground_truth,test_data,
                    scene_name,elapsed_time,dataset_type,dense)
                # Add result and rank to dictionary
                eval_dict[metric] = {'result':result,'rank':rank}
        return eval_dict

    @staticmethod
    def eval_metric(metric, ground_truth, test_data, 
        scene_name=None, elapsed_time=None, dataset_type=DatasetType.I,dense=True):
        """
        Run evaluation routine based on metric

        This function simplified the running of evaluation routine

        Parameters:
            metric (Eval.Metrics): Evaluation metrics to calculate
            ground_truth (numpy): 2D ground truth image
            test_data (numpy): 2D test image
            scene_name (string): name of scene from Middlebury stereo dataset (2014)
            elapsed_time (float): time to complete match (only used in time metrics)

        Returns:
            result (float): Result from evaluation routine
            rank (int): Rank of result compared to middlesbury stereo dataset
                        (none if no scene_name provided)
        """
        # Check metric is valid
        if metric not in Metrics.get_metrics_list():
            raise Exception("Invalid metric")
        # Initalise return variable
        result = None
        rank = None
        print("Evaluating {} metric...".format(metric))
        # Run evaluation routine based on metric chosen
        if metric == Metrics.bad050:
            result = Eval.bad_pix_error(ground_truth,test_data,0.5)
        elif metric == Metrics.bad100:
            result = Eval.bad_pix_error(ground_truth,test_data,1)
        elif metric == Metrics.bad200:
            result = Eval.bad_pix_error(ground_truth,test_data,2)
        elif metric == Metrics.bad400:
            result = Eval.bad_pix_error(ground_truth,test_data,4)
        elif metric == Metrics.rms:
            result = Eval.rmse(ground_truth,test_data)
        elif metric == Metrics.avgerr:
            result = Eval.avgerr(ground_truth,test_data)
        elif metric == Metrics.time:
            # Check elapsed time parameter is defined
            if elapsed_time is None:
                raise Exception("Elapsed time parameter is missing")
            result = elapsed_time
        elif metric == Metrics.time_mp:
            # Check elapsed time parameter is defined
            if elapsed_time is None:
                raise Exception("Elapsed time pparameter is missing")
            result = Eval.time_mp(test_data,elapsed_time)
        else:
            msg = "Evalutation method not yet avaiable for this metric: {}"
            msg = msg.format(metric)
            print(msg)
            return result, rank
            #raise Exception(msg.format(metric))

        # Assume rank requested if scene_name is given
        if scene_name is not None:
            # Compare result with Middlebury results and return rank
            print("Comparing {} result with online results...".format(metric))
            rank = Eval.get_metric_rank(metric,scene_name,result,dataset_type,dense)
        return result, rank

    @staticmethod
    def time_mp(test_data,match_time):
        """
        Time normalised by number of megapixels

        Parameters:
            test_data (numpy): test image (used to get number of megapixels)
            match_time (float): elapsed time for match (in seconds)

        Returns:
            norm_time (float): time normalised by megapixels
        """
        # Count megapixels in image
        num_of_megapixels = test_data.size/1000000
        # Calculate time normalised by megapixels
        norm_time = match_time/num_of_megapixels
        return norm_time

    @staticmethod
    def avgerr(ground_truth, test_data):
        """
        Average error

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth
        Returns:
            average (float): average error in test data
        """
        diff = Eval.diff(ground_truth,test_data)
        average = np.average(diff)
        return average

    @staticmethod
    def diff(ground_truth, test_data):
        """
        Difference between ground truth and test data

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth
        Returns:
            diff (numpy): test data subtracted from ground truth
        """
        # Check images are the same size
        ImageSizeNotEqual.validate(test_data,ground_truth)
        # Replace nan and inf values with zero
        test_data_no_nan = np.nan_to_num(test_data, nan=0.0,posinf=0.0,neginf=0.0)
        ground_truth_no_nan = np.nan_to_num(ground_truth, nan=0.0,posinf=0.0,neginf=0.0)
        # Subtract test data from ground truth to find difference
        diff = np.subtract(test_data_no_nan,ground_truth_no_nan)
        return diff

    @staticmethod
    def bad_pix_error(ground_truth,test_data,threshold=2.0):
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
        ImageSizeNotEqual.validate(test_data,ground_truth)
        # Calculate pixel difference between ground truth and test data
        diff = Eval.diff(ground_truth,test_data)
        # Get the absolute difference (positive only)
        abs_diff = np.abs(diff)
        # Count number of 'bad' pixels
        bad_count = float((~(abs_diff < threshold)).sum())
        # Convert number of 'bad' pixels to percentage
        total_size = float(ground_truth.shape[0] * ground_truth.shape[1])
        perc_bad = (bad_count/total_size)*100
        return perc_bad

    @staticmethod
    def rmse(ground_truth, test_data):
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
        # Check test data and ground truth are the same size
        ImageSizeNotEqual.validate(test_data,ground_truth)
        # Calculate the root of mse
        rmse = math.sqrt(Eval.mse(ground_truth,test_data))
        return rmse

    @staticmethod
    def mse(ground_truth, test_data):
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
        ImageSizeNotEqual.validate(test_data,ground_truth)
        # Calculate difference (error) between ground truth and test data
        diff = Eval.diff(ground_truth,test_data)
        # Calculate mean of the square error
        err = np.square(diff).mean()
        return err
