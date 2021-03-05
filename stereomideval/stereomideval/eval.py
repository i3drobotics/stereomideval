"""Evaluation module"""
import time
import math
import numpy as np
import cv2
import urllib3
import requests
from bs4 import BeautifulSoup
from stereomideval import image_resize, puttext_multiline
from stereomideval.structures import DatasetType, EvaluationData
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
        if not dataset_type == DatasetType.imperfect:
            scene_name_suffix = dataset_type
        return scene_name_suffix

    @staticmethod
    def get_table_url(test_or_training, sparsity, metric, mask, has_invalid):
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
        sparse_options = ["dense", "sparse"]
        test_train_options = ["test", "training"]
        metric_options = Metric.get_metrics_list()
        mask_options = ["nonocc", "all"]
        has_invalid_options = ["true", "false"]

        # check validitiy of parameters
        if isinstance(has_invalid, bool):
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
        url = base_url+db_arg_url.format(test_or_training, sparsity, metric, mask, has_invalid)
        return url

    @staticmethod
    def get_av_metric_vals(metric, dense=True):
        """
        Get list of metrics values from Middlebury website

        Currently assumes using training set with 'all' mask

        Parameters:
            metric (Eval.Metric): metric list to get values for
            scene_name (string): scene name to get values for

        Returns:
            alg_names (list(string)): list of algorithm names
            metric_list (list(string)): list of values for given metric
        """
        dense_or_sparse = "dense"
        if not dense:
            dense_or_sparse = "sparse"
        # Get url for table on Middlebury website
        url = WebscrapeMiddlebury.get_table_url("training", dense_or_sparse, metric, "all", False)
        # disable warnings to avoid message:
        # 'InsecureRequestWarning: Unverified HTTPS request
        # is being made to host 'vision.middlebury.edu'.
        # Adding certificate verification is strongly advised.'
        urllib3.disable_warnings()
        # Load url
        print("Loading results from Middlebury website...")
        page = requests.get(url, verify=False)
        # Parse webpage
        soup = BeautifulSoup(page.content, 'html.parser')
        # Get stereo table from webpage
        table = soup.find('table', {'id': 'stereoTable'})
        # Find table body
        table_body = table.find('tbody')
        # Find table body rows
        table_rows = table_body.findAll('tr')
        # Initalise lists for storing result of table
        alg_names_list = []
        metric_average_list = []
        # (e.g. trying to compare image that is not on website table)
        # Iteration through rows in table
        for table_row in list(table_rows):
            # Find algorithm name in row data
            alg_name = table_row.find('td', {'class': ['algName']})
            # Find overall metric average in row data
            metric_average = table_row.find('td', {'class': ['data wtavg']},
                                            partial=False)
            # Check if metric average was found
            if metric_average is None:
                # Find metric average with 'firstPlace'
                # (the first row in the data has this extra class definition)
                metric_average = table_row.find('td', {"class": ["data wtavg firstPlace"]},
                                                partial=False)
            # Check metric value and algorithm name were found
            if alg_name is not None and metric_average is not None:
                # Add metric value and algorithm name to lists
                metric_average_list.append(float(metric_average.string))
                alg_names_list.append(alg_name.string)
            else:
                raise Exception("Failed to find table data for metric average or algorithm name")
        return alg_names_list, metric_average_list

    @staticmethod
    def get_metric_vals(metric, scene_name, dataset_type=DatasetType.imperfect, dense=True):
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
        InvalidSceneName.validate_scene_info_list(scene_name, Dataset.get_training_scene_list())
        dense_or_sparse = "dense"
        if not dense:
            dense_or_sparse = "sparse"
        # Get url for table on Middlebury website
        url = WebscrapeMiddlebury.get_table_url("training", dense_or_sparse,
                                                metric, "nonocc", False)
        # disable warnings to avoid message:
        # 'InsecureRequestWarning: Unverified HTTPS request
        # is being made to host 'vision.middlebury.edu'.
        # Adding certificate verification is strongly advised.'
        urllib3.disable_warnings()
        # Load url
        print("Loading results from Middlebury website...")
        page = requests.get(url, verify=False)
        # Parse webpage
        soup = BeautifulSoup(page.content, 'html.parser')
        # Get stereo table from webpage
        table = soup.find('table', {'id': 'stereoTable'})
        # Find table body
        table_body = table.find('tbody')
        # Find table body rows
        table_rows = table_body.findAll('tr')
        # Initalise lists for storing result of table
        alg_names_list = []
        metric_list = []
        web_scene_name_suffix = WebscrapeMiddlebury.get_web_scene_name_suffix(dataset_type)
        # (e.g. trying to compare image that is not on website table)
        # Iteration through rows in table
        for table_row in list(table_rows):
            # Find algorithm name in row data
            alg_name = table_row.find('td', {'class': ['algName']})
            # Find metric value in row data
            metric_val = table_row.find('td', {
                "class": ["{} data datasetCol".format(scene_name+web_scene_name_suffix)]},
                partial=False)
            # Check if metric value was found
            if metric_val is None:
                # Find metric value with 'firstPlace'
                # (the first row in the data has this extra class definition)
                metric_val = table_row.find('td', {
                    "class": ["{} firstPlace data datasetCol".format(
                        scene_name+web_scene_name_suffix)]},
                    partial=False)
            # Check metric value and algorithm name were found
            if metric_val is not None and alg_name is not None:
                # Add metric value and algorithm name to lists
                metric_list.append(float(metric_val.string))
                alg_names_list.append(alg_name.string)
            else:
                raise Exception("Failed to find table data for metric value or algorithm name")
        return alg_names_list, metric_list


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


class Metric:
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
    time_gd = "time/Gdisp"
    coverage = "coverage"
    TIME_METRICS = [time, time_mp, time_gd]
    BAD_METRICS = [bad050, bad100, bad200, bad400]
    QUANTILE_METRICS = [a50, a90, a95, a99]
    NON_MIDEVAL_METRICS = [coverage]

    @staticmethod
    def get_metrics_list():
        """Get metric strings as a list"""
        m = Metric
        return [
            m.bad050, m.bad100, m.bad200, m.bad400,
            m.avgerr, m.rms,
            m.a50, m.a90, m.a95, m.a99,
            m.time, m.time_mp, m.time_gd, m.coverage
        ]

    @staticmethod
    def get_quantile_percentage(metric):
        """Get percentage value for quantile error metrics"""
        if metric not in Metric.QUANTILE_METRICS:
            raise Exception("Must provided quantile matric e.g. Metric.a50")
        if metric == Metric.a50:
            percentage = 0.5
        elif metric == Metric.a90:
            percentage = 0.9
        elif metric == Metric.a95:
            percentage = 0.95
        elif metric == Metric.a99:
            percentage = 0.99
        return percentage

    @staticmethod
    def get_bad_percentage(metric):
        """Get percentage value for bad pixel error metrics"""
        if metric not in Metric.BAD_METRICS:
            raise Exception("Must provided bad matric e.g. Metric.bad050")
        if metric == Metric.bad050:
            percentage = 0.5
        elif metric == Metric.bad100:
            percentage = 1
        elif metric == Metric.bad200:
            percentage = 2
        elif metric == Metric.bad400:
            percentage = 4
        return percentage

    @staticmethod
    def calc_time_mp(test_data, match_time):
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
    def calc_time_gd(test_data, match_time, ndisp):
        """
        Time normalised by number of disparity hypotheses

        Parameters:
            test_data (numpy): test image (used to get number of megapixels)
            match_time (float): elapsed time for match (in seconds)
            ndisp (int): number of disparities

        Returns:
            norm_time (float): time normalised by number of disparity hypotheses
        """
        # Count gigapixels in image
        num_of_gigapixels = test_data.size/1000000000
        # Calculate time normalised by megapixels
        norm_time = match_time/(num_of_gigapixels*ndisp)
        return norm_time

    @staticmethod
    def calc_quantile(ground_truth, test_data, quantile):
        """
        Q-th quantile of the data error

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth
            quantile (float): q-th quantile of the data

        Returns:
            quantile_error (float): Q-th quantile of the data error
        """
        diff = Metric.calc_diff(ground_truth, test_data)
        # Get the absolute difference (positive only)
        abs_diff = np.abs(diff)
        quantile_error = np.quantile(abs_diff, quantile)
        return quantile_error

    @staticmethod
    def calc_avgerr(ground_truth, test_data):
        """
        Average error

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth
        Returns:
            average (float): average error in test data
        """
        diff = Metric.calc_diff(ground_truth, test_data)
        # Get the absolute difference (positive only)
        abs_diff = np.abs(diff)
        average = np.average(abs_diff)
        return average

    @staticmethod
    def calc_diff(ground_truth, test_data):
        """
        Difference between ground truth and test data

        Parameters:
            ground_truth (numpy): 2D ground truth image to use for comparision
            test_data (numpy): 2D test image to compare against ground truth
        Returns:
            diff (numpy): test data subtracted from ground truth
        """
        # Check images are the same size
        ImageSizeNotEqual.validate(test_data, ground_truth)
        # Replace nan and inf values with zero
        test_data_no_nan = np.nan_to_num(test_data, nan=0.0, posinf=0.0, neginf=0.0)
        ground_truth_no_nan = np.nan_to_num(ground_truth, nan=0.0, posinf=0.0, neginf=0.0)
        # Subtract test data from ground truth to find difference
        diff = np.subtract(test_data_no_nan, ground_truth_no_nan)
        return diff

    @staticmethod
    def calc_bad_pix_error(ground_truth, test_data, threshold=2.0):
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
        ImageSizeNotEqual.validate(test_data, ground_truth)
        # Calculate pixel difference between ground truth and test data
        diff = Metric.calc_diff(ground_truth, test_data)
        # Get the absolute difference (positive only)
        abs_diff = np.abs(diff)
        # Count number of 'bad' pixels
        bad_count = float((~(abs_diff < threshold)).sum())
        # Convert number of 'bad' pixels to percentage
        total_size = float(ground_truth.shape[0] * ground_truth.shape[1])
        perc_bad = (bad_count/total_size)*100
        return perc_bad

    @staticmethod
    def calc_rmse(ground_truth, test_data):
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
        ImageSizeNotEqual.validate(test_data, ground_truth)
        # Calculate the root of mse
        rmse = math.sqrt(Metric.calc_mse(ground_truth, test_data))
        return rmse

    @staticmethod
    def calc_mse(ground_truth, test_data):
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
        ImageSizeNotEqual.validate(test_data, ground_truth)
        # Calculate difference (error) between ground truth and test data
        diff = Metric.calc_diff(ground_truth, test_data)
        # Calculate mean of the square error
        err = np.square(diff).mean()
        return err

    
    @staticmethod
    def calc_coverage(test_data,invalid_value=0):
        # count pixels with value of zero
        valid_count = np.count_nonzero(test_data!=invalid_value)
        print(valid_count)
        coverage = (valid_count/test_data.size)*100
        return coverage

class Eval:
    """Evaluate disparity image against ground truth"""
    @staticmethod
    def display_results(match_result, eval_result_list=None, window_name="Results", wait=1000):
        """Display match results and evaluation to OpenCV window"""
        match_result_image = match_result.match_result
        ground_truth = match_result.ground_truth
        left_image = match_result.left_image
        right_image = match_result.right_image
        # remove negative disparities
        match_result_image[match_result_image < 0] = 0.0
        ground_truth[ground_truth < 0] = 0.0
        # Replace nan and inf values with zero
        match_result_image = np.nan_to_num(match_result_image, nan=0.0, posinf=0.0, neginf=0.0)
        ground_truth = np.nan_to_num(ground_truth, nan=0.0, posinf=0.0, neginf=0.0)
        # normalise image
        match_result_image = cv2.normalize(match_result_image, None, 0, 255, cv2.NORM_MINMAX)
        ground_truth = cv2.normalize(ground_truth, None, 0, 255, cv2.NORM_MINMAX)
        # convert to uint8 (required by applyColorMap function)
        match_result_image = match_result_image.astype(np.uint8)
        ground_truth = ground_truth.astype(np.uint8)
        # apply colormap
        match_result_image = cv2.applyColorMap(match_result_image, cv2.COLORMAP_JET)
        ground_truth = cv2.applyColorMap(ground_truth, cv2.COLORMAP_JET)
        # Resize colormap image for displaying in window
        match_result_image = image_resize(match_result_image, width=320)
        ground_truth = image_resize(ground_truth, width=320)
        left_image = image_resize(left_image, width=320)
        right_image = image_resize(right_image, width=320)
        # Concatinate images
        disp_images = cv2.hconcat([ground_truth, match_result_image])
        stereo_image = cv2.hconcat([left_image, right_image])
        display_image = cv2.vconcat([stereo_image, disp_images])

        if eval_result_list is not None:
            # Print metrics on display image
            msg = ""
            for eval_result in eval_result_list:
                metric_result = eval_result.result
                metric_rank = eval_result.rank
                metric = eval_result.metric
                if metric_result is not None:
                    msg += metric+" {:.2f} ".format(metric_result)
                    if metric_rank is not None:
                        msg += "({})\n".format(metric_rank)
                    else:
                        msg += "\n"
            display_image = puttext_multiline(
                img=display_image, text=msg, org=(10, 10), font=cv2.FONT_HERSHEY_TRIPLEX,
                font_scale=0.7, color=(0, 0, 0), thickness=1, outline_color=(255, 255, 255))

        # Display in OpenCV window
        cv2.imshow(window_name, display_image)
        cv2.waitKey(wait)

    @staticmethod
    def evaluate_match_result_list(match_result_list, display_results=False,
                                   display_window_name="Results", display_wait=1000):
        """
        Evaluate match result list

        Paramaters:
            match_result_list (list(MatchResult)): list of match results
            display_results (bool): display results to OpenCV window while evaluating
            display_window (string): name of OpenCV window
            display_wait (int): time to wait, used in 'cv2.waitKey(display_wait)'

        Returns:
            metric_average_list (list(MatchResult)): list of average metrics across all test data
            eval_result_list_list (list(list(MatchResult))): list of lists of match results,
                                                        seperate list for each test data provided
        """
        eval_result_list_list = []
        for match_result in match_result_list:
            # Evaluate test data against all metrics
            eval_result_list = Eval.eval_all_metrics(match_result)
            if display_results:
                # Display evaluation results to OpenCV window
                Eval.display_results(match_result, eval_result_list,
                                     display_window_name, display_wait)
            eval_result_list_list.append(eval_result_list)

        metric_average_list = \
            Eval.average_all_metrics_across_scenes(eval_result_list_list)

        print("Metric average over all test data:")
        for metric_average in metric_average_list:
            msg = "{}: {} ({})"
            msg = msg.format(metric_average.metric, metric_average.result, metric_average.rank)
            print(msg)

        return metric_average_list, eval_result_list_list

    @staticmethod
    def evaluate_match_data_list(match_data_list, get_metric_rank=False, get_av_metric_rank=False,
                                 dense=True, display_results=False, display_window_name="Results",
                                 display_wait=1000):
        """
        Evaluate match result list

        Paramaters:
            match_result_list (list(MatchData)): list of match data
            get_metric_rank (bool): compare evaluation data against Middlesbury website ranking
            get_av_metric_rank (bool): compare average evaluation data against Middlesbury
                                         website ranking
            display_results (bool): display results to OpenCV window while evaluating
            display_window (string): name of OpenCV window
            display_wait (int): time to wait, used in 'cv2.waitKey(display_wait)'

        Returns:
            metric_average_list (list(MatchResult)): list of average metrics across all test data
            eval_data_list (list(EvaluationData)): list of evaluation data, seperate list for
                                                    each test data provided
        """
        eval_data_list = []
        eval_result_list_list = []
        for match_data in match_data_list:
            scene_info = match_data.scene_info

            if get_metric_rank:
                # Evaluate test data against all metrics
                eval_result_list = Eval.eval_all_metrics_rank(match_data, dense)
            else:
                # Evaluate test data against all metrics
                eval_result_list = Eval.eval_all_metrics(match_data.match_result)

            if display_results:
                Eval.display_results(match_data.match_result, eval_result_list,
                                     display_window_name, display_wait)
            eval_data = EvaluationData(scene_info, eval_result_list)
            eval_data_list.append(eval_data)
            eval_result_list_list.append(eval_result_list)

        if get_av_metric_rank:
            metric_average_list = \
                Eval.average_all_metrics_across_scenes_rank(eval_data_list, dense)
        else:
            metric_average_list = \
                Eval.average_all_metrics_across_scenes(eval_result_list_list)

        print("Weighted metric average over all test data:")
        for metric_average in metric_average_list:
            msg = "{}: {} ({})"
            msg = msg.format(metric_average.metric, metric_average.result, metric_average.rank)
            print(msg)

        return metric_average_list, eval_data_list

    @staticmethod
    def get_average_metric_rank(value, metric, dense=True):
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
        _, metric_avs = WebscrapeMiddlebury.get_av_metric_vals(metric, dense)
        if len(metric_avs) <= 0:
            raise Exception("Failed to receive metric averages from Middlebury webpage")
        print("Comparing against {} results...".format(len(metric_avs)))
        # add your metric value to the list
        metric_avs.append(value)
        metric_ranks = Eval.rank_vals(metric_avs)
        # get last element in list (this will be the element we added)
        return metric_ranks[-1]

    @staticmethod
    def get_metric_rank(metric, scene_name, value, dataset_type=DatasetType.imperfect, dense=True):
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
        _, metric_vals = WebscrapeMiddlebury.get_metric_vals(metric, scene_name,
                                                             dataset_type, dense)
        if len(metric_vals) <= 0:
            raise Exception("Failed to receive metric values from Middlebury webpage")
        print("Comparing against {} results...".format(len(metric_vals)))
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
        ivec = sorted(range(len(data)), key=data.__getitem__)
        svec = [data[rank] for rank in ivec]
        sumranks = 0
        dupcount = 0
        newarray = [0]*data_len
        for data_index in range(data_len):
            sumranks += data_index
            dupcount += 1
            if data_index == data_len-1 or svec[data_index] != svec[data_index+1]:
                averank = sumranks / float(dupcount) + 1
                for j in range(data_index-dupcount+1, data_index+1):
                    newarray[ivec[j]] = int(averank)
                sumranks = 0
                dupcount = 0
        return newarray

    @staticmethod
    def average_all_metrics_across_scenes_rank(evaluation_data_list, dense=True):
        """
        Evaluate all metrics across all scenes and compare rank

        Parameters:
            evaluation_data_list (list(EvaluationData)): list of evaluation data
            dense (bool): compare to dense online ranking table

        Returns:
            metric_average_list (list(EvaluationResult)): average result and rank across all scenes
        """
        # Initalise dictionary
        metric_average_list = []
        # Run evaluation for each metric
        for metric in Metric.get_metrics_list():
            metric_average = \
                Eval.average_metric_across_scenes_rank(metric, evaluation_data_list, dense)
            metric_average_list.append(metric_average)

        return metric_average_list

    @staticmethod
    def average_all_metrics_across_scenes(evaluation_result_list_list):
        """
        Evaluate all metrics across all scenes

        Parameters:
            evaluation_data_list (list(list(EvaluationResults))):
                list of lists of evaluation results
        Returns:
            metric_average_list (list(EvaluationResult)): average result across all scenes
        """
        # Initalise dictionary
        metric_average_list = []
        # Run evaluation for each metric
        for metric in Metric.get_metrics_list():
            metric_average = Eval.average_metric_across_scenes(metric, evaluation_result_list_list)
            metric_average_list.append(metric_average)
        return metric_average_list

    @staticmethod
    def average_metric_across_scenes_rank(metric, evaluation_data_list, dense=True):
        """
        Evaluate metric result and rank across all scenes

        Parameters:
            metric (Metric): metric to evaluate
            evaluation_data_list (list(EvaluationData)):
                list of evaluation data
            dense (bool): compare to dense online ranking table
        Returns:
            average_metric_result (list(EvaluationResult)):
                average result for metric across all scenes
        """
        metric_result_list = []
        for evaluation_data in evaluation_data_list:
            scene_info = evaluation_data.scene_info
            evaluation_result_list = evaluation_data.eval_result_list
            metric_found = False
            for evaluation_result in evaluation_result_list:
                if evaluation_result.metric == metric:
                    metric_result_list.append(evaluation_result)
                    metric_found = True
            # raise exception if missing metric
            if not metric_found:
                raise Exception("Missing metric {} in evaluation data".format(metric))

        average = 0
        scene_count = 0
        for metric_result in metric_result_list:
            metric_result_val = metric_result.result
            if metric_result_val is None:
                raise Exception("Missing metric result for {}".format(metric))
            average += metric_result_val * scene_info.weight
            scene_count += 1
        average = average / scene_count
        average_metric_result = EvaluationData.EvaluationResult(metric, average)

        if average_metric_result.result is None:
            raise Exception("Failed to find result value for {} metric".format(metric))

        # Compare result with Middlebury results and return rank
        msg = "Comparing average {} result of {:.2f} with online results..."
        msg = msg.format(metric, average_metric_result.result)
        print(msg)
        average_metric_result.rank = \
            Eval.get_average_metric_rank(average_metric_result.result, metric, dense)
        print("Average {} result of {:.2f} is rank {}".format(
            metric, average_metric_result.result, average_metric_result.rank))
        return average_metric_result

    @staticmethod
    def average_metric_across_scenes(metric, evaluation_result_list_list):
        """
        Evaluate metric result across all scenes

        Parameters:
            metric (Metric): metric to evaluate
            evaluation_data_list (list(list(EvaluationResults))):
                list of lists of evaluation results
        Returns:
            average_metric_result (list(EvaluationResult)):
                average result for metric across all scenes
        """
        metric_result_list = []
        for evaluation_result_list in evaluation_result_list_list:
            metric_found = False
            for evaluation_result in evaluation_result_list:
                if evaluation_result.metric == metric:
                    metric_result_list.append(evaluation_result)
                    metric_found = True
            # raise exception if missing metric
            if not metric_found:
                raise Exception("Missing metric {} in evaluation data".format(metric))

        average = 0
        scene_count = 0
        evaluation_result_list = []
        for metric_result in metric_result_list:
            metric_result_val = metric_result.result
            if metric_result_val is None:
                raise Exception("Missing metric result for {}".format(metric))
            average += metric_result_val
            scene_count += 1
        average = average / scene_count
        average_metric_result = EvaluationData.EvaluationResult(metric, average)
        return average_metric_result

    @staticmethod
    def eval_all_metrics_rank(match_data, dense=True):
        """
        Evaluate test data using all Middlesbury metrics and compare rank with online results

        Parameters:
            ground_truth (numpy): 2D ground truth image
            test_data (numpy): 2D test image
            elapsed_time (float): time to complete match (only used in time metrics)
            TODO

        Returns:
            TODO
        """
        # Initalise dictionary
        metric_result_list = []
        # Run evaluation for each metric
        for metric in Metric.get_metrics_list():
            # Evaluate metric and compare rank
            metric_result = Eval.eval_metric_rank(metric, match_data, dense)
            # Add metric result to list
            metric_result_list.append(metric_result)
        return metric_result_list

    @staticmethod
    def eval_all_metrics(match_result):
        """
        Evaluate test data using all Middlesbury metrics

        Parameters:
            ground_truth (numpy): 2D ground truth image
            test_data (numpy): 2D test image
            elapsed_time (float): time to complete match (only used in time metrics)

        Returns:
            TODO
        """
        # Initalise dictionary
        metric_result_list = []
        # Run evaluation for each metric
        for metric in Metric.get_metrics_list():
            # Evaluate metric and compare rank
            metric_result = Eval.eval_metric(metric, match_result)
            # Add metric result to list
            metric_result_list.append(metric_result)
        return metric_result_list

    @staticmethod
    def eval_metric_rank(metric, match_data, dense=True):
        """
        Run evaluation routine based on metric and compare rank with online results

        Parameters:
            metric (Eval.Metrics): Evaluation metrics to calculate
            match_data (Structures.MatchData): match data (see Structures.py for details)
            dense (bool): compare to dense online ranking table

        Returns:
            metric_result (Structures.EvaluationResult): metric result
        """
        # Check metric is valid
        if metric not in Metric.get_metrics_list():
            raise Exception("Invalid metric")
        scene_info = match_data.scene_info
        # Run evaluation routine based on metric chosen
        metric_result = Eval.eval_metric(metric, match_data.match_result)

        if metric_result.result is None:
            raise Exception("Failed to evaluate metric {}".format(metric))

        # Compare result with Middlebury results and return rank
        print("Comparing {} result of {:.2f} with online results...".format(
            metric, metric_result.result))
        scene_info = match_data.scene_info
        metric_result.rank = Eval.get_metric_rank(
            metric, scene_info.scene_name, metric_result.result,
            scene_info.dataset_type, dense)

        if metric_result.rank is None:
            raise Exception("Failed to get {} rank".format(metric))

        print("{} result of {:.2f} is rank {}".format(
            metric, metric_result.result, metric_result.rank))

        return metric_result

    @staticmethod
    def eval_metric(metric, match_result):
        """
        Run evaluation routine based on metric

        Parameters:
            metric (Eval.Metrics): Evaluation metrics to calculate
            match_data (Structures.MatchData): match data (see Structures.py for details)

        Returns:
            metric_result (Structures.EvaluationResult): metric result
        """
        # TODO fix 'PEP8 C901 function is too complex (12)'
        # Check metric is valid
        if metric not in Metric.get_metrics_list():
            raise Exception("Invalid metric")
        # Check match time parameter is defined when using time metrics
        if match_result.match_time is None:
            raise Exception("Match time parameter is missing")
        # Initalise return variable
        result = None
        print("Evaluating {} metric...".format(metric))
        # Run evaluation routine based on metric chosen
        if metric in Metric.BAD_METRICS:
            bad_perc = Metric.get_bad_percentage(metric)
            result = Metric.calc_bad_pix_error(
                match_result.ground_truth, match_result.match_result, bad_perc)
        elif metric in Metric.QUANTILE_METRICS:
            quantile_perc = Metric.get_quantile_percentage(metric)
            result = Metric.calc_quantile(
                match_result.ground_truth, match_result.match_result, quantile_perc)
        elif metric in Metric.TIME_METRICS:
            if metric == Metric.time:
                result = match_result.match_time
            elif metric == Metric.time_mp:
                result = Metric.calc_time_mp(match_result.match_result, match_result.match_time)
            elif metric == Metric.time_gd:
                if match_result.ndisp is None:
                    raise Exception("number of disparities parameter is missing")
                result = Metric.calc_time_gd(match_result.match_result,
                                             match_result.match_time,
                                             match_result.ndisp)
        elif metric == Metric.rms:
            result = Metric.calc_rmse(match_result.ground_truth, match_result.match_result)
        elif metric == Metric.avgerr:
            result = Metric.calc_avgerr(match_result.ground_truth, match_result.match_result)
        elif metric == Metric.coverage:
            result = Metric.calc_coverage(match_result.match_result)

        metric_result = EvaluationData.EvaluationResult(metric, result)
        return metric_result
