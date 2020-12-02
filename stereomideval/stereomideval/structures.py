"""Structures"""


class DatasetType:
    """Dataset type"""
    imperfect = 'I'
    exposure_changed = 'E'
    lighting_changed = 'L'
    perfect = 'P'
    SCENE_CHANGED_TYPES = [exposure_changed, lighting_changed]


class CalibrationData:
    """
    Calibration data

    Used to make returning and accessing calibration data simple.
    """
    def __init__(self, width, height, c_x, c_y, focal_length, doffs, baseline, ndisp):
        """
        Initalisaiton of CalibrationData structure

        Parameters:
            c_x (float): Principle point in X
            c_y (float): Principle point in Y
            focal_length (float): focal length
            doffs (float): x-difference of principal points, doffs = cx1 - cx0
            baseline (float): baseline
        """
        self.width = width
        self.height = height
        self.c_x = c_x
        self.c_y = c_y
        self.focal_length = focal_length
        self.doffs = doffs
        self.baseline = baseline
        self.ndisp = ndisp


class TestData:
    """
    Test data

    Used to make returning and accessing scene data simple.
    """
    def __init__(self, left_image, right_image, disp_image, depth_image, ndisp):
        """
        Initalisaiton of TestData structure

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
        self.ndisp = ndisp


class SceneInfo:
    """
    Scene info
    """
    def __init__(self, scene_name, dataset_type, weight=1.0):
        """
        Initalisaiton of SceneInfo structure

        Parameters:
        """
        self.scene_name = scene_name
        self.dataset_type = dataset_type
        self.weight = weight

    def get_unique_name(self):
        scene_uniq_name = self.scene_name
        if self.dataset_type != DatasetType.imperfect:
            scene_uniq_name = self.scene_name+self.dataset_type
        return scene_uniq_name


class MatchData:
    """Match data"""

    class MatchResult:
        """Match result"""
        def __init__(self, left_image, right_image, ground_truth, match_result, match_time, ndisp=None):
            self.left_image = left_image
            self.right_image = right_image
            self.ground_truth = ground_truth
            self.match_result = match_result
            self.match_time = match_time
            self.ndisp = ndisp

    def __init__(self, scene_info, match_result):
        self.scene_info = scene_info
        self.match_result = match_result


class EvaluationData:
    """Evaluation data"""
    class EvaluationResult:
        """Evaluation result"""
        def __init__(self, metric, result, rank=None, ndisp=None):
            self.metric = metric
            self.result = result
            self.rank = rank
            self.ndisp = ndisp

    def __init__(self, scene_info, eval_result_list):
        self.scene_info = scene_info
        self.eval_result_list = eval_result_list
