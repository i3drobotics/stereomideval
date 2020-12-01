"""Structures"""

class CalibrationData:
    """
    Calibration data

    Used to make returning and accessing calibration data simple.
    """
    def __init__(self,c_x,c_y,focal_length,doffs,baseline):
        """
        Initalisaiton of CalibrationData structure

        Parameters:
            c_x (float): Principle point in X
            c_y (float): Principle point in Y
            focal_length (float): focal length
            doffs (float): x-difference of principal points, doffs = cx1 - cx0
            baseline (float): baseline
        """
        self.c_x = c_x
        self.c_y = c_y
        self.focal_length = focal_length
        self.doffs = doffs
        self.baseline = baseline

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

class DatasetType:
    """Dataset type"""
    I = 'I'
    E = 'E'
    L = 'L'
    P = 'P'
