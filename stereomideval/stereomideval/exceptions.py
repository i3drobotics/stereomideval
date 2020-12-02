"""Exceptions"""
import os


class ImageSizeNotEqual(Exception):
    """Image size not equal exception"""
    def __str__(self):
        return "Image sizes must be equal"

    @staticmethod
    def validate(image_a, image_b):
        """
        Validate standard exception condition.
        Raises exception if validation fails.

        Paramaters:
            image_a (numpy): image to compare
            image_b (numpy): image to compare
        """
        if image_a.shape != image_b.shape:
            raise ImageSizeNotEqual()


class InvalidSceneName(Exception):
    """Invalid scene exception"""
    def __init__(self, scene_name):
        """
        Exception handelling for invalid scene name
        Parameters:
            scene_name (string): Scene name that failed
        """
        self.message = "Invalid scene name '{}'.".format(scene_name)
        super().__init__(self.message)

    def __str__(self):
        return self.message

    @staticmethod
    def validate_scene_list(scene_name, scene_list):
        """
        Validate standard exception condition.
        Raises exception if validation fails.

        Parameters:
            scene_name (string): scene name to test
            scene_list (list(string)): list of scene names to compare against
        """
        if scene_name not in scene_list:
            raise InvalidSceneName(scene_name)

    @staticmethod
    def validate_scene_info_list(scene_name, scene_info_list):
        """
        Validate standard exception condition.
        Raises exception if validation fails.

        Parameters:
            scene_name (string): scene name to test
            scene_info_list (list(SceneInfo)):
                list of scene info to compare against
        """
        for scene_info in scene_info_list:
            if scene_name == scene_info.scene_name:
                return
        raise InvalidSceneName(scene_name)


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
    def __init__(self, filepath, message="Path does not exist: "):
        """
        Exception handelling for path not found
        Parameters:
            filepath (string): Filepath that was not found.
                Will be displayed in exception message
            message (string): Message to display in exception,
                will use default message if no message is provided
        """
        self.filepath = filepath
        self.message = message
        super().__init__(self.message)

    @staticmethod
    def validate(filepath):
        """
        Validate standard exception condition.
        Raises exception if validation fails.

        Parameters:
            filepath (string): filepath to test
        """
        if not os.path.exists(filepath):
            raise PathNotFound(filepath)

    def __str__(self):
        """Overload of exception message"""
        return self.message+self.filepath
