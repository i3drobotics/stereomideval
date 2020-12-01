"""
Stereo Middlebury Evaluation package

This module is for loading the stereo Middlebury dataset loading
and includes tools evaluatating stereo matching algorithms
"""
import numpy as np
import cv2

# Helpful functions
def in_dictlist(key, value, my_dictlist):
    """Check if key value pair exists in dictionary"""
    for this in my_dictlist:
        if this[key] == value:
            return True
    return False

def puttext_multiline(img, text, org, font,  
        font_scale, color, thickness, line_type=cv2.LINE_AA,
        line_spacing=1.5, outline_color=None):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(org, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (_, height), _ = cv2.getTextSize(
            text=line,
            fontFace=font,
            fontScale=font_scale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, height]
        org = tuple(uv_bottom_left_i.astype(int))
        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=font,
                fontScale=font_scale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=line_type,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=line_type,
        )
        uv_top_left += [0, height * line_spacing]
    return img

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """
    Resize image based on height or width while maintaning aspect ratio
    :param image: image matrix
    :param width: desired width of output image (can only use width or height not both)
    :param height: desired height of output image (can only use width or height not both)
    :param inter: opencv resize method (default: cv2.INTER_AREA)
    :type image: numpy
    :type width: int
    :type height: int
    :type inter: int
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (current_height, current_width) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        ratio = height / float(current_height)
        dim = (int(current_width * ratio), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        ratio = width / float(current_width)
        dim = (width, int(current_height * ratio))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
        