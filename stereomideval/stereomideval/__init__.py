"""
Stereo Middlebury Evaluation package

This module is for loading the stereo Middlebury dataset loading
and includes tools evaluatating stereo matching algorithms
"""
import numpy as np
import cv2


def normalise_disp(disp_image):
    """
    Normalise disparity image

    Parameters:
        disp_image (numpy): disparity image
    Results:
        norm_disp_image (numpy): normalised disparity image
    """
    norm_disp_image = disp_image
    # remove negative disparities
    norm_disp_image[norm_disp_image < 0] = 0.0
    # Replace nan and inf values with zero
    norm_disp_image = np.nan_to_num(norm_disp_image, nan=0.0, posinf=0.0, neginf=0.0)
    # normalise image
    norm_disp_image = cv2.normalize(norm_disp_image, None, 0, 255, cv2.NORM_MINMAX)
    # convert to uint8
    norm_disp_image = norm_disp_image.astype(np.uint8)
    return norm_disp_image


def colormap_disp(disp_image):
    """
    Apply colormap to disparity

    Parameters:
        disp_image (numpy): disparity image
    Returns:
        colormap_disp_image (numpy): disparity image with colormap applied
    """
    colormap_disp_image = normalise_disp(disp_image)
    colormap_disp_image = cv2.applyColorMap(colormap_disp_image, cv2.COLORMAP_JET)
    return colormap_disp_image


# Helpful functions
def in_dictlist(key, value, my_dictlist):
    """Check if key value pair exists in dictionary"""
    for this in my_dictlist:
        if this[key] == value:
            return True
    return False


def puttext_multiline(img, text, org, font, font_scale, color, thickness,
                      line_type=cv2.LINE_AA, line_spacing=1.5, outline_color=None):
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


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image based on height or width while maintaning aspect ratio
    Parameters:
        image (numpy): image to resize
        width (int): desired width of output image (can only use width or height, not both)
        height (int): desired height of output image (can only use width or height, not both)
        inter (int): OpenCV resize method (default: cv2.INTER_AREA)

    Returns:
        resized (numpy): resized image
    """
    if width is not None and height is not None:
        raise Exception("Cannot set both height and width")
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
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
