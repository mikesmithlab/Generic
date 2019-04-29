import numpy as np
import cv2
from .basics import *
from .colors import *
import math

__all__ = ['resize', 'rotate', 'hstack', 'vstack']

def resize(img, percent=25.0):
    """
    Resizes an image to a given percentage

    Parameters
    ----------
    img: numpy array containing an image

    percent:
        the new size of the image as a percentage

    Returns
    -------
    resized_image:
        The image after it's been resized

    """
    width, height = get_width_and_height(img)
    dim = (int(width * percent / 100), int(height * percent / 100))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def rotate(img, angle):
    """
    Rotates an image without cropping it

    Parameters
    ----------
    img: input image
        Can have any number of channels

    angle: angle to rotate by in degrees
        Positive values mean clockwise rotation

    Returns
    -------
    out: output image
        May have different dimensions than the original image

    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (c_x, c_y) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rot_matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])

    # compute the new bounding dimensions of the image
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    rot_matrix[0, 2] += (n_w / 2) - c_x
    rot_matrix[1, 2] += (n_h / 2) - c_y

    # perform the actual rotation and return the image
    out = cv2.warpAffine(img, rot_matrix, (n_w, n_h))
    return out


def hstack(*args):
    """
    Stacks images horizontally

    If image depths are mismatched then converts grayscale images to bgr before stacking
    """
    depths = [get_depth(im) for im in args]
    gray = [d == 1 for d in depths]
    if all(gray):
        return np.hstack(args)
    else:
        ims = [grayscale_2_bgr(im) for im in args]
        return np.hstack(ims)


def vstack(*args):
    """
    Stacks images vertically

    If image depths are mismatched then converts grayscale images to bgr before stacking
    """
    depths = [get_depth(im) for im in args]
    gray = [d == 1 for d in depths]
    if all(gray):
        return np.vstack(args)
    else:
        ims = [grayscale_2_bgr(im) for im in args]
        return np.vstack(ims)