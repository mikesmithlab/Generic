import cv2
import numpy as np
from . import *

__all__ = ['extract_biggest_object', 'find_circles', 'find_colour',
           'find_color']


def extract_biggest_object(img):
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    stats = stats[1:][:]
    index = np.argmax(stats[:, cv2.CC_STAT_AREA]) + 1
    out = np.zeros(np.shape(img))
    try:
        out[labels == index] = 255
    except:
        print(output[0])
        display(img)
    return out


def find_circles(img, min_dist, p1, p2, min_rad, max_rad):
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT, 1,
        min_dist,
        param1=p1,
        param2=p2,
        minRadius=min_rad,
        maxRadius=max_rad)
    return np.squeeze(circles)


def find_colour(image, col):
    """
    LAB colorspace allows finding colours somewhat independent of
    lighting conditions.

    https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
    """
    # Swap to LAB colorspace
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if col == 'Blue':
        b = lab[:, :, 2]
        blue = threshold(b, mode=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return ~blue

find_color = find_colour
