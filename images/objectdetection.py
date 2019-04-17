import cv2
import numpy as np
from math import pi, cos, sin
import scipy.optimize as op

__all__ = ['find_connected_components']

def find_connected_components(thresh_img,connectivity=4, option=cv2.CV_32S):
    """

    :param thresh_img: thresholded image
    :param connectivity: can be 4 or 8
    :param option:
    :return: labels, stats, centroids
    labels is a matrix the same size as the image where each element has a
    value equal to its label
    stats[label, COLUMN] where available columns are defined below.
        cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive
        start of the bounding box in the horizontal direction.
        cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive
        start of the bounding box in the vertical direction.
        cv2.CC_STAT_WIDTH The horizontal size of the bounding box
        cv2.CC_STAT_HEIGHT The vertical size of the bounding box
        cv2.CC_STAT_AREA The total area (in pixels) of the connected component
    centroids is a matrix with the x and y locations of each centroid.
    The row in this matrix corresponds to the label number.
    """
    output = cv2.connectedComponentsWithStats(thresh_img, connectivity, option)

    #num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    return labels, stats, centroids
