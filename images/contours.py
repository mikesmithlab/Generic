import cv2
import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt
import scipy.spatial as sp

__all__ = ['find_contours', 'sort_contours', 'find_contour_corners']


def find_contours(img, hierarchy=False):
    contours, hier = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy:
        return contours, hier
    else:
        return contours


def sort_contours(cnts):
    """
    Sorts contours by area from smallest to largest.

    Parameters
    ----------
    cnts: list
        List containing contours.

    Returns
    -------
    cnts_new: list
        List of input contours sorted by area.
    """
    area = []
    for cnt in cnts:
        area.append(cv2.contourArea(cnt))
    sorted = np.argsort(area)
    cnts_new = []
    for arg in sorted:
        cnts_new.append(cnts[arg])
    return cnts_new


def find_contour_corners(cnt, n, aligned=True):
    """
    Find the corners of a contour forming a regular polygon

    Parameters
    ----------
    cnt: contour points along a regular polygon
    n: number of sides
    aligned: set true if corner in west direction.

    Returns
    -------
    corners: indices of cnt corresponding to corners

    """
    (xc, yc), r = cv2.minEnclosingCircle(cnt)
    cnt = np.squeeze(cnt)

    # calculate the angles of all contour points
    vectors = cnt - (xc, yc)
    if aligned:
        R = np.array(((cos(pi/6), -sin(pi/6)), (sin(pi/6), cos(pi/6))))
        vectors = np.dot(vectors, R)
    r = vectors[:, 0]**2 + vectors[:, 1]**2
    theta = np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / pi
    angles = np.linspace(-180, 180, n+1)
    corners = []
    for i in range(n):
        in_region = np.nonzero((theta >= angles[i])*(theta < angles[i+1]))
        max_r = np.argmax(r[in_region])
        corners.append(in_region[0][max_r])
    return corners, (xc, yc)






