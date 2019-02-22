import cv2
import numpy as np
from math import pi
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
    Find the corners of a contour forming a regular polygon.
    """
    (xc, yc), r = cv2.minEnclosingCircle(cnt)
    cnt = np.squeeze(cnt)

    # calculate the angles of all contour points
    angles = np.arctan2(cnt[:, 1]-yc, cnt[:, 0]-xc) * 180/pi

    # convert angles
    div = 360 / 2
    if aligned:
        angles[angles<-div/2] = 360 + angles[angles<-div/2]
    else:
        angles[angles<0] = 360 + angles[angles<0]

    # Find distance between all contour points and the centre of the contour
    dists = sp.distance.cdist(np.array([[xc, yc]]), cnt)
    dists = np.squeeze(dists)

    # Find the furthest point in n, 360/n degree segments.
    corners = []
    for segment in range(n):
        if aligned:
            index = np.nonzero(~((angles >= (-div/2 + div*segment))
                                 * (angles < (-div/2 + div*(segment+1)))))
        else:
            index = np.nonzero(~((angles >= div*segment) *
                                 (angles <= div*(segment+1))))
        temp_dists = dists.copy()
        temp_dists[index] = 0
        sort_list = np.argsort(temp_dists)
        corners.append(sort_list[-1])
    return corners, (xc, yc)

