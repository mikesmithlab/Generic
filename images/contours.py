import cv2
import numpy as np
from math import pi, cos, sin
import scipy.optimize as op
from skimage.transform import hough_ellipse
from Generic import images
import matplotlib.pyplot as plt

__all__ = ['find_contours','rotated_bounding_rectangle','separate_rects', 'sort_contours', 'find_contour_corners', 'fit_hex']


def find_contours(img, hierarchy=False):
    '''
    contours is a tuple containing (img, contours)
    '''
    # work for any version of opencv
    try:
        im, contours, hier = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, hier = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy:
        return contours, hier
    else:
        return contours

def rotated_bounding_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.rotatedRectangleIntersection(rect1, rect2)
    return box, rect

def separate_rects(contour, box):
    """

    :param contour:
    :param crop: crop is the box coords
    :return:
    """
    print(contour)

    crop = cv2.boundingRect(box)
    print(crop)
    img = np.zeros((crop[3],crop[2]),dtype=np.uint8)
    temp_contour = []
    for pt in contour:
        temp_contour.append([pt[0]-crop[0],pt[1]-crop[1]])
    img = images.draw_contours(img, contour, thickness=-1)
    plt.figure()
    plt.imshow(img)
    plt.show()
    img = images.skeleton(img)
    return img



def hough_ellipse(img, a,b):
    """
    https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
    :param img: grayscale image
    :param a: = max_size ellipse
    :param b: = min_size ellipse
    :return:
    """


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


def fit_hex(contour):
    """
    Fits a regular hexagon to a list of points.

    Uses scipy.optimize.minimize to minimise the distance between the points
    and the hexagon.
    """
    (xc, yc), radius = cv2.minEnclosingCircle(contour)
    res = op.minimize(hex_dist, (xc, yc, radius, 0), args=contour)
    (xc, yc, r, theta) = res.x
    hex_corners = hexagon(xc, yc, r, theta)
    return hex_corners


def hexagon(xc, yc, r, theta):
    points = [[xc + r * cos(t + theta), yc + r * sin(t + theta)]
              for t in [0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3]]
    return np.array(points)


def hex_dist(params, contour):
    """https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line"""
    xc, yc, r, theta = params
    points = hexagon(xc, yc, r, theta)
    distances = np.zeros((len(contour), 6))
    x0, y0 = np.split(contour, 2, axis=1)
    for p in range(6):
        x1, y1 = points[p-1, :]
        x2, y2 = points[p, :]
        distance = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances[:, p] = np.squeeze(distance)
    distances = np.min(distances, axis=1)
    return np.sum(distances)



