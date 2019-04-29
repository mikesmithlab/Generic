import cv2
import numpy as np
from . import *
from .colors import *

import scipy.spatial as sp


__all__ = ['draw_voronoi_cells', 'draw_polygons', 'draw_polygon',
           'draw_delaunay_tess', 'draw_circle', 'draw_circles',
           'draw_contours']


def draw_voronoi_cells(img, points):
    """
    Draws the voronoi cells for a set of points on an image

    Parameters
    ----------
    img: input image
        Any number of channels

    points: array of N points
        Shape (N, 2).
        points[:, 0] contains x coordinates
        points[:, 1] contains y coordinates

    Returns
    -------
    ing: annotated image
        Same shape and type as input image
    """
    img = check_image_depth(img)
    voro = sp.Voronoi(points)
    ridge_vertices = voro.ridge_vertices
    new_ridge_vertices = []
    for ridge in ridge_vertices:
        if -1 not in ridge:
            new_ridge_vertices.append(ridge)
    img = draw_polygons(img,
                        voro.vertices[new_ridge_vertices],
                        color=PINK)
    return img


def draw_polygons(img, polygons, color=RED):
    """
    Draws multiple polygons on an image from a list of polygons

    Parameters
    ----------
    img: input image
        Any number of channels

    polygons: array containing coordinates of polygons
        shape is (P, N, 2) where P is the number of polygons, N is the number
        of vertices in each polygon. [:, :, 0] contains x coordinates,
        [:, :, 1] contains y coordinates.

    color: BGR tuple

    Returns
    -------
    img: annotated image
        Same shape and type as input image
    """
    img = check_image_depth(img)
    for vertices in polygons:
        img = draw_polygon(img, vertices, color)
    return img


def draw_polygon(img, vertices, color=RED, thickness=1):
    """
    Draws a polygon on an image from a list of vertices

    Parameters
    ----------
    img: input image
        Any number of channels

    vertices: array of N vertices
        Shape (N, 2) where
            vertices[:, 0] contains x coordinates
            vertices[:, 1] contains y coordinates

    color: BGR tuple
        if input image is grayscale then circles will be black

    thickness: int
        Thickness of the lines

    Returns
    -------
    out: output image
        Same shape and type as input image
    """
    img = check_image_depth(img)
    vertices = vertices.astype(np.int32)
    out = cv2.polylines(img, [vertices], True, color, thickness=thickness)
    return out


def draw_delaunay_tess(img, points):
    """
    Draws the delaunay tesselation for a set of points on an image

    Parameters
    ----------
    img: input image
        Any number of channels

    points: array of N points
        Shape (N, 2).
        points[:, 0] contains x coordinates
        points[:, 1] contains y coordinates

    Returns
    -------
    ing: annotated image
        Same shape and type as input image
    """
    img = check_image_depth(img)
    tess = sp.Delaunay(points)
    img = draw_polygons(img,
                        points[tess.simplices],
                        color=LIME)
    return img


def draw_circle(img, cx, cy, rad, color=YELLOW, thickness=2):
    img = check_image_depth(img)
    cv2.circle(img, (int(cx), int(cy)), int(rad), color, thickness)
    return img


def draw_circles(img, circles, color=YELLOW, thickness=2):
    """
    Draws circles on an image.

    Parameters
    ----------
    img: Input image
        Any number of channels

    circles: array of shape (N, 3) containing x, y, and radius of N circles
        circles[:,0] contains the x-coordinates of the centers
        circles[:,1] contains the y-coordinates of the centers
        circles[:,2] contains the radii of the circles
            Can not have radii dimension

    color: BGR tuple
        If input image is grayscale circles will be black

    thickness: pixel thickness
        If -1, the circle is filled in

    Returns
    -------
    img: image with annotated circles
        Same height, width and channels as input image
    """
    img = check_image_depth(img)
    try:
        if np.shape(circles)[1] == 3:
            for x, y, rad in circles:
                cv2.circle(img, (int(x), int(y)), int(rad), color, thickness)
        elif np.shape(circles)[1] == 2:
            for x, y in circles:
                cv2.circle(img, (int(x), int(y)), int(5), color, thickness)
    except IndexError as error:
        print('no circles', error)
    return img


def draw_contours(img, contours, col=RED, thickness=1):
    """

    :param img:
    :param contours:
    :param col: Can be a defined colour in colors.py or a list of tuples(3,1) of colors of length contours
    :param thickness: -1 fills the contour.
    :return:
    """

    img = check_image_depth(img)

    if np.size(np.shape(col)) == 1:
        img = cv2.drawContours(img, contours, -1, col, thickness)
    else:
        for i, contour in enumerate(contours):
            img = cv2.drawContours(img, contour, -1, col[i], thickness)
    return img


def check_image_depth(img):
    depth = get_depth(img)
    if depth == 1:
        return grayscale_2_bgr(img)
    else:
        return img