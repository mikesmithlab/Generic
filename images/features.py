import cv2
import numpy as np
import matplotlib.pyplot as plt

from . import *
from Generic.images.drawing import draw_circles

__all__ = ['extract_biggest_object', 'find_circles', 'find_colour',
           'find_color', 'histogram_peak']


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


def find_circles(img, min_dist, p1, p2, min_rad, max_rad, dp=1):
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT, dp,
        min_dist,
        param1=p1,
        param2=p2,
        minRadius=min_rad,
        maxRadius=max_rad)
    return np.squeeze(circles)


def histogram_peak(im, disp=False):
    if len(np.shape(im)) == 2:
        data, bins = np.histogram(np.ndarray.flatten(im), bins=np.arange(20, 255, 1))
        peak = bins[np.argmax(data)]
    if disp:
        plt.figure()
        plt.plot(bins[:-1], data)
        plt.show()
    return peak


def find_colour(image, col):
    """
    LAB colorspace allows finding colours somewhat independent of
    lighting conditions.

    https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
    """
    # Swap to LAB colorspace
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2]
    if col == 'Blue':
        peak = histogram_peak(b, disp=False)
        blue = threshold(b, thresh=peak-8, mode=cv2.THRESH_BINARY)
        return ~blue


find_color = find_colour


if __name__ == '__main__':
    pass