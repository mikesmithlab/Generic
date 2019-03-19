import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import *

__all__ = ['extract_biggest_object', 'find_circles', 'find_colour',
           'find_color', 'histogram_peak', 'Circle_GUI']


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


class Circle_GUI:

    def __init__(self, img):
        self.im = img
        self.im0 = img.copy()
        self.distance = 5
        self.thresh1 = 200
        self.thresh2 = 5
        self.min_rad = 3
        self.max_rad = 7
        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('image', 960, 540)
        cv2.createTrackbar('Min Distance', 'image', 5, 51, self.change_d)
        cv2.createTrackbar('Thresh 1', 'image', 200, 255, self.change_t1)
        cv2.createTrackbar('Thresh 2', 'image', 5, 10, self.change_t2)
        cv2.createTrackbar('Min Radius', 'image', 3, 51, self.change_min_r)
        cv2.createTrackbar('Max Radius', 'image', 3, 51, self.change_max_r)
        while(1):
            cv2.imshow('image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                break
        cv2.destroyAllWindows()

    def change_d(self, d):
        if d == 0:
            d = 1
        self.distance = d
        self.update()

    def change_t1(self, d):
        self.thresh1 = d
        self.update()

    def change_t2(self, d):
        self.thresh2 = d
        self.update()

    def change_min_r(self, d):
        if d > self.max_rad:
            d = self.max_rad - 2
        self.min_rad = d
        self.update()

    def change_max_r(self, d):
        if d < self.min_rad:
            d = self.min_rad + 2
        self.max_rad = d
        self.update()

    def update(self):
        circles = find_circles(self.im0, self.distance,
                               self.thresh1, self.thresh2,
                               self.min_rad, self.max_rad)
        self.im = self.im0.copy()
        self.im = draw_circles(np.dstack((self.im, self.im, self.im)), circles)




find_color = find_colour
