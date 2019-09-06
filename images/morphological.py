import cv2
import numpy as np
from Generic import images
import matplotlib.pyplot as plt


__all__ = ['dilate', 'erode', 'closing', 'opening','skeleton', 'opening_slider', 'closing_slider']


def dilate(img, kernel=(3, 3), kernel_type=None, iterations=1):
    """
    Dilates an image by using a specific structuring element.

    The function dilates the source image using the specified structuring
    element that determines the shape of a pixel neighborhood over which
    the maximum is taken

    Parameters
    ----------
    img: binary input image
        Can have any number of channels which are processed separately

    kernel: tuple giving (width, height) for kernel
        Width and height should be positive and odd

    Returns
    -------
    out: output image
        Same size and type as img

    Notes
    -----
    It dilates the boundaries of the foreground object (Always try to keep
    foreground in white).

    A pixel in the original image (either 1 or 0) will be considered 1 if
    any of the pixels under the kernel is 1.

    """
    if kernel_type is not None:
        kernel = cv2.getStructuringElement(kernel_type, kernel)
    else:
        kernel = np.ones(kernel)
    out = cv2.dilate(img, kernel, iterations=iterations)
    return out


def erode(img, kernel=(3, 3), kernel_type=None, iterations=1):
    """
    Erodes an image by using a specific structuring element.

    The function erodes the source image using the specified structuring
    element that determines the shape of a pixel neighborhood over which
    the minimum is taken.

    Parameters
    ----------
    img: binary input image
        Number of channels can be arbitrary

    kernel: tuple giving (width, height) for kernel
        Width and height should be positive and odd

    Returns
    -------
    out: output image
        Same size and type as img

    Notes
    -----
    It erodes away the boundaries of foreground object
    (Always try to keep foreground in white).

    A pixel in the original image (either 1 or 0) will be considered 1
    only if all the pixels under the kernel is 1, otherwise it is eroded
    (made to zero).


    """
    if kernel_type is not None:
        kernel = cv2.getStructuringElement(kernel_type, kernel)
    else:
        kernel=np.ones(kernel)
    out = cv2.erode(img, kernel, iterations)
    return out


def closing(img, kernel=(3, 3)):
    """
    Performs a dilation followed by an erosion

    Parameters
    ----------
    img: binary input image
        Number of channels can be arbitrary

    kernel: tuple giving (width, height) for kernel
        Width and height should be positive and odd

    Returns
    -------
    out: output image
        Same size and type as img

    """
    out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return out


def opening(img, kernel=(3, 3), kernel_type=None, iterations=1):
    """
    Performs an erosion followed by a dilation

    Parameters
    ----------
    img: binary input image
        Number of channels can be arbitrary

    kernel: tuple giving (width, height) for kernel
        Width and height should be positive and odd

    kernel_type: Either None or cv2.MORPH_?????

    Returns
    -------
    out: output image
        Same size and type as img

    """
    if kernel_type is not None:
        kernel = cv2.getStructuringElement(kernel_type, kernel)
    else:
        kernel = np.ones(kernel)
    out = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return out

def skeleton(img):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done=False
    size=np.shape(img)
    skel=np.zeros(size,dtype=np.uint8)


    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()


        #zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

class opening_slider:

    def __init__(self, img, type=cv2.THRESH_BINARY):
        self.im = img
        self.im0 = img.copy()
        self.k = 0
        self.type = type
        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('image', 960, 540)
        cv2.createTrackbar('kernel (odd)', 'image', 0, 31, self.change)
        while(1):
            cv2.imshow('image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                break
        cv2.destroyAllWindows()

    def change(self, k):
        if k % 2 == 0:
            k += 1
        self.im = opening(self.im0, kernel=(k, k))
        self.k = k


class closing_slider:

    def __init__(self, img, type=cv2.THRESH_BINARY):
        self.im = img
        self.im0 = img.copy()
        self.k = 0
        self.type = type
        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('image', 960, 540)
        cv2.createTrackbar('kernel (odd)', 'image', 0, 31, self.change)
        while(1):
            cv2.imshow('image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                break
        cv2.destroyAllWindows()

    def change(self, k):
        if k % 2 == 0:
            k += 1
        self.im = closing(self.im0, kernel=(k, k))
        self.k = k