import cv2
import numpy as np
from .colors import *

__all__ = ['crop_img', 'crop_and_mask_image', 'mask_img', 'set_edge_white',
           'mask_right', 'mask_top', 'InteractiveCrop', 'imfill', 'CropShape']


def crop_img(img, crop):
    """
    Crops an image.

    Parameters
    ----------
    img: input image
        Any number of channels

    crop: list containing crop values
        crop = [[xmin, xmax], [ymin, ymax]] where x goes from left to right
        and y goes from top to bottom in the image

    Returns
    -------
    out: cropped image
    """
    if len(np.shape(img)) == 3:
        # out = img[ymin:ymax, xmin:xmax
        # crop = ([xmin, ymin], [xmax, ymax])
        out = img[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0], :]
    else:
        out = img[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]
    return out


def crop_and_mask_image(img, crop, mask, mask_color='black'):
    img = mask_img(img, mask, mask_color)
    img = crop_img(img, crop)
    return img


def mask_img(img, mask, color='black'):
    """
    Masks pixels in an image.

    Pixels in the image that are 1 in the mask are kept.
    Pixels in the image that are 0 in the mask are set to 0.

    Parameters
    ----------
    img: The input image
        Any number of channels

    mask: Mask image
        Same height and width as img containing 0 or 1 in each pixel

    color: Color of the mask

    Returns
    -------
    out: The masked image
        Same dimensions and type as img
    """
    if color == 'black':
        out = cv2.bitwise_and(img, img, mask=mask)
    else:
        out = cv2.bitwise_and(img, img, mask=mask)
        add = cv2.cvtColor(~mask, cv2.COLOR_GRAY2BGR)
        out = cv2.add(out, add)
    return out


def set_edge_white(img, column=200):
    img = img.copy()
    img[:, column] = 255
    img[:, column + 1] = 255
    img[:, column - 1] = 1
    img[:, column - 2] = 0
    img[:, column - 3] = 0
    img[:, column - 4] = 0
    img[:, column - 5] = 0
    img[:, column - 6] = 0
    img[:, column - 7] = 0
    img[:, column - 8] = 0
    img[:, column - 9] = 0
    mask = np.zeros(np.shape(img))
    mask[:, column] = 255
    mask[:, column] = 255
    mask[:, column+1] = 255
    mask[:, column+2] = 255
    return img, mask


def mask_right(img, column=-100):
    img = img.copy()
    rh_edge = np.shape(img)[1]
    img[:, rh_edge+column:1] = 0
    return img


def mask_top(img, row=0):
    img = img.copy()
    img[0:row, :] = 0
    return img


class InteractiveCrop:
    """ Take an interactive crop of a shape"""

    def __init__(self, input_image, no_of_sides=1):
        """
        Initialise with input image and the number of sides:

        Parameters
        ----------
        input_image: 2D numpy array
            contains an image
        no_of_sides: int
            The number of sides that the desired shape contains.
                1: Uses a circle
                >2: Uses a polygon

        """

        super().__init__()
        self.cropping = False
        self.refPt = []
        self.image = input_image
        self.no_of_sides = no_of_sides
        self.original_image = self.image.copy()

    def _click_and_crop(self, event, x, y, flags, param):
        """Internal method to manage the user cropping"""

        if event == cv2.EVENT_LBUTTONDOWN:
            # x is across, y is down
            self.refPt = [(x, y)]
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.cropping = False
            if self.no_of_sides == 1:
                self.refPt.append((x, y))
                cx = ((self.refPt[1][0] - self.refPt[0][0]) / 2 +
                      self.refPt[0][0])
                cy = ((self.refPt[1][1] - self.refPt[0][1]) / 2 +
                      self.refPt[0][1])
                rad = int((self.refPt[1][0] - self.refPt[0][0]) / 2)
                cv2.circle(self.image, (int(cx), int(cy)), rad, LIME, 2)
                cv2.imshow('crop: '+str(self.no_of_sides), self.image)
            print(self.refPt)

    def begin_crop(self):
        """Method to create the mask image and the crop region"""

        clone = self.image.copy()
        points = np.zeros((self.no_of_sides, 2))
        cv2.namedWindow('crop: '+str(self.no_of_sides), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('crop: '+str(self.no_of_sides), 960, 540)
        cv2.setMouseCallback('crop: '+str(self.no_of_sides),
                             self._click_and_crop)
        count = 0

        # keep looping until 'q' is pressed
        while True:
            cv2.imshow('crop: '+str(self.no_of_sides), self.image)
            key = cv2.waitKey(1) & 0xFF

            if self.cropping and self.no_of_sides > 1:
                # self.refPt = [(x, y)]
                points[count, 0] = self.refPt[0][0]
                points[count, 1] = self.refPt[0][1]
                self.cropping = False
                count += 1

            if key == ord("r"):
                self.image = clone.copy()
                count = 0
                points = np.zeros((self.no_of_sides, 2))

            elif key == ord("c"):
                break

        cv2.destroyAllWindows()
        if self.no_of_sides == 1:
            points = self.refPt
        return self.find_crop_and_mask(points)

    def find_crop_and_mask(self, points):

        if self.no_of_sides == 1:
            # self.refPt = [(xmin, ymin), (xmax, ymax)]
            cx = (points[1][0] - points[0][0]) / 2 + points[0][0]
            cy = (points[1][1] - points[0][1]) / 2 + points[0][1]
            rad = int((points[1][0] - points[0][0]) / 2)
            mask_img = np.zeros((np.shape(self.original_image)))\
                .astype(np.uint8)
            cv2.circle(mask_img, (int(cx), int(cy)), int(rad), [255, 255, 255],
                       thickness=-1)
            crop = ([int(points[0][0]), int(cy-rad)],
                    [int(points[1][0]), int(cy+rad)])
            # crop = ([xmin, ymin], [xmax, ymax])
            boundary = np.array((cx, cy, rad), dtype=np.int32)
            return mask_img[:, :, 0], np.array(crop, dtype=np.int32), \
                boundary, points

        else:
            mask_img = np.zeros(np.shape(self.original_image)).astype('uint8')
            cv2.fillPoly(mask_img, pts=np.array([points], dtype=np.int32),
                         color=(250, 250, 250))
            crop = ([min(points[:, 0]), min(points[:, 1])],
                    [max(points[:, 0]), max(points[:, 1])])
            # crop = ([xmin, ymin], [xmax, ymax])
            return mask_img[:, :, 0], np.array(crop, dtype=np.int32), points, \
                points

CropShape = InteractiveCrop


def imfill(img):
    """
    Fills holes in the input binary image.

    A hole is a set of background pixels that cannot be reached by filling in
    the background from the edge of the image.

    Parameters
    ----------
    img: binary input image

    Returns
    -------
    out: binary output image
        same size and type as img
    """
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels larger than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    out = img | im_floodfill_inv

    return out