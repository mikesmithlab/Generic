import cv2

__all__ = ['BLUE', 'LIME', 'RED', 'YELLOW', 'ORANGE', 'BLACK', 'WHITE',
           'MAGENTA', 'PINK', 'CYAN', 'NAVY', 'TEAL', 'PURPLE', 'GREEN',
           'MAROON', 'bgr_to_hsv', 'bgr_to_lab', 'hsv_to_bgr', 'lab_to_bgr']

BLUE = (255, 0, 0)
LIME = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 128, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
PINK = MAGENTA
CYAN = (255, 255, 0)
NAVY = (128, 0, 0)
TEAL = (128, 128, 0)
PURPLE = (128, 0, 128)
GREEN = (0, 128, 0)
MAROON = (0, 0, 128)


def bgr_to_hsv(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


def bgr_to_lab(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2LAB)


def hsv_to_bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_HSV2BGR)


def lab_to_bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
