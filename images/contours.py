import cv2


def find_contours(img, hierarchy=False):
    contours, hier = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy:
        return contours, hier
    else:
        return contours
