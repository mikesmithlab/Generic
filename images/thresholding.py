import cv2
import numpy as np
from . import *


__all__ = ['threshold', 'adaptive_threshold', 'distance_transform', 'watershed']


def threshold(img, thresh=None, mode=cv2.THRESH_BINARY):
    """
    Thresholds an image

    Pixels below thresh set to black, pixels above set to white
    """
    if thresh is None:
        mode = mode + cv2.THRESH_OTSU
        ret, out = cv2.threshold(
                img,
                0,
                255,
                mode)
    else:
        ret, out = cv2.threshold(
            img,
            thresh,
            255,
            mode)
    return out


def adaptive_threshold(img, block_size=5, constant=0, mode=cv2.THRESH_BINARY):
    """
    Performs an adaptive threshold on an image

    Uses cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
        threshold value is the weighted sum of neighbourhood values where
        weights are a gaussian window.

    Uses cv2.THRESH_BINARY:
        Pixels below the threshold set to black
        Pixels above the threshold set to white

    Parameters
    ----------
    img: numpy array containing an image

    block_size: the size of the neighbourhood area

    constant: subtracted from the weighted sum
    """
    out = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            mode,
            block_size,
            constant
            )
    return out



def watershed(img, watershed_threshold = 0.1, block_size=5,constant=0,mode=cv2.THRESH_BINARY):
    depth = get_depth(img)
    if depth == 3:
        grayscale_img = bgr_2_grayscale(img)
    else:
        grayscale_img = img.copy()
        img = grayscale_2_bgr(img)
    binary_img = adaptive_threshold(grayscale_img,block_size=block_size, constant=constant, mode=mode)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel,
                               iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform_img = distance_transform(binary_img)
    #sure foreground area
    sure_fg = threshold(dist_transform_img, thresh=watershed_threshold)
    sure_fg = np.uint8(sure_fg)

    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255,0,0]

    return img


def distance_transform(img, preprocess=False, block_size=5,constant=0,mode=cv2.THRESH_BINARY):
    """
    Calculates the distance to the closest zero pixel for each pixel.

    Calculates the approximate or precise distance from every binary image
    pixel to the nearest zero pixel. For zero image pixels, the distance will
    obviously be zero.

    Parameters
    ----------
    img: 8-bit image.
    preprocess = True applies a series of operations: binarise with adaptive threshold, denoise with opening.
               = False just applies the distance transform. img must already be a binary image.
    Remaining params are the input for the adaptive threshold.

    Returns
    -------
    out: Output image with calculated distances.
        It is a 8-bit or 32-bit floating-point, single-channel image of the
        same size as img.

    References
    ----------
    Pedro Felzenszwalb and Daniel Huttenlocher. Distance transforms of sampled
    functions. Technical report, Cornell University, 2004.
    """
    if preprocess:
        grayscale_img = images.bgr_2_grayscale(img)
        binary_img = adaptive_threshold(grayscale_img, block_size=block_size,
                                    constant=constant, mode=mode)
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel,
                               iterations=2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
    else:
        binary_img = img
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    return dist_transform


if __name__ == '__main__':
    from Generic import images
    from Generic import video

    read_vid = video.ReadVideo('/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.avi')
    im = read_vid.read_next_frame()

    images.display(im, title='a')