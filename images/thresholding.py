import cv2


__all__ = ['threshold', 'adaptive_threshold', 'distance_transform']


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


def distance_transform(img):
    """
    Calculates the distance to the closest zero pixel for each pixel.

    Calculates the approximate or precise distance from every binary image
    pixel to the nearest zero pixel. For zero image pixels, the distance will
    obviously be zero.

    Parameters
    ----------
    img: 8-bit, single-channel (binary) source image.

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
    out = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    return out


class threshold_slider:

    def __init__(self, img, type=cv2.THRESH_BINARY):
        self.im = img
        self.im0 = img.copy()
        self.g = 0
        self.type = type
        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('image', 960, 540)
        cv2.createTrackbar('thresh', 'image', 0, 255, self.change)
        while(1):
            cv2.imshow('image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                break
        cv2.destroyAllWindows()

    def change(self, g):
        # g = cv2.getTrackbarPos('thresh')
        if g != self.g:
            self.im = threshold(self.im0, thresh=g, mode=self.type)
            self.g = g
