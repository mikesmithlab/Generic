import cv2


__all__ = ['gaussian_blur']


def gaussian_blur(img, kernel=(3, 3)):
    """
    Blurs an image using a gaussian filter

    The function convolves the source image with the specified Gaussian kernel.

    Parameters
    ----------
    img: input image
        Can have any number of channels which are processed separately

    kernel: tuple giving (width, height) for kernel
        Width and height should be positive and odd

    Returns
    -------
    out: output image
        Same size and type as img
    """
    out = cv2.GaussianBlur(img, kernel, 0)
    return out

def median_blur(img, kernel=(3,3)):
    """
    Blurs an image using a median filter

    The function convolves the source image with the specified kernel.

    Parameters
    ----------
    img: input image
        Can have any number of channels which are processed separately

    kernel: tuple giving (width, height) for kernel
        Width and height should be positive and odd

    Returns
    -------
    out: output image
        Same size and type as img
    """
    out = cv2.medianBlur(img, kernel, 0)
    return out