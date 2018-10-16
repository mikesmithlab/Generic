"""
This module will contain simple short functions
"""

import cv2

def cv2im(image, title=''):
    """Uses cv2 to display an image then wait for a button press"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()