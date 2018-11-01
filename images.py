import cv2
import numpy as np
import Generic.filedialogs as fd


def get_width_and_height(img):
    width = get_width(img)
    height = get_height(img)
    return width, height


def get_width(img):
    return np.shape(img)[0]


def get_height(img):
    return np.shape(img)[1]


def resize(img, percent=25):
    width, height = get_width_and_height(img)
    dim = (int(height * percent / 100), int(width * percent / 100))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def display(image, title=''):
    """Uses cv2 to display an image then wait for a button press"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bgr_2_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def threshold(img, thresh=100):
    ret, out = cv2.threshold(
            img,
            thresh,
            255,
            cv2.THRESH_BINARY)
    return out


def adaptive_threshold(img, block_size=5, constant=0):
    out = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant
            )
    return out


def gaussian_blur(img, kernel=(3,3)):
    out = cv2.GaussianBlur(img, kernel, 0)
    return out


def dilate(img, kernel=(3,3)):
    out = cv2.dilate(img, kernel)
    return out


def erode(img, kernel=(3,3)):
    out = cv2.erode(img, kernel)
    return out


def closing(img, kernel=(3,3)):
    out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return out


def opening(img, kernel=(3,3)):
    out = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return out


def distance_transform(img):
    out = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    return out


def rotate(img, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH))


def imfill(img):
    # img should be a thresholded image
    # Copy the thresholded image.
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    out = img | im_floodfill_inv

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


def load_img(filename):
    img = cv2.imread(filename)
    return img


def write_img(img, filename):
    cv2.imwrite(filename, img)


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


def mask_img(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def crop_img(img, crop):
    if len(np.shape(img)) == 3:
        out = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :]
    else:
        out = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    return out

if __name__ == "__main__":
    filename = fd.load_filename()

    img = load_img(filename)
    display(img)

    width, height = get_width_and_height(img)

    img = resize(img, 50)
    display(img, 'resize')

    img = bgr_2_grayscale(img)
    display(img, 'grayscale')

    img = rotate(img, 45)
    display(img, 'rotate')

    thresh = threshold(img, 100)
    display(thresh, 'simple threshold')

    adap = adaptive_threshold(img)
    display(adap, 'adaptive threshold')

    blur = gaussian_blur(img)
    display(blur, 'blur')

    dil = dilate(thresh)
    display(dil, 'dilation')

    ero = erode(thresh)
    display(ero, 'erosion')

    clo = closing(thresh)
    display(clo, 'closing')

    ope = opening(thresh)
    display(ope, 'opening')

    dist = distance_transform(thresh)
    display(dist, 'distance_transform')

    white, mask = set_edge_white(img)
    display(white, 'set edge white')

    right = mask_right(img, int(width/2))
    display(right, 'mask right')

    top = mask_top(img, int(height/2))
    display(top, 'mask top')

    big = extract_biggest_object(img)
    display(big)