import cv2
import numpy as np
import Generic.filedialogs as fd
from scipy import spatial

# Color list BGR tuples
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


def get_width_and_height(img):
    """
    Returns width, height for an image

    Parameters
    ----------
    img: Array containing an image

    Returns
    -------
    width: int
        Width of the image
    height: int
        Height of the image

    Notes
    -----
    Width of an image is the first dimension for numpy arrays.
    Height of an image is the first dimension for openCV
    """
    width = get_width(img)
    height = get_height(img)
    return width, height


def get_width(img):
    """
    Returns width for img

    Parameters
    ----------
    img: Array containing an image

    Returns
    -------
    width: int
        Width of the image

    """
    return int(np.shape(img)[1])


def get_height(img):
    """
    Returns the height of an image

    Parameters
    ----------
    img: Array containing an image

    Returns
    -------
    height: int
        height of the image

    """
    return int(np.shape(img)[0])


def resize(img, percent=25.0):
    """
    Resizes an image to a given percentage

    Parameters
    ----------
    img: numpy array containing an image

    percent:
        the new size of the image as a percentage

    Returns
    -------
    resized_image:
        The image after it's been resized

    """
    width, height = get_width_and_height(img)
    dim = (int(height * percent / 100), int(width * percent / 100))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def display(image, title=''):
    """Uses cv2 to display an image then wait for a button press"""
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title, (960, 540))
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bgr_2_grayscale(img):
    """Converts a BGR image to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def threshold(img, thresh=100, type=cv2.THRESH_BINARY):
    """
    Thresholds an image

    Pixels below thresh set to black, pixels above set to white
    """
    ret, out = cv2.threshold(
            img,
            thresh,
            255,
            type)
    return out


def adaptive_threshold(img, block_size=5, constant=0):
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
            cv2.THRESH_BINARY,
            block_size,
            constant
            )
    return out


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


def dilate(img, kernel=(3, 3)):
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
    out = cv2.dilate(img, kernel)
    return out


def erode(img, kernel=(3, 3)):
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
    out = cv2.erode(img, kernel)
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


def opening(img, kernel=(3, 3)):
    """
    Performs an erosion followed by a dilation

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
    out = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
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
    out = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    return out


def rotate(img, angle):
    """
    Rotates an image without cropping it

    Parameters
    ----------
    img: input image
        Can have any number of channels

    angle: angle to rotate by in degrees
        Positive values mean clockwise rotation

    Returns
    -------
    out: output image
        May have different dimensions than the original image

    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (c_x, c_y) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rot_matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])

    # compute the new bounding dimensions of the image
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    rot_matrix[0, 2] += (n_w / 2) - c_x
    rot_matrix[1, 2] += (n_h / 2) - c_y

    # perform the actual rotation and return the image
    out = cv2.warpAffine(img, rot_matrix, (n_w, n_h))
    return out


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


def load_img(filepath, flag=1):
    """
    Reads an image from a filepath.

    The image should be in the working directory or a full path of image
    should be given.

    Parameters
    ----------
    filepath: filepath of the image

    flag: Specifies how the image is read
        1: Loads a color image. Any transparency will be neglected.
        0: Loads image in grayscale mode.
        -1: Loads image including alpha channel

    Returns
    -------
    img: output image
        Number of channels will be determined by the chosen flag.
        Equal to None if filepath does not exist
        Color images will have channels stored in BGR order

    """
    img = cv2.imread(filepath, flag)
    return img


def write_img(img, filename):
    """
    Saves an image to a specified file.

    The image format is chosen based on the filename extension

    Parameters
    ----------
    img: Image to be saved

    filename: Name of the file

    Notes
    -----
    Only 8-bit single channel or 3-channel (BGR order) can be saved. If
    the format, depth or channel order is different convert it first.

    It is possible to store PNG images with an alpha channel using this
    function. To do this, create 8-bit 4-channel image BGRA, where the alpha
    channel goes last. Fully transparent pixels should have alpha set to 0,
    fully opaque pixels should have alpha set to 255

    """
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


def mask_img(img, mask, colour='black'):
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

    Returns
    -------
    out: The masked image
        Same dimensions and type as img
    """
    if colour == 'black':
        out = cv2.bitwise_and(img, img, mask=mask)
    else:
        out = cv2.bitwise_and(img, img, mask=mask)
        add = cv2.cvtColor(~mask, cv2.COLOR_GRAY2BGR)
        out = cv2.add(out, add)
    return out


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

class CropShape:
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
            # self.refPt = [(xmin, ymin), (xmax, ymax)]
            cx = (self.refPt[1][0] - self.refPt[0][0]) / 2 + self.refPt[0][0]
            cy = (self.refPt[1][1] - self.refPt[0][1]) / 2 + self.refPt[0][1]
            rad = int((self.refPt[1][0] - self.refPt[0][0]) / 2)
            mask_img = np.zeros((np.shape(self.original_image)))\
                .astype(np.uint8)
            cv2.circle(mask_img, (int(cx), int(cy)), int(rad), [255, 255, 255], thickness=-1)
            crop = ([int(self.refPt[0][0]), int(cy-rad)],
                    [int(self.refPt[1][0]), int(cy+rad)])
            # crop = ([xmin, ymin], [xmax, ymax])
            boundary = np.array((cx, cy, rad), dtype=np.int32)
            return mask_img[:, :, 0], np.array(crop, dtype=np.int32), boundary

        else:
            mask_img = np.zeros(np.shape(self.original_image)).astype('uint8')
            cv2.fillPoly(mask_img, pts=np.array([points], dtype=np.int32),
                         color=(250, 250, 250))
            crop = ([min(points[:, 0]), min(points[:, 1])],
                    [max(points[:, 0]), max(points[:, 1])])
            # crop = ([xmin, ymin], [xmax, ymax])
            return mask_img[:, :, 0], np.array(crop, dtype=np.int32), points


def draw_circles(img, circles, color=YELLOW, thickness=2):
    """
    Draws circles on an image.

    Parameters
    ----------
    img: Input image
        Any number of channels

    circles: array of shape (N, 3) containing x, y, and radius of N circles
        circles[:,0] contains the x-coordinates of the centers
        circles[:,1] contains the y-coordinates of the centers
        circles[:,2] contains the radii of the circles
            Can not have radii dimension

    color: BGR tuple
        If input image is grayscale circles will be black

    thickness: pixel thickness
        If -1, the circle is filled in

    Returns
    -------
    img: image with annotated circles
        Same height, width and channels as input image
    """
    if circles is not None:
        if np.shape(circles)[1] == 3:
            for x, y, rad in circles:
                cv2.circle(img, (int(x), int(y)), int(rad), color, thickness)
        elif np.shape(circles)[1] == 2:
            for x, y in circles:
                cv2.circle(img, (int(x), int(y)), int(5), color, thickness)
    return img

def draw_circle(img, cx, cy, rad, color=YELLOW, thickness=2):
    cv2.circle(img, (int(cx), int(cy)), int(rad), color, thickness)
    return img


def draw_polygon(img, vertices, color=RED, thickness=1):
    """
    Draws a polygon on an image from a list of vertices

    Parameters
    ----------
    img: input image
        Any number of channels

    vertices: array of N vertices
        Shape (N, 2) where
            vertices[:, 0] contains x coordinates
            vertices[:, 1] contains y coordinates

    color: BGR tuple
        if input image is grayscale then circles will be black

    thickness: int
        Thickness of the lines

    Returns
    -------
    out: output image
        Same shape and type as input image
    """
    vertices = vertices.astype(np.int32)
    out = cv2.polylines(img, [vertices], True, color, thickness=thickness)
    return out


def draw_polygons(img, polygons, color=RED):
    """
    Draws multiple polygons on an image from a list of polygons

    Parameters
    ----------
    img: input image
        Any number of channels

    polygons: array containing coordinates of polygons
        shape is (P, N, 2) where P is the number of polygons, N is the number
        of vertices in each polygon. [:, :, 0] contains x coordinates,
        [:, :, 1] contains y coordinates.

    color: BGR tuple

    Returns
    -------
    img: annotated image
        Same shape and type as input image
    """
    print(polygons)
    for vertices in polygons:
        img = draw_polygon(img, vertices, color)
    return img


def draw_delaunay_tess(img, points):
    """
    Draws the delaunay tesselation for a set of points on an image

    Parameters
    ----------
    img: input image
        Any number of channels

    points: array of N points
        Shape (N, 2).
        points[:, 0] contains x coordinates
        points[:, 1] contains y coordinates

    Returns
    -------
    ing: annotated image
        Same shape and type as input image
    """
    tess = spatial.Delaunay(points)
    img = draw_polygons(img,
                        points[tess.simplices],
                        color=LIME)
    return img


def draw_voronoi_cells(img, points):
    """
    Draws the voronoi cells for a set of points on an image

    Parameters
    ----------
    img: input image
        Any number of channels

    points: array of N points
        Shape (N, 2).
        points[:, 0] contains x coordinates
        points[:, 1] contains y coordinates

    Returns
    -------
    ing: annotated image
        Same shape and type as input image
    """
    voro = spatial.Voronoi(points)
    ridge_vertices = voro.ridge_vertices
    new_ridge_vertices = []
    for ridge in ridge_vertices:
        if -1 not in ridge:
            new_ridge_vertices.append(ridge)
    img = draw_polygons(img,
                        voro.vertices[new_ridge_vertices],
                        color=PINK)
    return img


if __name__ == "__main__":
    filename = fd.load_filename()

    img = load_img(filename)
    display(img)

    #width, height = get_width_and_height(img)

    crop_inst = CropShape(img, 6)
    mask, crop, boundary = crop_inst.begin_crop()

    masked_im = mask_img(img, mask)
    masked_and_cropped = crop_img(masked_im, crop)
    display(masked_and_cropped, 'masked and cropped')
    #
    # img = resize(img, 50)
    # display(img, 'resize')
    #
    # img = bgr_2_grayscale(img)
    # display(img, 'grayscale')
    #
    # img = rotate(img, 45)
    # display(img, 'rotate')
    #
    # thresh = threshold(img, 100)
    # display(thresh, 'simple threshold')
    #
    # adap = adaptive_threshold(img)
    # display(adap, 'adaptive threshold')
    #
    # blur = gaussian_blur(img)
    # display(blur, 'blur')
    #
    # dil = dilate(thresh)
    # display(dil, 'dilation')
    #
    # ero = erode(thresh)
    # display(ero, 'erosion')
    #
    # clo = closing(thresh)
    # display(clo, 'closing')
    #
    # ope = opening(thresh)
    # display(ope, 'opening')
    #
    # dist = distance_transform(thresh)
    # display(dist, 'distance_transform')
    #
    # white, mask = set_edge_white(img)
    # display(white, 'set edge white')
    #
    # right = mask_right(img, int(width/2))
    # display(right, 'mask right')
    #
    # top = mask_top(img, int(height/2))
    # display(top, 'mask top')
    #
    # big = extract_biggest_object(img)
    # display(big)
