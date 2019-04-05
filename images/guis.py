from . import *
import cv2
import numpy as np

__all__ = ['CircleGui', 'ThresholdGui', 'AdaptiveThresholdGui', 'InrangeGui', 'ContoursGui']


class ParamGui:
    def __init__(self, img):
        self.im = img
        self.im0 = img.copy()

        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('image', 960, 540)
        for key in self.param_dict:
            cv2.createTrackbar(key, 'image', self.param_dict[key][0],
                               self.param_dict[key][1], self._update_dict)

        while(1):
            cv2.imshow('image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                break
        cv2.destroyAllWindows()

    def _update_dict(self, new_value):
        for key in self.param_dict:
            self.param_dict[key] = (cv2.getTrackbarPos(key, 'image'),
                                    self.param_dict[key][1])
        self.update()

    def _update_trackbars(self):
        for key in self.param_dict:
            cv2.setTrackbarPos(key, 'image', self.param_dict[key][0])

    def params(self, show=False):
        print('--------------------')
        print('Parameters:')
        print('--------------------')
        for key in self.param_dict:
            print(key, self.param_dict[key])
        print('--------------------')
        return self.param_dict


class CircleGui(ParamGui):
    def __init__(self, img):
        self.param_dict = {
                    'distance': (25, 51),
                    'thresh1': (200, 255),
                    'thresh2': (5, 20),
                    'min_rad': (17, 50),
                    'max_rad': (19, 50)
                    }
        ParamGui.__init__(self, img)

    def update(self):
        circles = find_circles(self.im0, self.param_dict['distance'][0],
                               self.param_dict['thresh1'][0], self.param_dict['thresh2'][0],
                               self.param_dict['min_rad'][0], self.param_dict['max_rad'][0])
        self.im = draw_circles(np.dstack((self.im, self.im, self.im)), circles)


class ThresholdGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'threshold': (1, 255),
                           'invert': (0, 1)}
        ParamGui.__init__(self, img)

    def update(self):
        if self.param_dict['invert'][0] == 0:
            self.im = threshold(self.im0, thresh=self.param_dict['threshold'][0])
        else:
            self.im = threshold(self.im0, thresh=self.param_dict['threshold'][0],
                                type=cv2.THRESH_BINARY_INV)


class AdaptiveThresholdGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'window': (1, 101),
                           'constant+30': (0, 60),
                           'invert': (0, 1)}
        ParamGui.__init__(self, img)

    def update(self):
        window = self.param_dict['window'][0]
        if window % 2 == 0:
            window += 1
            self.param_dict['window'] = (window,self.param_dict['window'][1])
            self._update_trackbars()

        if self.param_dict['invert'] == 0:
            self.im = adaptive_threshold(self.im0, self.param_dict['window'][0],
                                     self.param_dict['constant+30'][0]-30)
        else:
            self.im = adaptive_threshold(self.im0, self.param_dict['window'][0],
                                         self.param_dict['constant+30'][0] - 30,
                                         type=cv2.THRESH_BINARY_INV)





class ContoursGui(ParamGui):
    '''
    This applies adaptive threshold (this is what you are adjusting and is the
    value on the slider. It then applies findcontours and draws them to display result
    '''
    def __init__(self, img, thickness=2):

        self.param_dict = {'window': (1, 101),
                           'constant+30': (0, 60),
                           'invert': (0, 1)}

        self.thickness = thickness
        blurred_img = gaussian_blur(img)
        self.blurred_img = blurred_img
        self.orig_img = img

        self.orig_img0 = img.copy()
        img = np.hstack((img, img))
        ParamGui.__init__(self, img)
        self.update()


    def update(self):
        window = self.param_dict['window'][0]
        if window % 2 == 0:
            window += 1
            self.param_dict['window'] = (window, self.param_dict['window'][1])
            self._update_trackbars()
        const = self.param_dict['constant+30'][0]
        if const % 2 == 0:
            const += 1
            self.param_dict['constant+30'] = (const, self.param_dict['constant+30'][1])

        if self.param_dict['invert'][0] == 0:
            thresh = adaptive_threshold(self.blurred_img, self.param_dict['window'][0],
                                     self.param_dict['constant+30'][0] - 30)
        else:
            thresh = adaptive_threshold(self.blurred_img, self.param_dict['window'][0],
                                         self.param_dict['constant+30'][0] - 30,
                                         type=cv2.THRESH_BINARY_INV)
        contours = find_contours(thresh)
        contour_img = draw_contours(stack_3(self.orig_img0.copy()), contours, thickness=self.thickness)
        self.im = np.hstack((stack_3(thresh), contour_img))

class WatershedGui(ParamGui):
    def __init__(self, img):

        self.param_dict = {'window': (1, 101),
                           'constant+30': (0, 60),
                           'invert': (0, 1)}
        self.orig_img = img
        self.orig_img0 = img.copy()
        self.blurred_img = img.copy()
        img = np.hstack((img, img))
        ParamGui.__init__(self, img)
        self.update()


    def update(self):
        window = self.param_dict['window'][0]
        if window % 2 == 0:
            window += 1
            self.param_dict['window'] = (window, self.param_dict['window'][1])
            self._update_trackbars()
        const = self.param_dict['constant+30'][0]
        if const % 2 == 0:
            const += 1
            self.param_dict['constant+30'] = (const, self.param_dict['constant+30'][1])

        if self.param_dict['invert'][0] == 0:
            thresh = adaptive_threshold(self.blurred_img, self.param_dict['window'][0],
                                     self.param_dict['constant+30'][0] - 30)
        else:
            thresh = adaptive_threshold(self.blurred_img, self.param_dict['window'][0],
                                         self.param_dict['constant+30'][0] - 30,
                                         type=cv2.THRESH_BINARY_INV)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)
        self.im2 = self.orig_img0.copy()
        self.im2[markers == -1] = [255, 0, 0]
        self.im = np.hstack((stack_3(thresh), self.im2))

class InrangeGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'bottom': (1, 255),
                           'top': (200, 255)}
        ParamGui.__init__(self, img)

    def update(self):
        bottom = self.param_dict['bottom'][0]
        top = self.param_dict['top'][0]
        if top <= bottom:
            top = bottom + 1
        self.param_dict['top'] = (top,self.param_dict['top'][1])
        self._update_trackbars()

        self.im = cv2.inRange(self.im0, self.param_dict['bottom'][0],
                              self.param_dict['top'][0])

if __name__ == "__main__":
    from Generic import video

    vid = video.ReadVideo()
    frame = vid.read_next_frame()
    frame = bgr_2_grayscale(frame)
    CircleGui(frame)
    ThresholdGui(frame)
    AdaptiveThresholdGui(frame)
    InrangeGui(frame)
