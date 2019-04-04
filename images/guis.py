from . import *
import cv2
import numpy as np

__all__ = ['CircleGui', 'ThresholdGui', 'AdaptiveThresholdGui', 'InrangeGui']


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
            self.param_dict[key] = cv2.getTrackbarPos(key, 'image')
        self.im = self.im0.copy()
        self.update()

    def _update_trackbars(self):
        for key in self.param_dict:
            cv2.setTrackbarPos(key, 'image', self.param_dict[key])


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
        circles = find_circles(self.im0, self.param_dict['distance'],
                               self.param_dict['thresh1'], self.param_dict['thresh2'],
                               self.param_dict['min_rad'], self.param_dict['max_rad'])
        self.im = draw_circles(np.dstack((self.im, self.im, self.im)), circles)


class ThresholdGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'threshold': (1, 255)}
        ParamGui.__init__(self, img)

    def update(self):
        self.im = threshold(self.im0, thresh=self.param_dict['threshold'])


class AdaptiveThresholdGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'window': (1, 101),
                           'constant+30': (0, 60)}
        ParamGui.__init__(self, img)

    def update(self):
        window = self.param_dict['window']
        if window % 2 == 0:
            window += 1
            self.param_dict['window'] = window
            self._update_trackbars()

        self.im = adaptive_threshold(self.im0, self.param_dict['window'],
                                     self.param_dict['constant+30']-30)


class InrangeGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'bottom': (1, 255),
                           'top': (200, 255)}
        ParamGui.__init__(self, img)

    def update(self):
        bottom = self.param_dict['bottom']
        top = self.param_dict['top']
        if top <= bottom:
            top = bottom + 1
        self.param_dict['top'] = top
        self._update_trackbars()

        self.im = cv2.inRange(self.im0, self.param_dict['bottom'],
                              self.param_dict['top'])


if __name__ == "__main__":
    from Generic import video

    vid = video.ReadVideo()
    frame = vid.read_next_frame()
    frame = bgr_2_grayscale(frame)
    CircleGui(frame)
    ThresholdGui(frame)
    AdaptiveThresholdGui(frame)
    InrangeGui(frame)
