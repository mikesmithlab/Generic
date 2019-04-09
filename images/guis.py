from . import *
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QApplication,
                             QSlider, QHBoxLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
import qimage2ndarray as qim

__all__ = ['CircleGui', 'ThresholdGui', 'AdaptiveThresholdGui', 'InrangeGui',
           'ContoursGui']


class ParamGui:
    """
    Parent Gui for image gui classes.

    Child classes should have attribute self.param_dict where each item should
    be a list containing:
        [initial value, lowest value, highest value, step size]
    Currently step sizes can only be 1 or 2.
    """
    def __init__(self, img):
        self.im = img
        self.im0 = img.copy()

        self.init_ui()

    def init_ui(self):
        app = QApplication(sys.argv)
        self.win = QWidget()
        self.lbl = QLabel()
        self._update_im()
        self.vbox = QVBoxLayout(self.win)
        self.vbox.addWidget(self.lbl)

        self.add_sliders()

        self.win.setWindowTitle('ParamGui')
        self.win.setLayout(self.vbox)
        self.win.show()
        sys.exit(app.exec_())

    def add_sliders(self):

        self.sliders = {}
        self.labels = {}

        for key in self.param_dict:
            widget = QWidget()
            hbox = QHBoxLayout()

            params = self.param_dict[key]
            val, bottom, top, step = params

            lbl = QLabel()
            lbl.setText(key + ': ' + str(val))

            slider = QSlider(Qt.Horizontal)
            if step == 2:
                length = (top - bottom) / 2
                slider.setRange(0, length)
                slider.setValue((val-bottom)/2)
            else:
                slider.setRange(bottom, top)
                slider.setValue(val)
            slider.valueChanged.connect(self._update_sliders)

            hbox.addWidget(lbl)
            hbox.addWidget(slider)
            self.sliders[key] = slider
            self.labels[key] = lbl
            widget.setLayout(hbox)
            self.vbox.addWidget(widget)

    def _update_sliders(self):
        for key in self.param_dict:
            params = self.param_dict[key]
            val, bottom, top, step = params

            val = self.sliders[key].value()
            if params[3] == 2:
                val = 2*val + bottom
            self.labels[key].setText(key + ': ' + str(val))
            self.param_dict[key][0] = val
        self.update()
        self._update_im()

    def _update_im(self):
        pixmap = QPixmap.fromImage(qim.array2qimage(self.im))
        self.lbl.setPixmap(pixmap.scaled(1280, 720, Qt.KeepAspectRatio))


class CircleGui(ParamGui):
    def __init__(self, img):
        self.param_dict = {
                    'distance': [25, 3, 51, 2],
                    'thresh1': [200, 0, 255, 1],
                    'thresh2': [5, 0, 20, 1],
                    'min_rad': [17, 3, 50, 1],
                    'max_rad': [19, 3, 50, 1]
                    }
        ParamGui.__init__(self, img)

    def update(self):
        circles = find_circles(self.im0, self.param_dict['distance'][0],
                               self.param_dict['thresh1'][0],
                               self.param_dict['thresh2'][0],
                               self.param_dict['min_rad'][0],
                               self.param_dict['max_rad'][0])
        self.im = draw_circles(stack_3(self.im0), circles)


class ThresholdGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'threshold': [1, 0, 255, 1],
                           'invert': [0, 0, 1, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        if self.param_dict['invert'][0] == 0:
            self.im = threshold(self.im0,
                                thresh=self.param_dict['threshold'][0])
        else:
            self.im = threshold(self.im0,
                                thresh=self.param_dict['threshold'][0],
                                mode=cv2.THRESH_BINARY_INV)


class AdaptiveThresholdGui(ParamGui):

    def __init__(self, img):
        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        self.im = adaptive_threshold(self.im0, self.param_dict['window'][0],
                                 self.param_dict['constant'][0])


class ContoursGui(ParamGui):
    '''
    This applies adaptive threshold (this is what you are adjusting and is the
    value on the slider. It then applies findcontours and draws them to display result
    '''
    def __init__(self, img, thickness=2):

        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1]}

        self.thickness = thickness
        blurred_img = gaussian_blur(img)
        self.blurred_img = blurred_img
        self.orig_img = img

        self.orig_img0 = img.copy()
        img = np.hstack((img, img))
        ParamGui.__init__(self, img)
        self.update()

    def update(self):
        thresh = adaptive_threshold(self.blurred_img,
                                    self.param_dict['window'][0],
                                    self.param_dict['constant'][0])
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
        self.param_dict = {'bottom': [1, 0, 255, 1],
                           'top': [200, 0, 255, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        self.im = cv2.inRange(self.im0, self.param_dict['bottom'][0],
                              self.param_dict['top'][0])

if __name__ == "__main__":
    """
    Relative import will break when running this file as top-level
    
    Run functions from images as images.function_name
    """
    from Generic import video
    from Generic import images
    vid = video.ReadVideo()
    frame = vid.read_next_frame()
    frame = images.bgr_2_grayscale(frame)
    images.CircleGui(frame)
    images.ThresholdGui(frame)
    images.AdaptiveThresholdGui(frame)
    images.ContoursGui(frame)
    images.InrangeGui(frame)
