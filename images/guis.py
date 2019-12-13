import sys

import cv2
import numpy as np
import qimage2ndarray as qim
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QApplication,
                             QSlider, QHBoxLayout,
                             QCheckBox)

from Generic.pyqt5_widgets import QtImageViewer, QWidgetMod
from . import *

__all__ = ['CircleGui', 'ThresholdGui', 'AdaptiveThresholdGui', 'InrangeGui',
           'ContoursGui', 'RotatedBoxGui','DistanceTransformGui','WatershedGui', 'ParamGui',
           'CannyGui', 'Inrange3GUI']

'''
------------------------------------------------------------------------------
Parent class
------------------------------------------------------------------------------
'''

class ParamGui:
    """
    Parent Gui for image gui classes.

    Child classes should have attribute self.param_dict where each item should
    be a list containing:
        [initial value, lowest value, highest value, step size]
    Currently step sizes can only be 1 or 2.
    """
    def __init__(self, img_or_vid, num_imgs=1):
        self.num_imgs = num_imgs
        self._file_setup(img_or_vid)
        self.im0 = self.im.copy()
        if num_imgs == 1:
            self._display_img(self.im0)
        elif num_imgs == 2:
            self._display_img(self.im0, self.im0)
        self.init_ui()

    def _file_setup(self, img_or_vid):
        if isinstance(img_or_vid, video.ReadVideo):
            self.read_vid = img_or_vid
            self.frame_no = 0
            self.num_frames = self.read_vid.num_frames
            self.read_vid.grayscale = self.grayscale
            self.im = self.read_vid.find_frame(self.frame_no)
            self.type = 'multiframe'
        else:
            if self.grayscale:
                self.im = bgr_2_grayscale(img_or_vid)
            else:
                self.im = img_or_vid
            self.type = 'singleframe'

    def init_ui(self):
        # Create window and layout
        app = QApplication(sys.argv)
        self.win = QWidgetMod(self.param_dict)
        self.vbox = QVBoxLayout(self.win)

        # Create Image viewer
        self.viewer = QtImageViewer()
        self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.leftMouseButtonPressed.connect(self.get_coords)
        self.viewer.canZoom = True
        self.viewer.canPan = True
        self._update_im()
        self.vbox.addWidget(self.viewer)

        # Create live update checkbox
        cb = QCheckBox('Update')
        cb.toggle()
        cb.stateChanged.connect(self._update_cb)
        self.live_update = True
        self.vbox.addWidget(cb)

        # Add sliders
        self.add_sliders()

        # Finalise window
        self.win.setWindowTitle('ParamGui')
        self.win.setLayout(self.vbox)
        self.win.show()
        sys.exit(app.exec_())

    def add_sliders(self):

        self.sliders = {}
        self.labels = {}

        if self.type == 'multiframe':
            widget = QWidget()
            hbox = QHBoxLayout()
            self.frame_lbl = QLabel()
            self.frame_lbl.setText('frame: ' + str(0))

            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setRange(0, self.num_frames)
            self.frame_slider.setValue(0)
            self.frame_slider.valueChanged.connect(self._update_sliders)
            hbox.addWidget(self.frame_lbl)
            hbox.addWidget(self.frame_slider)
            widget.setLayout(hbox)
            self.vbox.addWidget(widget)


        for key in sorted(self.param_dict.keys()):
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

    def _update_cb(self, state):
        if state == Qt.Checked:
            self.live_update = True
            self._update_sliders()
        else:
            self.live_update = False

    def _update_sliders(self):
        if self.type == 'multiframe':
            self.frame_no = self.frame_slider.value()
            self.frame_lbl.setText('frame: ' + str(self.frame_no))
            self.im0 = self.read_vid.find_frame(self.frame_no)
        for key in self.param_dict:
            params = self.param_dict[key]
            val, bottom, top, step = params

            val = self.sliders[key].value()
            if params[3] == 2:
                val = 2*val + bottom
            self.labels[key].setText(key + ': ' + str(val))
            self.param_dict[key][0] = val
        if self.live_update == True:
            self.update()
            self._update_im()

    def get_coords(self, x, y):
        print('cursor position (x, y) = ({}, {})'.format(int(x), int(y)))

    def _display_img(self, *ims):
        if len(ims) == 1:
            self.im = ims[0]
        else:
            self.im = hstack(*ims)

    def _update_im(self):
        pixmap = QPixmap.fromImage(qim.array2qimage(self.im))
        self.viewer.setImage(pixmap)



'''
------------------------------------------------------------------------------
Single image display
------------------------------------------------------------------------------
'''

class CircleGui(ParamGui):
    def __init__(self, img):
        self.grayscale = True
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
        self._display_img(draw_circles(self.im0, circles))


class ThresholdGui(ParamGui):

    def __init__(self, img):
        self.grayscale = True
        self.param_dict = {'threshold': [1, 0, 255, 1],
                           'invert': [0, 0, 1, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        if self.param_dict['invert'][0] == 0:
            self._display_img(threshold(self.im0,
                                thresh=self.param_dict['threshold'][0]))
        else:
            self._display_img(threshold(self.im0,
                                thresh=self.param_dict['threshold'][0],
                                mode=cv2.THRESH_BINARY_INV))


class AdaptiveThresholdGui(ParamGui):

    def __init__(self, img, mode=cv2.THRESH_BINARY):
        self.grayscale = True
        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1],
                           'invert': [0, 0, 1, 1]}
        self.mode = mode
        ParamGui.__init__(self, img)

    def update(self):
        if self.param_dict['invert'][0] == 0:
            self._display_img(adaptive_threshold(self.im0,
                                                 self.param_dict['window'][0],
                                                 self.param_dict['constant'][0],
                                                 mode=self.mode)
                              )
        else:
            self._display_img(adaptive_threshold(self.im0,
                                                 self.param_dict['window'][0],
                                                 self.param_dict['constant'][0],
                                                 mode=cv2.THRESH_BINARY_INV)
                              )


class InrangeGui(ParamGui):

    def __init__(self, img):
        self.grayscale = True
        self.param_dict = {'bottom': [1, 0, 255, 1],
                           'top': [200, 0, 255, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        self.im = cv2.inRange(self.im0, self.param_dict['bottom'][0],
                              self.param_dict['top'][0])


class Inrange3GUI(ParamGui):

    def __init__(self, img):
        self.grayscale = False
        self.param_dict = {'0 bottom': [1, 0, 255, 1],
                           '0 top': [200, 0, 255, 1],
                           '1 bottom': [1, 0, 255, 1],
                           '1 top': [200, 0, 255, 1],
                           '2 bottom': [1, 0, 255, 1],
                           '2 top': [200, 0, 255, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        self.im = cv2.inRange(
            self.im0,
            (self.param_dict['0 bottom'][0], self.param_dict['1 bottom'][0],
             self.param_dict['2 bottom'][0]),
            (self.param_dict['0 top'][0], self.param_dict['1 top'][0],
             self.param_dict['2 top'][0]))

class CannyGui(ParamGui):

    def __init__(self, img):
        self.grayscale = True
        self.param_dict = {'p1': [1, 0, 255, 1],
                           'p2': [1, 0, 255, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        self.im = cv2.Canny(self.im0,
                            self.param_dict['p1'][0],
                            self.param_dict['p2'][0])
'''
------------------------------------------------------------------------------
Double image display
------------------------------------------------------------------------------
'''

class ContoursGui(ParamGui):
    '''
    This applies adaptive threshold (this is what you are adjusting and is the
    value on the slider. It then applies findcontours and draws them to display result
    '''
    def __init__(self, img, thickness=2):
        self.param_dict = {'window': [53, 3, 101, 2],
                           'constant': [-26, -30, 30, 1],
                           'invert': [0, 0, 1, 1]}
        self.thickness = thickness
        self.grayscale = True
        ParamGui.__init__(self, img, num_imgs=2)
        self.blurred_img = self.im.copy()
        self.update()

    def update(self):
        self.blurred_img = gaussian_blur(self.im0.copy())
        thresh = adaptive_threshold(self.blurred_img,
                                    self.param_dict['window'][0],
                                    self.param_dict['constant'][0],
                                    self.param_dict['invert'][0])

        contours = find_contours(thresh)
        self._display_img(thresh, draw_contours(self.im0.copy(), contours, thickness=self.thickness))

class RotatedBoxGui(ParamGui):
    '''
    This applies adaptive threshold (this is what you are adjusting and is the
    value on the slider. It then applies findcontours and draws them to display result
    '''
    def __init__(self, img, thickness=2):
        self.param_dict = {'window': [53, 3, 101, 2],
                           'constant': [-26, -30, 30, 1],
                           'invert': [0, 0, 1, 1]}
        self.thickness = thickness
        self.grayscale = True
        ParamGui.__init__(self, img, num_imgs=2)
        self.blurred_img = self.im.copy()
        self.update()

    def update(self):
        self.blurred_img = gaussian_blur(self.im0.copy())
        thresh = adaptive_threshold(self.blurred_img,
                                    self.param_dict['window'][0],
                                    self.param_dict['constant'][0],
                                    self.param_dict['invert'][0])

        contours = images.find_contours(thresh)
        box=[]
        for contour in contours:
            box_guess, rect_guess = images.rotated_bounding_rectangle(contour)
            print(rect_guess[1][0])
            if rect_guess[1][0] < 15:
                box.append(box_guess)
            else:
                img = separate_rects(contour, box_guess)



        box = np.array(box)
        self._display_img(thresh, draw_contours(self.im0.copy(), box, thickness=self.thickness))


class DistanceTransformGui(ParamGui):
    def __init__(self, img):
        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1],
                           'invert': [0, 0, 1, 1],
                           }
        self.grayscale = True
        ParamGui.__init__(self, img, num_imgs=2)
        self.blurred_img = self.im.copy()
        self.update()

    def update(self):
        self.blurred_img = gaussian_blur(self.im0.copy())
        thresh = adaptive_threshold(self.blurred_img,
                                    self.param_dict['window'][0],
                                    self.param_dict['constant'][0],
                                    self.param_dict['invert'][0]
                                    )

        dist_transform_img = distance_transform(self.blurred_img,
                                                preprocess=True,
                                                block_size=self.param_dict['window'][0],
                                                constant=self.param_dict['constant'][0],
                                                mode=self.param_dict['invert'][0]
                                                )
        dist_transform_img = 255*dist_transform_img/np.max(dist_transform_img)
        self._display_img(thresh, dist_transform_img)


class WatershedGui(ParamGui):
    def __init__(self, img):

        self.param_dict = {'window': [41, 3, 101, 2],
                           'constant': [-26, -30, 30, 1],
                           'invert': [0, 0, 1, 1],
                           'watershed_thresh': [1, 0, 255, 1]
                            }
        self.grayscale = True
        ParamGui.__init__(self, img, num_imgs=2)
        self.blurred_img = self.im.copy()
        self.update()

    def update(self):
        self.blurred_img = gaussian_blur(self.im0.copy())
        thresh = adaptive_threshold(self.blurred_img,
                                    self.param_dict['window'][0],
                                    self.param_dict['constant'][0],
                                    self.param_dict['invert'][0]
                                    )

        watershed_img = watershed(self.im0.copy(),
                                  watershed_threshold=self.param_dict['watershed_thresh'][0],
                                  block_size=self.param_dict['window'][0],
                                  constant=self.param_dict['constant'][0],
                                  mode=self.param_dict['invert'][0]
                                  )
        self._display_img(thresh, watershed_img)



if __name__ == "__main__":

    """
    Relative import will break when running this file as top-level

    Run functions from images as images.function_name
    """
    from Generic import video, images, filedialogs
    file = filedialogs.load_filename('Load a video')
    vid = video.ReadVideo(file)
    # frame = images.bgr_2_grayscale(frame)
    images.CircleGui(vid)
    # images.ThresholdGui(vid)
    # images.AdaptiveThresholdGui(vid)
    #images.ContoursGui(vid,thickness=-1)
    #images.InrangeGui(vid)
    #images.DistanceTransformGui(vid)
    # images.WatershedGui(vid)
    # images.RotatedBoxGui(vid)