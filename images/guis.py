import sys
import cv2
import numpy as np
import qimage2ndarray as qim
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QT_VERSION_STR
from PyQt5.QtGui import QPixmap, QImage, QPainterPath, QCloseEvent
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QApplication,
                             QSlider, QHBoxLayout, QGraphicsView, QGraphicsScene,
                             QFileDialog)
from skimage import filters
from Generic import video, images
from . import *

__all__ = ['CircleGui', 'ThresholdGui', 'AdaptiveThresholdGui', 'InrangeGui',
           'ContoursGui', 'DistanceTransformGui','WatershedGui', 'ParamGui']

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
            self._display_img(self.im0, img2=self.im0)
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
        app = QApplication(sys.argv)
        self.win = QWidgetMod(self.param_dict)
        self.viewer = QtImageViewer()
        self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.canZoom = True
        self.viewer.canPan = True
        self._update_im()
        self.vbox = QVBoxLayout(self.win)
        self.vbox.addWidget(self.viewer)

        self.add_sliders()

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

    def _update_sliders(self):
        if self.type == 'multiframe':
            frame_no = self.frame_slider.value()
            self.frame_lbl.setText('frame: ' + str(frame_no))
            self.im0 = self.read_vid.find_frame(frame_no)
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

    def _display_img(self, img1, img2=None):
        if img2 is None:
            self.im = img1
        else:
            if np.size(np.shape(img1)) == 2:
                img1 = stack_3(img1)
            if np.size(np.shape(img2)) == 2:
                img2 = stack_3(img2)
            self.im = np.hstack((img1, img2))

    def _update_im(self):
        pixmap = QPixmap.fromImage(qim.array2qimage(self.im))
        self.viewer.setImage(pixmap)
        # if self.num_imgs == 1:
        #     self.lbl.setPixmap(pixmap.scaled(1280, 720, Qt.KeepAspectRatio))
        # elif self.num_imgs == 2:
        #     self.lbl.setPixmap(pixmap.scaled(1280, 720, Qt.KeepAspectRatio))

class QWidgetMod(QWidget):
    """
    Overrides the closeEvent method of QWidget to print out the parameters set
    in the gui. Is used by ParamGui.
    """
    def __init__(self,param_dict):
        QWidget.__init__(self)
        self.param_dict = param_dict

    def closeEvent(self, a0: QCloseEvent) -> None:
        print('Final Parameters')
        print('------------------------------')
        for key in sorted(self.param_dict.keys()):
            print(key + ' : ' +  str(self.param_dict[key][0]))
        print('------------------------------')

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
        self._display_img(draw_circles(stack_3(self.im0), circles))



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

    def __init__(self, img):
        self.grayscale = True
        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1],
                           'invert': [0, 0, 1, 1]}
        ParamGui.__init__(self, img)

    def update(self):
        if self.param_dict['invert'][0] == 0:
            self._display_img(adaptive_threshold(self.im0,
                                                 self.param_dict['window'][0],
                                                 self.param_dict['constant'][0])
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
        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1],
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
        self._display_img(thresh, draw_contours(stack_3(self.im0.copy()),contours, thickness=self.thickness))


class DistanceTransformGui(ParamGui):
    def __init__(self, img):
        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1],
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

        dist_transform_img = distance_transform(self.im0.copy(),
                                                preprocess=False
                                                )
        self._display_img(thresh, dist_transform_img)


class WatershedGui(ParamGui):
    def __init__(self, img):

        self.param_dict = {'window': [3, 3, 101, 2],
                           'constant': [0, -30, 30, 1],
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



class QtImageViewer(QGraphicsView):
    """ PyQt image viewer widget for a QPixmap in a QGraphicsView scene with mouse zooming and panning.
    Displays a QImage or QPixmap (QImage is internally converted to a QPixmap).
    To display any other image format, you must first convert it to a QImage or QPixmap.
    Some useful image format conversion utilities:
        qimage2ndarray: NumPy ndarray <==> QImage    (https://github.com/hmeine/qimage2ndarray)
        ImageQt: PIL Image <==> QImage  (https://github.com/python-pillow/Pillow/blob/master/PIL/ImageQt.py)
    Mouse interaction:
        Left mouse button drag: Pan image.
        Right mouse button drag: Zoom box.
        Right mouse button doubleclick: Zoom to show entire image.

        __author__ = "Marcel Goldschen-Ohm <marcel.goldschen@gmail.com>"
    """

    # Mouse button signals emit image scene (x, y) coordinates.
    # !!! For image (row, column) matrix indexing, row = y and column = x.
    leftMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = pyqtSignal(float, float)

    def __init__(self):
        QGraphicsView.__init__(self)

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Store a local handle to the scene's current image pixmap.
        self._pixmapHandle = None

        # Image aspect ratio mode.
        # !!! ONLY applies to full image. Aspect ratio is always ignored when zooming.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio

        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Stack of QRectF zoom boxes in scene coordinates.
        self.zoomStack = []

        # Flags for enabling/disabling mouse interaction.
        self.canZoom = True
        self.canPan = True

    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def clearImage(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None

    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None

    def image(self):
        """ Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None

    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()

    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
            self.fitInView(self.zoomStack[-1], Qt.IgnoreAspectRatio)  # Show zoomed rect (ignore aspect ratio).
        else:
            self.zoomStack = []  # Clear the zoom stack (in case we got here because of an invalid zoom).
            self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).

    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.updateViewer()

    def mousePressEvent(self, event):
        """ Start mouse pan or zoom mode.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if self.canPan:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.setDragMode(QGraphicsView.RubberBandDrag)
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """ Stop mouse pan or zoom mode (apply zoom if valid).
        """
        QGraphicsView.mouseReleaseEvent(self, event)
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                viewBBox = self.zoomStack[-1] if len(self.zoomStack) else self.sceneRect()
                selectionBBox = self.scene.selectionArea().boundingRect().intersected(viewBBox)
                self.scene.setSelectionArea(QPainterPath())  # Clear current selection area.
                if selectionBBox.isValid() and (selectionBBox != viewBBox):
                    self.zoomStack.append(selectionBBox)
                    self.updateViewer()
            self.setDragMode(QGraphicsView.NoDrag)
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.zoomStack = []  # Clear zoom stack.
                self.updateViewer()
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

if __name__ == "__main__":

    """
    Relative import will break when running this file as top-level

    Run functions from images as images.function_name
    """
    from Generic import video
    from Generic import images
    vid = video.ReadVideo()

    # frame = images.bgr_2_grayscale(frame)
    #images.CircleGui(vid)
    # images.ThresholdGui(vid)
    # images.AdaptiveThresholdGui(vid)
    #images.ContoursGui(vid)
    #images.InrangeGui(vid)
    # images.DistanceTransformGui(vid)
    images.WatershedGui(vid)
