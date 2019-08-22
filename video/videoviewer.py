from Generic.video import ReadVideo
from Generic.pyqt5_widgets import QtImageViewer, Slider
from Generic.images import hstack
import numpy as np
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QHBoxLayout,
                                 QWidget,
                                 QVBoxLayout, QAction, QLabel)
from qimage2ndarray import array2qimage



class MainWindow(QtImageViewer):
    def __init__(self, filename=None):
        app = QApplication(sys.argv)
        super().__init__()
        self.filename=filename
        self.setup_main_window()
        self.load_vid()

        sys.exit(app.exec_())


    def setup_main_window(self):
        # Create window and layout


        self.win = QWidget()
        self.vbox = QVBoxLayout(self.win)


        # Create Image viewer
        self.viewer_setup()
        self.vbox.addWidget(self.viewer)
        self.framenum_slider = Slider(self.win, 'frame number', self.slider_update, 0, 5, 1)
        self.vbox.addWidget(self.framenum_slider)

        # Finalise window
        self.win.setWindowTitle('ParamGui')
        self.win.setLayout(self.vbox)
        self.win.show()


    def viewer_setup(self):
        self.viewer = QtImageViewer()
        self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.leftMouseButtonPressed.connect(self.get_coords)
        self.viewer.scrollMouseButton.connect(self._update_frame)
        self.viewer.canZoom = True
        self.viewer.canPan = True
        self.win.resize(1024, 720)


    def setup_menubar(self):
        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(app.quit)

        loadVid = QAction('&Load', self)
        loadVid.setShortcut(('Ctrl-O'))
        loadVid.triggered.connect(self.load_video)


        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction((loadVid))
        fileMenu.addAction(exitAct)

        preferences = menubar.addMenu('&Preferences')

    def load_video(self):
        self.filename=None
        self.load_vid()

    def load_vid(self):
        self.readvid=ReadVideo(filename=self.filename)
        self.filename = self.readvid.filename
        self.framenum = 0
        self.framenum_slider.setSliderRangeValues(0, self.readvid.num_frames -1)
        self.load_frame()

    def slider_update(self, val):
        print(val)
        self.framenum = self.framenum_slider.value()
        self.load_frame()

    def _update_frame(self, wheel_change):
        self.framenum = self.framenum + wheel_change
        if self.framenum < 0:
            self.framenum = 0
        elif self.framenum >= (self.readvid.num_frames - 1):
            self.framenum =  (self.readvid.num_frames - 1)
        self.framenum_slider.sliderCallback(self.framenum)
        self.load_frame()



    def load_frame(self):
        im = self.readvid.find_frame(self.framenum)
        pixmap = QPixmap.fromImage(array2qimage(im))
        self.viewer.setImage(pixmap)

    def _display_img(self, *ims):
        if len(ims) == 1:
            self.im = ims[0]
        else:
            self.im = hstack(*ims)

    def get_coords(self, x, y):
        print('cursor position (x, y) = ({}, {})'.format(int(x), int(y)))

if __name__ == "__main__":
    main = MainWindow(filename='/media/NAS/ActiveMatter/bacteriadata/Alessandro/bacteriaswarm.mp4')
    #main = MainWindow(filename='/media/ppzmis/data/ActiveMatter/Microscopy/190709MRaggregates/videos/test2_annotated.mp4')

