from Generic.video import ReadVideo
from Generic.pyqt5_widgets import QtImageViewer
from Generic.images import hstack
import numpy as np
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, 
from PyQt5.QtWidgets import (QApplication, QHBoxLayout,
                                 QWidget,
                                 QVBoxLayout, QAction, QLabel)
from qimage2ndarray import array2qimage

from Generic import pyqt5_widgets
import cv2


class MainWindow(QtImageViewer):
    def __init__(self, filename=None):
        app = QApplication(sys.argv)
        super().__init__()
        self.filename=filename
        self.setup_main_window()
        self.load_vid()
        #self.setup_main_widget()

        sys.exit(app.exec_())


    def setup_main_window(self):
        # Create window and layout


        self.win = QWidget()
        self.vbox = QVBoxLayout(self.win)

        # Create Image viewer
        self.viewer_setup()
        self.vbox.addWidget(self.viewer)


        # Finalise window
        self.win.setWindowTitle('ParamGui')
        self.win.setLayout(self.vbox)
        self.win.show()


    def viewer_setup(self):
        self.viewer = QtImageViewer()
        self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer.leftMouseButtonPressed.connect(self.get_coords)
        self.viewer.canZoom = True
        self.viewer.canPan = True
        self.win.resize(1024, 720)





    def setup_main_widget(self):
        pass
        #self.setCentralWidget(self.main_widget)
        #vbox = QVBoxLayout(self.main_widget)

        #vbox.addWidget(label)

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

        self.load_frame(self.framenum)


    def load_frame(self, framenum):
        im = self.readvid.find_frame(framenum)
        #img = cv2.cvtColor(self._display_img(im),
        #                   cv2.COLOR_BGR2RGB)
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

    main = MainWindow(filename='/media/ppzmis/data/ActiveMatter/Microscopy/190709MRaggregates/videos/test2_annotated.mp4')

