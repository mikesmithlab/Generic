import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QSlider, QCheckBox, QHBoxLayout,
                             QLabel, QComboBox, QSizePolicy, QVBoxLayout,
                             QApplication)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
from matplotlib.figure import Figure


class Slider(QWidget):

    def __init__(
            self,
            parent,
            label,
            function,
            start=0,
            end=10,
            dpi=1,
            initial=0):
        QWidget.__init__(self, parent)

        start *= dpi
        end *= dpi
        initial *= dpi

        self.dpi = dpi
        self.function = function
        self.checked = True
        self.value = initial

        lbl = QLabel(self)
        lbl.setText(label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(start, end)
        self.slider.setSliderPosition(initial)
        self.slider.valueChanged[int].connect(self.slider_changed)
        self.slider.setTickPosition(QSlider.TicksBelow)

        self.val_label = QLabel(parent)
        self.val_label.setText(str(initial / dpi))

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(lbl)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.val_label)

    def slider_changed(self, value):
        self.value = value / self.dpi
        self.val_label.setText(str(self.value))
        self.call_function()

    def call_function(self):
        if self.checked:
            self.function(self.value)
        else:
            self.function(None)


class CheckedSlider(Slider):

    def __init__(
            self,
            parent,
            label,
            function,
            start=0,
            end=10,
            dpi=1,
            initial=0):
        Slider.__init__(self, parent, label, function, start, end, dpi,
                        initial)
        self.checked = False

        self.cb = QCheckBox(parent)
        self.cb.stateChanged.connect(self.cb_changed)
        self.layout().addWidget(self.cb)

    def cb_changed(self, state):
        if state == Qt.Checked:
            self.checked = True
        else:
            self.checked = False


class ComboBox(QWidget):

    def __init__(self, parent, label, items, function):
        QWidget.__init__(self, parent)

        lbl = QLabel(label, parent)
        combo = QComboBox(parent)
        combo.addItems(items)
        combo.activated[str].connect(function)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(lbl)
        self.layout().addWidget(combo)
        self.layout().addStretch(1)


class CheckBox(QCheckBox):
    def __init__(self, parent, label, function, initial='off'):
        QCheckBox.__init__(self, label, parent)
        self.stateChanged.connect(function)
        if initial == 'on':
            self.setCheckState(Qt.Checked)


class MatplotlibFigure(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.setLayout(QVBoxLayout())
        self.canvas = PlotCanvas(self)
        self.fig = self.canvas.fig
        self.draw = self.canvas.draw
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent):
        self.fig = Figure()
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)


if __name__ == "__main__":
    import numpy as np

    # Matplotlib Example
    app = QApplication(sys.argv)

    main = MatplotlibFigure(None)
    main.ax = main.fig.add_subplot(111)
    x = np.arange(10)
    y = x ** 2
    main.ax.plot(x, y)
    main.draw()
    main.show()

    sys.exit(app.exec_())
