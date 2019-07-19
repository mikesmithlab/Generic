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
        initial *= dpi
        start *= dpi
        end *= dpi

        self._function = function
        self._dpi = dpi
        self._sliderValue = initial
        self._value = initial

        QWidget.__init__(self, parent)

        lbl = QLabel(label, parent=self)

        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(start, end)
        slider.setSliderPosition(initial)
        slider.valueChanged[int].connect(self.sliderCallback)
        slider.setTickPosition(QSlider.TicksBelow)

        self.lbl = QLabel(str(initial / dpi), self)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(lbl)
        self.layout().addWidget(slider)
        self.layout().addWidget(self.lbl)

    def sliderCallback(self, value):
        value = self.calculateValue(value)
        if self.value() is not None:
            self.setValue(value)
        self.setSliderValue(value)
        self.callFunction()

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value

    def setSliderValue(self, value):
        self._sliderValue = value
        self.lbl.setText(str(value))

    def sliderValue(self):
        return self._sliderValue

    def calculateValue(self, value):
        return value / self._dpi

    def callFunction(self):
        self._function(self.value())


class CheckedSlider(Slider):

    def __init__(
            self,
            parent,
            label,
            function,
            **kwargs):
        Slider.__init__(self, parent, label, function, **kwargs)
        self.setValue(None)
        checkbox = QCheckBox(self)
        checkbox.stateChanged.connect(self.checkboxCallback)
        self.layout().addWidget(checkbox)

    def checkboxCallback(self, state):
        if state == Qt.Checked:
            self.setValue(self.sliderValue())
        else:
            self.setValue(None)
        self.callFunction()


class ArraySlider(Slider):

    def __init__(self, parent, label, function, array):
        start = 0
        end = len(array) - 1
        dpi = 1
        initial = 0
        Slider.__init__(self, parent, label, function,
                        start=start, end=end, dpi=dpi,
                        initial=initial)
        self.array = array

    def calculateValue(self, value):
        return self.array[value]


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
