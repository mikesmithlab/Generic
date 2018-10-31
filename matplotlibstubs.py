import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog


'''
Basic plot
'''
class Plotter():

    def __init__(self, figsize=(8,6), dpi=80, subplot=None, sharex='none', sharey='none'):

        if subplot is None:
            subplot = (1,1)
        self._num_subplots = subplot[0]*subplot[1]
        # row and column sharing
        self.fig, self.subplot_handles = plt.subplots(subplot[0], subplot[1],
                                                      sharex=sharex, sharey=sharey,
                                                      figsize=figsize, dpi=dpi)
        self._plots = -1
        self._subplots = {}

    def add_plot(self, xdata, ydata, marker='rx', subplot=0):
        if subplot > self._num_subplots:
            print('subplot does not exist')
        else:
            self.subplot_handles[subplot].plot(xdata, ydata, marker)
            self._plots += 1
            self._subplots[str(self._plots)] = subplot

    def remove_plot(self, num_of_plot):

        print(self._subplots[str(num_of_plot)])
        #self._subplot_handles[]
        self.fig.draw()

    def list_plots(self):
        print('{plot number : subplot number}')
        print(self._subplots)

    def configure_title(self, title='', fontsize=20):
        self.fig.suptitle(title, fontsize=fontsize)

    def configure_subplot_title(self, subplot=0, title='', fontsize=20):
        self.subplot_handles[subplot].set_title(title, fontsize=fontsize)

    def configure_xaxis(self,subplot=0, xlabel='x', fontsize=20, xlim=(None, None)):
        self.subplot_handles[subplot].set_xlabel(xlabel, fontsize=fontsize)
        self.subplot_handles[subplot].set_xlim(left=xlim[0], right=xlim[1])

    def configure_yaxis(self, subplot=0, ylabel='y', fontsize=20, ylim=(None, None)):
        self.subplot_handles[subplot].set_ylabel(ylabel, fontsize=fontsize)
        self.subplot_handles[subplot].set_ylim(bottom=ylim[0], top=ylim[1])

    def save_figure(self, filename='*.png', initialdir='~ppzmis/Documents', dpi=80):
        options = {}
        options['defaultextension'] = '.png'
        options['filetypes'] = [('PNG', '.png'), ('JPG', '.jpg'), ('TIFF', '.tiff')]
        options['initialdir'] = initialdir
        options['initialfile'] = filename
        options['title'] = 'save image'

        if filename == '*.png':
            filename = filedialog.asksaveasfilename()
        self.fig.savefig(filename, dpi=dpi)

    def show_figure(self):
        plt.show()




if __name__=='__main__':
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    y2, y1 = np.cos(X), np.sin(X)

    #fig, subplot_handles = plt.subplots(2, 1)
    #plt.show()

    f = Plotter(subplot=(2,1))
    f.add_plot(X, y1, marker='r-')
    f.configure_title('test_title')
    f.configure_xaxis()
    f.configure_yaxis()
    f.list_plots()
    f.save_figure()
    #f.remove_plot(0)


