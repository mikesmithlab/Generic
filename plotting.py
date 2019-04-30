import matplotlib.pyplot as plt
import numpy as np
import Generic.filedialogs as fd


class Plotter:
    """
    Generic plotting object which allows multipanel graphs in x,y format

    Inputs:
    subplot in __init__ = tuple(rows,columns) of graphs
    subplot everywhere else = a single number indicating which subplot to put data on. Numbers go
                              top to bottom and then left to right.
    sharex and sharey specifies shared axes
    xdata and ydata = data to be plotted
    marker = a string to indicate how the data should be represented
    num_of_plot = unique identifier for a set of plotted data


    methods:
    add_plot  =  add single scatter to a subplot
    add_bar = add bar graph to subplot
    remove_plot = remove the dataset specified by num_of_plot from the subplot
    list_plots = access to dictionary indicating num_of_plot: subplot numbers, markers used for data
    configure.... = access to the various labels
    save_figure  = enables saving the figure
    show_figure = show figure. This will make a call to plt.show() and will open all currently
                  live figures regardless of origin.




    """

    def __init__(self, figsize=(8, 6), dpi=80, subplot=None, sharex='none',
                 sharey='none'):

        if subplot is None:
            subplot = (1, 1)
        self._num_subplots = subplot[0] * subplot[1]
        # row and column sharing
        self.fig, self._subplot_handles = plt.subplots(subplot[0], subplot[1],
                                                       sharex=sharex,
                                                       sharey=sharey,
                                                       figsize=figsize, dpi=dpi)
        self.nrows = subplot[0]
        self.ncols = subplot[1]
        if subplot == (1, 1):
            # This is to force _subplot_handles to be a list and prevent
            # code failing when we try to index it
            self._subplot_handles = [self._subplot_handles, 'dummy_var']
        self._plots = -1
        self._dict_plots = {}

    def add_plot(self, xdata, ydata, subplot=0, polar=False, **kwargs):
        """
        Descriptions of kwargs can be found here:
           https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html

        and then here:
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html

        """
        if subplot > self._num_subplots:
            print('subplot does not exist')
        else:
            if polar:
                self.set_subplot_polar(subplot)
            plot_handle = self._subplot_handles[subplot].errorbar(
                xdata, ydata, **kwargs)
            line_handle = plot_handle.get_children()[0]
            c = line_handle.get_color()
            m = line_handle.get_marker()
            ls = line_handle.get_linestyle()
            key = get_key(m, ls, c)
            self._plots += 1
            self._dict_plots[self._plots] = (subplot, key, plot_handle)

    def add_bar(self, xdata, ydata, subplot=0, polar=False, **kwargs):
        """
        Descriptions of kwargs can be found here:
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html
        """
        if subplot > self._num_subplots:
            print('subplot does not exist')
        else:
            if polar:
                self.set_subplot_polar(subplot)
            plot_handle = self._subplot_handles[subplot].bar(
                xdata, ydata, width=xdata[1] - xdata[0], align='edge', **kwargs)
            self._plots += 1
            self._dict_plots[self._plots] = (subplot, 'bar', plot_handle)

    def add_quiver(self, x, y, u, v, subplot=0, polar=False, **kwargs):
        if subplot > self._num_subplots:
            print('subplot does not exist')
        else:
            if polar:
                self.set_subplot_polar(subplot)
            plot_handle = self._subplot_handles[subplot].quiver(
                x, y, u, v, **kwargs)
            self._plots += 1
            self._dict_plots[self._plots] = (subplot, 'quiver', plot_handle)

    def add_hexbin(self, xdata, ydata, subplot=0, **kwargs):
        """
        Descriptions of kwargs can be found here:
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hexbin.html
        """
        if subplot > self._num_subplots:
            print('subplot does not exist')
        else:
            plot_handle = self.ax[subplot].hexbin(xdata, ydata, **kwargs)
            self._plots += 1
            self._dict_plots[self._plots] = (subplot, 'hexbin', plot_handle)

    def remove_plot(self, num_of_plot):
        self._subplot_handles[self._dict_plots[int(num_of_plot)][0]].lines[
            0].remove()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        del self._dict_plots[num_of_plot]

    def add_img(self, matrix, subplot=0, normalise=True):
        if subplot > self._num_subplots:
            print('subplot does not exist')
        else:
            if normalise:
                matrix = 255 * matrix / np.max(np.max(matrix))
            marker = 'img'
            plot_handle = self._subplot_handles[subplot].imshow(
                matrix.astype(np.uint8))
            self._plots += 1
            self._dict_plots[self._plots] = (subplot, marker, plot_handle)

    def remove_img(self, num_of_plot):
        self._subplot_handles[self._dict_plots[int(num_of_plot)][0]].lines[
            0].remove()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        del self._dict_plots[num_of_plot]

    def list_plots(self):
        print('plot number : subplot number , marker')
        for key in self._dict_plots.keys():
            print(key, ':', self._dict_plots[key][0], ',',
                  self._dict_plots[key][1])

    def set_subplot_polar(self, subplot):
        self._subplot_handles[subplot].remove()
        self._subplot_handles[subplot] = self.fig.add_subplot(
            self.nrows, self.ncols, subplot + 1, projection='polar')

    def configure_title(self, title='', fontsize=20):
        self.fig.suptitle(title, fontsize=fontsize)

    def configure_subplot_title(self, subplot=0, title='', fontsize=20):
        self._subplot_handles[subplot].set_title(title, fontsize=fontsize)

    def configure_xaxis(self, subplot=0, xlabel='x', fontsize=20,
                        xlim=(None, None)):
        self._subplot_handles[subplot].set_xlabel(xlabel, fontsize=fontsize)
        if (xlim[0] is not None) or (xlim[1] is not None):
            self._subplot_handles[subplot].set_xlim(left=xlim[0], right=xlim[1])

    def configure_yaxis(self, subplot=0, ylabel='y', fontsize=20,
                        ylim=(None, None)):
        self._subplot_handles[subplot].set_ylabel(ylabel, fontsize=fontsize)
        if (ylim[0] is not None) or (ylim[1] is not None):
            self._subplot_handles[subplot].set_ylim(bottom=ylim[0], top=ylim[1])

    def configure_legend(self, subplot=0, **kwargs):
        self._subplot_handles[subplot].legend(**kwargs)

    def save_figure(self, filename='*.png', initialdir='~ppzmis/Documents',
                    dpi=80):
        if filename == '*.png':
            filename = fd.save_filename(caption='select filename',
                                        file_filter='*.png;;*.jpg;;*.tiff')
        self.fig.savefig(filename, dpi=dpi)

    def show_figure(self):
        plt.show()


def get_key(marker, linestyle, color):
    if marker == 'None':
        if linestyle == 'None':
            key = ''
        else:
            key = linestyle
    else:
        if linestyle == 'None':
            key = marker
        else:
            key = marker + linestyle
    key = color + key
    return key


def histogram(data, bins=10, marker='rx', normalise=False, ax=None, show=False):
    '''
    If you feed a sequence to bins it will use these as the binedges
    '''
    if ax is None:
        fig, ax = plt.subplots()
    freq, binedges = np.histogram(data, bins=bins, density=normalise)
    if normalise:
        freq = freq / np.sum(freq)

    bins = 0.5 * (binedges[:-1] + binedges[1:])
    ax.plot(bins, freq, marker)
    if show:
        plt.show()
    return ax, bins, freq


def next_colour(colour):
    colour_vals = ['r', 'b', 'g', 'y', 'k', 'm', 'c']
    if colour >= len(colour_vals):
        colour = colour % len(colour_vals)
    return colour_vals[colour]


def next_type(index):
    type_vals = ['x', 'o', '+', '.']
    if index >= len(type_vals):
        index = index % len(type_vals)
    return type_vals[index]


if __name__ == '__main__':
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    y2, y1 = np.cos(X), np.sin(X)

    f = Plotter(subplot=(3, 1))
    f.add_plot(X, y1, fmt='r-')
    f.add_plot(X, y2, fmt='b-')
    f.add_plot(X, y2, fmt='g-', subplot=0)
    # f.save_figure()

    f.remove_plot(0)
    f.add_plot(X, y2, fmt='b-', subplot=1)
    f.configure_title('test_title')
    f.configure_xaxis(subplot=1, xlim=(0, 1))
    f.configure_yaxis()
    # x = np.arange(0, 10)
    # y = x ** 2
    # im = np.random.rand(50, 50)
    # f = Plotter(subplot=(1, 4))
    # f.add_plot(x, y, yerr=y/2)
    # f.add_plot(x, y, polar=True, subplot=1)
    # # f.add_polar_scatter(x, y, subplot=1)
    # f.add_img(im, subplot=2)
    # f.show_figure()
    # f.list_plots()

    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
    U = np.cos(X)
    V = np.sin(Y)
    f.add_quiver(X, Y, U, V, subplot=2)
    f.show_figure()

