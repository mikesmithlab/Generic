import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def smooth(xdata, window_len=0, window='bartlett', show=False):
    """smooth a pandas data series or numpy array using a window with requested size.
    Return has same format as input.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an even integer. Set to 0 as default
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if type(data) == type(pd.Series()):
        series = True
        name_val = xdata.name
        x = xdata.values
        x_index = xdata.index
    else:
        series = False
        x = xdata


    if window_len > 3:
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')

        new_y1 = y[int(window_len / 2 - 1):-int(window_len / 2)]
        new_y2 = y[int(window_len / 2 - 1) + 1:-int(-1 + window_len / 2)]
        new_y = 0.5 * (new_y1 + new_y2)

        # Pack back into pd.Series
        smoothed_vals = pd.Series(new_y, name=name_val, index=x_index)

        if show:
            plt.figure(name_val)
            plt.plot(x_series[x_series.index < 200].index, x_series[x_series.index < 200], 'rx')
            plt.plot(smoothed_vals[smoothed_vals.index < 200].index, smoothed_vals[smoothed_vals.index < 200], 'b-')
        if series:
            smoothed_series = pd.Series(data=smoothed_vals,index=x_index,name=name_val)
            return smoothed_series
        else:
            return smoothed_vals

    else:
        # If window_len < 3 leave unchanged
        return x_data

def fft(tdata,ydata,show=False):

    Y = np.fft.fft(ydata)
    freq = np.fft.fftfreq(len(y), tdata[1] - tdata[0])

    if show:
        plt.figure('fft')
        plt.subplot(2, 1, 1)
        plt.plot(tdata, ydata, 'rx')
        plt.title('Original data')
        plt.xlabel('tdata')
        plt.ylabel('ydata')

        plt.subplot(2, 1, 2)
        plt.plot(freq, Y, 'b-')
        plt.title('Fourier Transform')
        plt.xlabel('Frequency')
        plt.ylabel('Fourier Amplitude')

        plt.show()

    return (freq, Y)


if __name__ == '__main__':
    freq = 0.1
    noise = 0.1
    amplitude = 3.0

    t = np.arange(0,1000,1)
    y = amplitude*np.sin(2*np.pi*freq*t) + noise*np.random.normal(size=1000)

    F,Y = fft(t,y,show=True)