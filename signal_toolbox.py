import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal


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
    if isinstance(xdata, pd.Series):
        series = True
        name_val = xdata.name
        x = xdata.values
        x_index = xdata.index
    else:
        name_val = ' '
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

        if show:
            plt.figure(name_val)
            plt.plot(xdata, 'rx')
            plt.plot(new_y, 'b-')
            plt.show()

        if series:
            smoothed_series = pd.Series(data=new_y, index=x_index, name=name_val)
            return smoothed_series
        else:
            return new_y

    else:
        # If window_len < 3 leave unchanged
        print('window_len < 3 leaves the data unchanged')
        return x_data


def fft_power_spectrum(tdata, ydata, limits=None, show=False):
    """
    Calculates the power spectrum on a 1D signal. It also obtains
    a quick estimate of the peak freq.

    :param tdata: time series
    :param ydata: signal to fft
    :param limits: adjusts displayed freq axis. Either None - no limits
                    or a tuple (lower limit, upper limit)
    :param show: set to True if you want to visualise the fft

    :return: 3 part tuple (freqs,powerspectrum amplitudes, peak freq)
    """
    y_fft = fftpack.rfft(ydata)
    step_size = tdata[2] - tdata[1]
    sample_freq = fftpack.fftfreq(ydata.size, d=step_size)

    power_spectrum = np.abs(y_fft)

    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power_spectrum[pos_mask]
    peak_freq = freqs[powers.argmax()]

    if show:
        plt.figure('fft')
        plt.subplot(2, 1, 1)
        plt.plot(tdata, ydata, 'rx')
        plt.title('Original data')
        plt.xlabel('tdata')
        plt.ylabel('ydata')

        plt.subplot(2, 1, 2)
        plt.plot(freqs, powers, 'b-')

        axis = plt.gca()
        if limits is not None:
            axis.set_xlim(left=limits[0], right=limits[1])
        plt.title('FFT')
        plt.xlabel('Frequency')
        plt.ylabel('Fourier Amplitude')

        plt.show()

    return freqs, powers, peak_freq


def fft_freq_filter(tdata, ydata, cutoff_freq, high_pass=True, show=False):
    """

    :param tdata: Time series data
    :param ydata: Y data
    :param cutoff_freq: cutoff frequency for filter, must be specified
    :param high_pass: If True filters high frequency if False filters low frequency
    :param show: Plots the before and after data

    :return: time data, filtered y data.
    """

    y_fft = fftpack.fftfreq(ydata.size, d=tdata[1] - tdata[0])
    f_signal = fftpack.rfft(ydata)

    # If our original signal time was in seconds, this is now in Hz
    cut_f_signal = f_signal.copy()

    if high_pass:
        cut_f_signal[(y_fft > cutoff_freq)] = 0
    else:
        cut_f_signal[(y_fft < cutoff_freq)] = 0

    cut_signal = fftpack.irfft(cut_f_signal)

    if show:
        plt.figure('fft filter')
        plt.plot(tdata, ydata, 'rx')
        plt.plot(tdata, cut_signal, 'b-')
        plt.title('Original data')
        plt.xlabel('tdata')
        plt.ylabel('ydata')
        plt.show()
    return tdata, cut_signal


def correlation(x1=None, x2=None, time_step=1.0, show=False):
    '''
    Performs the correlation of a signal with either itself or another signal
    The returned signal is normalised
    :param x1: 1D dataset as numpy array
    :param x2: optional second dataset. If you want to do autocorrelation leave blank
    x1 and x2 should be the same length
    :param time_step: convenience function which converts array index to a time.
    :param show: plots the data and correlation

    :return: returns the lags and correlation coeffs as numpy arrays
    '''
    len_data = np.shape(x1)[0]
    if x2 == None:
        x2 = x1.copy()
    time = time_step*np.arange(len_data)
    lags = time.copy()
    corr = signal.correlate(x1, x2, mode='same') / len_data

    if show:
        fig, (ax_orig, ax_corr) = plt.subplots(2, 1, sharex=True)
        ax_orig.plot(time,x1,'b-')
        ax_orig.plot(time,x2, 'r-')
        ax_orig.set_title('Original signal')
        ax_corr.plot(lags,corr,'g-')
        ax_corr.set_title('Correlation')
        ax_orig.margins(0, 0.1)
        fig.tight_layout()
        fig.show()

    return lags, corr

if __name__ == '__main__':
    # Seed the random number generator
    np.random.seed(1234)

    time_step = 0.02
    period = 5.

    time_vec = np.arange(0, 20, time_step)
    sig = (np.sin(2 * np.pi / period * time_vec)+ np.sin(2 * np.pi / (0.1*period) * time_vec) + 0.5 * np.random.randn(time_vec.size))

    fft_power_spectrum(time_vec, sig, show=True)
    time_vec,sig = fft_freq_filter(time_vec, sig, cutoff_freq=2 , high_pass=True, show=True)
    fft_power_spectrum(time_vec, sig, show=True)
    #smoothed_signal = smooth(signal, window_len=20, show=True)

    lags,corr = correlation(sig,time_step=time_step,show=True)