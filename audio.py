import os
import subprocess
from scipy.io.wavfile import read
import numpy as np

"""Functions for extracting the frequencies of square waves in a 
sound file."""

def extract_wav(file):
    """
    Extract a wav file from an mp4

    Parameters
    ----------
    file: str
        Full path to the mp4

    Returns
    -------
    wav: list
        wav[0] = rate Hz
        wav[1] = audio signal
    """
    out_file = os.path.splitext(file)[0] + '.wav'
    if os.path.exists(out_file):
        pass
    else:
        command = "ffmpeg -i '{}' -ab 160k -ac 2 -ar 44100 -vn '{}'".format(
            file, out_file)
        subprocess.call(command, shell=True)
    wav = read_wav(out_file)
    return wav

def read_wav(file):
    return read(file)


def digitise(sig):
    """Makes a noisy square signal, perfectly square."""
    out = np.zeros(len(sig))
    out[sig < 0.8*np.min(sig)] = -1
    out[sig > 0.8*np.max(sig)] = 1
    out[(sig > 0.8*np.min(sig))*(sig < 0.8*np.max(sig))] = 0
    return out


def fourier_transform_peak(sig, time_step):
    """Find the peak frequency in a signal"""
    ft = abs(np.fft.fft(sig))
    freq = np.fft.fftfreq(len(sig), time_step)
    peak = np.argmax(ft)
    return abs(freq[peak])


def frame_frequency(wave, audio_rate, video_rate):
    """Returns the peak frequency in an audio file for each video frame"""
    window = int(audio_rate/video_rate)
    windows = int(np.ceil(len(wave)/window))
    freq = np.zeros(windows)
    for i in range(windows):
        b = i*window
        t = (i+1)*window
        if t > len(wave):
            t = len(wave)
        freq[i] = int(fourier_transform_peak(wave[b:t], 1/audio_rate))
    return freq
