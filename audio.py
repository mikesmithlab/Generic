import os
import subprocess
from scipy.io.wavfile import read
import numpy as np
from moviepy.editor import AudioFileClip
import matplotlib.pyplot as plt

"""Functions for extracting the frequencies of square waves in a 
sound file."""

def digitise(sig):
    """Makes a noisy square signal, perfectly square."""
    out = np.zeros(len(sig))
    out[sig < 0.8*np.min(sig)] = -1
    out[sig > 0.8*np.max(sig)] = 1
    out[(sig > 0.8*np.min(sig))*(sig < 0.8*np.max(sig))] = 0
    return out


def fourier_transform_peak(sig, time_step):
    """Find the peak frequency in a signal"""
    ft = abs(np.fft.fft(sig, n=50000))
    # freq = np.fft.fftfreq(len(sig), time_step)
    freq = np.fft.fftfreq(50000, time_step*2)
    peak = np.argmax(ft)
    return abs(freq[peak])


def frame_frequency(wave, frames, audio_rate):
    """Returns the peak frequency in an audio file for each video frame"""
    window = int(len(wave)/frames)
    windows = frames
    freq = np.zeros(windows)
    for i in range(windows):
        b = i*window
        t = (i+1)*window
        if t > len(wave):
            t = len(wave)
        freq[i] = int(fourier_transform_peak(wave[b:t], 1/audio_rate))
    return freq


def extract_wav(file):
    audioclip = AudioFileClip(file)
    audioclip_arr = audioclip.to_soundarray(fps=48000, nbytes=2)
    return audioclip_arr
