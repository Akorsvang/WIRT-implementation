# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np

def plot_spectrum(samples, fs=2.4e9):
    plt.figure()
    freq, response = signal.welch(samples, fs=fs, detrend=False, nperseg=4096, return_onesided=False)
    
    response = np.fft.fftshift(response)
    response_adjusted = 10*np.log10(response * response.conj())
    response_adjusted -= response_adjusted.max()
    
    plt.plot(np.fft.fftshift(freq),  response_adjusted )
    

