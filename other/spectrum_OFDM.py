#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:10:07 2019

@author: alexander
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from misc.modulation import qpsk_modulate
from misc.filters import rrcosfilter
from ofdm.OFDM import OFDMModulator

PLOT_OFDM = True
ONLY_SHAPED = False

np.random.seed(1032)

F0 = 700e6  # IF frequency
FS = 2.4e9  # Sample rate of the IF
N_bits = 1020600  # Number of bits in total
samp_per_symbol = 4

bits = (np.random.rand(N_bits) > 0.5).astype(np.int)
bits_packed = 2 * bits[::2] + bits[1::2]

# OFDM
ofdm_mod = OFDMModulator(fft_size=1024, num_cyclic_prefix=2, add_pilot=False, num_guardbands=(80,80))

# Insert only symbols in specific subcarriers to create a nice spectrum.
# subcarrier_index = np.concatenate((np.arange(-26, -3), np.arange(4, 27))) + 32
# subcarrier_index = np.concatenate((np.arange(-30, 0), np.arange(1, 31))) + 32

# Good looking pattern where the subcarriers are easily visible.
# subcarrier_index = np.array([7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57]) - 6
#bits_split = np.array(np.split(qpsk_modulate(bits_packed), (len(bits_packed) // len(subcarrier_index))))
#bits_padded = np.zeros((len(bits_split), 56), dtype=np.complex)
#bits_padded[:, subcarrier_index] = bits_split
#bits_padded = np.hstack(bits_padded)

# Perform OFDM modulation
OFDM_symbols = ofdm_mod.modulate(qpsk_modulate(bits_packed))

# Upsample to FS
OFDM_samples = np.zeros(len(OFDM_symbols) * samp_per_symbol, dtype=np.complex)
OFDM_samples[::samp_per_symbol] = 1000 * OFDM_symbols * samp_per_symbol

IQ_data = np.repeat(OFDM_symbols, samp_per_symbol)

# Filter for upsampling
# h = signal.firwin(time_bandwidth_product * samp_per_symbol, 1/samp_per_symbol)
OFDM_bandwidth = FS / samp_per_symbol
# h = signal.firwin(20 * samp_per_symbol, OFDM_bandwidth/2, fs=FS)

FILTER_LEN = 31
FILTER_ALPHA = 0.15
upsamp_filter = rrcosfilter(FILTER_LEN, FILTER_ALPHA, samp_per_symbol / FS, FS)
upsamp_filter /= (upsamp_filter.sum() / samp_per_symbol)


# Apply upsampling filter
OFDM_samples_shaped = signal.fftconvolve(OFDM_samples, upsamp_filter, mode='same') # Apply filter

# Scipy also has a way to do it.
OFDM_samples_resampled = signal.resample(1000*OFDM_symbols, len(OFDM_symbols) * samp_per_symbol)


# Plot
N_samples_total = len(OFDM_samples)
t = np.linspace(0, 1/FS * N_samples_total, N_samples_total)

# To place the waveforms on an IF wave for better looking spectrums
carrier_I = np.cos(2 * np.pi * t * F0)
carrier_Q = np.sin(2 * np.pi * t * F0)

###
if not ONLY_SHAPED:
    OFDM_time_series = IQ_data.real * carrier_I + IQ_data.imag * carrier_Q
    _, axes = plt.subplots(1, 2)
    axes[0].plot(t * 1e6, IQ_data)
    axes[0].set_xlabel('Time [µs]')
    axes[0].set_title("OFDM time series")

    freq2, frequency = signal.welch(OFDM_time_series, fs=FS, detrend=False, nperseg=4096, return_onesided=False)
    axes[1].plot(freq2,  10*np.log10(np.abs(frequency)))
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_title("OFDM frequency response")
    plt.axvline(F0, c='C1')
    plt.axvline(F0 + OFDM_bandwidth/2, c='C2')
    plt.axvline(F0 - OFDM_bandwidth/2, c='C2')

###
OFDM_time_series_shaped = OFDM_samples_shaped.real * carrier_I + OFDM_samples_shaped.imag * carrier_Q
_, axes = plt.subplots(1, 2)
axes[0].plot(t * 1e6, OFDM_time_series_shaped)
axes[0].set_xlabel('Time [µs]')
axes[0].set_title("OFDM (pulse shaped) time series")

freq_shaped, frequency_shaped = signal.welch(OFDM_time_series_shaped, fs=FS, detrend=False, nperseg=4096, return_onesided=False)
axes[1].plot(freq_shaped,  10*np.log10(np.abs(frequency_shaped)))
axes[1].set_xlabel('Frequency [Hz]')
axes[1].set_title("OFDM (pulse shaped) frequency response")

plt.axvline(F0, c='C1')
plt.axvline(F0 + OFDM_bandwidth/2, c='C2')
plt.axvline(F0 - OFDM_bandwidth/2, c='C2')


###
OFDM_time_series_resamp = OFDM_samples_resampled.real * carrier_I + OFDM_samples_resampled.imag * carrier_Q
_, axes = plt.subplots(1, 2)
axes[0].plot(t * 1e6, OFDM_time_series_resamp)
axes[0].set_xlabel('Time [µs]')
axes[0].set_title("OFDM (resampled) time series")

freq_resamp, frequency_resamp = signal.welch(OFDM_time_series_resamp, fs=FS, detrend=False, nperseg=4096, return_onesided=False)
axes[1].plot(freq_resamp,  10*np.log10(np.abs(frequency_resamp)))
axes[1].set_xlabel('Frequency [Hz]')
axes[1].set_title("OFDM (resampled) frequency response")

plt.axvline(F0, c='C1')
plt.axvline(F0 + OFDM_bandwidth/2, c='C2')
plt.axvline(F0 - OFDM_bandwidth/2, c='C2')


###
plt.figure()
plt.title("OFDM symbols directly")
freq_direct, frequency_direct = signal.welch(OFDM_samples_shaped, fs=FS, detrend=False, nperseg=4096)
plt.plot(np.fft.fftshift(freq_direct),  np.fft.fftshift(10*np.log10(np.abs(frequency_direct))))


#%%
frequency_shaped_log = 10*np.log10(np.abs(frequency_shaped))
frequency_shaped_log -= frequency_shaped_log.max()
plt.figure(); plt.plot(freq_shaped,  frequency_shaped_log)

plt.xlabel('Frequency [Hz]')
plt.title("Filtered frequency response")

plt.axvline(F0, c='C1')
plt.axvline(F0 + OFDM_bandwidth/2, c='C2')
plt.axvline(F0 - OFDM_bandwidth/2, c='C2')

plt.xlim([F0 - OFDM_bandwidth, F0 + OFDM_bandwidth])
plt.ylim(-40, 5)


#%%
frequency_log = 10*np.log10(np.abs(frequency))
frequency_log -= frequency_log.max()
plt.figure(); plt.plot(np.fft.fftshift(freq2),  np.fft.fftshift(frequency_log))

plt.xlabel('Frequency [Hz]')
plt.title("Unfiltered frequency response")

plt.axvline(F0, c='C1')
plt.axvline(F0 + OFDM_bandwidth/2, c='C2')
plt.axvline(F0 - OFDM_bandwidth/2, c='C2')

plt.xlim([F0 - OFDM_bandwidth, F0 + OFDM_bandwidth])
plt.ylim(-40, 5)
