import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from misc.modulation import qpsk_modulate
from ofdm.OFDM import OFDMModulator

PLOT_OOK = False
PLOT_BPSK = False
PLOT_QPSK = False
PLOT_OFDM = True
PLOT_ANY = PLOT_BPSK or PLOT_OOK or PLOT_QPSK or PLOT_OFDM

if PLOT_ANY:
    plt.close('all')


def plot_response(title, t, freq, time_series, time_series_shaped, only_shaped=False):
    frequency = np.fft.fft(time_series)[:len(time_series) // 2]
    frequency_shaped = np.fft.fft(time_series_shaped)[:len(time_series_shaped) // 2]

    if not only_shaped:
        _, axes = plt.subplots(1, 2)
        axes[0].plot(t, time_series)
        axes[0].set_xlabel('Time [s]')
        axes[0].set_title("{} time series".format(title))

        axes[1].plot(freq,  10*np.log10(np.abs(frequency)))
        axes[1].set_xlabel('Frequency [Hz]')
        axes[1].set_title("{} frequency response".format(title))

    _, axes = plt.subplots(1, 2)
    axes[0].plot(t, time_series_shaped)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_title("{} (pulse shaped) time series".format(title))

    axes[1].plot(freq, 10*np.log10(np.abs(frequency_shaped)))
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_title("{} (pulse shaped) frequency response".format(title))


# Parameters
F0 = 5e3
FS = 24000
symbol_rate = 1000
samp_per_symb = int(FS / symbol_rate)
N_bits = 2000
t = np.arange(0, 2, 1 / FS)
freq = np.linspace(0, 0.5, len(t) // 2) * FS
np.random.seed(10032)

# Common signals
bits = (np.random.rand(N_bits) > 0.5).astype(np.int)
carrier_I = np.cos(2 * np.pi * t * F0)
carrier_Q = np.sin(2 * np.pi * t * F0)


# On-off keying
OOK_symbols = bits.copy()
OOK_samples = np.repeat(OOK_symbols, samp_per_symb)

OOK_time_series = OOK_samples * carrier_I
OOK_frequency = np.fft.fft(OOK_time_series)[:len(OOK_samples) // 2]

time_bandwidth_product = 4
h = signal.firwin(time_bandwidth_product * samp_per_symb, 1 / samp_per_symb)
OOK_samples_shaped = np.zeros(samp_per_symb * N_bits)
OOK_samples_shaped[::samp_per_symb] = OOK_symbols
OOK_samples_shaped = signal.fftconvolve(OOK_samples_shaped, h, mode='same')

OOK_time_series_shaped = OOK_samples_shaped * carrier_I
OOK_time_series_shaped /= OOK_time_series_shaped.max()
OOK_frequency_shaped = np.fft.fft(OOK_time_series_shaped)[:len(OOK_samples_shaped) // 2]
if PLOT_OOK:
    plot_response('On-Off Keying', t, freq, OOK_time_series, OOK_time_series_shaped)


# BPSK
BPSK_symbols = 2 * bits.copy() - 1
BPSK_samples = np.repeat(BPSK_symbols, samp_per_symb)

BPSK_time_series = BPSK_samples * carrier_I

time_bandwidth_product = 4
h = signal.firwin(time_bandwidth_product * samp_per_symb, 1 / samp_per_symb)
BPSK_samples_shaped = np.zeros(samp_per_symb * N_bits)
BPSK_samples_shaped[::samp_per_symb] = BPSK_symbols
BPSK_samples_shaped = signal.fftconvolve(BPSK_samples_shaped, h, mode='same')

BPSK_time_series_shaped = BPSK_samples_shaped * carrier_I
BPSK_time_series_shaped /= BPSK_time_series_shaped.max()
if PLOT_BPSK:
    plot_response('Binary PSK', t, freq, BPSK_time_series, BPSK_time_series_shaped)


# QPSK
bits_packed = 2 * bits[::2] + bits[1::2]
QPSK_samp_per_symb = 2 * samp_per_symb
QPSK_symbols = qpsk_modulate(bits_packed)
QPSK_samples = np.repeat(QPSK_symbols, QPSK_samp_per_symb)

time_bandwidth_product = 4
h = signal.firwin(time_bandwidth_product * QPSK_samp_per_symb, 1 / QPSK_samp_per_symb)
QPSK_samples_shaped = np.zeros(QPSK_samp_per_symb * N_bits // 2, dtype=np.complex)
QPSK_samples_shaped[::QPSK_samp_per_symb] = QPSK_symbols * QPSK_samp_per_symb
QPSK_samples_shaped = np.convolve(QPSK_samples_shaped, h, mode='same')

QPSK_time_series = QPSK_samples.real * carrier_I + QPSK_samples.imag * carrier_Q

QPSK_time_series_shaped = QPSK_samples_shaped.real * carrier_I + QPSK_samples_shaped.imag * carrier_Q
if PLOT_QPSK:
    plot_response('Quadrature PSK', t, freq, QPSK_time_series, QPSK_time_series_shaped, only_shaped=True)


# OFDM
#TODO This does not work completely. Check spectrum_OFDM.py
ofdm_mod = OFDMModulator(fft_size=32, num_cyclic_prefix=2, add_pilot=True)

OFDM_symbols = ofdm_mod.modulate(bits_packed)
OFDM_samp_per_symb = len(QPSK_samples) // len(OFDM_symbols)
OFDM_samples = signal.resample(OFDM_symbols, OFDM_samp_per_symb * len(OFDM_symbols))

t_short = t[:len(OFDM_samples)]
freq_short = np.linspace(0, 0.5, len(t_short) // 2) * FS
carrier_I_short = np.cos(2 * np.pi * t_short * F0)
carrier_Q_short = np.sin(2 * np.pi * t_short * F0)

OFDM_time_series = OFDM_samples.real * carrier_I_short + OFDM_samples.imag * carrier_Q_short

if PLOT_OFDM:
    plot_response('OFDM', t_short, freq_short, OFDM_time_series, OFDM_time_series, only_shaped=True)



if PLOT_ANY:
    plt.show()
