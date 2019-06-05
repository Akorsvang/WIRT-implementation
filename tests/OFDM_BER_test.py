# -*- coding: utf-8 -*-
from time import perf_counter as pf

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ofdm import OFDM
from misc.zadoff_chu import generate_zc_sequence
from misc.modulation import qpsk_modulate, qpsk_demodulate, qpsk_demodulate_soft, packed_to_unpacked, unpacked_to_packed, qpsk_hard_decision
from misc.ecc.ecc_utils import channel_AWGN
from polar.polar import polar_encode
from polar.polar_ssc import polar_decode_ssc
from instrument_control.waveforms.waveform import iq_to_waveform
from instrument_control.socket_manager import SocketConnection
import instrument_control.tester_tools as tt


#%% Parameters
# Base
SEED = 123
FS = 2.4e9
samp_per_symbol = 4
bits_per_symbol = 2
filter_len = 3 * samp_per_symbol
num_runs = 20
ESNO = 10
polar_size = 2**13
PLOT_POWER = False
PLOT_CONSTELLATION = False
NO_TESTER = False

# WIRT parameters
wanted_subcarrier_spacing = 960e3
fft_size = 625  # See report
num_subcarriers = 524
num_pilots = 124
minimum_slot_size = 20e-6
num_databytes = 50
# num_databits = 8 * num_databytes
num_databits = 400
# ecc_rate = 1/32
ecc_rate = num_databits / polar_size

EBNO_expected = ESNO - 10 * np.log10(2*ecc_rate)

cycl_prefix = np.ceil(69e-9 * FS).astype(np.int)
gb_count = fft_size - num_subcarriers
guardbands = (np.floor(gb_count / 2).astype(int), np.ceil(gb_count / 2).astype(int))

# Create the OFDM object
mod = OFDM.OFDMModulator(fft_size=fft_size, num_cyclic_prefix=cycl_prefix, add_pilot=True, num_pilots=num_pilots, num_guardbands=guardbands)

# Calculate some stats
achieved_subcarrier_spacing = (FS / samp_per_symbol) / fft_size

num_total_qpsksymbols = (FS / samp_per_symbol) * minimum_slot_size * (mod.num_data_carriers / mod.get_modulated_size(1))
num_enc_qpsksymbol = 4 * np.ceil(num_databytes * 1/ecc_rate).astype(int)
num_ofdm_repeats = np.ceil(num_total_qpsksymbols / num_enc_qpsksymbol).astype(int)

ofdm_symbol_samples_count = mod.get_modulated_size(num_enc_qpsksymbol)
symbol_count = ofdm_symbol_samples_count * num_ofdm_repeats

expected_duration = symbol_count * samp_per_symbol * (1 / FS)


# Resampling filters
# Using this increased filter size is not optimal, but since we have the guardbands anyway
# we can reduce the required number of operations in filtering by using a "slower" filter
upsamp_filter   = signal.firwin(filter_len, (1.1 * num_subcarriers * wanted_subcarrier_spacing) / 2, fs=FS)
downsamp_filter = signal.firwin(filter_len, (1.1 * num_subcarriers * wanted_subcarrier_spacing) / 2, fs=FS)

# Split it up for polyphase implementation.
# See https://www.dsprelated.com/showarticle/191.php and others on that site.
h = np.array([upsamp_filter[i::samp_per_symbol] for i in [3,0,1,2]]) * samp_per_symbol
h_down = np.array([downsamp_filter[i::samp_per_symbol] for i in [3,0,1,2]])


# Zadoff-Chu sequence parameters
zc_N = 73
u = 7

# Approximate offset and the uncertainty in number of samples
approximate_start = 1700
n_samples_uncertainty = 400

# Generate Zadoff-Chu sequence
zc_sequence = generate_zc_sequence(zc_N, u)
zc_seq_upsamp = np.zeros(len(zc_sequence) * samp_per_symbol, dtype=np.complex)
zc_seq_upsamp[::samp_per_symbol] = zc_sequence * samp_per_symbol
zc_seq_upsamp = np.convolve(zc_seq_upsamp, upsamp_filter, mode='same') # Apply filter

zc_duration = len(zc_seq_upsamp) / FS

#%% Loop init
# Save some info between runs
first_samples = np.full(num_runs, -1)
BLERs_raw = np.full(num_runs, -1, dtype=float)
BLERs_ecc = np.full(num_runs, -1, dtype=float)
BERs_raw = np.full(num_runs, -1, dtype=float)
BERs_ecc = np.full(num_runs, -1, dtype=float)

# Socket connection
if not NO_TESTER:
    sock = SocketConnection()
    tt.tester_default_setup(sock)
    tt.tester_set_capturetime(10e-6 + num_runs * (zc_duration + expected_duration), sock)

# Seed the RNG
np.random.seed(SEED)

#%% Generate some waves
st = pf()

data_orig = np.random.bytes(num_runs * num_databytes)
data_orig_uint = np.frombuffer(data_orig, np.uint8).reshape(num_runs, num_databytes)

IQ_data_full = np.empty((num_runs, samp_per_symbol * (symbol_count + len(zc_sequence))), np.complex)
for i in range(num_runs):
    bytes_idx = slice(i * num_databytes, (i + 1) * num_databytes)
    data = np.frombuffer(data_orig[bytes_idx], dtype=np.uint8)
    data_bits = np.unpackbits(data)

    # Encode with a simple (systematic) conv ECC from
    # data_enc = conv_encode(data, 3, 4, 5)
    # data_enc = np.concatenate(data_enc)
    data_polar = polar_encode(polar_size, num_databits, data_bits)
    data_enc = np.packbits(data_polar)

    # QPSK modulate
    data_enc_mod = qpsk_modulate(packed_to_unpacked(data_enc))

    # OFDM modulate
    IQ_data_clean = mod.modulate(data_enc_mod)

    # Repeat the package to match the size of the transmission window
    IQ_data_clean = np.tile(IQ_data_clean, (1, num_ofdm_repeats)).ravel()

    # Upsampling, Polyphase
    IQ_data = np.zeros(len(IQ_data_clean) * samp_per_symbol, dtype=np.complex)
    for j in range(samp_per_symbol):
        IQ_data[j::samp_per_symbol] = np.convolve(IQ_data_clean, h[j], mode='same')

    # Normalize the power to mean 0 dBm
    signal_power = (IQ_data.conj() * IQ_data).mean().real
    IQ_data /= np.sqrt(signal_power)

    # Add the preample sequence
    IQ_data_zc = np.hstack((zc_seq_upsamp, IQ_data))

    IQ_data_full[i] = IQ_data_zc

print("Generation/modulation took {:.2f} ms".format(1000 * (pf() - st)))


#%% Transmit to tester
st = pf()

if NO_TESTER:
    # channel_output = channel_AWGN(IQ_data_full.ravel(), ESNO)
    channel_output = IQ_data_full.ravel()
    # Add some zeros to get the same pre-padded length as the tester
    samples = np.pad(channel_output, (approximate_start, 0), 'constant')
else:
    # Convert to waveform
    waveform_bytes = iq_to_waveform(IQ_data_full.ravel(), fs=FS, template_file='instrument_control/waveforms/template.iqvsg')

    ###
    # Handle the instrument
    remote_name = tt.upload_waveform(waveform_bytes, sock=sock, play_immediate=False)
    tt.tester_setup_repeat(remote_name, sock, n_repeats=1)

    # Triggers
    tt.tester_setup_ext_triggers(sock)
    tt.tester_arm(sock)
    tt.tester_ext_trigger(sock)

    # Download the actual samples
    samples = tt.download_waveform(sock, send_trigger=False)

# Downsample
samples_downsamp = np.convolve(samples, downsamp_filter, mode='same')[::samp_per_symbol]

print("Instrument took {:.2f} ms".format(1000 * (pf() - st)))


#%% Handle the decoding.
st = pf()
for run_i in range(num_runs):
    run_offset = run_i * (symbol_count + len(zc_sequence)) + approximate_start//4

    ###
    # Sync
    # Synchronize based on Zadoff-Chu sequence
    ZC_indicies = np.arange(run_offset, run_offset + 1.5 * len(zc_sequence), dtype=int)
    autocorrelation_postresamp = np.correlate(samples_downsamp[ZC_indicies], zc_sequence, mode='same')
    max_idx_post = np.argmax(np.abs(autocorrelation_postresamp)) + run_offset
    sequence_start_idx = max_idx_post - zc_N // 2
    # For an uneven ZC length, the "extra" sample is at the end
    sequence_end_idx = max_idx_post + np.ceil(zc_N / 2).astype(int)

    # The ZC indexes
    sequence_sampled_post = samples_downsamp[sequence_start_idx:sequence_end_idx]
    first_samples[run_i] = first_sample_zc = sequence_end_idx

    # Pick out the synchronized samples
    samples_all = samples_downsamp[first_sample_zc:first_sample_zc+symbol_count].reshape((-1, ofdm_symbol_samples_count))
    samples_single = samples_all[0]
    samples_combined = samples_all.sum(axis=0)

    # Demodulate the samples
    symbols_all = np.array([mod.demodulate(samples_all[j])[:num_enc_qpsksymbol] for j in range(len(samples_all))])
    symbols_single = mod.demodulate(samples_single)[:num_enc_qpsksymbol]
    symbols_combined = mod.demodulate(samples_combined)[:num_enc_qpsksymbol]

    # Demodulate in a soft manner and combine the LLRs
    LLRs = qpsk_demodulate_soft(symbols_all, ESNO).mean(axis=0)

    symbols_combined_LLR = unpacked_to_packed(qpsk_hard_decision(LLRs))

    # QPSK demodulate and compare
    data_recv = unpacked_to_packed(qpsk_demodulate(symbols_combined))
    # data_recv = symbols_combined_LLR
    data_recv_bits = np.unpackbits(data_recv)

    # dr = data_recv[::2]
    # BERs_raw[run_i] = (np.unpackbits(dr) != np.unpackbits(data_orig_uint[run_i])).mean()
    # BLERs_raw[run_i] = (dr != data_orig_uint[run_i]).mean()

    # dr_ecc = conv_decode(data_recv[::2], data_recv[1::2], 3, 4, 5)
    dr_ecc = polar_decode_ssc(polar_size, num_databits, LLRs.flatten())

    BERs_ecc[run_i] = (dr_ecc != np.unpackbits(data_orig_uint[run_i])).mean()
    BLERs_ecc[run_i] = (np.packbits(dr_ecc) != data_orig_uint[run_i]).any().astype(np.uint8)


    ###
    # Plotting
    if PLOT_CONSTELLATION and run_i in list(range(10)):
        plt.figure("Constellation plot {}".format(run_i))
        plt.plot(symbols_single.real, symbols_single.imag, 'ro', label="No power combining")
        plt.plot(symbols_combined.real, symbols_combined.imag, 'bo', label="Power combining")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.legend()

    if PLOT_POWER:
        # Calculate first sample based on power
        samples_power = 10 * np.log10((samples * samples.conj()).real) # I/Q are in units of sqrt(mW), so this is dBm
        plt.plot(np.arange(0, len(samples_power)/FS, 1/FS) * 1e6, samples_power)
        plt.xlabel("t [µs]")

    if num_runs > 10 and run_i % (num_runs // 10) == 0:
        print(int(run_i / num_runs * 100), end=',')


if not NO_TESTER:
    sock.close()

print("\nDemodulation took {:.2f} ms".format(1000 * (pf() - st)))

if PLOT_POWER:
    plt.show()