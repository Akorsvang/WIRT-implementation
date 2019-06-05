# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as signal

from misc.zadoff_chu import generate_zc_sequence

from instrument_control.socket_manager import SocketConnection
import instrument_control.tester_tools as tt
from instrument_control.waveforms import waveform

iqvsg_filename = "tmp_sync_test.iqvsg"
FS = int(2.4e9)
samp_per_symbol = 4
N = 353
u = 7

expected_bandwidth = FS / samp_per_symbol
upsamp_filter = signal.firwin(30 * samp_per_symbol, expected_bandwidth / 2, fs=FS)

###
# Generate sequence
zc_sequence = generate_zc_sequence(N, u)

# Upsample
zc_seq_upsamp = np.zeros(len(zc_sequence) * samp_per_symbol, dtype=np.complex)
zc_seq_upsamp[::samp_per_symbol] = zc_sequence * samp_per_symbol
# zc_seq_upsamp = np.pad(zc_seq_upsamp, (10, 10), 'constant')  # I've seen issues with the padding..?
zc_seq_upsamp = np.convolve(zc_seq_upsamp, upsamp_filter, mode='same') # Apply filter

# Pad to make the detection less obvious
zc_sequence_padded = np.pad(zc_seq_upsamp, (245, 50), 'constant')

# Pack into the waveform format the tester expects
IQ_bytes = waveform.iq_to_bytes(zc_seq_upsamp)
waveform_bytes = waveform.bytes_to_waveform(IQ_bytes, fs=FS, template_file='instrument_control/waveforms/template.iqvsg')



###
# Socket connection
expected_duration = len(zc_sequence_padded) * (1 / expected_bandwidth)

sock = SocketConnection()
tt.tester_default_setup(sock)

remote_name = tt.upload_waveform(waveform_bytes, sock=sock)
tt.tester_setup_repeat(remote_name, sock, n_repeats=1)
tt.tester_set_capturetime(1e-6 + expected_duration, sock)
tt.tester_setup_ext_triggers(sock)
tt.tester_arm(sock)
tt.tester_ext_trigger(sock)

samples = tt.download_waveform(sock, send_trigger=False)

sock.close()



### Sample analysis
# The sample where the sequence approximately starts.
approximate_start = 1750
indicies_pre = np.arange(approximate_start, approximate_start + 1.5 * len(zc_seq_upsamp), dtype=int)
indicies_post = np.arange(approximate_start//4, approximate_start//4 + 1.5 * len(zc_sequence), dtype=int)

# Autocorrelation before reducing the amount of samples
autocorrelation_preresamp = np.correlate(samples[indicies_pre], zc_seq_upsamp, mode='same')
max_idx_pre = np.argmax(np.abs(autocorrelation_preresamp)) + approximate_start
sequence_start_idx = max_idx_pre - samp_per_symbol * (N // 2)
sequence_end_idx = max_idx_pre + samp_per_symbol * (N // 2)

sequence_sampled_pre_noresamp = samples[sequence_start_idx:sequence_end_idx]
sequence_sampled_pre = np.convolve(sequence_sampled_pre_noresamp, upsamp_filter, mode='same')[::samp_per_symbol]


# Synchronization after reducing the amount of samples
samples_downsamp = np.convolve(samples, upsamp_filter, mode='same')[::samp_per_symbol]
autocorrelation_postresamp = np.correlate(samples_downsamp[indicies_post], zc_sequence, mode='same')
max_idx_post = np.argmax(np.abs(autocorrelation_postresamp)) + approximate_start // 4
sequence_start_idx = max_idx_post - N // 2
sequence_end_idx = max_idx_post + N // 2

sequence_sampled_post = samples_downsamp[sequence_start_idx:sequence_end_idx]

