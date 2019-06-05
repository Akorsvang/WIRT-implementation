# -*- coding: utf-8 -*-
from time import perf_counter as pf
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


from wirt.WIRT import wirt
from instrument_control.waveforms.waveform import iq_to_waveform
from instrument_control.socket_manager import SocketConnection
import instrument_control.tester_tools as tt


#%% Parameters
# Base
SEED = 123
ENABLE_EQ = True
WRITE_FILE = False
PLOT_POWER = False

FS = 2.4e9
num_runs = 100

# The approximate offset of the VSA vs VSG trigger.
approximate_start = 1800


#%%
encoder = wirt(enable_equalization=ENABLE_EQ)
expected_duration = encoder.num_samples_per_package/encoder.FS

if ENABLE_EQ:
    FILENAME = "output/WIRT_OTA_eq.npz"
else:
    FILENAME = "output/WIRT_OTA_noeq.npz"


#%% Loop

BLERs = np.full(num_runs, -1, dtype=float)
BERs = np.full(num_runs, -1, dtype=float)

# Socket connection
sock = SocketConnection()
tt.tester_default_setup(sock)
tt.tester_set_capturetime(10e-6 + num_runs * (expected_duration), sock)

# Seed the RNG
np.random.seed(SEED)


#%% Generate some waves
st = pf()

data = np.random.binomial(1, 0.5, (num_runs, wirt.DATA_SIZE)).astype(np.uint8)

IQ_data_full = np.empty((num_runs, encoder.num_samples_per_package), np.complex)
for i in range(num_runs):
    encoded = encoder.encode(data[i])
    IQ_data_full[i] = encoded

print("Generation/modulation took {:.2f} ms".format(1000 * (pf() - st)))


#%% Transmit to tester
st = pf()

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

sock.close()

print("Instrument took {:.2f} ms".format(1000 * (pf() - st)))


#%% Handle the decoding.
st = pf()
for run_i in range(num_runs):
    run_offset = run_i * encoder.num_samples_per_package + approximate_start
    run_samples = samples[run_offset:run_offset + encoder.num_samples_per_package]

    decoded = encoder.decode(run_samples, noise_power=-45)

    errors = (decoded != data[run_i])
    BERs[run_i] = errors.sum()
    BLERs[run_i] = errors.any()


if PLOT_POWER:
    # Calculate first sample based on power
    samples_power = 10 * np.log10((samples * samples.conj()).real) # I/Q are in units of sqrt(mW), so this is dBm
    plt.plot(np.arange(0, len(samples_power)/FS, 1/FS) * 1e6, samples_power)
    plt.xlabel("t [µs]")




print("Demodulation took {:.2f} ms".format(1000 * (pf() - st)))


#%% Save file

if WRITE_FILE:
    np.savez(FILENAME,
             results_BER=BERs, results_BLER=BLERs,
             config={
            "seed": SEED,
            "num_runs": num_runs,
            "Time": datetime.now(),
            }
             )


