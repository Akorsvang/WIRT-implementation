#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:50:36 2019

@author: alexander
"""

from datetime import datetime

import numpy as np

from wirt.WIRT import wirt
from misc.ecc.ecc_utils import channel_AWGN

#%% Parameters
# Base
SEED = 123
num_runs_max = 50000
num_frame_errors_max = 150
ESNOs = np.arange(-20, -10, 0.25)
ENABLE_EQ = False
WRITE_FILE = True

if ENABLE_EQ:
    FILENAME = "output/WIRT_AWGN_eq.npz"
else:
    FILENAME = "output/WIRT_AWGN_noeq.npz"

#%% Create objects
encoder = wirt(enable_equalization=ENABLE_EQ)

results_BER = np.empty((len(ESNOs)))
results_BLER = np.empty((len(ESNOs)))
results_frame_errors = np.empty((len(ESNOs)))

np.random.seed(SEED)
#%% Perform the test
for i, esno in enumerate(ESNOs):
    print("ESNO", esno)
    total_errors = 0
    total_frame_errors = 0
    total_bits = 0
    total_frames = 0

    for _ in range(num_runs_max):
        data = np.random.binomial(1, 0.5, wirt.DATA_SIZE).astype(np.uint8)

        # Encoder
        encoded = encoder.encode(data)

        # Channel
        received = channel_AWGN(np.pad(encoded, (20, 40), 'constant'), esno)

        # Decoder
        decoded = encoder.decode(received, estimate_esno=esno)


        ###
        errors = (decoded != data)
        total_errors += errors.sum()
        total_frame_errors += errors.any()

        total_bits += wirt.DATA_SIZE
        total_frames += 1

        if total_frame_errors >= num_frame_errors_max:
            break
    else:
        print("Timeout at {} dB".format(esno))

    results_BER[i] = total_errors / total_bits
    results_BLER[i] = total_frame_errors / total_frames
    results_frame_errors[i] = total_frame_errors

#%% Save file
config = {
        "seed": SEED,
        "num_runs": num_runs_max,
        "ESNOs": ESNOs,
        "Time": datetime.now(),
        'num_runs_max':num_runs_max,
        'num_frame_errors_max':num_frame_errors_max
        }

if WRITE_FILE:
    np.savez(FILENAME,
             ESNOs=ESNOs,
             results_BER=results_BER, results_BLER=results_BLER, frame_errors=results_frame_errors,
             config=config
             )
