# -*- coding: utf-8 -*-

import numpy as np


def channel_AWGN(data, esno):
    """
    Adds AWGN noise corresponding to the specified ES/NO (in dB) and the average power of the input.
    """
    data = data.reshape(-1)

    signal_power = ((data.conjugate() * data).mean()).real

    esno_real = 10**(esno / 10)
    noise_power = signal_power / esno_real

    noise = np.random.normal(0, np.sqrt(noise_power/2), size=(len(data), 2)) @ np.array([1, 1j])

    return data + noise


def channel(data, BER):
    bits = np.unpackbits(np.array(data, dtype=np.uint8))

    error_mask = np.random.binomial(1, BER, len(bits))
    error_count = error_mask.sum()

    errored_ints = np.packbits(bits ^ error_mask)
    return (errored_ints, error_count)
