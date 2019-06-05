# -*- coding: utf-8 -*-

import numpy as np


def rrcosfilter(N, alpha, Ts, Fs):
    """
    Based on Wikipedia (https://en.wikipedia.org/wiki/Root-raised-cosine_filter)
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are (0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("Alpha must between (0 and 1]")

    T_delta = 1 / float(Fs)
    time_idx = np.linspace(-N//2, N//2, N, endpoint=False) * T_delta
    h_rrc = np.zeros(N, dtype=float)

    for i, t in enumerate(time_idx):
        if t == 0.0:
            h_rrc[i] = (1 / Ts) * (1.0 + alpha * (4 / np.pi - 1))

        elif np.abs(t) == Ts / (4 * alpha):
            h_rrc[i] = ((alpha / (Ts * np.sqrt(2))) *
                (((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))) +
                 ((1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))))

        else:
            pi_t_ts = np.pi * (t / Ts)
            h_rrc[i] = (1 / Ts) * \
                (((np.sin(pi_t_ts * (1 - alpha)) + 4 * alpha * (t / Ts) * np.cos(pi_t_ts * (1 + alpha)))) /
                (pi_t_ts * (1 - (4 * alpha * t / Ts)**2)))

    return h_rrc


def rrcosfilter_purdue(N, beta, Ts, Fs):
    T_delta = 1 / float(Fs)
    t = np.linspace(-N//2, N//2, N, endpoint=False) * T_delta

    t_norm = t / Ts
    amplitude = (2 * beta) / (np.pi * np.sqrt(Ts))
    denom = 1 - (4 * beta * t_norm)**2
    nom = np.cos((1 + beta) * np.pi * t_norm) + ((np.sin((1 - beta) * np.pi * t_norm)) / ((4 * beta * t_norm)**2))

    h = amplitude * nom / denom
    return h


def rcosfilter(N, alpha, T, Fs):
    """
    Based on SC-FDMA, OFDM in LTE (Thesis Humbert)
    Generates a raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    T  : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    """

    Ts = 1 / Fs
    h_rrc = np.zeros(N)

    for i, k in enumerate(np.linspace(-N//2, N//2, N, endpoint=False)):
        absk = np.abs(k)

        if 0 <= absk <= ((1 - alpha) * T/(2*Ts)):
            val = 1
        elif absk >= ((1 + alpha) * T/(2*Ts)):
            val = 0

        else:
            val = (1 / 2) * (1 - np.sin((np.pi / alpha) * ((Ts / T) * absk - 0.5)))

        h_rrc[i] = val

    return h_rrc



if __name__ == '__main__':
    N = 8000
    alpha = 0.35
    FS = 2.4e9
    carrier_spacing = ((FS / 4) / 1024)
    Ts = 1 / carrier_spacing

    filt = rrcosfilter(N, alpha, Ts, FS)
    filt2 = rcosfilter(N, alpha, Ts, FS)

