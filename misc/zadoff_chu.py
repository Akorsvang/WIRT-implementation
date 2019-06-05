# -*- coding: utf-8 -*-
import numpy as np

def generate_zc_sequence(N, u):
    # Add a Zadoff-Chu sequence for
    n = np.arange(N)
    zadoff_chu_seq = np.exp(-1j * (np.pi * u * n * (n + (N % 2)))/N)

    return zadoff_chu_seq


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 353
    u = 73
    seq = generate_zc_sequence(N, u)
    fig, axes = plt.subplots(2, 1)
    axes[0].set_title('Zadoff Chu sequence, N = {}, u = {}'.format(N, u))
    axes[0].plot(seq.real)
    axes[0].set_ylabel('Real part')
    axes[1].plot(seq.imag)
    axes[1].set_ylabel('Imag part')

    plt.figure()
    autocorrelation = np.correlate(seq, seq, mode='same')
    plt.plot(autocorrelation)
    plt.title('Autocorrelation of Zadoff Chu sequence, N = {}, u = {}'.format(N, u))
