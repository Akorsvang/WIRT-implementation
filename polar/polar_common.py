"""
Shared functions used for polar encoding and decoding
"""

from functools import lru_cache

import numpy as np

def idx(phi, beta, lamb):
    return phi + (beta << lamb)

def polar_calculate_weights(N):
    """
    Calculate polar weights
    Right now only supports N <= 2**8, for more look at
    np.unpackbits(np.array([3]).byteswap().view(np.uint8))
    """
    if np.log2(N) > 8:
        raise ValueError("Ordering calculation does not support above 2**8")

    beta = 2**(1 / 4)

    I = np.arange(N, dtype=np.uint8)
    beta_power = (beta**np.arange(7, -1, -1))

    W = np.empty(I.shape)
    for i in I:
        W[i] = (np.unpackbits(i) * beta_power).sum()

    W_index = np.argsort(W)
    return W_index

@lru_cache()
def polar_hpw(N):
    """
    Calculate polar weights using the higher order method
    """

    beta = 2**(1 / 4)

    I = np.arange(N, dtype='>u4') # Creating this as a big-endian array, so we don't have to byteswap
    beta_power = (beta**np.arange(31, -1, -1))
    beta_power_quad = (beta**((1 / 4) * np.arange(31, -1, -1)))

    elem_bits = np.unpackbits(I.view(np.uint8)).reshape(-1, 32)
    W = (elem_bits * (beta_power + (1/4) * beta_power_quad)).sum(axis=1)

    W_index = np.argsort(W)
    return W_index

@lru_cache()
def polar_find_ordering(N):
    o = np.arange(N, dtype='>u4')
    n_bits = np.log2(N).astype(int)

    # We use some view tricks here to find the bits that correspond to the entries in o
    elem_bits = np.unpackbits(o.view(np.uint8)).reshape(-1, 32)

    # Flip the bit order, roll the bits down to the correct order and revert the view from before
    return np.packbits(np.roll(np.fliplr(elem_bits), 32 - n_bits)).view('>u4')

@lru_cache()
def polar_find_G(N, reorder=True):
    n = np.log2(N).astype(int)

    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = F.copy()
    for _ in range(1, n):
        G = np.kron(G, F)

    if reorder:
        G_shuffled = G[polar_find_ordering(N)]
    else:
        G_shuffled = G

    return G_shuffled


def polar_transform_pipelined(u, reorder=True):
    N = len(u)
    N_half = N//2
    n = np.log2(N).astype(int)

    working_bits = u.copy()
    for n_i in range(n):
        u2 = working_bits[1::2].copy()

        working_bits[:N_half] = working_bits[::2] ^ u2
        working_bits[N_half:] = u2

    if reorder:
        order = polar_find_ordering(N)
        working_bits = working_bits[order]

    return working_bits


def polar_transform(u):
    """
    This should be a way to use encode using the bit reversal structure -> more efficient in HW.

    Based on http://pfister.ee.duke.edu/courses/ecen655/polar.pdf
    """
    if len(u) == 1:
        x = u
    else:
        u1u2 = np.mod(u[::2] + u[1::2], 2)
        u2 = u[1::2]

        x = np.concatenate((polar_transform(u1u2), polar_transform(u2)))
    return x