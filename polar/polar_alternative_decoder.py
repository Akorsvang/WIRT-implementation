# -*- coding: utf-8 -*-
"""
This decoder is based on the message passing principle instead of the right to left decoder from
Arikans work. It forms the basis of the SSC decoder.
"""
import numpy as np

from polar.polar_common import polar_find_ordering, polar_hpw

##############################
def f_op_exact(LLR0, LLR1):
    """
    Alternative representations, but equivalent.
    """
    # return 2 * np.arctanh(np.tanh(LLR0/2) * np.tanh(LLR1/2))
    return np.log1p(np.exp(LLR0 + LLR1)) - np.logaddexp(LLR0, LLR1)

def f_op_llr_alt(LLR0, LLR1):
    """
    Perform the F operation in a vector manner.
    The original implementation uses tanh and arctanh
    val = 2*np.arctanh(np.tanh(LLR0/2) * np.tanh(LLR1/2)).
    tanh can be simplified to this operation
    val = np.log((np.exp(LLR0 + LLR1) + 1)/(np.exp(LLR0) + np.exp(LLR1))).
    See f_op_exact for implementation.
    At the cost of some EC performance, the operation can be approximated as
    val = sign(LLR0) * sign(LLR1) * min(abs(LLR0), abs(LLR1)).
    """

    return np.sign(LLR0) * np.sign(LLR1) * np.minimum(np.abs(LLR0), np.abs(LLR1))


def g_op_llr_alt(LLR0, LLR1, u):
    """
    Perform the G operation. For numpy it is faster to create a memory mask and use indexing
    instead of taking powers of x:
    avr = x**(1 - 2*bvl) + y
    """

    mask = (u == 1)
    LLR0_signed = LLR0
    LLR0_signed[mask] = -LLR0_signed[mask]

    return LLR1 + LLR0_signed



def polar_alt_recurse(av, all_bits, A_set, use_f_approx=False):
    """
    Perform the recursion central to the SSC decoding of the polar codes.

    av are the log likelihoods from the parent (caller).
    all_bits is list for the final bits.
    A_set is a set of the non-frozen bit locations.
    """
    if len(av) == 1:
        # This is a leaf node
        bit_i = len(all_bits)
        if bit_i in A_set:
            cur_bit = np.array(av < 0, np.int8)
        else:
            cur_bit = np.array([0], np.int8)

        all_bits.extend(cur_bit)
        return cur_bit

    av_len_half = len(av) // 2
    x = av[:av_len_half]
    y = av[av_len_half:]

    # Left node
    if use_f_approx:
        avl = f_op_llr_alt(x,y)
    else:
        avl = f_op_exact(x,y)

    bvl = polar_alt_recurse(avl, all_bits, A_set, use_f_approx=use_f_approx)

    # Right node
    avr = g_op_llr_alt(x, y, bvl)
    bvr = polar_alt_recurse(avr, all_bits, A_set, use_f_approx=use_f_approx)

    # Combine and return
    bv0 = bvl ^ bvr
    bv1 = bvr
    result = np.concatenate((bv0, bv1))

    return result


def polar_decode_alternate(N, K, LLR, use_f_approx=False):
    # Sort the LLRs to match what this algorithm expects.
    # This sorting can be skipped if we simply don't reorder in the encoding phase
    LLR0 = LLR[polar_find_ordering(N)]

    # Sets are significantly faster when checking for contents, so converting is a major speedup
    A = polar_hpw(N)[-K:]
    A_set = set(A)

    all_bits = []
    # This operation modifies the all_bits list with the correct bits
    polar_alt_recurse(LLR0, all_bits, A_set, use_f_approx=use_f_approx)

    all_bits_np = np.array(all_bits, np.uint8)

    return all_bits_np[A]
