"""
LLR based polar SC decoders
"""

import numpy as np

from polar.polar_common import idx, polar_hpw
from polar.polar_p0 import recursivelyUpdateB

##############################
def f_op_exact(LLR0, LLR1):
    return 2 * np.arctanh(np.tanh(LLR0/2) * np.tanh(LLR1/2))

def f_op_llr(LLR0, LLR1):
    val = min(abs(LLR0), abs(LLR1))

    if (LLR0 > 0) ^ (LLR1 > 0) == 0:  # The signs are the same
        return val
    else:  # The signs are different
        return -val

#    return np.sign(LLR0) * np.sign(LLR1) * val

def g_op_llr(LLR0, LLR1, u):
    if u:
        return LLR1 - LLR0
    else:
        return LLR1 + LLR0

def recursivlyCalcP_llr(lamb, phi, P, B, n, use_f_exact=False):
    if lamb == 0:
        return

#    print(f"RCP ({lamb}, {phi})")

    psi = phi // 2
    if phi % 2 == 0:
        recursivlyCalcP_llr(lamb - 1, psi, P, B, n)

    for beta in range(2**(n - lamb)):
        idx0 = 2*beta
        idx1 = 2*beta + 1

        if phi % 2 == 0:
            if use_f_exact:
                P[lamb, beta] = f_op_exact(P[lamb - 1, idx0], P[lamb - 1, idx1])
            else:
                P[lamb, beta] = f_op_llr(P[lamb - 1, idx0], P[lamb - 1, idx1])

        else:
            u_dot = B[lamb, idx(phi-1, beta, lamb)]
            P[lamb, beta] = g_op_llr(P[lamb - 1, idx0], P[lamb - 1, idx1], u_dot)


def polar_decode_llr(N, K, LLR0, use_f_exact=False):
    """
    Decode a (N, K) polar code.
    P0 must be log likelihoods
    """
    n = np.log2(N).astype(int)
    A = polar_hpw(N)[-K:]

    # We're not using all the elements in the P array, as each layer lamb
    # only uses 2**(n-lamb) elements. Given the current indexing it's still fine.
    P = np.empty((n + 1, N), dtype=float)
    B = np.zeros((n + 1, N), dtype=np.uint8)

    # Init
    P[0] = LLR0.copy()

    for phi in range(N):
        recursivlyCalcP_llr(n, phi, P, B, n)
#        print(f"Setting bit {phi}")

        if not phi in A:
            B[n, phi] = 0
        else:
            if P[n, 0] > 0:
                B[n, phi] = 0
            else:
                B[n, phi] = 1

        if phi % 2 == 1:
            recursivelyUpdateB(n, phi, B, n)


    u_hat_full = B[n]
    return u_hat_full[A]

##############################
def recursivelyUpdateB_debug(lamb, phi, B, n, B_idx):
    psi = phi // 2

#    print(f"RCB ({lamb}, {phi})")
    for beta in range(2**(n - lamb)):
        B[lamb - 1, idx(psi, 2 * beta,     lamb - 1)] = (B[lamb, idx(phi - 1, beta, lamb)] ^
                                                         B[lamb, idx(phi    , beta, lamb)])
        B[lamb - 1, idx(psi, 2 * beta + 1, lamb - 1)] =  B[lamb, idx(phi,     beta, lamb)]

        print(f"B[{lamb-1},{idx(psi, 2 * beta,     lamb - 1)}] = B[{lamb}, {idx(phi - 1, beta, lamb)}] ^ B[{lamb}, {idx(phi, beta, lamb)}]")
        print(f"B[{lamb-1},{idx(psi, 2 * beta + 1, lamb - 1)}] = B[{lamb}, {idx(phi, beta, lamb)}]")

        B_idx[lamb - 1, idx(psi, 2 * beta,     lamb - 1)] = (B_idx[lamb, idx(phi - 1, beta, lamb)] +
                                                             B_idx[lamb, idx(phi    , beta, lamb)])
        B_idx[lamb - 1, idx(psi, 2 * beta + 1, lamb - 1)] =  B_idx[lamb, idx(phi,     beta, lamb)]


    if psi % 2 == 1:
        recursivelyUpdateB_debug(lamb - 1, psi, B, n, B_idx)

def recursivlyCalcP_debug(lamb, phi, P, B, n):
    if lamb == 0:
        return

#    print(f"RCP ({lamb}, {phi})")

    psi = phi // 2
    if phi % 2 == 0:
        recursivlyCalcP_debug(lamb - 1, psi, P, B, n)

    for beta in range(2**(n - lamb)):
        idx0 = 2*beta
        idx1 = 2*beta + 1

        if phi % 2 == 0:
            P[lamb, beta] = f_op_llr(P[lamb - 1, idx0], P[lamb - 1, idx1])
            print(f"P[{lamb},{beta}] <= f(P[{lamb-1},{idx0}], P[{lamb-1},{idx1}])")

        else:
            u_dot = B[lamb, idx(phi-1, beta, lamb)]
            P[lamb, beta] = g_op_llr(P[lamb - 1, idx0], P[lamb - 1, idx1], u_dot)
            print(f"P[{lamb},{beta}] <= g(P[{lamb-1},{idx0}], P[{lamb-1},{idx1}], B[{lamb}, {idx(phi-1, beta, lamb)}])")



def polar_decode_debug(N, K, LLR0):
    """
    Decode a (N, K) polar code.
    P0 must be log lieklihoods
    """
    n = np.log2(N).astype(int)
    A = polar_hpw(N)[-K:]

    # We're not using all the elements in the P array, as each layer lamb
    # only uses 2**(n-lamb) elements. Given the current indexing it's still fine.
    P = np.empty((n + 1, N), dtype=float)
    B = np.zeros((n + 1, N), dtype=np.uint8)
    B_idx = np.zeros((n + 1, N), dtype=tuple)
    B_idx[n] = [(i, ) for i in range(N)]

    # Init
    P[0] = LLR0.copy()

    for phi in range(N):
        recursivlyCalcP_debug(n, phi, P, B, n)
        print(f"Setting bit {phi}")

        if not phi in A:
            B[n, phi] = 0
        else:
            if P[n, 0] > 0:
                B[n, phi] = 0
            else:
                B[n, phi] = 1

        if phi % 2 == 1:
            recursivelyUpdateB_debug(n, phi, B, n, B_idx)


    u_hat_full = B[n]
    return (u_hat_full[A], P, B, B_idx)