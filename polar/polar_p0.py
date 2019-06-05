"""
Probability normalized (P1) based polar decoder
"""

import numpy as np

from polar.polar_common import idx, polar_hpw

##############################
def f_op_p1(P0, P1):
    res0 = 0.5 * (P0[0] * P1[0] + P0[1] * P1[1])
    res1 = 0.5 * (P0[1] * P1[0] + P0[0] * P1[1])

    return (res0, res1)

def g_op_p1(P0, P1, u):
    res0 = 0.5 * (P0[u]   * P1[0])
    res1 = 0.5 * (P0[1^u] * P1[1])

    return (res0, res1)

def recursivlyCalcP(lamb, phi, P, B, n):
    if lamb == 0:
        return

#    print(f"RCP ({n - lamb}, {phi})")
    psi = phi // 2
    if phi % 2 == 0:
        recursivlyCalcP(lamb - 1, psi, P, B, n)

    for beta in range(2**(n - lamb)):
        idx_main = beta
        idx0 = 2*beta
        idx1 = 2*beta + 1

        if phi % 2 == 0:
            P[lamb, idx_main] = f_op_p1(P[lamb - 1, idx0], P[lamb - 1, idx1])
            P[lamb, idx_main] /= (P[lamb, idx_main, 0] + P[lamb, idx_main, 1])

        else:
            u_dot = B[lamb, idx(phi - 1, beta, lamb)]

            P[lamb, idx_main] = g_op_p1(P[lamb - 1, idx0], P[lamb - 1, idx1], u_dot)
            P[lamb, idx_main] /= (P[lamb, idx_main, 0] + P[lamb, idx_main, 1])


def recursivelyUpdateB(lamb, phi, B, n):
    psi = phi // 2

    for beta in range(2**(n - lamb)):
        B[lamb - 1, idx(psi, 2 * beta,     lamb - 1)] = (B[lamb, idx(phi - 1, beta, lamb)] ^
                                                         B[lamb, idx(phi    , beta, lamb)])
        B[lamb - 1, idx(psi, 2 * beta + 1, lamb - 1)] =  B[lamb, idx(phi,     beta, lamb)]

    if psi % 2 == 1:
        recursivelyUpdateB(lamb - 1, psi, B, n)


def polar_decode(N, K, P0):
    """
    Decode a (N, K) polar code.
    P0 must be 1-normalized probabilities
    """
    n = np.log2(N).astype(int)
    A = polar_hpw(N)[-K:]

    # We're not using all the elements in the P array, as each layer lamb
    # only uses 2**(n-lamb) elements. Given the current indexing it's still fine.
    P = np.empty((n + 1, N, 2), dtype=float)
    B = np.zeros((n + 1, N), dtype=np.uint8)

    for beta in range(N):
        P[0, beta, 0] = P0[beta]
        P[0, beta, 1] = 1 - P0[beta]

    for phi in range(N):
        recursivlyCalcP(n, phi, P, B, n)

        if not phi in A:
            B[n, phi] = 0
        else:
            if P[n, 0, 0] > P[n, 0, 1]:
                B[n, phi] = 0
            else:
                B[n, phi] = 1

        if phi % 2 == 1:
            recursivelyUpdateB(n, phi, B, n)

    u_hat_full = B[n]
    return u_hat_full[A]
