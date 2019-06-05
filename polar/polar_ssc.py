import numpy as np

from polar.polar_common import polar_find_ordering, polar_transform_pipelined, polar_hpw
from polar.fast_ssc_compiler import polar_calc_R0_R1_set


##############################
def f_op_ssc(LLR0, LLR1):
    """
    Perform the F operation in a vector manner.
    The original implementation uses tanh and arctanh
    val = 2*np.arctanh(np.tanh(LLR0/2) * np.tanh(LLR1/2))
    tanh can be simplified to this operation
    val = np.log((np.exp(LLR0 + LLR1) + 1)/(np.exp(LLR0) + np.exp(LLR1)))
    At the cost of some EC performance, the operation can be approximated as
    val = sign(LLR0) * sign(LLR1) * min(abs(LLR0), abs(LLR1))
    """

#    val = np.minimum(np.abs(LLR0), np.abs(LLR1))
#
#    mask = (LLR0 > 0) ^ (LLR1 > 0) != 0
#    val[mask] = -val[mask]
#
#    return val
    return np.sign(LLR0) * np.sign(LLR1) * np.minimum(np.abs(LLR0), np.abs(LLR1))


def g_op_ssc(LLR0, LLR1, u):
    """
    Perform the G operation. For numpy it is faster to create a memory mask and use indexing
    instead of taking powers of x:
    avr = x**(1 - 2*bvl) + y
    """

    mask = (u == 1)
    LLR0_signed = LLR0
    LLR0_signed[mask] = -LLR0_signed[mask]

    return LLR1 + LLR0_signed


def polar_transform_pipelined_LLR(LLR, reorder=True):
    """
    If we need soft outputs from R1 nodes we have to perform the F operation in the same
    manner as the normal transform.
    """
    N = len(LLR)
    N_half = N//2
    n = np.log2(N).astype(int)

    working_reg = LLR.copy()
    for n_i in range(n):
        u2 = working_reg[1::2].copy()

        working_reg[:N_half] = f_op_ssc(working_reg[::2], u2)
        working_reg[N_half:] = u2

    if reorder:
        order = polar_find_ordering(N)
        working_reg = working_reg[order]

    return working_reg


def polar_ssc_recurse(av, beta, all_bits, A_set, R0R1_tuple, soft_output=False):
    """
    Perform the recursion central to the SSC decoding of the polar codes.

    Av are the log likelihoods from the parent (caller).
    beta is a tuple of the current position of the node (depth, index).
    all_bits is list for the final bits.
    A_set is a set of the non-frozen bit locations.
    R0_set is a set of the positions of R0 nodes.
    """
    if len(av) == 1:
        bit_i = len(all_bits)
        if bit_i in A_set:
            cur_bit = np.array(av < 0, np.int8)
        else:
            cur_bit = np.array([0], np.int8)

        if soft_output:
            all_bits.extend(av)
        else:
            all_bits.extend(cur_bit)

        return cur_bit

    av_len_half = len(av) // 2
    x = av[:av_len_half]
    y = av[av_len_half:]

    ##
    # Left child
    child_left = (beta[0] - 1, 2 * beta[1])

    # R0 node
    if child_left in R0R1_tuple[0]:
        # The R0 set is constructed up front and is ensured to only contain frozen bits. Therefore
        # we don't have to check for them here.
        bvl = np.zeros(av_len_half, np.int8)
        if soft_output:
            all_bits.extend(np.full(av_len_half, 1000))  # 1000 represents highly probable 0
        else:
            all_bits.extend(bvl)

    # R1 node
    elif child_left in R0R1_tuple[1]:
        assert False, "Left R1 node"

    # REP node
    elif child_left in R0R1_tuple[2]:
        avl = f_op_ssc(x,y)
        bit_estimate = (avl.sum() < 0)
        bvl = np.full(4, bit_estimate, dtype=np.uint8)

        if soft_output:
            all_bits.extend(4 * [avl.sum()])
        else:
            all_bits.extend(bvl)

    # SPC node
    elif child_left in R0R1_tuple[3]:
        avl = f_op_ssc(x,y)
        bvl = np.array(avl < 0, np.int8)

        if np.mod(bvl.sum(), 2) != 0:
            least_likely_idx = np.argmin(np.abs(avl))
            bvl[least_likely_idx] = not bvl[least_likely_idx]

        if soft_output:
            leaf_bits = polar_transform_pipelined_LLR(avl, reorder=False)
        else:
            leaf_bits = polar_transform_pipelined(bvl, reorder=False)

        all_bits.extend(leaf_bits)


    else:
        avl = f_op_ssc(x,y)
        bvl = polar_ssc_recurse(avl, child_left, all_bits, A_set, R0R1_tuple, soft_output)

    ##
    # Right child
    child_right = (child_left[0], child_left[1] + 1)
    avr = g_op_ssc(x, y, bvl)

    # R0 node
    if child_right in R0R1_tuple[0]:
        assert False, "Right R0 node"

    # R1 node
    elif child_right in R0R1_tuple[1]:
        bvr = np.array(avr < 0, np.int8)

        if soft_output:
            leaf_bits = polar_transform_pipelined_LLR(avr, reorder=False)
        else:
            leaf_bits = polar_transform_pipelined(bvr, reorder=False)

        all_bits.extend(leaf_bits)

    # REP node
    elif child_right in R0R1_tuple[2]:
        bit_estimate = (avr.sum() < 0)
        bvr = np.full(4, bit_estimate, dtype=np.uint8)

        if soft_output:
            all_bits.extend(4 * [avr.sum()])
        else:
            all_bits.extend(bvr)

    # SPC node
    elif child_right in R0R1_tuple[3]:
        bvr = np.array(avr < 0, np.int8)

        if np.mod(bvr.sum(), 2) != 0:
            least_likely_idx = np.argmin(np.abs(avr))
            bvr[least_likely_idx] = not bvr[least_likely_idx]

        if soft_output:
            leaf_bits = polar_transform_pipelined_LLR(avr, reorder=False)
        else:
            leaf_bits = polar_transform_pipelined(bvr, reorder=False)
        all_bits.extend(leaf_bits)

    else:
        bvr = polar_ssc_recurse(avr, child_right, all_bits, A_set, R0R1_tuple, soft_output)

    bv0 = bvl ^ bvr
    bv1 = bvr
    result = np.concatenate((bv0, bv1))

    return result


def polar_decode_ssc(N, K, LLR, soft_output=False):
    n = np.log2(N).astype(int)

    # Sort the LLRs to match what this algorithm expects.
    # This sorting can be skipped if we simply don't reorder in the encoding phase
    LLR0 = LLR[polar_find_ordering(N)]

    # Sets are significantly faster when checking for contents, so converting is a major speedup
    A = polar_hpw(N)[-K:]
    A_set = set(A)

    R0R1_set = polar_calc_R0_R1_set(N, K)

    all_bits = []
    # This operation modifies the all_bits list with the correct bits
    polar_ssc_recurse(LLR0, (n, 0), all_bits, A_set, R0R1_set, soft_output)

    if soft_output:
        all_bits_np = np.array(all_bits)
    else:
        all_bits_np = np.array(all_bits, np.uint8)

    return all_bits_np[A]
