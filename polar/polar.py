# -*- coding: utf-8 -*-

from time import perf_counter as pf

import numpy as np

from polar.polar_common import polar_find_G, polar_hpw, polar_transform_pipelined
from polar.polar_ssc import polar_decode_ssc
from polar.polar_alternative_decoder import polar_decode_alternate

def polar_encode_slow(N, K, u):
    G = polar_find_G(N)
    A = polar_hpw(N)[-K:]

    x = np.mod(u @ G[A], 2)
    return x

def polar_encode(N, K, u, reorder=True):
    A = polar_hpw(N)[-K:]
    u_full = np.zeros(N, dtype=np.uint8)
    u_full[A] = u

    return polar_transform_pipelined(u_full, reorder)


##############################
if __name__ == '__main__':
    num_runs = 100
    N = 2048
    K = 400
    print("Rate: R = 1/{}".format(N/K))

    st = pf()
    success_demod = np.zeros(num_runs, dtype=bool)
    impl_identical = np.zeros(num_runs, dtype=bool)
    for i in range(num_runs):
        # Generate bits
        u = np.random.binomial(1, 0.5, K)

        # Encode
        x = polar_encode(N, K, u)

#        expected_result = polar_transform(x)[A]
#        if not (expected_result == u).all():
#            print("Decoding is going to be hard..")

        # Channel
        L0 = np.zeros(x.shape)  # Probability that received bit is zero
        L0[x == 0] = 0.9
        L0[x == 1] = 0.1

        LR0 = L0 / (1 - L0)    # Likeliky ratio = (prob. bit is zero) / (prob. bit is one)
        LLR0 = np.log(LR0)

        # Demodulation
#        polar_result_p0 = polar_decode(N, K, L0)
#        polar_result_llr = polar_decode_llr(N, K, LLR0, use_f_exact=False)
#        (polar_result_ssc, P, B, B_idx) = polar_decode_debug(N, K, LLR0)
        polar_result_alt = polar_decode_alternate(N, K, LLR0, use_f_approx=False)
        polar_result_alt_approx = polar_decode_alternate(N, K, LLR0, use_f_approx=True)
        polar_result_ssc = polar_decode_ssc(N, K, LLR0)
        # polar_result_ssc_soft = polar_decode_ssc(N, K, LLR0, soft_output=True)

        # impl_identical[i] = (polar_result_ssc == (polar_result_ssc_soft < 0).astype(np.uint8)).all()
        impl_identical[i] = ((polar_result_ssc == polar_result_alt).all() and
                             (polar_result_ssc == polar_result_alt_approx).all())

        success_demod[i] = np.all(polar_result_ssc == u)

        if num_runs > 10 and i % (num_runs // 10) == 0:
            print(i, end=', ')

    print("Total time: {} s".format(pf() - st))
