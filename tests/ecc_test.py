# -*- coding: utf-8 -*-

from datetime import datetime
from time import perf_counter as pc

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

from misc.modulation import (packed_to_unpacked, unpacked_to_packed, qpsk_demodulate,
                             qpsk_modulate, qpsk_demodulate_soft, qpsk_hard_decision)
from misc.ecc.ecc_utils import channel_AWGN
from misc.ecc.convolutional import conv_encode, conv_decode
from polar.polar import polar_encode
from polar.polar_ssc import polar_decode_ssc
from polar.polar_alternative_decoder import polar_decode_alternate


def test_BER(filename='output/BER_compare.npz', enable_conv=False):
    num_runs_max = 20000
    num_frame_errors = 150
    ESNOs = np.arange(-1, 10, 0.25)

    N = 2**13
    K = N // 2

    types = ["Uncoded", "Repetition code (Hard)", "Repetition code (Soft)", "Polar code"]
    if enable_conv:
        conv_params = (3, 7, 5)
        types += "Convolutional code"

    N_types = len(types)

    results_BER = np.empty((N_types, len(ESNOs)))
    results_BLER = np.empty((N_types, len(ESNOs)))

    if enable_conv:
        conv_params = (3, 7, 5)

    for i, esno in enumerate(ESNOs):
        print("ESNO", esno)

        # A counter for the total number of frame errors, so we can stop when some of the
        # schemes have reached their limit.
        total_frame_errors = np.zeros(N_types, np.int)
        total_bit_errors = np.zeros(N_types, np.int)
        total_frames = np.zeros(N_types, np.int)
        total_bits = np.zeros(N_types, np.int)

        for _ in range(num_runs_max):
            data = np.frombuffer(np.random.bytes(K // 8), dtype=np.uint8)
            data_unpacked = packed_to_unpacked(data)
            data_bits = np.unpackbits(data)

            ###
            if total_frame_errors[0] < num_frame_errors:
                data_modulated = qpsk_modulate(data_unpacked)
                uncoded = channel_AWGN(data_modulated, esno)
                uncoded_bits = np.unpackbits(unpacked_to_packed(qpsk_demodulate(uncoded)))

                total_frame_errors[0] += (uncoded_bits != data_bits).any()
                total_bit_errors[0] += np.count_nonzero(uncoded_bits != data_bits)
                total_frames[0] += 1
                total_bits[0] += K

            ###
            if (total_frame_errors[1] < num_frame_errors) and (total_frame_errors[2] < num_frame_errors):
                rep_modulated = np.tile(data_modulated, (1, 2)).ravel()

                rep_coded = channel_AWGN(rep_modulated, esno)
                rep_coded_reshaped = rep_coded.reshape((-1, len(uncoded)))

                rep_coded_hard = np.array([1, 2], np.uint8) @ (np.stack((rep_coded_reshaped.real < 0, rep_coded_reshaped.imag < 0)).sum(axis=1) >= 1)
                rep_coded_hard_bits = np.unpackbits(unpacked_to_packed(rep_coded_hard))

                rep_coded_LLRs = np.roll(qpsk_demodulate_soft(rep_coded_reshaped, esno).mean(axis=0), 1, axis=1)
                rep_coded_soft_bits = np.unpackbits(unpacked_to_packed(qpsk_hard_decision(rep_coded_LLRs)))

                total_frame_errors[1] += (rep_coded_hard_bits != data_bits).any()
                total_bit_errors[1] += np.count_nonzero(rep_coded_hard_bits != data_bits)

                total_frame_errors[2] += (rep_coded_soft_bits != data_bits).any()
                total_bit_errors[2] += np.count_nonzero(rep_coded_soft_bits != data_bits)

                total_frames[1:3] += 1
                total_bits[1:3] += K

            ###
            if total_frame_errors[3] < num_frame_errors:
                polar_data = polar_encode(N, K, data_bits)
                polar_data_repacked = packed_to_unpacked(np.packbits(polar_data))
                polar_data_modulated = qpsk_modulate(polar_data_repacked)

                polar_coded = channel_AWGN(polar_data_modulated, esno)

                polar_coded_demod_soft = qpsk_demodulate_soft(polar_coded, esno).flatten()
                polar_result = polar_decode_ssc(N, K, polar_coded_demod_soft)

                total_frame_errors[3] += (polar_result != data_bits).any()
                total_bit_errors[3] += np.count_nonzero(polar_result != data_bits)
                total_frames[3] += 1
                total_bits[3] += K

            ###
            if enable_conv and total_frame_errors[4] < num_frame_errors:
                conv_data = conv_encode(data, *conv_params)
                conv_data_modulated = qpsk_modulate(packed_to_unpacked(conv_data))

                conv_coded = channel_AWGN(conv_data_modulated, esno)

                conv_coded_demod = np.frombuffer(unpacked_to_packed(qpsk_demodulate(conv_coded)), dtype=np.uint8)
                conv_decoded = conv_decode(conv_coded_demod[::2], conv_coded_demod[1::2], *conv_params)

                total_frame_errors[4] += (np.unpackbits(conv_decoded) != data_bits).any()
                total_bit_errors[4] += np.count_nonzero(np.unpackbits(conv_decoded) != data_bits)
                total_frames[4] += 1
                total_bits[4] += K

        results_BER[:, i] = total_bit_errors / total_bits
        results_BLER[:, i] = total_frame_errors / total_frames

    qfunc = lambda x : 0.5 * erfc(x / np.sqrt(2))
    expected_uncoded_BER = qfunc(np.sqrt(10**(ESNOs/10)))

    np.savez(filename,
         N=N, K=K, ESNOs=ESNOs, results_BER=results_BER, results_BLER=results_BLER,
         expected_BER_uncoded=expected_uncoded_BER, legend=types,
         config={
                 'num_runs_max':num_runs_max,
                 'num_frame_errors':num_frame_errors,
                 }
         )


def test_polar_limit(filename='output/polar_limit_rate05_nossc.npz'):
    num_runs_max = 20000
    num_frame_errors = 100
    ESNOs = np.arange(-1, 5, 0.25)

    SEED = 456
    N = 8192
    K = 4096

    np.random.seed(SEED)

    results_BER = np.empty((len(ESNOs)))
    results_BLER = np.empty((len(ESNOs)))
    results_frame_errors = np.empty((len(ESNOs)))
    for i, esno in enumerate(ESNOs):
        print("ESNO", esno)
        total_errors = 0
        total_frame_errors = 0
        total_bits = 0
        total_frames = 0

        for _ in range(num_runs_max):

            ###
            data = np.frombuffer(np.random.bytes(K // 8), dtype=np.uint8)
            data_bits = np.unpackbits(data)

            ###
            polar_data = polar_encode(N, K, data_bits)
            polar_data_repacked = packed_to_unpacked(np.packbits(polar_data))
            polar_data_modulated = qpsk_modulate(polar_data_repacked)

            ###
            polar_coded = channel_AWGN(polar_data_modulated, esno)

            ###
            polar_coded_demod_soft = qpsk_demodulate_soft(polar_coded, esno).flatten()
            # polar_result = polar_decode_ssc(N, K, polar_coded_demod_soft)
            polar_result = polar_decode_alternate(N, K, polar_coded_demod_soft, use_f_approx=False)

            ###
            errors = (polar_result != data_bits)

            total_errors += errors.sum()
            total_frame_errors += errors.any()
            total_frames += 1
            total_bits += len(data_bits)

            if total_frame_errors >= num_frame_errors:
                break
        else:
            print("Timeout at {} dB".format(esno))

        results_BER[i] = total_errors / total_bits
        results_BLER[i] = total_frame_errors / total_frames
        results_frame_errors[i] = total_frame_errors

    #% Save file
    config = {
            "seed": SEED,
            "num_runs": num_runs_max,
            "ESNOs": ESNOs,
            "Time": datetime.now(),
            'num_runs_max':num_runs_max,
            'num_frame_errors_max':num_frame_errors
            }

    np.savez(filename,
            N=N, K=K, ESNOs=ESNOs,
            results_BER=results_BER, results_BLER=results_BLER, frame_errors=results_frame_errors,
            config=config
            )



def test_polar_rates(filename='output/polar_various_rates.npz'):
    print("Polar compare performance at different rates")
    num_runs_max = 1000
    num_frame_errors = 100
    ESNOs = np.arange(-11, 3, 0.5)

    rates = 1 / np.array([2, 3, 5, 10, 16, 20.48])
    N = 2**13
    K = (N * rates).astype(int)

    results_BER = np.empty((len(rates), len(ESNOs)))
    results_BLER = np.empty((len(rates), len(ESNOs)))

    for i, esno in enumerate(ESNOs):
        print("ESNO", esno)

        for K_i, cur_K in enumerate(K):
            total_errors = 0
            total_frame_errors = 0
            total_bits = 0
            total_frames = 0

            for _ in range(num_runs_max):
                data_bits = np.random.binomial(1, 0.5, cur_K).astype(np.uint8)

                polar_data = polar_encode(N, cur_K, data_bits)
                polar_data_repacked = packed_to_unpacked(np.packbits(polar_data))
                polar_data_modulated = qpsk_modulate(polar_data_repacked)

                polar_coded = channel_AWGN(polar_data_modulated, esno)

                polar_coded_demod_soft = qpsk_demodulate_soft(polar_coded, esno).flatten()
                polar_result = polar_decode_ssc(N, cur_K, polar_coded_demod_soft)

                ######
                errors = (polar_result != data_bits)

                total_errors += errors.sum()
                total_frame_errors += errors.any()
                total_bits += cur_K
                total_frames += 1

                if total_frame_errors >= num_frame_errors:
                    break
            else:
                print("Timeout at {} dB".format(esno))

            results_BER[K_i, i] = total_errors / total_bits
            results_BLER[K_i, i] = total_frame_errors / total_frames

    np.savez(filename,
             N=N, K=K, ESNOs=ESNOs, results_BER=results_BER, results_BLER=results_BLER,
             legend=[f"Rate: {r}" for r in rates],
             config={
                 'num_runs_max':num_runs_max,
                 'num_frame_errors':num_frame_errors,
                 'total_bits':total_bits,
                 }
             )


def test_polar_combining(K=32, repetitions=2, plot=False):
    num_runs = 100
    ESNOs = np.arange(-13, -2, 0.5)

    # First just repeating and combining soft LLRs
    # Then a rate 1/2 code, repeated a number of times
    # Then polar code with the lowest rate possible

    results_uncoded_BER = np.empty((num_runs, len(ESNOs)))
    results_uncoded_BLER = np.empty((num_runs, len(ESNOs)))
    results_rate_half_BER = np.empty((num_runs, len(ESNOs)))
    results_rate_half_BLER = np.empty((num_runs, len(ESNOs)))
    results_rate_min_BER = np.empty((num_runs, len(ESNOs)))
    results_rate_min_BLER = np.empty((num_runs, len(ESNOs)))

    for n_run in range(num_runs):
        if num_runs > 10 and n_run % (num_runs // 10) == 0:
            print(n_run)

        # Generate data
        data = np.frombuffer(np.random.bytes(K // 8), dtype=np.uint8)
        data_bits = np.unpackbits(data)

        # Match N
        uncoded_data = np.tile(data_bits, (2*repetitions))
        polar_rate_half = np.tile(polar_encode(2 * K, K, data_bits), repetitions)
        polar_rate_min = polar_encode(repetitions * 2 * K, K, data_bits)

        # Pack
        uncoded_repacked = packed_to_unpacked(np.packbits(uncoded_data))
        polar_rate_half_repacked = packed_to_unpacked(np.packbits(polar_rate_half))
        polar_rate_min_repacked = packed_to_unpacked(np.packbits(polar_rate_min))

        # Modulate
        uncoded_modulated = qpsk_modulate(uncoded_repacked)
        polar_rate_half_modulated = qpsk_modulate(polar_rate_half_repacked)
        polar_rate_min_modulated = qpsk_modulate(polar_rate_min_repacked)

        for i, esno in enumerate(ESNOs):
            # print("{}: esno {}".format(i, esno))

            uncoded_channel = channel_AWGN(uncoded_modulated, esno)
            uncoded_reshaped = uncoded_channel.reshape((-1, K//2))
            uncoded_channel_LLRs = np.mean(qpsk_demodulate_soft(uncoded_reshaped, esno), axis=0)
            uncoded_bits = np.unpackbits(unpacked_to_packed(qpsk_hard_decision(uncoded_channel_LLRs)))
            results_uncoded_BER[n_run, i] = (uncoded_bits != data_bits).mean()
            results_uncoded_BLER[n_run, i] = (uncoded_bits != data_bits).any()

            ###
            polar_rate_half_channel = channel_AWGN(polar_rate_half_modulated, esno)
            polar_rate_half_reshaped = polar_rate_half_channel.reshape((-1, K))
            polar_rate_half_channel_LLRs = np.mean(qpsk_demodulate_soft(polar_rate_half_reshaped, esno), axis=0)

            polar_result_rate_half = polar_decode_ssc(2 * K, K, polar_rate_half_channel_LLRs.flatten())
            results_rate_half_BER[n_run, i] = (polar_result_rate_half != data_bits).mean()
            results_rate_half_BLER[n_run, i] = (polar_result_rate_half != data_bits).any()

            ###
            polar_rate_min_channel = channel_AWGN(polar_rate_min_modulated, esno)
            polar_rate_min_demod = qpsk_demodulate_soft(polar_rate_min_channel, esno).flatten()
            polar_result_rate_min = polar_decode_ssc(repetitions * 2 * K, K, polar_rate_min_demod)
            results_rate_min_BER[n_run, i] = (polar_result_rate_min != data_bits).mean()
            results_rate_min_BLER[n_run, i] = (polar_result_rate_min != data_bits).any()


    if plot:
        ###
        plt.figure("Polar BER test")
        plt.title(f"Comparison of different rate polar codes, QPSK, AWGN channel, K={K}")
        plt.plot(ESNOs, results_uncoded_BER.mean(axis=0), label=f'Repetition code R = 1 / {1/(2*repetitions)}')
        plt.plot(ESNOs, results_rate_half_BER.mean(axis=0), label=f'Polar code (SC decoder) R = 1 / {1/2}, repeated = {repetitions}')
        plt.plot(ESNOs, results_rate_min_BER.mean(axis=0), label=f'Polar code (SC decoder) R = 1 / {1/(2*repetitions)}')
        plt.ylabel("BER")
        plt.yscale('log')
        plt.legend()


        ###
        plt.figure("Polar BLER test")
        plt.title(f"Comparison of different rate polar codes, QPSK, AWGN channel, K={K}")
        plt.plot(ESNOs, results_uncoded_BLER.mean(axis=0), label=f'Repetition code R = 1 / {1/(2*repetitions)}')
        plt.plot(ESNOs, results_rate_half_BLER.mean(axis=0), label=f'Polar code (SC decoder) R = 1 / {1/2}, repeated = {repetitions}')
        plt.plot(ESNOs, results_rate_min_BLER.mean(axis=0), label=f'Polar code (SC decoder) R = 1 / {1/(2*repetitions)}')
        plt.xlabel("ESNO")
        plt.ylabel("BLER")
        plt.yscale('log')
        plt.legend()

        ###
        plt.show()


def test_polar_rate_vs_rep(filename='output/polar_rates_vs_rep.npz'):
    num_runs_max = 50000
    num_frame_errors = 100
    ESNOs = np.arange(-12, -6, 0.25)

    total_bits = 14400
    K = 400
    Ns = 2**(np.arange(9, 14))
    repeats = (total_bits / Ns).astype(int)

    results_BER = np.empty((len(repeats), len(ESNOs)))
    results_BLER = np.empty((len(repeats), len(ESNOs)))

    for i, esno in enumerate(ESNOs):
        print("ESNO", esno)

        for n_rep in range(len(repeats)):
            total_errors = 0
            total_frame_errors = 0
            total_bits = 0
            total_frames = 0

            for _ in range(num_runs_max):
                ###
                data = np.frombuffer(np.random.bytes(K // 8), dtype=np.uint8)
                data_bits = np.unpackbits(data)

                ###
                polar_data = np.tile(polar_encode(Ns[n_rep], K, data_bits), repeats[n_rep])
                polar_data_repacked = packed_to_unpacked(np.packbits(polar_data))
                polar_data_modulated = qpsk_modulate(polar_data_repacked)

                ###
                polar_coded = channel_AWGN(polar_data_modulated, esno)

                ###
                polar_reshaped = polar_coded.reshape((repeats[n_rep], -1))
                polar_channel_LLRs = np.mean(qpsk_demodulate_soft(polar_reshaped, esno), axis=0)

                polar_result = polar_decode_ssc(Ns[n_rep], K, polar_channel_LLRs.flatten())

                ###
                errors = (polar_result != data_bits)

                total_errors += errors.sum()
                total_frame_errors += errors.any()
                total_bits += K
                total_frames += 1

                if total_frame_errors >= num_frame_errors:
                    break
            else:
                print("Timeout at {} dB".format(esno))

            results_BER[n_rep, i] = total_errors / total_bits
            results_BLER[n_rep, i] = total_frame_errors / total_frames

    np.savez(filename,
             N=Ns, K=K, repeats=repeats, ESNOs=ESNOs, results_BER=results_BER, results_BLER=results_BLER,
             legend=[f"N: {Ns[i]}, Rep: {repeats[i]}" for i in range(len(Ns))],
             config={
                     'num_runs_max':num_runs_max,
                     'num_frame_errors':num_frame_errors,
                     'total_bits':total_bits,
                     }
             )


def test_polar_soft_combine(filename='output/polar_soft_combine.npz'):
    print("Polar compare post- vs pre-combining.")
    num_runs_max = 50000
    num_frame_errors = 100
    ESNOs = np.arange(-12, -9, 0.25)

    total_bits = 14400
    K = 400
    Ns = np.array([2048, 2048, 4096, 4096])
    repeats = (total_bits / Ns).astype(int)

    results_BER = np.empty((len(repeats), len(ESNOs)))
    results_BLER = np.empty((len(repeats), len(ESNOs)))

    for i, esno in enumerate(ESNOs):
        print("ESNO", esno)

        for n_rep in range(len(repeats)):
            total_errors = 0
            total_frame_errors = 0
            total_bits = 0
            total_frames = 0

            for _ in range(num_runs_max):
                ###
                data = np.frombuffer(np.random.bytes(K // 8), dtype=np.uint8)
                data_bits = np.unpackbits(data)

                ###
                polar_data = np.tile(polar_encode(Ns[n_rep], K, data_bits), repeats[n_rep])
                polar_data_repacked = packed_to_unpacked(np.packbits(polar_data))
                polar_data_modulated = qpsk_modulate(polar_data_repacked)

                ###
                polar_coded = channel_AWGN(polar_data_modulated, esno)

                ######
                polar_reshaped = polar_coded.reshape((repeats[n_rep], -1))
                polar_channel_LLRs = qpsk_demodulate_soft(polar_reshaped, esno)

                #####
                if n_rep % 1 == 0:
                    polar_result = polar_decode_ssc(Ns[n_rep], K, polar_channel_LLRs.mean(axis=0).flatten())

                ######
                else:
                    polar_results_soft_all = np.empty((repeats[n_rep], K))
                    for j in range(len(polar_channel_LLRs)):
                        polar_results_soft_all[j] = polar_decode_ssc(Ns[n_rep], K, polar_channel_LLRs[j], soft_output=True)

                    polar_result = polar_results_soft_all.mean(axis=0)

                ######
                errors = (polar_result != data_bits)

                total_errors += errors.sum()
                total_frame_errors += errors.any()
                total_bits += K
                total_frames += 1

                if total_frame_errors >= num_frame_errors:
                    break
            else:
                print("Timeout at {} dB".format(esno))

            results_BER[n_rep, i] = total_errors / total_bits
            results_BLER[n_rep, i] = total_frame_errors / total_frames

    np.savez(filename,
             N=Ns, K=K, repeats=repeats, ESNOs=ESNOs, results_BER=results_BER, results_BLER=results_BLER,
             legend=["Hard combine", "Soft combine", "Hard combine", "Soft combine"],
             config={
                 'num_runs_max': num_runs_max,
                 'num_frame_errors': num_frame_errors,
                 'total_bits': total_bits
             }
             )


def test_polar_compare_impl_runtime():
    K = 400
    N = 2048
    num_runs = 10000

    ###
    data = np.frombuffer(np.random.bytes(K // 8), dtype=np.uint8)
    data_bits = np.unpackbits(data)

    ###
    polar_data = polar_encode(N, K, data_bits)
    polar_data_repacked = packed_to_unpacked(np.packbits(polar_data))
    polar_data_modulated = qpsk_modulate(polar_data_repacked)
    polar_channel_LLRs = qpsk_demodulate_soft(polar_data_modulated, 2).flatten()


    ###
    st = pc()
    for _ in range(num_runs):
        polar_decode_alternate(N, K, polar_channel_LLRs, use_f_approx=False)
    time_none = (pc() - st) / num_runs

    st = pc()
    for _ in range(num_runs):
        polar_decode_alternate(N, K, polar_channel_LLRs, use_f_approx=True)
    time_f = (pc() - st) / num_runs

    st = pc()
    for _ in range(num_runs):
        polar_decode_ssc(N, K, polar_channel_LLRs)
    time_ssc = (pc() - st) / num_runs


    print(f"Results from timing test, N={N}, K={K}, num_runs={num_runs}")

    print("None                     &  {:.2f} \\\\".format(1000 * time_none))
    print("$F$ approximation        &  {:.2f} \\\\".format(1000 * time_f))
    print("SSC + $F$ approximation  &  {:.2f}".format(1000 * time_ssc))



def test_polar_compare_impl(filename='output/polar_compare_impl.npz'):
    print("Polar with and without approximation.")
    num_runs_max = 20000
    num_frame_errors = 150
    ESNOs = np.arange(-12, -9, 0.25)

    total_bits = 14400
    K = 400
    N = 2048
    reps = total_bits // N

    run_types = ["None", "F approx", "SSC + F_approx"]

    results_BER = np.empty((len(run_types), len(ESNOs)))
    results_BLER = np.empty((len(run_types), len(ESNOs)))

    for i, esno in enumerate(ESNOs):
        print("ESNO", esno)

        for rt_i, _ in enumerate(run_types):
            total_errors = 0
            total_frame_errors = 0
            total_bits = 0
            total_frames = 0

            for _ in range(num_runs_max):
                ###
                data = np.frombuffer(np.random.bytes(K // 8), dtype=np.uint8)
                data_bits = np.unpackbits(data)

                ###
                polar_data = np.tile(polar_encode(N, K, data_bits), reps)
                polar_data_repacked = packed_to_unpacked(np.packbits(polar_data))
                polar_data_modulated = qpsk_modulate(polar_data_repacked)

                ###
                polar_coded = channel_AWGN(polar_data_modulated, esno)

                ######
                polar_reshaped = polar_coded.reshape((reps, -1))
                polar_channel_LLRs = qpsk_demodulate_soft(polar_reshaped, esno)

                #####
                mean_LLRs = polar_channel_LLRs.mean(axis=0).flatten()

                if rt_i == 0:
                    polar_result = polar_decode_alternate(N, K, mean_LLRs, use_f_approx=False)
                elif rt_i == 1:
                    polar_result = polar_decode_alternate(N, K, mean_LLRs, use_f_approx=True)
                elif rt_i == 2:
                    polar_result = polar_decode_ssc(N, K, mean_LLRs)

                ######
                errors = (polar_result != data_bits)

                total_errors += errors.sum()
                total_frame_errors += errors.any()
                total_bits += K
                total_frames += 1

                if total_frame_errors >= num_frame_errors:
                    break
            else:
                print("Timeout at {} dB".format(esno))

            results_BER[rt_i, i] = total_errors / total_bits
            results_BLER[rt_i, i] = total_frame_errors / total_frames


    np.savez(filename,
             N=N, K=K, repeats=reps, ESNOs=ESNOs, results_BER=results_BER, results_BLER=results_BLER,
             legend=run_types,
             config={
                     'num_runs_max':num_runs_max,
                     'num_frame_errors':num_frame_errors,
                     'total_bits':total_bits
                     }
             )


if __name__ == "__main__":
#    test_BER()
#    test_polar_rates()
#    test_polar_rate_vs_rep()
#    test_polar_soft_combine()
    test_polar_compare_impl()
#    test_polar_limit()
#    test_polar_combining(K=2**9, repetitions=16, plot=True)
#    test_polar_compare_impl_runtime()
