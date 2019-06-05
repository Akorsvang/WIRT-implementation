# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from ofdm.OFDM import OFDMModulator

from misc.zadoff_chu import generate_zc_sequence
from misc.modulation import qpsk_modulate, repack_bits, qpsk_demodulate_soft
from misc.resample import upsample, upsample_scipy, upsample_poly, downsample
from misc.filters import rrcosfilter
from other.plot_spectrum import plot_spectrum

from polar.polar import polar_encode
from polar.polar_ssc import polar_decode_ssc


PLOT_CONSTELLATION = False
PLOT_SPECTRUM = False


class wirt:
    """
    This class encapsulates and abstracts the WIRT system.
    The constants are described in the report
    """
    # Resampling parameters
    FS = 2.4e9
    SAMP_PER_SYMBOL = 4
    # FILTER_LEN = 200 * SAMP_PER_SYMBOL - 1
    FILTER_LEN = 31
    FILTER_ALPHA = 0.15
    FILTER_DELAY = ((FILTER_LEN // SAMP_PER_SYMBOL) * SAMP_PER_SYMBOL)

    # Synchronization parameters
    ZC_N = 17
    ZC_U = 7
    MAXIMUM_SAMPLE_OFFSET = 20

    # OFDM parameters
    FFT_SIZE = 625
    NUM_SUBCARRIERS = 524
    SUBCARRIER_SPACING = 960e3
    NUM_PILOTS = 124
    NUM_CYCLIC_PREFIX = np.ceil(69e-9 * (FS / SAMP_PER_SYMBOL)).astype(np.int)
    NUM_GUARDBAND = FFT_SIZE - NUM_SUBCARRIERS

    # Various lengths
    DATA_SIZE = 400
    ENCODED_SIZE = 14400
    POLAR_SIZE = 2048
    NUM_REPETITIONS = ENCODED_SIZE // POLAR_SIZE
    USED_SIZE = NUM_REPETITIONS * POLAR_SIZE


    def __init__(self, enable_equalization=True):
        self.mod = OFDMModulator(fft_size=wirt.FFT_SIZE, num_cyclic_prefix=wirt.NUM_CYCLIC_PREFIX,
                                 add_pilot=enable_equalization, num_pilots=wirt.NUM_PILOTS,
                                 num_guardbands=wirt.NUM_GUARDBAND)

        self.upsamp_filter = rrcosfilter(wirt.FILTER_LEN, wirt.FILTER_ALPHA,
                                         wirt.SAMP_PER_SYMBOL / wirt.FS, wirt.FS)

        self.upsamp_filter /= (self.upsamp_filter.sum() / wirt.SAMP_PER_SYMBOL)

        self.zc_sequence = generate_zc_sequence(wirt.ZC_N, wirt.ZC_U)
        self.num_samples_per_package = int(wirt.SAMP_PER_SYMBOL * (
                self.mod.get_modulated_size(wirt.NUM_REPETITIONS * wirt.POLAR_SIZE / 2)
                + len(self.zc_sequence))
                + wirt.FILTER_DELAY )


    def encode(self, data_bits):
        if len(data_bits) != 400:
            raise ValueError("WIRT only supports 400 bits data")

        # Polar coding
        data_polar = polar_encode(wirt.POLAR_SIZE, wirt.DATA_SIZE, data_bits)
        data_enc = repack_bits(data_polar, 1, 2)
        data_enc_rep = np.tile(data_enc, (1, wirt.NUM_REPETITIONS)).ravel()

        # QPSK modulate
        data_enc_mod = qpsk_modulate(data_enc_rep)

        # OFDM modulation (subcarrier mapping, pilot insertion, IFFT, cyclic prefix insert)
        IQ_data_clean = self.mod.modulate(data_enc_mod)
        IQ_data_power = IQ_data_clean.var()

        # Add preamble sequence
        # The ZC sequence is given the same variance as the IQ data in order to normalize the power
        zc_sequence_norm = self.zc_sequence * np.sqrt(IQ_data_power)
        IQ_data_preamp = np.hstack((zc_sequence_norm, IQ_data_clean))

        # Upsampling, Polyphase
        IQ_data = upsample_poly(IQ_data_preamp, wirt.SAMP_PER_SYMBOL, self.upsamp_filter)


        # Normalize the power to mean 0 dBm
        signal_power = (IQ_data.conj() * IQ_data).mean().real
        IQ_data /= np.sqrt(signal_power)

        return IQ_data


    def generate_random(self):
        data = np.random.binomial(1, 0.5, wirt.DATA_SIZE)
        return (data, self.encode(data))


    def decode(self, received_samples, estimate_esno=None, noise_power=None, post_not_precombine=False, do_downsample=True):
        """
        Decode a WIRT package.

        - received_samples: The samples to decode. The sample rate is assumed to be RF sample rate,
        unless do_downsample is False
        - estimate_esno: If the ESNO is known, specify it here in dB.
        - noise_power: if the ESNO is not known, specify the noise power in dB.
        - post_not_precombine: If the system should combine the LLRs before or after the decoding.
        - do_downsample: Perform downsampling operation.
        """

        if do_downsample:
            filt_delay = np.ceil(wirt.FILTER_LEN / wirt.SAMP_PER_SYMBOL).astype(np.int)
            samples = downsample(received_samples, wirt.SAMP_PER_SYMBOL, self.upsamp_filter)[filt_delay:]
        else:
            samples = received_samples


        ###
        # Synchronize based on the Zadoff-Chu sequence
        # The system is estimated to be approximately synchronized so the package begins after
        # the first included sample.
        N_samples_sync = int(wirt.MAXIMUM_SAMPLE_OFFSET * 1.25)
        ZC_indicies = np.arange(0, N_samples_sync, dtype=int)

        # Find the autocorrelation within the indicies from that the location of the ZC sequence.
        autocorrelation_postresamp = np.correlate(samples[ZC_indicies], self.zc_sequence, mode='valid')
        max_idx_post = np.argmax(np.abs(autocorrelation_postresamp))

        # The samples containing the ZC sequence
        # For an uneven ZC length, the "extra" sample is at the end
        # sequence_start_idx = max_idx_post - wirt.ZC_N // 2
        sequence_end_idx = max_idx_post + wirt.ZC_N

        # Pick out the synchronized samples
        symbol_count = (self.num_samples_per_package - wirt.FILTER_DELAY) // 4 - wirt.ZC_N
        samples_selector = slice(sequence_end_idx, (sequence_end_idx + symbol_count))
        samples_all = samples[samples_selector]

        # OFDM demodulate
        if estimate_esno is None:
            symbols_all, est_signal_power = self.mod.demodulate(samples_all, estimate_signal_power=True)
            symbols_all = symbols_all[:wirt.USED_SIZE//2].reshape((wirt.NUM_REPETITIONS, -1))

            estimate_esno = est_signal_power - noise_power
        else:
            symbols_all = self.mod.demodulate(samples_all, estimate_signal_power=False)[:wirt.USED_SIZE//2].reshape((wirt.NUM_REPETITIONS, -1))


        if PLOT_CONSTELLATION:
            plt.figure("Constellation plot")
            plt.plot(symbols_all[0].real, symbols_all[0].imag, 'bo', label="Received")
            #plt.xlim([-2, 2])
            #plt.ylim([-2, 2])
            plt.legend()


        # QPSK demodulate
        polar_channel_LLRs = qpsk_demodulate_soft(symbols_all, estimate_esno)

        # Polar decode
        if post_not_precombine:
            data_recv_all = np.empty((wirt.NUM_REPETITIONS, wirt.DATA_SIZE))
            for j in range(wirt.NUM_REPETITIONS):
                data_recv_all[j] = polar_decode_ssc(wirt.POLAR_SIZE, wirt.DATA_SIZE,
                         polar_channel_LLRs[j].flatten(), soft_output=True)

            data_recv = np.array(data_recv_all.mean(axis=0) < 0, dtype=np.uint8)

        else:
            combined_LLR = polar_channel_LLRs.mean(axis=0).flatten()
            data_recv = polar_decode_ssc(wirt.POLAR_SIZE, wirt.DATA_SIZE, combined_LLR)

        return data_recv


if __name__ == "__main__":
    num_runs = 1000
    encoder = wirt()

    encoded_data = []
    data = np.random.binomial(1, 0.5, (num_runs, wirt.DATA_SIZE)).astype(np.uint8)
    for i in range(num_runs):
        encoded = encoder.encode(data[i])
        encoded_data.append(encoded)
        decoded = encoder.decode(encoded, estimate_esno=20)

    if PLOT_SPECTRUM:
        plot_spectrum(np.concatenate(encoded_data), fs = wirt.FS)

    print("All equal:", (decoded == data[i]).all())

