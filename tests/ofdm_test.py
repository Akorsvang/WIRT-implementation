from time import perf_counter

import numpy as np
from scipy.io import loadmat

from ofdm import OFDM
from misc.modulation import qpsk_modulate, qpsk_demodulate, packed_to_unpacked, unpacked_to_packed


def matlab_test():
    print("Matlab test")
    matlab_data = loadmat('tests/data/ofdm_test.mat')

    FFT_SIZE = int(matlab_data['fft_size'])
    CYCLIC_PREFIX_LEN = int(matlab_data['num_cyclic_prefix'])

    # Structures
    mod = OFDM.OFDMModulator(fft_size=FFT_SIZE, num_cyclic_prefix=CYCLIC_PREFIX_LEN, add_pilot=False, num_guardbands=(0, 0))

    data = matlab_data['data'].flatten()
    python_modulated = mod.modulate(qpsk_modulate(data)) / (len(data) / 2)  # Python normalizes power, matlab does not
    matlab_modulated = matlab_data['moddat'].flatten()
    print("Modulation correct:", np.allclose(python_modulated, matlab_modulated))
    print("Demodulation identical:", np.allclose(mod.demodulate(python_modulated), mod.demodulate(matlab_modulated)))
    print("Demodulation correct:", (data == qpsk_demodulate(mod.demodulate(matlab_modulated))[:len(data)]).all())


def benchmark_test():
    """
    Test the chain through with timing output
    """
    print("Benchmark test")
    # Options
    FFT_SIZE = int(2**10)
    CYCLIC_PREFIX_LEN = 2

    # Structures
    mod = OFDM.OFDMModulator(fft_size=FFT_SIZE, num_cyclic_prefix=CYCLIC_PREFIX_LEN, add_pilot=True)

    # Generate some bytes to send
    st = perf_counter()
    data_orig = np.random.bytes(int(1e7))
    print("Generation {:.3f} ms".format(1000 * (perf_counter() - st)))

    st = perf_counter()
    data = packed_to_unpacked(data_orig)
    print("Unpacking {:.3f} ms".format(1000 * (perf_counter() - st)))

    # Transmitter
    st = perf_counter()
    transmit_data = mod.modulate(qpsk_modulate(data))
    print("Modulation {:.3f} ms".format(1000 * (perf_counter() - st)))

    # Receiver
    st = perf_counter()
    data_recv = qpsk_demodulate(mod.demodulate(transmit_data))
    print("Demodulation {:.3f} ms".format(1000 * (perf_counter() - st)))

    data_recv = data_recv[:len(data)]
    st = perf_counter()
    data_recv_packed = unpacked_to_packed(data_recv)
    print("Packing {:.3f} ms".format(1000 * (perf_counter() - st)))
    # print(data_recv_packed)

    # Decoded! Find BER:
    bit_error_rate = 1 - (np.frombuffer(data_recv_packed[:len(data_orig)], dtype=np.byte) == np.frombuffer(data_orig, dtype=np.byte)).mean()
    if bit_error_rate != 0:
        print("Bit errors! BER:", bit_error_rate)
    else:
        print("No bit errors!")


if __name__ == '__main__':
    benchmark_test()
    matlab_test()
