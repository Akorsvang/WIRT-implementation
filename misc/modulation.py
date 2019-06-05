import numpy as np

QPSK_MAP = (np.sqrt(2) / 2) * np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
QPSK_MAP_GRAY = (np.sqrt(2) / 2) * np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j])


def packed_to_unpacked(data, bits_per_byte=2):
    """
    Converts packed bytes (ordinary data) into unpacked bytes.
    The bytes in data are split into new bytes with no more than "bits_per_chunk" per bit
    Modelled after gnuradio block. The inverse is "unpacked_to_packed"
    """
    if not bits_per_byte in [1, 2]:
        raise ValueError("Bits per byte should be one of {}".format( [1, 2] ))

    data_bits = np.unpackbits(np.array(bytearray(data), dtype=np.uint8))

    if bits_per_byte == 1:
        unpacked_data = data_bits
    elif bits_per_byte == 2:
        unpacked_data = 2 * data_bits[::2] + data_bits[1::2]

    return unpacked_data


def unpacked_to_packed(data, bits_per_byte=2):
    """
    Converts unpacked bytes (symbols) into packed bytes (data).
    The bytes in data are split into new bytes with no more than "bits_per_chunk" per bit
    Modelled after gnuradio block. The inverse is "packed_to_unpacked"
    """
    if not bits_per_byte in [1, 2]:
        raise ValueError("Bits per byte should be one of {}".format( [1, 2] ))

    data_uint = np.array(bytearray(data), dtype=np.uint8)

    if bits_per_byte == 1:
        result = np.packbits(data_uint)
    elif bits_per_byte == 2:
        data_uint = data_uint.reshape((-1, 4))
        result = np.left_shift(data_uint, [6, 4, 2, 0]).sum(axis=1).astype(np.uint8)

    return result


def repack_bits(data, in_bits_per_byte, out_bits_per_byte):
    return packed_to_unpacked(unpacked_to_packed(data, in_bits_per_byte), out_bits_per_byte)


def qpsk_modulate(data, gray=True):
    """
    Perform QPSK modulation.
    Data should be an array of integers between 0 and 3.
    """

    if gray:
        modulated_data = QPSK_MAP_GRAY[data]
    else:
        modulated_data = QPSK_MAP[data]

    return modulated_data


def qpsk_demodulate_soft(data, esno):
    """
    Soft demodulate the QPSK bits based on an AWGN channel.
    Returns the LLRs on the I and Q channels in the real and complex parts of the result respectively.
    This is NOT the symbol LLR.
    """
    # Assume the signal power is normalized to one, the noise power is simply the inverse
    # of the ESNO
    inv_noisepwr = 1 / (10**(-esno / 10))

    llrs = (np.sqrt(2) * inv_noisepwr) * data

    return np.stack((llrs.imag, llrs.real), axis=-1)


def qpsk_hard_decision(llrs, gray=True):
    """
    Makes a hard decision based on LLRs from a QPSK soft demodulation.
    Assumes the format presented in qpsk_demodulate_soft
    """

    if gray:
        symbols = (    (llrs[:, 0] < 0) +
                   2 * (llrs[:, 1] < 0)
                   ).ravel().astype(np.uint8)
    else:
        symbols = (2 * (llrs[:, 0] > 0) +
                       (llrs[:, 1] > 0)
                   ).ravel().astype(np.uint8)
    return symbols


def qpsk_demodulate(data, gray=True):
    """
    Demodulate QPSK constellation.
    Gray parameter controls the constellation used.
    """
    if gray:
        symbols = (    (data.real < 0) +
                   2 * (data.imag < 0)
                   ).ravel().astype(np.uint8)
    else:
        symbols = (2 * (data.real > 0) +
                       (data.imag > 0)
                   ).ravel().astype(np.uint8)

    return symbols

