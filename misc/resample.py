# -*- coding: utf-8 -*-

import numpy as np

from scipy.signal import upfirdn, fftconvolve

from misc.filters import rrcosfilter

def upsample(data, samples_per_sample, filt):
    dlen = len(data) * samples_per_sample
    lenh = len(filt) - samples_per_sample

    filt_offset = len(filt) // 2

    data_upsamp = np.zeros(dlen + lenh + 1, dtype=data.dtype)
    data_upsamp[filt_offset: filt_offset+ dlen:samples_per_sample] = data
    data_upsamp = np.convolve(data_upsamp, filt, mode='same')
    # data_upsamp = fftconvolve(data_upsamp, filt, mode='same')

    return data_upsamp * samples_per_sample


def upsample_scipy(data, samples_per_sample, filt):
    return upfirdn(filt, data, samples_per_sample, 1) * samples_per_sample


def upsample_poly(data, samples_per_sample, filt):
    dlen = len(data) * samples_per_sample
    lenh = len(filt) - samples_per_sample

    filt = np.pad(filt, (0, samples_per_sample - (len(filt) % samples_per_sample)), 'constant')
    filt_split = filt.reshape(-1, samples_per_sample).T
    data_pad = np.pad(data, ((samples_per_sample - 1), len(filt_split)), 'constant')

    data_upsamp = np.zeros(dlen + lenh + 1, data.dtype)
    for i in range(samples_per_sample):
        data_upsamp[i::samples_per_sample] = np.convolve(data_pad, filt_split[i], mode='same')

    return data_upsamp * samples_per_sample


def downsample(data, samples_per_sample, filt):
    data_downsamp = upfirdn(filt, data, 1, samples_per_sample)
    return data_downsamp


if __name__ == '__main__':
    np.random.seed(123)

    filt = rrcosfilter(31, 0.15, 4, 1)
    data = np.arange(12086) + np.random.randn(12086)
    res0 = upsample(data, 4, filt)
    res1 = upsample_scipy(data, 4, filt)
    res2 = upsample_poly(data, 4, filt)

    print("All filters agree:", np.allclose(res0, res1) and np.allclose(res0, res2))
