import numpy as np
from misc.modulation import qpsk_modulate

TIMING_DEBUG = False


class OFDMModulator:
    """OFDM modulator object
    fft_size: size of the FFT, including the pilot and guardbands, but excluding cyclic prefix
    num_cyclic_prefix: number of cyclic prefix samples to prepend
    add_pilot: bool if the modulator should add a default pilot. If false, the pilot is assumed included and equalization must be done externally.
    num_guardbands: number of subcarriers used as guardbands given as a turple (before, after)
    """

    def __init__(self, fft_size=32, num_cyclic_prefix=2, add_pilot=True,
                 num_pilots=4, num_guardbands=None):
        super(OFDMModulator, self).__init__()

        self.fft_size = fft_size
        self.num_cyclic_prefix = num_cyclic_prefix
        self.num_data_carriers = self.fft_size

        if num_guardbands is None:
            self.num_guardbands = (2, 2)
        elif type(num_guardbands) == tuple:
            self.num_guardbands = num_guardbands
        elif type(num_guardbands) == int:
            gb = (np.floor(num_guardbands / 2).astype(int), np.ceil(num_guardbands / 2).astype(int))
            self.num_guardbands = gb
        else:
            raise ValueError("num_guardbands must be integer or tuple")


        self.num_data_carriers -= sum(self.num_guardbands)

        self.add_pilot = add_pilot
        if self.add_pilot:
            self.num_data_carriers -= num_pilots
            first_pilot = self.num_data_carriers / (num_pilots)

            self.pilot_index = np.round(np.linspace(first_pilot,  (num_pilots - 1) * first_pilot, num_pilots)).astype(np.int)
            self.pilot = np.tile(qpsk_modulate(np.array([0, 1])), (1, num_pilots // 2)).ravel()


    def get_modulated_size(self, num_symbols):
        """
        Get the number of samples resulting from a modulation with the OFDMModulator object
        """

        mod_len = num_symbols
        if num_symbols % self.num_data_carriers != 0:
            mod_len += self.num_data_carriers - (num_symbols % self.num_data_carriers)

        num_ofdm_symbols = mod_len // self.num_data_carriers

        if self.add_pilot:
            mod_len += num_ofdm_symbols * len(self.pilot)

        mod_len += num_ofdm_symbols * sum(self.num_guardbands)
        mod_len += num_ofdm_symbols * self.num_cyclic_prefix

        return mod_len


    def modulate(self, data):
        """
        Perform OFDM modulation of a stream of data.
        Assumes symbol modulation has already been performed (to match matlab).
        """
        # Pad the data to always fit the size.
        if len(data) % self.num_data_carriers != 0:
            num_rows = len(data) // self.num_data_carriers + 1
            data2 = np.zeros((num_rows * self.num_data_carriers), dtype=data.dtype)
            data2[:len(data)] = data
            data = data2

        # Serial to parallel => Split into chunks of size "num_data_carriers"
        split_data = data.reshape((-1, self.num_data_carriers))

        # Add pilot
        if self.add_pilot:
            split_data_pilot = np.empty((split_data.shape[0], split_data.shape[1] + len(self.pilot)), dtype=np.complex)

            pilot_mask = np.zeros_like(split_data_pilot, dtype=bool)
            pilot_mask[:, self.pilot_index] = 1

            split_data_pilot[pilot_mask == 1] = np.tile(self.pilot, (1, len(split_data))).ravel()
            split_data_pilot[pilot_mask == 0] = data

            split_data = split_data_pilot

        # Sub-carrier mapping
        # Mainly adding guardbands to the side
        if sum(self.num_guardbands) > 0:
            # Numpy pad is significantly slower than this.
            large_split_data = np.zeros((split_data.shape[0], split_data.shape[1] + sum(self.num_guardbands)), dtype=split_data.dtype)
            large_split_data[:, self.num_guardbands[0]:-self.num_guardbands[1]] = split_data
        else:
            large_split_data = split_data

        # Ifft
        ifft_data = np.fft.ifft(np.fft.ifftshift(large_split_data, axes=1), norm='ortho')

        # Cyclic prefix
        prefixed_data = np.empty(
            (ifft_data.shape[0], ifft_data.shape[1] + self.num_cyclic_prefix),
            dtype=np.complex)

        prefixed_data[:, self.num_cyclic_prefix:] = ifft_data
        prefixed_data[:, :self.num_cyclic_prefix] = ifft_data[:, -self.num_cyclic_prefix:]

        # Parallel to serial
        stacked_data = prefixed_data.ravel()

        return stacked_data


    def demodulate(self, input_data, estimate_signal_power=False):
        # Serial to parallel
        split_synced_data = input_data.reshape((-1, self.fft_size + self.num_cyclic_prefix))

        # Remove cyclic prefix
        cycl_removed = split_synced_data[:, self.num_cyclic_prefix:]

        # FFT
        fft_data = np.fft.fftshift(np.fft.fft(cycl_removed, norm='ortho'), axes=1)

        # Remove guardbands
        mask = np.zeros_like(fft_data, dtype=bool)
        mask[:, self.num_guardbands[0]: mask.shape[1] - self.num_guardbands[1]] = True
        fft_data = np.reshape(fft_data[mask], (fft_data.shape[0], -1))

        if self.add_pilot:
            # Seperate the data and the pilot
            data_idx = np.delete(np.arange(fft_data.shape[1]), self.pilot_index)
            fft_data_no_pilot = fft_data[:, data_idx]
            pilot_recv        = fft_data[:, self.pilot_index]

            # Channel estimation
            h_est = pilot_recv / self.pilot

            # Linear interpolation
            eq_channel_est = np.empty_like(fft_data_no_pilot)
            for i in range(len(h_est)):
                eq_channel_est[i, :] = np.interp(data_idx, self.pilot_index, h_est[i,:])

            # Equalization
            fft_data_no_pilot /= eq_channel_est

            if estimate_signal_power:
                est_signal_power = 10*np.log10((np.abs(h_est)**2).mean())

        else:
            fft_data_no_pilot = fft_data

        if estimate_signal_power:
            return (fft_data_no_pilot.ravel(), est_signal_power)
        else:
            return fft_data_no_pilot.ravel()
