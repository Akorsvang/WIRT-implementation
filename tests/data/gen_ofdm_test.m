fft_size = 16;
num_cyclic_prefix = 2;
num_pilot = 0;

data = [0, 1, 0, 0, 1, 2, 1, 0]';

qpskmod = comm.QPSKModulator;
qpskdat = qpskmod(data);
qpskdat_pad = [qpskdat; zeros(fft_size - num_pilot - length(data), 1)];

pilot = repmat([0, 1]', (num_pilot/2) ,1);
qpskdat_pilot = [qpskmod(pilot); qpskdat_pad];

mod = comm.OFDMModulator('CyclicPrefixLength', num_cyclic_prefix, 'FFTLength', fft_size, 'NumGuardBandCarriers', [0;0]);
moddat = mod(qpskdat_pilot);

save('ofdm_test.mat', 'data', 'moddat', 'fft_size', 'num_cyclic_prefix', 'num_pilot')