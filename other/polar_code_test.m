num_runs_max = 1000;
num_frame_errors = 100;
ESNOs = -1:0.25:10;
count_ESNOs = length(ESNOs);

N = 2^13;
K = N / 2;

%% Init
initPC(N, K, 'AWGN', 0);

results_BLER = zeros(count_ESNOs, 1);

%% Run
for esno_i = 18:count_ESNOs
    fprintf("ESNO %d\n", ESNOs(esno_i));
    total_frame_errors = 0;
    total_frames = 0;

    for n_run = 1:num_runs_max
        u = (rand(K, 1) > 0.5);  %random message
        x = pencode(u);      		  %polar encoding
        y = OutputOfChannel(x, 'AWGN', ESNOs(esno_i)); %simulate AWGN channel at Eb/N0=1dB
        uhat = pdecode(y, 'AWGN', ESNOs(esno_i)); 	  %decode under the same AWGN channel setting.

        total_frame_errors = total_frame_errors + (1 - logical( sum( uhat == u ) == K ));	%check for Decoding success!
        total_frames = total_frames + 1;
        
        if total_frame_errors > num_frame_errors
            break;
        end
    end
    if total_frame_errors < num_frame_errors
        fprintf("Timeout!\n");
    end
    
    results_BLER(esno_i) = total_frame_errors / total_frames;
end
