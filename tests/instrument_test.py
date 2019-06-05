from time import perf_counter as pc

import numpy as np

from ofdm.zadoff_chu import generate_zc_sequence
from instrument_control.socket_manager import SocketConnection
import instrument_control.tester_tools as tt
from instrument_control.waveforms import waveform

def timing_test_ascii():
    IP = "192.168.100.254"
    PORT = 24000
    NUM_MEASUREMENTS = 100


    # Default TCP socket and default setup
    start_time = pc()

    sock = SocketConnection(IP, PORT)
    tt.tester_default_setup(sock)
    sock.send("FORM:READ:DATA ASC")  # ASCII mode download

    print("Setup took {:.2f} s".format(pc() - start_time))

    # Ensure the tester is ready.
    sock.wait_op_complete()

    results = np.empty(NUM_MEASUREMENTS)
    for i in range(NUM_MEASUREMENTS):
        # Trigger!
        sock.send("VSA1; INIT")

        # Ensure the trigger is done before downloading measurement, but don't block
        sock.send("*WAI")

        start_time = pc()
        power_samples_bytes = sock.send_recv_all('CAPT:SEGM1:SIGN1:DATA:SUBS:PVT? 16000000, 0, 5e-3')

        results[i] = pc() - start_time

    sock.close()

    print("Sample download time; mean: {:.3f} s, variance: {:.3f} s".format(results.mean(), results.var()))


def timing_test_binary():
    IP = "192.168.100.254"
    PORT = 24000
    NUM_MEASUREMENTS = 100


    # Default TCP socket and default setup
    start_time = pc()

    sock = SocketConnection(IP, PORT)
    tt.tester_default_setup(sock)
    sock.send("FORM:READ:DATA PACK")

    print("Setup took {:.2f} s".format(pc() - start_time))

    # Ensure the tester is ready.
    sock.wait_op_complete()

    results = np.empty(NUM_MEASUREMENTS)
    for i in range(NUM_MEASUREMENTS):
        # Trigger!
        sock.send("VSA1; INIT")

        # Ensure the trigger is done before downloading measurement, but don't block
        sock.send("*WAI")

        start_time = pc()

        SEPERATE = False
        if SEPERATE:
            # Download I measurement
            sock.send('CAPT:SEGM1:SIGN1:DATA:SUBS:I? 16000000, 0, 5e-3')
            I_samples_bytes = sock.recv_arbitrary()

            # Q measurement
            sock.send('CAPT:SEGM1:SIGN1:DATA:SUBS:Q? 16000000, 0, 5e-3')
            Q_samples_bytes = sock.recv_arbitrary()

        else:
            # Download I/Q measurement
            sock.send('CAPT:SEGM1:SIGN1:DATA:IQ?')
            IQ_samples_bytes = sock.recv_arbitrary()

        results[i] = pc() - start_time

    sock.close()

    print("Sample download time; mean: {:.3f} s, variance: {:.3f} s".format(results.mean(), results.var()))


def timing_test_iqvsa():
    IP = "192.168.100.254"
    PORT = 24000
    NUM_MEASUREMENTS = 100


    # Default TCP socket and default setup
    start_time = pc()

    sock = SocketConnection(IP, PORT)
    tt.tester_default_setup(sock)
    sock.send("FORM:READ:DATA PACK")

    print("Setup took {:.2f} s".format(pc() - start_time))

    # Ensure the tester is ready.
    sock.wait_op_complete()

    results = np.empty(NUM_MEASUREMENTS)
    for i in range(NUM_MEASUREMENTS):
        # Trigger!
        sock.send("VSA1; INIT")

        # Ensure the trigger is done before downloading measurement, but don't block
        sock.send("*WAI")

        start_time = pc()

        # Save data to iqsva file
        iqvsa_filename = "'/Capture/test_uwb_capture.iqvsa'"
        sock.send("CHAN1;capt:segm1:stor " + iqvsa_filename + ";")

        sock.send("*WAI; MMEM:DATA? " + iqvsa_filename)
        power_samples_bytes = sock.recv_arbitrary()

        results[i] = pc() - start_time

    sock.close()

    print("Sample download time; mean: {:.3f} s, variance: {:.3f} s".format(results.mean(), results.var()))


def basic_transfer_test():
    IP = "192.168.100.254"
    PORT = 24000
    IQVSG_FILE = "waveforms/testfile.iqvsg"
    REMOTE_IQVSG_FILE = "/user/uploaded_waveform.iqvsg"

    # Default TCP socket and default setup
    start_time = pc()
    sock = SocketConnection(IP, PORT)
    tt.tester_default_setup(sock)

    # Get response in binary format
    sock.send("FORM:READ:DATA PACK")

    # Prepare....
    sock.wait_op_complete()

    print("Setup took {:.2f} s".format(pc() - start_time))

    ###
    # Upload the example waveform file.

    # Read file to upload
    with open(IQVSG_FILE, 'rb') as f:
        file_data = f.read()

    # Construct and send upload command
    dlen = len(file_data)
    dlen_len = len(str(dlen))

    upload_data = b"MMEM:DATA '" + REMOTE_IQVSG_FILE.encode() + b"', " + \
                  b"#" + str(dlen_len).encode() + str(dlen).encode() + \
                  file_data

    sock.send(upload_data)

    # Load and execute the new waveform
    sock.send("*WAI; VSG1;" +
              "WAVE:LOAD '" + REMOTE_IQVSG_FILE + "';" +
              "WAVE:EXEC ON; WAI*")


    ###
    # Download the measured samples back

    # Trigger!
    sock.send("VSA1; INIT")

    # Download samples
    sock.send('*WAI; CAPT:SEGM1:SIGN1:DATA:IQ?')
    IQ_samples_bytes = sock.recv_arbitrary()

    # Done with the socket connection
    sock.close()

    # Convert ot float and compare
    IQ_samples_bin = np.frombuffer(IQ_samples_bytes, dtype=np.float32)

    I_samples_bin = IQ_samples_bin[::2]
    Q_samples_bin = IQ_samples_bin[1::2]


def samples_download_time():
    # We use a Zadoff-Chu sequence to get some random data, because that's what I had at hand ¯\_(ツ)_/¯

    FS = 2.4e9

    zc_sequence = generate_zc_sequence(997, 7)
    IQ_bytes = waveform.iq_to_bytes(zc_sequence)
    waveform_bytes = waveform.bytes_to_waveform(IQ_bytes, fs=FS, template_file='instrument_control/waveforms/template.iqvsg')

    expected_duration = len(zc_sequence) * (1 / FS)

    ###
    # Socket connection
    sock = SocketConnection()
    tt.tester_default_setup(sock)
    tt.tester_set_capturetime(1e-6 + expected_duration, sock)

    remote_name = tt.upload_waveform(waveform_bytes, sock=sock)

    # Setup the tester
    tt.tester_setup_repeat(remote_name, sock, n_repeats=1)
    tt.tester_setup_ext_triggers(sock)
    tt.tester_arm(sock)
    tt.tester_ext_trigger(sock)

    st = pc()
    # Download the samples
    samples = tt.download_waveform(sock, send_trigger=False)

    print("No wait time:", pc() - st)

    sock.close()

    ###
    # Socket connection
    sock = SocketConnection()
    tt.tester_default_setup(sock)
    tt.tester_set_capturetime(1e-6 + expected_duration, sock)

    remote_name = tt.upload_waveform(waveform_bytes, sock=sock)

    # Setup the tester
    tt.tester_setup_repeat(remote_name, sock, n_repeats=1)
    tt.tester_setup_ext_triggers(sock)
    tt.tester_arm(sock)
    tt.tester_ext_trigger(sock)

    st = pc()
    sock.wait_op_complete()
    print("Tester wait time:", pc() - st)

    st = pc()
    # Download the samples
    samples = tt.download_waveform(sock, send_trigger=False)

    print("With wait time:", pc() - st)

    sock.close()


if __name__ == '__main__':
    pass