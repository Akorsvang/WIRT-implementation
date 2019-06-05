"""
Contains default setups for the tester
"""
import numpy as np

from instrument_control.socket_manager import SocketConnection

REMOTE_IQVSG_DIR = "/user/"

def tester_default_setup(sock):
    """
    Takes an open socket connection to the tester and perfoms a reset and basic setup
    """
    sock.send_recv("*IDN?;*RST;")
    sock.send_recv("*WAI;SYST:ERR:ALL?;")

    # Setup routing
    sock.send("ROUT1;")
    sock.send("PORT:RES:ADD RF1A,VSA1;")  # VSA on port RF1A
    sock.send("PORT:RES:ADD RF2A,VSG1;")  # VSG on port RF2A

    # Select the channel
    sock.send("CHAN1;")

    # VSG settings
    sock.send("VSG1;")
    sock.send("POW:LEV -30")  # Output power
    sock.send("WAVE:EXEC OFF")

    # VSA settings
    tester_set_capturetime(1e-3, sock)

    sock.send("VSA1;")
    sock.send("RLEV -10")  # Reference power level
    sock.send_recv("*WAI;SYST:ERR:ALL?;")


def tester_load_default_waveform(sock):
    sock.send("WAVE:EXEC OFF")
    sock.send("WAVE:LOAD '/UWBP/UWBP_DRATE_P11_SYNC64.iqvsg'")
    sock.send("WAVE:EXEC ON")


def tester_set_capturetime(capture_time, sock):
    if capture_time > 25e-3:
        import sys
        sys.exit("The tester does not support longer captures times than 25 ms")

    sock.send("VSA1;")
    sock.send("CAPT:TIME {}".format(capture_time))
    resp = sock.send_recv("*WAI;SYST:ERR:ALL?;")
    return resp


def tester_setup_repeat(remote_waveform_name, sock, n_repeats=1):
    command =  "VSG1;"   # Select the VSG
    command += "WAVE:EXEC OFF;"  # Disable currently playing wave
    command += "WLIST:WSEG1:DATA '" + remote_waveform_name + "';"  # Create a waveform playlist
    command += "WLIST:WSEG1:SAVE;"  # Save the waveform playlist
    command += "WLIST:COUNT:ENABLE WSEG1;"  # Enable the number counter
    command += "WLIS:COUN {};".format(int(n_repeats))  # Set the number of repeats

    sock.send(command)


def tester_setup_ext_triggers(sock, trigger_level=-40, trigger_offset=0):
    """
    Setup external triggering based on rising edge of the EXT1 channel.
    trigger_level is in dBm
    trigger_offset is in seconds
    """

    command = "BP; MARK:EXT1:SOURCE LOW"  # Ensure that the external marker is LOW
    sock.send(command)

    command =  "VSG1;"  # Select the VSG
    command += "TRIG:SOUR EXTernal1;"  # Trigger on the EXT1 channel
    command += "TRIG:TYPE EDGE;"  # Edge trigger
    command += "TRIG:LEV {};".format(int(trigger_level))  # Power in dBm for triggering
    sock.send(command)

    command =  "VSA1;"
    command += "TRIG:SOUR EXTernal1;"
    command += "TRIG:TYPE EDGE;"
    command += "TRIG:LEV {};".format(int(trigger_level))
    command += "TRIG:OFFS:TIME {};".format(trigger_offset)  # Offset in seconds for the trigger

    sock.send(command)

def tester_arm(sock, arm_VSG=True, arm_VSA=True):
    """
    Arms the triggers in the VSA and VSG. VSG assumes there is a waveform loaded into slot WSEG1
    """
    command = ""
    if arm_VSA:
        command += "VSA1; INIT:NBLocked;"
    if arm_VSG:
        command += "VSG1;WAVE:EXEC ON, WSEG1"

    sock.send(command)


def tester_ext_trigger(sock):
    """
    Active the external trigger.
    Will only trigger the instrument if the external triggers have been setup.
    """
    command = "BP; MARK:EXT1:SOURCE HIGH"  # Set the EXT1 to high to trigger devices
    sock.send(command)


def upload_waveform_file(waveform_filename, sock=None, play_immediate=False):
    # Read file to upload
    with open(waveform_filename, 'rb') as f:
        file_data = f.read()

    remote_name = waveform_filename[waveform_filename.rfind('/') + 1:]
    return upload_waveform(file_data, remote_name=remote_name, sock=sock, play_immediate=play_immediate)


def upload_waveform(waveform, remote_name=None, sock=None, play_immediate=False):
    if remote_name is None:
        remote_name = "WIRT_tmp.iqvsg"

    if sock is None:
        sock = SocketConnection()

    # Get response in binary format
    sock.send("FORM:READ:DATA PACK")

    # Construct and send upload command
    dlen = len(waveform)
    dlen_len = len(str(dlen))

    remote_filename = REMOTE_IQVSG_DIR + remote_name
    upload_data = b"MMEM:DATA '" + remote_filename.encode() + b"', " + \
                  b"#" + str(dlen_len).encode() + str(dlen).encode() + \
                  waveform

    sock.send(upload_data)

    # Load and execute the new waveform
    sock.send("VSG1;WAVE:EXEC OFF")
    sock.send("*WAI; VSG1;" +
              "WAVE:LOAD '" + remote_filename + "';")

    if play_immediate:
        sock.send("WAVE:EXEC ON; WAI*")

    sock.wait_op_complete()

    return remote_filename



def download_waveform(sock=None, send_trigger=True):
    if send_trigger:
        sock.send("VSA1; INIT")

    # Download samples
    sock.send('*WAI; CAPT:SEGM1:SIGN1:DATA:IQ?')
    IQ_samples_bytes = sock.recv_arbitrary()

    # Convert ot float and compare
    IQ_samples_bin = np.frombuffer(IQ_samples_bytes, dtype=np.float32)

    return IQ_samples_bin[::2] + 1j * IQ_samples_bin[1::2]


if __name__ == "__main__":
    IP = "192.168.100.254"
    PORT = 24000

    sock = SocketConnection(IP, PORT)

    print("Performing setup")
    tester_default_setup(sock)

    print("Sending trigger, status: ")
    sock.send("INIT")
    print(sock.send_recv("*WAI;SYST:ERR:ALL?;"))

    sock.close()
