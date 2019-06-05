from datetime import datetime

import numpy as np

TEMPLATE_FILE = "template.iqvsg"
IQVSG_FILE = "outfile.iqvsg"
FS = int(2.4e9)


def iq_to_bytes(IQ_data):
    IQ_split = np.zeros(2 * len(IQ_data))
    IQ_split[::2] = IQ_data.real
    IQ_split[1::2] = IQ_data.imag

    return IQ_split.astype('float32').tobytes()

def iq_to_waveform(IQ_data, fs=FS, template_file=TEMPLATE_FILE):
    """
    Combines the IQ to bytes and bytes to waveform translations
    """

    return bytes_to_waveform(iq_to_bytes(IQ_data), fs=FS, template_file=template_file)

def bytes_to_waveform(data, fs=FS, template_file=TEMPLATE_FILE):
    """
    Packs a byte stream into a waveform for use with the equipment.
    The data should be bytes packed in [0_real,0_imag,1_real,1_imag,...] format.
    FS must be a multiple of 300, max 2400 (300, 600, 1200, 2400)
    """
    with open(template_file, 'r') as f:
        file_data = f.read()

    # The data is packed into alternating float32 (4 byte per point) I and Q (2 points per sample)
    sample_count = len(data) // (2 * 4)

    # Replace template options
    file_data = file_data.replace("[[datetime]]", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    file_data = file_data.replace("[[sample_rate]]", str(FS))
    file_data = file_data.replace("[[sample_count]]", str(sample_count))

    # Add data
    new_file_data = file_data.encode() + data

    return new_file_data


def bytes_to_waveform_file(data, fs=FS, template_file=TEMPLATE_FILE, iqvsg_file=IQVSG_FILE):
    new_file_data = bytes_to_waveform(data, fs, template_file)

    with open(iqvsg_file, 'wb') as f:
        f.write(new_file_data)


if __name__ == "__main__":
    ###
    # Generate some data
    t = np.arange(0.00001, 2048, (1 / 8))  # Expect a tone at 1/8 FS

    I_data = np.cos(2 * np.pi * t)
    Q_data = np.sin(2 * np.pi * t)

    # The data is interleaved in I/Q pairs
    sample_count = I_data.shape[0]
    IQ_data = np.zeros(2 * sample_count)
    IQ_data[::2] = I_data
    IQ_data[1::2] = Q_data

    IQ_bytes = IQ_data.astype('float32').tobytes()

    ###
    # Write the file
    bytes_to_waveform(IQ_bytes)
