"""
Helper class for common socket operations
"""

import socket


class SocketConnection:
    DEBUG = False
    DEFAULT_IP = "192.168.100.254"
    DEFAULT_PORT = 24000

    def __init__(self, ip=DEFAULT_IP, port=DEFAULT_PORT, timeout=0.5, buf_size=4096):
        self.connected = False
        self.buf_size = buf_size

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)

        self.socket_error = self._sock.connect((ip, port))

        if self.socket_error == 0:
            self.connected = True

    def send(self, command):
        if type(command) == str:
            command = command.encode('UTF-8')

        if command[:-1] != b'\n':
            command += b'\n'

        self._sock.sendall(command)
        if self.DEBUG:
            print(b"Socket sent:", command)

    def recv(self):
        _dat = self._sock.recv(self.buf_size)
        if self.DEBUG:
            print(b"Socket got:", _dat)

        return _dat

    def recv_until_term(self):
        _dat = self.recv()
        while 1:
            if _dat[-1] == ord('\n'):
                break

            _dat += self.recv()

        return _dat

    def recv_bytes(self, num_bytes):
        """
        Receive a specified number of bytes, returning when that number is received.
        NOT protected against timeout.
        """
        data = bytearray(num_bytes)
        data_view = memoryview(data)
        remaining_bytes = num_bytes

        while remaining_bytes > 0:
            remaining_bytes -= self._sock.recv_into(
                data_view[(num_bytes - remaining_bytes):],
                min(remaining_bytes, self.buf_size))

        if self.DEBUG:
            print(b"Socket got:", data)

        return data

    def recv_all(self):
        data = []
        try:
            while(1):
                new_data = self.recv()
                data.append(new_data)
        except socket.timeout:
            # Done!
            full_data = b''.join(data)
            return full_data

    def recv_arbitrary(self):
        delim = self.recv_bytes(1)  # Delimiter
        if delim != b"#":
            print("Failed delimiter! Got: {} + {}".format(delim.decode('UTF-8'), self.recv().decode('UTF-8')))

        len_len = int(self.recv_bytes(1))
        data_len = int(self.recv_bytes(len_len))
        data = self.recv_bytes(data_len)

        self.recv_bytes(1)  # Discard the final newline.

        return data

    def send_recv(self, command: str) -> bytes:
        self.send(command)
        return self.recv_until_term()

    def send_recv_all(self, command):
        self.send(command)
        return self.recv_all()
    
    def wait_op_complete(self):
        """
        Wait for the current operation 
        """
        while not int(self.send_recv("*WAI; *OPC?")):
            pass

    def close(self):
        self._sock.close()
