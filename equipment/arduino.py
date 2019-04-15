import serial
import time
import Generic.filedialogs as fd
import os

"""

Linux:
If access to port is denied on linux then run the following command in the
terminal:

sudo adduser $USER dialout

Login out and back in again for it to take effect.
"""


class Arduino:

    def __init__(self, port=None, rate=9600, wait=True):
        """Open the selected serial port"""
        self.port = serial.Serial()
        if port:
            self.port.port = port
        else:
            self.choose_port()
        self.port.baudrate = rate
        self.port.timeout = 0
        if self.port.isOpen() == False:
            self.port.open()
            self.port_status = True
            time.sleep(0.5)
            print('port opened')
        else:
            print("Select a COMPORT")
        if wait:
            self.wait_for_ready()
        else:
            time.sleep(0.5)

    def choose_port(self, os='linux'):
        if os == 'linux':
            self.port.port = fd.load_filename(
                'Choose a comport',
                directory='/dev/')

    def wait_for_ready(self):
        """Ensure the arduino has initialised by waiting for the
        'ready' command"""
        serial_length = 0
        while (serial_length < 5):
            serial_length = self.port.inWaiting()
        print(self.port.readline().decode())

    def quit_serial(self):
        """ Close the serial port """
        self.port.close()
        print('port closed')

    def send_serial_line(self, text):
        """
        Send a string over the serial port making sure it ends in a new
        line .

        Input:
            text    the string to be sent to the arduino
        """
        if text[-2:] != '\n':
            text += '\n'
        text_in_bytes = bytes(text, 'utf8')
        self.port.write(text_in_bytes)

    def read_serial_bytes(self, no_of_bytes):
        """ Read a given no_of_bytes from the serial port"""
        size_of_input_buffer = 0
        while size_of_input_buffer < no_of_bytes:
            size_of_input_buffer = self.port.inWaiting()
        text = self.port.read(no_of_bytes)
        print(text.decode())

    def read_serial_line(self):
        """
        Waits for data in the input buffer then
        reads a single line from the serial port.

        Outputs:
            text    the data from serial in unicode
        """
        size_of_input_buffer = 0
        while size_of_input_buffer == 0:
            size_of_input_buffer = self.port.inWaiting()
            time.sleep(0.1)
        text = self.port.readline()
        return text.decode()

    def readlines(self, n):
        out = [self.port.readline().decode() for i in range(n)]
        return out

    def ignorelines(self, n):
        [self.port.readline() for i in range(n)]

    def flush(self):
        while self.port.inWaiting() > 1:
            self.port.reset_input_buffer()

    def read_all(self):
        string = ''
        while self.port.inWaiting() > 1:
            l = self.port.readline()
            string += l.decode("utf-8")
        return string


def find_port():
    items = os.listdir('/dev/')
    newlist = []
    for names in items:
        if names.startswith("ttyA"):
            newlist.append(names)
    return newlist[0]


if __name__ == "__main__":
    # a = find_port()
    # print(a)
    ard = Arduino()
    ard.send_serial_line("100")
    text = ard.read_serial_line()
    print(text)
    ard.send_serial_line('200')
    text = ard.read_serial_line()
    print(text)
    ard.quit_serial()