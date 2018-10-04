# -*- coding: utf-8 -*-
"""
Created on Wed Oct 4 09:32:00 2018

@author: ppzjd3
"""

import sys
import serial
import time

class Arduino:

    def __init__(self, comport='/dev/ttyACM0'):
        """Open the selected serial port"""
        self.port = serial.Serial()
        self.port.port = comport
        self.port.baudrate = 9600
        self.port.timeout = 0
        if self.port.isOpen() == False:
            self.port.open()
            self.port_status = True
            time.sleep(2)
            print('port opened')
        else:
            print("Select a COMPORT")
        self.wait_for_ready()

    def wait_for_ready(self):
        """Ensure the arduino has initialised by waiting for the
        'ready' command"""
        serial_length = 0
        while(serial_length<5):
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

if __name__=="__main__":
    ard = Arduino()
    ard.send_serial_line("r")
    text = ard.read_serial_line()
    print(text)
    ard.quit_serial()
