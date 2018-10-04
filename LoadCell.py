# -*- coding: utf-8 -*-
"""
Created on Wed Oct 3 10:11:00 2018

@author: ppzjd3
"""

import sys
import serial
import time

class LoadCell:

    def __init__(self,comport='/dev/ttyACM0'):
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

    def quit_serial(self):
        """Method to close the serial port when the Tk window is closed"""
        self.port.close()
        print('port closed')
        time.sleep(1)

    def read_force(self):
        """Method to listen to the serial port and print the results"""
        self.port.write(b'r')
        time.sleep(1)
        print(self.port.readline().decode())

    def wait_for_ready(self):
        """Method to ensure the arduino has initialised"""
        serial_length = 0
        while(serial_length<5): #always print 'Ready' when arduino has initialised
            serial_length = self.port.inWaiting()
        print(self.port.readline().decode())

if __name__=="__main__":
    lc = LoadCell()
    lc.read_force()
    lc.quit_serial()
