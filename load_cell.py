# -*- coding: utf-8 -*-
"""
Created on Wed Oct 3 10:11:00 2018

@author: ppzjd3
"""

import sys
import serial
import time
import arduino

class LoadCell():
    """
    This is a class to manage reading the values
    from load cells
    """
    def __init__(self, ard):
        """
        Initialise with an instance of arduino.Arduino
        """
        self.ard = ard

    def read_force(self):
        """
        Read the force from the load cell and print the results.
        """
        self.ard.send_serial_line('r')
        force = self.ard.read_serial_line()
        print(force)

if __name__=="__main__":
    ard = arduino.Arduino()
    lc = LoadCell(ard)
    lc.read_force()
    ard.quit_serial()
