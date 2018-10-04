# -*- coding: utf-8 -*-
"""
Created on Wed Oct 3 10:11:00 2018

@author: ppzjd3
"""

import sys
import serial
import time
import arduino

class LoadCell(arduino.Arduino):

    def __init__(self,comport='/dev/ttyACM0'):
        super().__init__()

    def read_force(self):
        """
        Method to return the force from the load cell and print the results
        """
        self.send_serial_line('r')
        force = self.read_serial_line()
        print(force)

if __name__=="__main__":
    #ard = arduino.Arduino()
    lc = LoadCell()
    lc.read_force()
    lc.quit_serial()
