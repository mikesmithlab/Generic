# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:49:19 2018

@author: ppzmis
"""


import sys
import glob
import serial
import time
import struct

class DualStepper:
 
    def __init__(self,comport='COM5'):
        """Open the selected serial port"""
        self.port = serial.Serial()
        self.port.port = comport
        self.port.baudrate = 9600
        self.port.timeout = 0
        if port.isOpen() == False:
            self.port.open()
            self.port_status = True
            time.sleep(2)
            print('port opened')
        else:
            print("Select a COMPORT")
        
    
    def quit_serial(self):
        """Method to close the serial port when the Tk window is closed"""
        self.port.close()
        sys.exit()
        
    def send_command(self,cmd,steps=0):
        '''
        The barriers move by sending messages to the arduino code newStepperBoard which handles the hardware. 
        The commands are as follows.
    
        RX - Reset the X barrier position to zero by moving until it hits the limit switch
        RY - Reset the Y barrier position to zero by moving until it hits the limit switch
        MX+10 - Move X barrier towards middle by 10 steps
        MX-10 - Move barrier towards end by 10 steps
        MY+10 - Move Y barrier towards the middle by 10 steps
        MY-10 - Move Y barrier towards the end by 10 steps
    
        To enter the command supply a string to cmd of RX,RY, MX+,MX-,MY+,MY- etc
        Supply the number of steps to move
    
        Function returns the new position
        '''

    
        if cmd =='RX':
            print('RX')
            self.serialObj.write(b'RX\n')
            self.posx = 0
        if cmd =='RY':
            print('RY')
            self.serialObj.write(b'RY\n')
            self.posy=0
        if cmd =='MX+':
            print('MX+')
            message  = b'MX+'
            self.serialObj.write(message)
            self.serialObj.write(bytes(str(steps),'utf8'))
            self.serialObj.write(b'\n')
            self.posx=self.posx + steps
        if cmd =='MX-':
            print('MX-')
            message  = b'MX-'
            self.serialObj.write(message)
            self.serialObj.write(bytes(str(steps),'utf8'))
            self.serialObj.write(b'\n')
            self.posx=self.posx - steps
        if cmd =='MY+':
            print('MY+')
            message  = b'MY+'
            self.serialObj.write(message)
            self.serialObj.write(bytes(str(steps),'utf8'))
            self.serialObj.write(b'\n')
            self.posy=self.posy + steps
        if cmd =='MY-':
            print('MY-')
            message  = b'MY-'
            self.serialObj.write(message)
            self.serialObj.write(bytes(str(steps),'utf8'))
            self.serialObj.write(b'\n')
            self.posy=self.posy - steps
    
          
        print('values written')
        time.sleep(0.1)
        return pos

 
    
if __name__ == '__main__':
    try:
        s.quit_serial(s)
    except:
        pass
    s=DualStepper()
    
    #Reset x barrier
    s.send_command('RX')

        
    #Move x barrier 2000 steps towards the middle
    s.send_command('MX+',steps=2000)
    
    s.quit_serial()
    
        