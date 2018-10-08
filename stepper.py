# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:49:19 2018

@author: ppzmis
"""


import sys
import serial
import time
import arduino

class DualStepper:

    def __init__(self,comport='COM4'):
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


    def quit_serial(self):
        """Method to close the serial port when the Tk window is closed"""
        self.port.close()
        time.sleep(1)


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
            self.port.write(b'RX\n')
            self.posx = 0
        if cmd =='RY':
            print('RY')
            self.port.write(b'RY\n')
            self.posy=0
        if cmd =='MX+':
            print('MX+')
            message  = b'MX+'
            self.port.write(message)
            self.port.write(bytes(str(steps),'utf8'))
            self.port.write(b'\n')
            self.posx=self.posx + steps
        if cmd =='MX-':
            print('MX-')
            message  = b'MX-'
            self.port.write(message)
            self.port.write(bytes(str(steps),'utf8'))
            self.port.write(b'\n')
            self.posx=self.posx - steps
        if cmd =='MY+':
            print('MY+')
            message  = b'MY+'
            self.port.write(message)
            self.port.write(bytes(str(steps),'utf8'))
            self.port.write(b'\n')
            self.posy=self.posy + steps
        if cmd =='MY-':
            print('MY-')
            message  = b'MY-'
            self.port.write(message)
            self.port.write(bytes(str(steps),'utf8'))
            self.port.write(b'\n')
            self.posy=self.posy - steps


        print('values written')
        time.sleep(0.5)

class Stepper():
    """
    Class to manage the movement of stepper motors
    """
    def __init__(self, ard):
        """ Initialise with an instance of arduino.Arduino"""
        self.ard = ard

    def move_motor(self, motor_no, steps, direction):
        """
        Generate the message to be sent to self.ard.send_serial_line

        Inputs:
        motor_no: 1 or 2
        steps: int
        direction: either '+' or '-'
        """
        motor_assignment = {1: 'A', 2:'B', 3:'C'}
        message = 'M' + motor_assignment[1] + direction + str(steps) + '\n'
        self.ard.send_serial_line(message)
        print('done')


if __name__ == '__main__':

    class_choice = input("Enter '1' to test DualStepper or '2' to test Stepper ")
    if class_choice == '1':
        s=DualStepper()
        #Reset y barrier
        s.send_command('RY')
        #Move y barrier 2000 steps towards the middle
        s.send_command('MY+',steps=1000)
        s.quit_serial()

    elif class_choice == '2':
        ard = arduino.Arduino()
        s = Stepper(ard)
        s.move_motor(1, 100, '+')
        ard.quit_serial()
