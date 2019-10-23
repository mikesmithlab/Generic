import time

import numpy as np
import serial

from Generic import filedialogs


class Lauda(serial.Serial):

    def __init__(self, port=None):
        if port is None:
            filedialogs.load_filename('Select a serial device',
                                      directory='/dev/serial/by-id/',
                                      file_filter=None)
        super().__init__(port)
        self.read_all()

    def read_current_temp(self):
        try:
            self.read_all()
            self.write(b'IN_PV_01\r\n')
            time.sleep(0.3)
            txt = self.read_all()
            val = float(txt)
        except:
            val = np.NaN
        return val

    def start(self):
        self.read_all()
        self.write(b'START\r\n')

    def stop(self):
        self.read_all()
        self.write(b'STOP\r\n')

    def set_temp(self, new_temp):
        self.read_all()
        self.write(bytes('OUT_SP_00_{:06.2f}\r\n'.format(new_temp),
                         encoding='utf-8', errors='strict'))

    def set_pumping_speed(self, val):
        self.read_all()
        self.write(bytes('OUT_SP_01_{:03d}\r\n'.format(val),
                         encoding='utf-8', errors='strict'))


if __name__ == "__main__":
    l = Lauda(
        '/dev/serial/by-id/usb-LAUDA_DR._R._WOBSER_GMBH_CO.KG_LAUDA_Constant_Temperature_Equipment_Virtual_COM_Port_34FFD90543523238-if00')

    print(l.read_current_temp())
    l.close()
