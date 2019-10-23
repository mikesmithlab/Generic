import serial


class Probe(serial.Serial):

    def __init__(self,
                 port="/dev/serial/by-id/usb-Omega_Engineering_RH-USB_N13012205-if00-port0"):
        super().__init__(port)
        self.write(b'C\r')
        self.readline()

    def get_temp_C(self):
        self.write(b'C\r')
        txt = self.readline()
        return float(txt.decode().split(' ')[0][1:])

    def get_relative_humidity(self):
        self.write(b'H\r')
        txt = self.readline()
        return txt


if __name__ == "__main__":
    p = Probe(
        "/dev/serial/by-id/usb-Omega_Engineering_RH-USB_N13012205-if00-port0")
    for i in range(100):
        print(p.get_temp_C())
    p.close()
