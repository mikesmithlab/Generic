import arduino
import stepper
import load_cell

class Balancer():
    """
    Class to balance the shaker using stepper motors and loadcells
    """
    def __init__(self):
        """ Initise the Arduino, Stepper and LoadCell classes """
        self.ard = arduino.Arduino()
        self.Stepper = stepper.Stepper(self.ard)
        self.LoadCell = load_cell.LoadCell(self.ard)

    def read_forces(self):
        """ Read the force from a load_cell"""
        force = self.LoadCell.read_force()

    def move_motor(self):
        """ Move a stepper motor """
        self.Stepper.move_motor(1, 1000, '+')

    def clean_up(self):
        """ link to quit serial """
        self.ard.quit_serial()


if __name__=="__main__":
    bal = Balancer()
    bal.read_forces()
    bal.move_motor()
    bal.clean_up()
