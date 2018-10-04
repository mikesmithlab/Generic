# Generic
This is a repository for all the highly generic code that will be reused in project after project. It consists of the following modules
- Camera --> Allows taking images and videos with the different cameras in the lab with generic user interface
- Video --> Allows reading writing videos
- stepper --> Contains class DualStepper which handles Serial communication with Arduino running newStepperBoard.ino to control up to 2 stepper motors.
              Contains class Stepper which controls a number of stepper motors with Arduino running StepperMotor.ino
- load_cell --> Contains class LoadCell which inherits from arduino.Arduino to read the value of a load cell from an Arduino running LoadCellSingle.ino
- arduino --> Contains class Arduino which handles Serial communication with Arduinos


Also contains a number of arduino sketches:
- LoadCellSingle.ino --> Allows the reading of load cell values by the load_cell.LoadCell class
- newStepperBoard.ino --> Allows the movement of two stepper motors using the stepper.DualStepper class
- StepperMotor.ino --> Allows the movement of a number of stepper motors using the stepper.Stepper class (needs additional stepper motors adding)
