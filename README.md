# Generic
This is a repository for all the highly generic code that will be reused in project after project. It consists of the following modules
- Camera --> Allows taking images and videos with the different cameras in the lab with generic user interface
- Video --> Allows reading writing videos
- stepper --> Contains class Stepper which handles Serial communication with Arduino running newStepperBoard.ino to control up to 2 stepper motors
- load_cell --> Contains class LoadCell which inherits from arduino.Arduino to read the value of a load cell from an Arduino running LoadCellSingle.ino
- arduino --> Contains class Arduino which handles Serial communication with Arduinos
