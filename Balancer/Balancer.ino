// This sketch is used to read load cells and provide instructions to stepper motors attached 
// to the same board.

#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include <WheatstoneBridge.h>

// Set up strain gauge on port 1
WheatstoneBridge wsb_strain1(A0, 365, 675, 0, 1000);

// Create the motor shield object with the default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield();

// Connect a stepper motor with 200 steps per revolution to 
// motor port #1 (M1 and M2)
Adafruit_StepperMotor *myMotor = AFMS.getStepper(64, 1);

void setup() {
  Serial.begin(9600);
  Serial.println("Ready");

  AFMS.begin(100); 
  myMotor->setSpeed(15);
}


int load1;

void loop() {
  char inBytes[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  String mode = "";
  String dir = "";
  String steps = "";
  int steps_val;
  if (Serial.available()) { // wait for serial input
    delay(50);

    int size_t = Serial.readBytesUntil('\n', inBytes, sizeof(inBytes)-1);

    if (inBytes[0] == 'M') { //move commands
      myMotor->step(100, FORWARD, SINGLE);
      }
    if (inBytes[0] == 'r') { //read load cell
      load1 = wsb_strain1.measureForce();
      Serial.println(load1, DEC);
    }
  }

}
