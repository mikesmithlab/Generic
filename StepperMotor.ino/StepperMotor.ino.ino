// This Sketch is used to control a number of Stepper Motors attached to the same board. 
// Require Serial commands with the following structure:
//     MA+100
// 1st Character is the command, M=move
// 2nd Character is the motor to move
// 3rd Character is the direction, +=forwards, -=backwards
// remaining characters are the number of steps to move up to a maximum of 999,999


#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Create the motor shield object with the default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield();

// Connect a stepper motor with 200 steps per revolution to 
// motor port #1 (M1 and M2)
Adafruit_StepperMotor *myMotor = AFMS.getStepper(64, 1);

void setup() {
  Serial.begin(9600);
  Serial.println("Ready");
  AFMS.begin(100); // create with the default frequency 1.6 KHz
  myMotor->setSpeed(15); // 10 rpm
}

void loop() {
  char inBytes[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  String mode = "";
  String dir = "";
  String steps = "";
  int steps_val;
  if (Serial.available()) { // wait for serial input
    delay(50);

    int size_t = Serial.readBytesUntil('\n', inBytes, sizeof(inBytes)-1);
    for (int i = 3; i < size_t; i++) {
      steps += (char)inBytes[i];
    }
    steps_val = steps.toInt();
    
    if (inBytes[0] == 'M'){ // move commands
      if (inBytes[1] == 'A') { // motor A
              if (inBytes[2] == '+') { // + direction
                Serial.println("Motor moves");
                myMotor->step(steps_val, FORWARD, SINGLE);
              }
      }
    }
  }

}
