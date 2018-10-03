#include <WheatstoneBridge.h>

// Set up strain gauge on port 1
WheatstoneBridge wsb_strain1(A0, 365, 675, 0, 1000);

void setup() {
  Serial.begin(9600);
  Serial.println("Ready");
}

// Variables to store the readings
int val1;
int valRaw1;

void loop() {
  char inByte = ' '; 
  if (Serial.available()){ // wait for serial input
    char inByte = Serial.read(); // read the incoming data
    if (inByte == 'r'){ // if input is 'r' then measure force and print to serial
      val1 = wsb_strain1.measureForce();
      Serial.println(val1, DEC);
    }
  }
}
