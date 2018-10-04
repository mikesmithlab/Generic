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
  char inBytes[5]; 
  if (Serial.available()){ // wait for serial input
    delay(50); // make sure line has finished
    
    for(int i = 0; i < sizeof(inBytes); ++i){ // reset inBytes
      inBytes[i] = char(0);
    }
    
    int size_t = Serial.readBytesUntil('\n', inBytes, sizeof(inBytes));
    if (inBytes[0] == 'r'){ // if input is 'r' then measure force and print to serial
      val1 = wsb_strain1.measureForce();
      Serial.println(val1, DEC);
    }
  }
}
