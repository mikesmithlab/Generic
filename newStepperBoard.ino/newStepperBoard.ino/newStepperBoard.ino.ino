int xStepPin = 2;
int xDirPin = 3;
int xPower = A2;

int yStepPin = 6;
int yDirPin = 7;
int yPower = A0;

int yLimit = 11;
int xLimit = 10;

int Speed = 200;
float Pause;
int repeats;

int yval;
int xval;
int interval_pause;
int intervals;
/*
 * Variables used for receiving Serial instructions
 */


String value = "";
String mode = "";
int val;
String setting = "";

void setup() {
  pinMode(xStepPin, OUTPUT);
  pinMode(xDirPin, OUTPUT);
  pinMode(yStepPin, OUTPUT);
  pinMode(yDirPin, OUTPUT);
  pinMode(yPower, OUTPUT);
  pinMode(xPower, OUTPUT);
  pinMode(yLimit, INPUT);
  pinMode(xLimit, INPUT);
  Pause = Speed;
  Pause = 500/Pause;

  digitalWrite(yPower, HIGH);
  digitalWrite(xPower, HIGH);
  Serial.begin(9600);
}

void loop() 
{
    // Read serial input:
    while (Serial.available() > 0) 
    {
        int inChar = Serial.read();
        if (isDigit(inChar)) 
        {
            // convert the incoming byte to a char and add it to the string:
            value += (char)inChar;
        }
        else{
            setting += (char)inChar;
        }
    
        // if you get a newline, print the string, then the string's value:
        if (inChar == '\n') 
        {
            
            val = value.toInt();
            // clear the string for new input:
            value = "";
            mode =setting;
            setting = "";
            
            if (mode.equals("RX\n")){resetx(); delay(2);Serial.println('f');}
            if (mode.equals("RY\n")){resety();Serial.println('f');}
            if (mode.equals("MX+\n")){moveXP();Serial.println('f');}
            if (mode.equals("MX-\n")){moveXN();Serial.println('f');}
            if (mode.equals("MY+\n")){moveYP();Serial.println('f');}
            if (mode.equals("MY-\n")){moveYN();Serial.println('f');}
            if (mode.equals("MXY-\n")){moveXYN();Serial.println('f');}
            if (mode.equals("MXY+\n")){moveXYP();Serial.println('f');}
    
              
         }
  
    
    
    }
}


void resetx() {
  //Serial.println("Resetting X to zero");
  //Resets the x barrier to zero using the limit switch
  //Cmd = "RX"
  xval = digitalRead(xLimit);
  digitalWrite(xPower, LOW);
  digitalWrite(xDirPin, true);
  while (xval==LOW) {
     digitalWrite(xStepPin, HIGH);
     delayMicroseconds(Pause*1000);
     digitalWrite(xStepPin, LOW);
     delayMicroseconds(Pause*1000);
     xval = digitalRead(xLimit);
   }
   digitalWrite(xPower, HIGH);
}  
 

void resety() {
    //Serial.println("Resetting Y to zero");
    //Resets the y barrier to zero using the limit switch
    //Cmd = "RY"
    yval = digitalRead(yLimit);
    digitalWrite(yPower, LOW);
    digitalWrite(yDirPin, false);
    while (yval==LOW) {
        digitalWrite(yStepPin, HIGH);
        delayMicroseconds(Pause*1000);
        digitalWrite(yStepPin, LOW);
        delayMicroseconds(Pause*1000);
        yval = digitalRead(yLimit);
       
    }
    digitalWrite(yPower, HIGH);
}


void moveXP(){
  //Moves the x barrier towards the centre
  //Cmd = "MX+"
  //Serial.println("Moving X inwards");
  movestep(false, false, LOW, HIGH, val);
  }

void moveXN(){
  
  //Moves the x barrier towards the end
  //Cmd = "MX-"
  Serial.println("Moving X outwards");
  movestep(true, false, LOW, HIGH, val);
  }

  void moveYP(){
  //Moves the y barrier towards the centre
  //Cmd = "MY+"
  Serial.println("Moving Y inwards");
  movestep(false, true, HIGH, LOW, val);
  }

  void moveYN(){
  //Moves the y barrier towards the end
  //Cmd = "MY-"
  Serial.println("Moving Y outwards");
  movestep(false,false,HIGH,LOW,val);
  }

  void moveXYP(){
    //move both barriers towards centre
    movestep(false,true,LOW,LOW,val);
    }

  void moveXYN(){
    //move both barriers towards ends
    movestep(true,false,LOW,LOW,val);
    }

void movestep(boolean xdir, boolean ydir, int xMove, int yMove, int steps) {
  //Turn on and set direction of stepper motors.
  digitalWrite(xPower, xMove);
  digitalWrite(yPower, yMove);
  digitalWrite(xDirPin, xdir);
  digitalWrite(yDirPin, ydir);

  //Toggle output to drive stepper motors.
  for (int i=0;i<steps;i++) {
    digitalWrite(yStepPin, HIGH);
    digitalWrite(xStepPin, HIGH);
    delayMicroseconds(Pause*1000);
    digitalWrite(yStepPin, LOW);
    digitalWrite(xStepPin, LOW);
    delayMicroseconds(Pause*1000);
  }
  digitalWrite(xPower, HIGH);
  digitalWrite(yPower, HIGH);
}


void interval(boolean xdir, boolean ydir, int xmove, int ymove, unsigned int steps) {
  digitalWrite(xPower, xmove);
  digitalWrite(yPower, ymove);
  digitalWrite(xDirPin, xdir);
  digitalWrite(yDirPin, ydir);
  for (int i=0;i<intervals;i++) {
    for (int j=0;i<steps;j++) {
      digitalWrite(yStepPin, HIGH);
      delayMicroseconds(Pause*1000);
      digitalWrite(yStepPin, LOW);
      delayMicroseconds(Pause*1000);
    }
    delay(interval_pause*1000);
  }
  digitalWrite(xPower, HIGH);
  digitalWrite(yPower, HIGH);
}





