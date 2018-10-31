int RecordPin = 0;
int WaitB4Rec = 10;//Time in seconds
int RecPeriod = 1000;//Time in seconds
int NumbRec = 5;
int GapBetweenRec = 1000;//Time in seconds


void setup() {
  // put your setup code here, to run once:
  pinMode(RecordPin, OUTPUT);//Specify GPIO output
  delay(WaitB4Rec);
}

void loop() {
  // put your main code here, to run repeatedly:
  for(int j = 1; j <= NumbRec; j++){
    digitalWrite(RecordPin, HIGH);//Start Movie
    delay(RecPeriod);
    digitalWrite(RecordPin, LOW);//Stop Movie
    delay(GapBetweenRec);
  }
  abort();
}
