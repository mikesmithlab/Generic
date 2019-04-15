#define halfPressPin 12
#define fullPressPin 13

void setup() {
  pinMode(halfPressPin, OUTPUT);
  pinMode(fullPressPin, OUTPUT);
}

void loop() {
  digitalWrite(halfPressPin, 1);
  delay(1000);

  digitalWrite(fullPressPin, 1);
  delay(1000);

  digitalWrite(halfPressPin, 0);
  digitalWrite(fullPressPin, 0);
  delay(1000);

}
