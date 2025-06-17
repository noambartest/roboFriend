const int joyX = A0;
const int joyY = A1;
const int joySW = 8;

void setup() {
  Serial.begin(9600);
  pinMode(joySW, INPUT_PULLUP);
}

void loop() {
  int x = analogRead(joyX);
  int y = analogRead(joyY);
  bool pressed = digitalRead(joySW) == LOW;

  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(",");
  Serial.println(pressed ? "1" : "0");

  delay(100);
}
