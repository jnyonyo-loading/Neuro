// === Arduino sketch: receive a voltage command over serial, output it, read back ===

const int DAC_PIN = A0;   // R4 Minima's DAC output pin (confirm this against your board's pinout)
const int READ_PIN = A1;  // use a different analog pin to read back, since A0 is now output

void setup() {
  Serial.begin(9600);
  analogWriteResolution(12); // R4's DAC supports higher resolution than the default 8-bit
}

void loop() {
  if (Serial.available() > 0) {
    float commandedVoltage = Serial.parseFloat(); // read a number sent from Python
    
    if (commandedVoltage >= 0 && commandedVoltage <= 5.0) {
      int dacValue = (commandedVoltage / 5.0) * 4095; // scale 0-5V to 12-bit DAC range
      analogWrite(DAC_PIN, dacValue);
      
      int raw = analogRead(READ_PIN);
      float measuredVoltage = raw * (5.0 / 1023.0);
      
      Serial.print("commanded:");
      Serial.print(commandedVoltage);
      Serial.print(",measured:");
      Serial.println(measuredVoltage);
    }
  }
}