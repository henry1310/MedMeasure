#include <ArduinoJson.h>

const int LED_PIN = 9;
int brightnessPWM = 128;  // Default 50% duty cycle

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  analogWrite(LED_PIN, brightnessPWM);
}

void loop() {
  // Read incoming serial commands from Python.
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    if (cmd.startsWith("SET_BRIGHTNESS:")) {
      brightnessPWM = cmd.substring(15).toInt();
      analogWrite(LED_PIN, constrain(brightnessPWM, 0, 255));
    }
  }

  // Read telemetry (simulated analog photodiode lux value).
  int rawADC = analogRead(A0);
  float estimatedLux = (rawADC / 1023.0) * 1000.0;

  // Stream a JSON packet over UART.
  StaticJsonDocument<128> doc;
  doc["lux"] = estimatedLux;
  doc["pwm"] = brightnessPWM;
  doc["status"] = "OK";
  serializeJson(doc, Serial);
  Serial.println();

  delay(200);  // 5 Hz telemetry stream
}
