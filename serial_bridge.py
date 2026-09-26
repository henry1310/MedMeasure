"""Serial interface for MedMeasure's optional Arduino telemetry payload."""

import json
import random

import serial
import serial.tools.list_ports


class HardwareBridge:
    """Read MedMeasure telemetry and send brightness commands to an Arduino."""

    def __init__(self, baudrate=115200):
        self.ser = None
        self.connected = False
        self.find_hardware(baudrate)

    def find_hardware(self, baudrate):
        """Connect to the first detected Arduino-compatible serial device."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if (
                "Arduino" in port.description
                or "CH340" in port.description
                or "FTDI" in port.description
            ):
                try:
                    self.ser = serial.Serial(port.device, baudrate, timeout=1)
                    self.connected = True
                    return
                except serial.SerialException:
                    pass
        self.connected = False

    def read_telemetry(self):
        """Return the next hardware packet, or mock telemetry when offline."""
        if self.connected and self.ser and self.ser.in_waiting:
            try:
                line = self.ser.readline().decode("utf-8").strip()
                return json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError, serial.SerialException):
                pass

        # Fallback mock telemetry when no physical microcontroller is plugged in.
        return {
            "lux": round(random.uniform(320.0, 450.0), 1),
            "pwm": 128,
            "status": "MOCK_HARDWARE",
        }

    def set_brightness(self, pwm_value):
        """Send a PWM brightness command to the connected microcontroller."""
        if self.connected and self.ser:
            cmd = f"SET_BRIGHTNESS:{pwm_value}\n"
            self.ser.write(cmd.encode("utf-8"))
