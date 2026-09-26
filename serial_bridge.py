"""Serial interface for MedMeasure's optional Arduino telemetry payload."""

import json
import os
import random
from importlib.util import find_spec

if find_spec("serial") is not None:
    import serial
    import serial.tools.list_ports
else:
    serial = None


class HardwareBridge:
    """Read MedMeasure telemetry and send brightness commands to an Arduino."""

    def __init__(self, baudrate=115200, port=None):
        self.ser = None
        self.connected = False
        self.baudrate = baudrate
        self.port = None
        self.find_hardware(port=port)

    def find_hardware(self, port=None):
        """Connect to an Arduino-compatible serial device.

        ``port`` (or the ``MEDMEASURE_SERIAL_PORT`` environment variable) is
        useful when an adapter does not advertise an Arduino/CH340/FTDI name.
        Without it, common Arduino-compatible USB serial adapters are detected
        automatically.
        """
        self.close()
        if serial is None:
            return
        requested_port = port or os.getenv("MEDMEASURE_SERIAL_PORT")
        ports = serial.tools.list_ports.comports()
        candidates = [
            device for device in ports
            if requested_port is None or device.device == requested_port
        ]
        if requested_port and not candidates:
            # A manually supplied device may not be listed yet (for example in
            # a container), but pyserial can still attempt to open it.
            candidates = [type("Port", (), {"device": requested_port})()]

        for device in candidates:
            description = " ".join(
                str(getattr(device, field, "") or "")
                for field in ("description", "manufacturer", "hwid")
            ).lower()
            if (
                requested_port
                or any(name in description for name in ("arduino", "ch340", "ftdi"))
            ):
                try:
                    self.ser = serial.Serial(device.device, self.baudrate, timeout=1)
                    self.connected = True
                    self.port = device.device
                    return
                except serial.SerialException:
                    pass
        self.connected = False
        self.port = None

    def close(self):
        """Close the serial device, if one is currently open."""
        if self.ser:
            try:
                self.ser.close()
            except serial.SerialException:
                pass
        self.ser = None
        self.connected = False
        self.port = None

    def read_telemetry(self):
        """Return the next hardware packet, or mock telemetry when offline."""
        if self.connected and self.ser and self.ser.in_waiting:
            try:
                line = self.ser.readline().decode("utf-8").strip()
                packet = json.loads(line)
                if not isinstance(packet, dict):
                    raise json.JSONDecodeError("Telemetry must be an object", line, 0)
                packet.setdefault("status", "OK")
                return packet
            except (UnicodeDecodeError, json.JSONDecodeError, serial.SerialException):
                self.close()

        # Fallback mock telemetry when no physical microcontroller is plugged in.
        return {
            "lux": round(random.uniform(320.0, 450.0), 1),
            "pwm": 128,
            "status": "MOCK_HARDWARE",
        }

    def set_brightness(self, pwm_value):
        """Send a PWM brightness command to the connected microcontroller."""
        pwm_value = int(pwm_value)
        if not 0 <= pwm_value <= 255:
            raise ValueError("Brightness must be between 0 and 255.")
        if self.connected and self.ser:
            cmd = f"SET_BRIGHTNESS:{pwm_value}\n"
            try:
                self.ser.write(cmd.encode("utf-8"))
            except serial.SerialException:
                self.close()
