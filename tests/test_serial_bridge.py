"""Unit tests for the Arduino serial bridge without requiring hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import serial_bridge


@pytest.fixture
def fake_serial(monkeypatch):
    """Install a minimal pyserial substitute so tests do not need hardware."""
    fake = SimpleNamespace(
        Serial=Mock(),
        SerialException=OSError,
        tools=SimpleNamespace(list_ports=SimpleNamespace(comports=Mock())),
    )
    monkeypatch.setattr(serial_bridge, "serial", fake)
    return fake


def test_bridge_connects_to_detected_adapter_and_reads_json_telemetry(fake_serial):
    port = Mock(device="/dev/ttyACM0", description="Arduino Uno", manufacturer="")
    serial_device = Mock(in_waiting=1)
    serial_device.readline.return_value = b'{"lux": 410.5, "pwm": 64}\n'

    fake_serial.tools.list_ports.comports.return_value = [port]
    fake_serial.Serial.return_value = serial_device
    bridge = serial_bridge.HardwareBridge()
    telemetry = bridge.read_telemetry()

    fake_serial.Serial.assert_called_once_with("/dev/ttyACM0", 115200, timeout=1)
    assert bridge.connected
    assert bridge.port == "/dev/ttyACM0"
    assert telemetry == {"lux": 410.5, "pwm": 64, "status": "OK"}


def test_bridge_uses_explicit_port_and_validates_brightness(fake_serial):
    serial_device = Mock()
    fake_serial.tools.list_ports.comports.return_value = []
    fake_serial.Serial.return_value = serial_device
    bridge = serial_bridge.HardwareBridge(port="/dev/custom-arduino")

    bridge.set_brightness(200)
    serial_device.write.assert_called_once_with(b"SET_BRIGHTNESS:200\n")
    with pytest.raises(ValueError, match="between 0 and 255"):
        bridge.set_brightness(256)


def test_bridge_returns_mock_telemetry_when_no_adapter_is_available(fake_serial):
    fake_serial.tools.list_ports.comports.return_value = []
    bridge = serial_bridge.HardwareBridge()

    telemetry = bridge.read_telemetry()
    assert not bridge.connected
    assert telemetry["status"] == "MOCK_HARDWARE"
    assert 320.0 <= telemetry["lux"] <= 450.0
