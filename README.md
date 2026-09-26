## Perspective-measurement test setup

The automated regression suite builds a white, flat synthetic test plane at
100 pixels per centimeter. It places a **2.0 cm × 1.0 cm blue reference
marker** and a **3.0 cm × 1.5 cm red target** on that plane, then projects the
plane through three known homographies that simulate mild, rolled, and steep
camera tilts. The generated image can also be written to a temporary PNG when
a failed test needs visual inspection.

Run the suite with:

```bash
pytest
```

## Optional Arduino telemetry

The optional hardware payload lives in
[`firmware/medmeasure_payload.ino`](firmware/medmeasure_payload.ino). Install
the ArduinoJson library, upload the sketch to an Arduino-compatible board, and
connect its simulated photodiode to `A0` and PWM lighting to pin `9`. The
sketch streams JSON telemetry at 115200 baud and accepts commands such as
`SET_BRIGHTNESS:128` over serial.

`serial_bridge.py` provides the Python `HardwareBridge` interface. It detects
Arduino, CH340, and FTDI serial adapters, returns parsed telemetry when a board
is connected, and returns mock telemetry when hardware is unavailable.

## Target segmentation

The simulated red/pink target is segmented using a combined HSV and CIE Lab
colour mask rather than a single red hue range.  It accepts red, pink, and
shadowed-red pixels while requiring Lab red chroma plus *brightness-normalized*
red dominance. It additionally compares both measures against a
resolution-scaled local neighbourhood: subtle red/pink tissue is included when
it contrasts with nearby skin, while a broad, uniformly warm skin region is
not treated as a target. Strongly red pixels are retained inside a large target
even when its centre has little local contrast. Small resolution-scaled
morphological cleanup removes noise and lighting gaps without smoothing away
the target's irregular outer boundary.

The Streamlit result includes a black-and-white segmentation-mask preview:
white pixels are included in the measurement.  This is an inspection aid for
the hackathon simulation only, not clinical validation or diagnosis.

For every simulated tilt, the tests require `reference_corners` to locate the
blue marker within 4 pixels of its projected corners and require
`correct_perspective` to restore the marker to within 3 pixels of its expected
200 px × 100 px size. `analyze` must recover the target's 4.50 cm² area,
3.00 cm length, 1.50 cm width, and 9.00 cm perimeter. The acceptable relative
errors are 9% for area, 4% for length, and 6% for width and perimeter; these
allow for normal rasterization and contour-boundary effects while detecting
meaningful calibration regressions.
