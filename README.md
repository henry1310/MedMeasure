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

For every simulated tilt, the tests require `reference_corners` to locate the
blue marker within 4 pixels of its projected corners and require
`correct_perspective` to restore the marker to within 3 pixels of its expected
200 px × 100 px size. `analyze` must recover the target's 4.50 cm² area,
3.00 cm length, 1.50 cm width, and 9.00 cm perimeter. The acceptable relative
errors are 9% for area, 4% for length, and 6% for width and perimeter; these
allow for normal rasterization and contour-boundary effects while detecting
meaningful calibration regressions.
