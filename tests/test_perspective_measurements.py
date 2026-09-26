"""Synthetic perspective-regression tests for MedMeasure's CV pipeline.

The images are deliberately generated rather than captured so the marker and
target dimensions are known exactly, while each parametrized scene simulates a
different camera tilt by projecting one flat physical plane with a homography.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

# Pytest's import mode may put ``tests/`` first without adding the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (
    analyze,
    correct_perspective,
    detect_reference_and_target,
    reference_corners,
    target_mask,
)


PIXELS_PER_CM = 100
REFERENCE_SIZE_CM = (2.0, 1.0)
TARGET_SIZE_CM = (3.0, 1.5)
EXPECTED_MEASUREMENTS = {
    "area_cm2": TARGET_SIZE_CM[0] * TARGET_SIZE_CM[1],
    "length_cm": max(TARGET_SIZE_CM),
    "width_cm": min(TARGET_SIZE_CM),
    "perimeter_cm": 2 * sum(TARGET_SIZE_CM),
}

# Tolerances include contour rasterization and one-pixel bounding-box effects.
RELATIVE_TOLERANCES = {
    "area_cm2": 0.09,
    "length_cm": 0.04,
    "width_cm": 0.06,
    "perimeter_cm": 0.06,
}


def _scene_at_tilt(destination_corners):
    """Return a projected test scene and the expected marker image corners."""
    plane_width, plane_height = 800, 500
    plane = np.full((plane_height, plane_width, 3), 255, dtype=np.uint8)

    reference_width_px = int(REFERENCE_SIZE_CM[0] * PIXELS_PER_CM)
    reference_height_px = int(REFERENCE_SIZE_CM[1] * PIXELS_PER_CM)
    target_width_px = int(TARGET_SIZE_CM[0] * PIXELS_PER_CM)
    target_height_px = int(TARGET_SIZE_CM[1] * PIXELS_PER_CM)

    # The objects have a clear white margin and do not overlap after projection.
    reference_top_left = (80, 80)
    target_top_left = (360, 230)
    cv2.rectangle(
        plane,
        reference_top_left,
        (reference_top_left[0] + reference_width_px,
         reference_top_left[1] + reference_height_px),
        (255, 0, 0),
        thickness=-1,
    )
    cv2.rectangle(
        plane,
        target_top_left,
        (target_top_left[0] + target_width_px,
         target_top_left[1] + target_height_px),
        (0, 0, 255),
        thickness=-1,
    )

    source_corners = np.float32(
        [[0, 0], [plane_width - 1, 0],
         [plane_width - 1, plane_height - 1], [0, plane_height - 1]]
    )
    transform = cv2.getPerspectiveTransform(source_corners, np.float32(destination_corners))
    projected = cv2.warpPerspective(
        plane, transform, (1000, 800), borderValue=(255, 255, 255)
    )

    marker_corners = np.float32(
        [[reference_top_left[0], reference_top_left[1]],
         [reference_top_left[0] + reference_width_px, reference_top_left[1]],
         [reference_top_left[0] + reference_width_px,
          reference_top_left[1] + reference_height_px],
         [reference_top_left[0], reference_top_left[1] + reference_height_px]]
    )
    return projected, cv2.perspectiveTransform(marker_corners[None, :, :], transform)[0]


TILT_SCENES = [
    pytest.param([[80, 100], [910, 70], [850, 650], [130, 700]], id="mild-tilt"),
    pytest.param([[180, 50], [820, 150], [930, 690], [80, 620]], id="roll-and-tilt"),
    pytest.param([[100, 210], [890, 80], [760, 720], [210, 610]], id="steep-tilt"),
]


@pytest.mark.parametrize("destination_corners", TILT_SCENES)
def test_reference_corners_recovers_projected_marker(destination_corners):
    image, expected_corners = _scene_at_tilt(destination_corners)
    reference, _, _, _ = detect_reference_and_target(image)

    assert reference is not None
    recovered_corners = reference_corners(reference)

    assert recovered_corners is not None
    np.testing.assert_allclose(recovered_corners, expected_corners, atol=4.0)


@pytest.mark.parametrize("destination_corners", TILT_SCENES)
def test_correct_perspective_restores_reference_marker(destination_corners):
    image, _ = _scene_at_tilt(destination_corners)
    reference, _, _, _ = detect_reference_and_target(image)

    corrected, error = correct_perspective(image, reference, *REFERENCE_SIZE_CM)

    assert error is None
    corrected_reference, _, _, _ = detect_reference_and_target(corrected)
    x, y, width_px, height_px = cv2.boundingRect(corrected_reference)
    assert width_px == pytest.approx(REFERENCE_SIZE_CM[0] * PIXELS_PER_CM, abs=3)
    assert height_px == pytest.approx(REFERENCE_SIZE_CM[1] * PIXELS_PER_CM, abs=3)


@pytest.mark.parametrize("destination_corners", TILT_SCENES)
def test_analyze_reports_physical_measurements_across_tilts(destination_corners):
    image, _ = _scene_at_tilt(destination_corners)

    measurements, error = analyze(image, *REFERENCE_SIZE_CM)

    assert error is None
    for name, expected in EXPECTED_MEASUREMENTS.items():
        assert measurements[name] == pytest.approx(
            expected, rel=RELATIVE_TOLERANCES[name]
        ), name


def test_generated_tilt_fixture_can_be_written_for_visual_inspection(tmp_path):
    """Keep a writable image fixture option for debugging failed regressions."""
    image, _ = _scene_at_tilt(TILT_SCENES[0].values[0])
    output = Path(tmp_path) / "synthetic_mild_tilt.png"

    assert cv2.imwrite(str(output), image)
    assert output.exists()


def test_target_segmentation_keeps_irregular_red_pink_target_out_of_skin_background():
    """Colour segmentation should include shade changes without selecting skin."""
    # This warm, moderately saturated skin tone passes a simple HSV red/pink
    # range and has a substantial *absolute* red-channel difference.  It must
    # not become the measured target.
    image = np.full((360, 520, 3), (90, 130, 180), dtype=np.uint8)
    image = np.full((360, 520, 3), (125, 170, 205), dtype=np.uint8)
    target = np.array(
        [[170, 80], [285, 66], [351, 124], [326, 201], [355, 260],
         [246, 287], [163, 238], [137, 158]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [target], (90, 80, 185))
    # These overlapping patches simulate pink tissue, a dark shadow, and an
    # unevenly illuminated red area without changing the outer boundary.
    cv2.ellipse(image, (220, 145), (58, 45), 0, 0, 360, (150, 145, 235), -1)
    cv2.ellipse(image, (290, 205), (45, 42), 0, 0, 360, (45, 35, 100), -1)
    cv2.ellipse(image, (235, 220), (38, 32), 0, 0, 360, (70, 65, 155), -1)

    mask = target_mask(image)
    reference, detected_target, _, segmented = detect_reference_and_target(image)

    assert reference is None  # This fixture has no blue calibration marker.
    assert detected_target is not None
    assert np.array_equal(mask, segmented)
    assert mask[145, 220] == 255  # pale pink
    assert mask[205, 290] == 255  # shadowed red
    assert mask[20, 20] == 0  # skin-like background
    assert cv2.countNonZero(mask) / mask.size < 0.25

    overlap = cv2.countNonZero(
        cv2.bitwise_and(mask, cv2.fillPoly(np.zeros(mask.shape, np.uint8), [target], 255))
    )
    target_area = cv2.contourArea(target)
    assert overlap / target_area > 0.75
