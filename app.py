
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import date
import csv
from html import escape
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile

from serial_bridge import HardwareBridge

st.set_page_config(page_title="MedMeasure", page_icon="🩺", layout="wide")


# ---------- Measurement history ----------

def add_history_measurement(history, measurement_date, area_cm2):
    """Return a date-sorted copy of ``history`` with one saved measurement.

    Keeping this small piece of state handling separate from the Streamlit UI
    makes it clear that every plotted value is an actual saved measurement,
    rather than a fixed comparison value.
    """
    entry = {
        "Date": measurement_date.isoformat(),
        "Wound size (cm²)": round(float(area_cm2), 2),
    }
    return sorted([*history, entry], key=lambda item: item["Date"])


def history_chart_data(history):
    """Convert saved entries into the column-oriented data used by Streamlit."""
    return {
        "Date": [entry["Date"] for entry in history],
        "Wound size (cm²)": [entry["Wound size (cm²)"] for entry in history],
    }


def remove_history_measurements(history, indices_to_remove):
    """Return history without the zero-based rows selected for removal."""
    removal_indices = set(indices_to_remove)
    return [
        entry for index, entry in enumerate(history)
        if index not in removal_indices
    ]


def history_export_zip(history):
    """Build a ZIP containing the saved measurements and a portable SVG trend."""
    csv_buffer = StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=("Date", "Wound size (cm²)"))
    writer.writeheader()
    writer.writerows(history)

    width, height, padding = 760, 360, 55
    values = [entry["Wound size (cm²)"] for entry in history]
    minimum, maximum = min(values), max(values)
    value_range = maximum - minimum or 1
    plot_width = width - (padding * 2)
    plot_height = height - (padding * 2)
    points = []
    for index, value in enumerate(values):
        x = padding + (plot_width * index / max(len(values) - 1, 1))
        y = padding + plot_height * (1 - ((value - minimum) / value_range))
        points.append(f"{x:.1f},{y:.1f}")
    labels = "".join(
        f'<text x="{padding + (plot_width * index / max(len(history) - 1, 1)):.1f}" '
        f'y="{height - 18}" text-anchor="middle">{escape(entry["Date"])}</text>'
        for index, entry in enumerate(history)
    )
    dots = "".join(
        f'<circle class="point" cx="{point.split(",")[0]}" '
        f'cy="{point.split(",")[1]}" r="4"/>'
        for point in points
    )
    graph = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>text {{ font: 12px sans-serif; fill: #334155; }} .title {{ font-size: 18px; font-weight: bold; }} .axis {{ stroke: #94a3b8; }} .trend {{ fill: none; stroke: #0f766e; stroke-width: 3; }} .point {{ fill: #0f766e; }}</style>
  <rect width="100%" height="100%" fill="white"/>
  <text class="title" x="{padding}" y="30">Wound size history</text>
  <line class="axis" x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}"/><line class="axis" x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}"/>
  <text x="8" y="{padding + 4}">{maximum:.2f} cm²</text><text x="8" y="{height - padding}">{minimum:.2f} cm²</text>
  <polyline class="trend" points="{' '.join(points)}"/>{dots}
  {labels}
</svg>'''

    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr("measurement_history.csv", csv_buffer.getvalue())
        zip_file.writestr("measurement_history_graph.svg", graph)
    return archive.getvalue()


# ---------- Computer vision ----------

def read_image(uploaded):
    data = uploaded.getvalue()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def target_mask(img_bgr):
    """Segment a red/pink simulated target while tolerating normal lighting.

    Hue alone is brittle: shadows reduce saturation and a warm background can
    have a red hue.  This mask combines hue/saturation with the Lab ``a``
    channel (red-to-green) and channel redness.  Those chromatic cues remain
    useful when the brightness changes across a photograph.  A small, scaled
    cleanup is deliberately used instead of smoothing the contour so that
    irregular target edges are retained for the perimeter measurement.

    This remains a demonstration-only colour segmentation method; it is not
    a clinical wound-segmentation model.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    blue, green, red = cv2.split(img_bgr.astype(np.int16))
    hue, saturation, _ = cv2.split(hsv)

    # OpenCV hue wraps red around both ends of its 0--179 hue scale.  The
    # broader range includes orange-red, bruised red, and pale pink targets.
    warm_hue = (hue <= 25) | (hue >= 155)
    red_green = lab[:, :, 1]
    # Absolute red-channel differences are misleading on skin: a bright tan
    # pixel can have a large R-G difference simply because it is bright.  Use
    # a brightness-normalized red dominance instead.  Healthy skin generally
    # remains low on this measure, while blood-red and pink wound tissue stays
    # high even in a shadow.
    intensity = np.maximum(red + green + blue, 1)
    red_dominance = 255.0 * (red - np.maximum(green, blue)) / intensity

    # Skin can be warm, saturated, and redder than green, so those absolute
    # colour thresholds alone are not a reliable wound boundary. Compare each
    # pixel with its local neighbourhood in both normalised red dominance and
    # Lab red chroma. This retains a subtle pink region when it is visibly
    # redder than the surrounding skin, while rejecting a broad, uniformly
    # warm arm region. Very strongly red pixels are retained even in the
    # centre of a large target, where the local comparison is naturally flat.
    shortest_side = min(img_bgr.shape[:2])
    local_sigma = max(5.0, shortest_side / 35.0)
    local_dominance = cv2.GaussianBlur(
        red_dominance, (0, 0), sigmaX=local_sigma
    )
    local_red_green = cv2.GaussianBlur(
        red_green.astype(np.float32), (0, 0), sigmaX=local_sigma
    )
    strong_red = (red_dominance >= 40) & (red_green >= 155)
    locally_redder = (
        (red_dominance >= 28)
        & (red_green >= 145)
        & (red_dominance - local_dominance >= 4)
        & (red_green - local_red_green >= 3)
    )
    target_pixels = (
        warm_hue
        & (saturation >= 25)
        & (strong_red | locally_redder)
    )
    mask = np.where(target_pixels, 255, 0).astype(np.uint8)

    # Remove isolated camera noise, bridge small lighting gaps, and fill
    # interior holes without erasing genuinely irregular outer boundaries.
    open_size = max(3, (shortest_side // 500) * 2 + 1)
    close_size = max(5, (shortest_side // 300) * 2 + 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
    )
    return mask

def detect_reference_and_target(img_bgr):
    """
    Hackathon MVP:
    - Reference marker: blue rectangle/card.
    - Target: red/pink simulated wound.

    The known reference width is supplied by the user in centimeters.
    This is intentionally a controlled prototype rather than a clinical
    wound-diagnosis system.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Blue reference marker
    lower_blue = np.array([90, 80, 60])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )

    # Red/pink target, including uneven illumination and shadowed portions.
    red_mask = target_mask(img_bgr)

    def largest_contour(mask, min_area):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not candidates:
            return None
        return max(candidates, key=cv2.contourArea)

    ref_contour = largest_contour(blue_mask, 300)
    target_contour = largest_contour(red_mask, 150)

    return ref_contour, target_contour, blue_mask, red_mask

def order_corners(corners):
    """Return quadrilateral corners in top-left, top-right, bottom-right,
    bottom-left order.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = corners.sum(axis=1)
    differences = np.diff(corners, axis=1).flatten()

    ordered[0] = corners[np.argmin(sums)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[1] = corners[np.argmin(differences)]
    ordered[3] = corners[np.argmax(differences)]
    return ordered

def reference_corners(ref_contour):
    """Find the four physical corners of the blue reference marker."""
    perimeter = cv2.arcLength(ref_contour, True)
    approximation = cv2.approxPolyDP(ref_contour, 0.02 * perimeter, True)
    if len(approximation) != 4:
        return None
    return order_corners(approximation.reshape(4, 2).astype(np.float32))

def correct_perspective(img_bgr, ref_contour, reference_width_cm,
                        reference_height_cm):
    """Warp the image so the blue reference marker is viewed face-on."""
    pixels_per_cm = 100
    marker_width_px = round(reference_width_cm * pixels_per_cm)
    marker_height_px = round(reference_height_cm * pixels_per_cm)

    if marker_width_px <= 0 or marker_height_px <= 0:
        return None, "Reference marker dimensions must be positive."

    source_corners = reference_corners(ref_contour)
    if source_corners is None:
        return None, "Could not identify four corners of the blue reference marker."
    marker_corners = np.array(
        [
            [0, 0],
            [marker_width_px - 1, 0],
            [marker_width_px - 1, marker_height_px - 1],
            [0, marker_height_px - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_corners, marker_corners)

    image_height, image_width = img_bgr.shape[:2]
    image_corners = np.array(
        [[[0, 0], [image_width - 1, 0],
          [image_width - 1, image_height - 1], [0, image_height - 1]]],
        dtype=np.float32,
    )
    warped_corners = cv2.perspectiveTransform(image_corners, transform)[0]
    minimum = np.floor(warped_corners.min(axis=0)).astype(int)
    maximum = np.ceil(warped_corners.max(axis=0)).astype(int)

    translation = np.array(
        [[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]],
        dtype=np.float32,
    )
    output_width, output_height = maximum - minimum + 1
    corrected = cv2.warpPerspective(
        img_bgr,
        translation @ transform,
        (int(output_width), int(output_height)),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    return corrected, None

def analyze(img_bgr, reference_width_cm, reference_height_cm):
    ref, _, _, _ = detect_reference_and_target(img_bgr)

    if ref is None:
        return None, "Could not find the blue reference marker."

    corrected, error = correct_perspective(
        img_bgr, ref, reference_width_cm, reference_height_cm
    )
    if error:
        return None, error

    ref, target, _, segmentation_mask = detect_reference_and_target(corrected)
    if target is None:
        return None, "Could not find the red/pink target."

    # The perspective warp maps the marker to 100 pixels per known centimeter.
    cm_per_px = 0.01

    target_area_px = cv2.contourArea(target)
    area_cm2 = target_area_px * (cm_per_px ** 2)

    x, y, w, h = cv2.boundingRect(target)
    length_cm = max(w, h) * cm_per_px
    width_cm = min(w, h) * cm_per_px

    perimeter_px = cv2.arcLength(target, True)
    perimeter_cm = perimeter_px * cm_per_px

    annotated = corrected.copy()
    if ref is not None:
        cv2.drawContours(annotated, [ref], -1, (255, 255, 0), 3)
    cv2.drawContours(annotated, [target], -1, (0, 255, 0), 3)
    cv2.rectangle(
        annotated, (x, y), (x + w, y + h), (0, 255, 255), 2
    )

    cv2.putText(
        annotated,
        f"Area: {area_cm2:.2f} cm^2",
        (max(10, x), max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return {
        "area_cm2": area_cm2,
        "length_cm": length_cm,
        "width_cm": width_cm,
        "perimeter_cm": perimeter_cm,
        "annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "segmentation_mask": segmentation_mask,
    }, None

# ---------- UI ----------

st.title("🩺 MedMeasure")
st.subheader("Computer vision for quantitative medical measurements")

if "measurement_history" not in st.session_state:
    st.session_state.measurement_history = []
if "history_editor_version" not in st.session_state:
    st.session_state.history_editor_version = 0
if "hardware_bridge" not in st.session_state:
    st.session_state.hardware_bridge = HardwareBridge()
if "hardware_brightness" not in st.session_state:
    st.session_state.hardware_brightness = 128
if "camera_enabled" not in st.session_state:
    st.session_state.camera_enabled = True

st.info(
    "Hackathon prototype: this demonstration measures a simulated target, "
    "not a real patient wound. It is not a diagnostic or clinical device."
)

with st.sidebar:
    st.header("Arduino integration")
    bridge = st.session_state.hardware_bridge
    if bridge.connected:
        st.success(f"Connected to Arduino on `{bridge.port}`")
    else:
        st.warning("Arduino not connected — showing simulated telemetry.")
    if st.button("Reconnect Arduino"):
        bridge.find_hardware()
        st.rerun()

    telemetry = bridge.read_telemetry()
    telemetry_status = telemetry.get("status", "UNKNOWN")
    telemetry_lux = telemetry.get("lux")
    if telemetry_lux is not None:
        st.metric("Ambient light", f"{float(telemetry_lux):.1f} lux")
    else:
        st.caption("No light telemetry received yet.")
    st.caption(f"Telemetry status: {telemetry_status}")

    brightness = st.slider(
        "Arduino LED brightness",
        min_value=0,
        max_value=255,
        key="hardware_brightness",
        help="Sends SET_BRIGHTNESS to the Arduino PWM output when applied.",
    )
    if st.button("Apply LED brightness"):
        try:
            bridge.set_brightness(brightness)
            if bridge.connected:
                st.success("Brightness command sent to Arduino.")
            else:
                st.info("Arduino is offline; brightness was not sent.")
        except ValueError as error:
            st.error(str(error))

    st.divider()
    st.header("Calibration")
    reference_width = st.number_input(
        "Blue reference marker width (cm)",
        min_value=0.1,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )
    reference_height = st.number_input(
        "Blue reference marker height (cm)",
        min_value=0.1,
        max_value=20.0,
        value=1.0,
        step=0.1,
    )
    st.caption(
        "Enter both dimensions of the blue rectangle/card placed next to "
        "the simulated target."
    )

    st.header("Add past measurement")
    historical_date = st.date_input(
        "Measurement date",
        value=date.today(),
        key="historical_measurement_date",
    )
    historical_area = st.number_input(
        "Wound size (cm²)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="historical_measurement_area",
    )
    if st.button("Add to history"):
        if historical_area <= 0:
            st.sidebar.warning("Enter a wound size greater than 0 cm².")
        else:
            st.session_state.measurement_history = add_history_measurement(
                st.session_state.measurement_history,
                historical_date,
                historical_area,
            )
            st.sidebar.success("Measurement added to history.")

camera = None
camera_toggle_label = (
    "Turn camera off" if st.session_state.camera_enabled else "Turn camera on"
)
if st.button(
    camera_toggle_label,
    help="Stops or enables the browser camera for this session.",
):
    st.session_state.camera_enabled = not st.session_state.camera_enabled
    st.rerun()

if st.session_state.camera_enabled:
    camera = st.camera_input("Take a measurement photo")
else:
    st.info("Camera is off. Turn it on when you are ready to take another photo.")

uploaded = st.file_uploader(
    "Or upload a test image",
    type=["jpg", "jpeg", "png"],
)

source = camera if camera is not None else uploaded

if source is None:
    st.markdown(
        """
### How to demonstrate it

1. Put a **blue reference rectangle** beside a simulated wound.
2. Make the target red/pink so the prototype can segment it.
3. Hold the paper reasonably flat and take a photo.
4. MedMeasure detects both objects and converts pixels into centimeters.
        """
    )

    st.markdown("### Prototype pipeline")
    st.code(
        "Webcam → reference detection → target segmentation → "
        "pixel-to-cm calibration → measurement → trend",
        language="text",
    )

else:
    img_bgr = read_image(source)

    result, error = analyze(img_bgr, reference_width, reference_height)

    if error:
        st.error(error)
        st.warning(
            "For the demo, use a clearly visible blue rectangular marker "
            "and a red/pink target on a light background."
        )
    else:
        st.success("Measurement completed.")

        left, right = st.columns(2)

        with left:
            st.image(
                result["annotated"],
                caption="Perspective-corrected computer-vision result",
            )
            st.image(
                result["segmentation_mask"],
                caption="Red/pink segmentation mask (white = measured target)",
                clamp=True,
            )

        with right:
            st.markdown("### Quantitative measurement")

            c1, c2 = st.columns(2)
            c1.metric("Area", f'{result["area_cm2"]:.2f} cm²')
            c2.metric("Length", f'{result["length_cm"]:.2f} cm')

            c3, c4 = st.columns(2)
            c3.metric("Width", f'{result["width_cm"]:.2f} cm')
            c4.metric("Perimeter", f'{result["perimeter_cm"]:.2f} cm')

            measurement_date = st.date_input(
                "Date for this measurement",
                value=date.today(),
                key="current_measurement_date",
            )
            if st.button("Save current measurement", type="primary"):
                st.session_state.measurement_history = add_history_measurement(
                    st.session_state.measurement_history,
                    measurement_date,
                    result["area_cm2"],
                )
                st.success("Current measurement saved to history.")


st.markdown("---")
st.markdown("### Measurement history")

history = st.session_state.measurement_history
if history:
    st.caption(
        "Select unwanted rows, then remove them. The trend updates immediately."
    )
    editable_history = {
        "Date": [entry["Date"] for entry in history],
        "Wound size (cm²)": [entry["Wound size (cm²)"] for entry in history],
        "Remove": [False] * len(history),
    }
    edited_history = st.data_editor(
        editable_history,
        column_order=("Date", "Wound size (cm²)", "Remove"),
        column_config={
            "Date": st.column_config.TextColumn("Date"),
            "Wound size (cm²)": st.column_config.NumberColumn(
                "Wound size (cm²)", format="%.2f cm²"
            ),
            "Remove": st.column_config.CheckboxColumn(
                "Remove", help="Select this measurement for removal."
            ),
        },
        disabled=("Date", "Wound size (cm²)"),
        hide_index=True,
        key=f"history_removal_editor_{st.session_state.history_editor_version}"
    )
    selected_indices = [
        index for index, selected in enumerate(edited_history["Remove"])
        if selected
    ]
    if selected_indices and st.button(
        f"Remove {len(selected_indices)} selected measurement(s)",
        type="secondary",
    ):
        st.session_state.measurement_history = remove_history_measurements(
            history, selected_indices
        )
        # A new key clears selections so an old checkbox cannot remove a new row.
        st.session_state.history_editor_version += 1
        st.rerun()

    st.download_button(
        "Download measurement history and graph",
        data=history_export_zip(history),
        file_name="medmeasure_history.zip",
        mime="application/zip",
        help="Downloads a ZIP with the measurements CSV and trend graph SVG.",
    )

    st.line_chart(
        history_chart_data(st.session_state.measurement_history),
        x="Date",
        y="Wound size (cm²)",
        x_label="Date",
        y_label="Wound size (cm²)",
    )
else:
    st.info(
        "No saved measurements yet. Save a photo result or add a past "
        "measurement from the sidebar to build the trend."
    )
