
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import date

st.set_page_config(page_title="MedMeasure", page_icon="🩺", layout="wide")

# ---------- Computer vision ----------

def read_image(uploaded):
    data = uploaded.getvalue()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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

    # Red/pink target
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([165, 70, 50])
    upper_red2 = np.array([179, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask |= cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )

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

def analyze(img_bgr, reference_width_cm):
    ref, target, _, _ = detect_reference_and_target(img_bgr)

    if ref is None:
        return None, "Could not find the blue reference marker."
    if target is None:
        return None, "Could not find the red/pink target."

    # Use the average of the reference rectangle's width/height as its
    # pixel size. This makes the demo tolerant of portrait/landscape marker.
    rw, rh = cv2.minAreaRect(ref)[1]
    ref_px = max(rw, rh)

    if ref_px <= 0:
        return None, "Reference marker was detected but has invalid size."

    cm_per_px = reference_width_cm / ref_px

    target_area_px = cv2.contourArea(target)
    area_cm2 = target_area_px * (cm_per_px ** 2)

    x, y, w, h = cv2.boundingRect(target)
    length_cm = max(w, h) * cm_per_px
    width_cm = min(w, h) * cm_per_px

    perimeter_px = cv2.arcLength(target, True)
    perimeter_cm = perimeter_px * cm_per_px

    annotated = img_bgr.copy()
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
    }, None

# ---------- UI ----------

st.title("🩺 MedMeasure")
st.subheader("Computer vision for quantitative medical measurements")

st.info(
    "Hackathon prototype: this demonstration measures a simulated target, "
    "not a real patient wound. It is not a diagnostic or clinical device."
)

with st.sidebar:
    st.header("Calibration")
    reference_width = st.number_input(
        "Blue reference marker width (cm)",
        min_value=0.1,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )
    st.caption(
        "Place a blue rectangle/card of this known width next to the "
        "simulated target."
    )

    st.header("Demo data")
    previous_area = st.number_input(
        "Previous measurement (cm²)",
        min_value=0.0,
        value=12.0,
        step=0.1,
    )

camera = st.camera_input(
    "Take a measurement photo",
    resolution="1080p",
)

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

    result, error = analyze(img_bgr, reference_width)

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
            st.image(result["annotated"], caption="Computer-vision result")

        with right:
            st.markdown("### Quantitative measurement")

            c1, c2 = st.columns(2)
            c1.metric("Area", f'{result["area_cm2"]:.2f} cm²')
            c2.metric("Length", f'{result["length_cm"]:.2f} cm')

            c3, c4 = st.columns(2)
            c3.metric("Width", f'{result["width_cm"]:.2f} cm')
            c4.metric("Perimeter", f'{result["perimeter_cm"]:.2f} cm')

            if previous_area > 0:
                change = (
                    (result["area_cm2"] - previous_area)
                    / previous_area
                    * 100
                )
                st.metric(
                    "Change from previous measurement",
                    f"{change:+.1f}%",
                    delta=f"{change:+.1f}%",
                )

        st.markdown("---")
        st.markdown("### Measurement history")

        history = np.array(
            [previous_area, result["area_cm2"]],
            dtype=float,
        )

        st.line_chart(
            {
                "Area (cm²)": history
            },
            x_label="Visit",
            y_label="Area (cm²)",
        )

        st.caption(
            f"Visit 1: {previous_area:.2f} cm²  →  "
            f"Current: {result['area_cm2']:.2f} cm²"
        )
