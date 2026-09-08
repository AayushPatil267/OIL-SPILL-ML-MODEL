"""Streamlit dashboard for oil spill detection and drift visualization."""

import base64

import cv2
import folium
import numpy as np
import requests
import streamlit as st
from streamlit_folium import st_folium


API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Oil Spill Detection Dashboard",
    page_icon=":ocean:",
    layout="wide",
)

st.title("Oil Spill Detection Dashboard")
st.caption("Upload a satellite image to classify, segment, and map a possible oil spill.")

with st.sidebar:
    st.header("About")
    st.write(
        "This dashboard sends satellite imagery to the oil spill detection API. "
        "The service classifies the image, segments the detected spill, estimates "
        "its area and severity, and can project drift from local wind conditions."
    )
    st.info("The FastAPI service must be running at http://localhost:8000.")

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png", "tif", "tiff", "webp"],
)


def decode_mask(mask_image, target_shape):
    """Decode a base64 PNG mask and resize it to the original image dimensions."""
    try:
        mask_bytes = base64.b64decode(mask_image)
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    except (ValueError, TypeError):
        return None

    if mask is None:
        return None
    height, width = target_shape[:2]
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 127


def create_overlay(image_bgr, mask):
    """Blend red over spill pixels while preserving the original image elsewhere."""
    overlay = image_bgr.copy()
    red = np.array([0, 0, 255], dtype=np.float32)
    blended_pixels = image_bgr[mask].astype(np.float32) * 0.6 + red * 0.4
    overlay[mask] = blended_pixels.astype(np.uint8)
    return overlay


def display_severity(severity):
    """Display the severity label using the requested color coding."""
    colors = {"small": "green", "medium": "orange", "large": "red"}
    color = colors.get(str(severity).lower(), "gray")
    st.markdown(
        f'<p><strong>Severity:</strong> '
        f'<span style="color:{color}; font-weight:700;">{severity}</span></p>',
        unsafe_allow_html=True,
    )


def build_map(result):
    """Build a Folium map for the detected spill and predicted drift path."""
    location = result.get("spill_location")
    if not location:
        return None

    latitude = float(location["lat"])
    longitude = float(location["lon"])
    spill_map = folium.Map(location=[latitude, longitude], zoom_start=8)
    folium.Marker(
        [latitude, longitude],
        tooltip="Detected spill",
        icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(spill_map)

    drift_points = result.get("drift_prediction") or []
    path = []
    for point in drift_points:
        if isinstance(point, dict):
            path.append([float(point["lat"]), float(point["lon"])])
        else:
            path.append([float(point[0]), float(point[1])])

    if path:
        folium.PolyLine(
            path,
            color="blue",
            weight=4,
            opacity=0.8,
            tooltip="Predicted drift path",
        ).add_to(spill_map)

    return spill_map


if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    with st.spinner("Analyzing image..."):
        try:
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        image_bytes,
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as error:
            st.error(f"Could not reach the prediction service: {error}")
            st.stop()
        except ValueError:
            st.error("The prediction service returned invalid JSON.")
            st.stop()

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    original_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if original_bgr is None:
        st.error("The uploaded file is not a valid image.")
        st.stop()
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Analysis")
    confidence = float(result.get("confidence", 0.0))
    spill_detected = bool(result.get("spill_detected", False))
    metric_columns = st.columns(3)
    metric_columns[0].metric("Spill detected", "Yes" if spill_detected else "No")
    metric_columns[1].metric("Confidence", f"{confidence:.1%}")
    if result.get("area_km2") is not None:
        metric_columns[2].metric("Area", f"{float(result['area_km2']):.3f} km²")
        display_severity(result.get("severity", "unknown"))

    if not spill_detected:
        st.image(original_rgb, caption="Original image", use_container_width=True)
        st.success("No oil spill detected in this image.")
    else:
        mask = decode_mask(result.get("mask_image", ""), original_bgr.shape)
        if mask is None:
            st.warning("The API response did not contain a valid segmentation mask.")
        else:
            overlay_bgr = create_overlay(original_bgr, mask)
            image_columns = st.columns(2)
            image_columns[0].image(original_rgb, caption="Original image", use_container_width=True)
            image_columns[1].image(
                cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
                caption="Segmented spill overlay",
                use_container_width=True,
            )

        spill_map = build_map(result)
        if spill_map is not None:
            st.subheader("Spill location and drift")
            st_folium(spill_map, use_container_width=True, height=500)
        elif result.get("spill_location") is None:
            st.info("No spill coordinates were provided, so a map cannot be displayed.")
