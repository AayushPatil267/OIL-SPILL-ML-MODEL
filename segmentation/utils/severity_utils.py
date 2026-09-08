"""Provide utilities for estimating and categorizing spill severity."""

from datetime import datetime, timezone

import numpy as np


def calculate_area_and_severity(binary_mask, meters_per_pixel=10):
	"""Calculate spill area in km2 and classify its severity."""
	if meters_per_pixel <= 0:
		raise ValueError("meters_per_pixel must be greater than zero")

	pixel_count = int(np.count_nonzero(np.asarray(binary_mask)))
	area_km2 = pixel_count * meters_per_pixel**2 / 1_000_000

	if area_km2 < 1:
		severity = "small"
	elif area_km2 <= 10:
		severity = "medium"
	else:
		severity = "large"

	return {
		"area_km2": float(area_km2),
		"severity": severity,
		"pixel_count": pixel_count,
	}


def generate_report(prediction_result):
	"""Format a prediction response as a readable incident report."""
	result = prediction_result or {}
	spill_detected = bool(result.get("spill_detected", False))
	confidence = result.get("confidence")
	area = result.get("area_km2")
	severity = result.get("severity")
	lines = [
		"Oil Spill Detection Incident Report",
		f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
		f"Detection result: {'Oil spill detected' if spill_detected else 'No oil spill detected'}",
		f"Confidence: {float(confidence):.3f}" if confidence is not None else "Confidence: N/A",
		f"Area: {float(area):.3f} km2" if area is not None else "Area: N/A",
		f"Severity: {severity if severity is not None else 'N/A'}",
	]

	location = result.get("spill_location")
	if location:
		lines.append(f"Coordinates: latitude {location['lat']:.6f}, longitude {location['lon']:.6f}")
	else:
		lines.append("Coordinates: N/A")

	drift_prediction = result.get("drift_prediction") or []
	if drift_prediction:
		first_point = drift_prediction[0]
		last_point = drift_prediction[-1]
		if isinstance(first_point, dict):
			first_lat, first_lon = first_point["lat"], first_point["lon"]
			last_lat, last_lon = last_point["lat"], last_point["lon"]
		else:
			first_lat, first_lon = first_point
			last_lat, last_lon = last_point
		lines.append(
			f"Drift prediction: {len(drift_prediction)} points, "
			f"from ({float(first_lat):.6f}, {float(first_lon):.6f}) "
			f"to ({float(last_lat):.6f}, {float(last_lon):.6f})"
		)
	else:
		lines.append("Drift prediction: N/A")

	return "\n".join(lines)
