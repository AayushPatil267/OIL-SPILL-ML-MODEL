"""Provide geospatial helpers for mapping detected oil spills."""

import cv2
import math

import numpy as np


def mask_centroid_to_latlon(
	binary_mask,
	top_left_lat,
	top_left_lon,
	bottom_right_lat,
	bottom_right_lon,
):
	"""Map the largest foreground region centroid from pixels to latitude/longitude."""
	mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
	if mask.ndim != 2 or not np.any(mask):
		return None

	_, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
	largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
	centroid_x, centroid_y = centroids[largest_label]
	height, width = mask.shape

	x_fraction = centroid_x / (width - 1) if width > 1 else 0.0
	y_fraction = centroid_y / (height - 1) if height > 1 else 0.0
	latitude = top_left_lat + y_fraction * (bottom_right_lat - top_left_lat)
	longitude = top_left_lon + x_fraction * (bottom_right_lon - top_left_lon)

	return {"lat": float(latitude), "lon": float(longitude)}


def predict_drift(
	lat,
	lon,
	wind_speed_mps,
	wind_direction_degrees,
	hours=6,
):
	"""Estimate hourly oil spill drift positions from wind speed and bearing."""
	if hours < 0:
		raise ValueError("hours must be greater than or equal to zero")
	if wind_speed_mps < 0:
		raise ValueError("wind_speed_mps must be greater than or equal to zero")

	earth_radius_m = 6_371_000
	drift_speed_mps = wind_speed_mps * 0.03
	direction_radians = math.radians(wind_direction_degrees)
	positions = []

	for hour in range(1, hours + 1):
		distance_m = drift_speed_mps * hour * 3_600
		north_offset_m = distance_m * math.cos(direction_radians)
		east_offset_m = distance_m * math.sin(direction_radians)
		latitude = lat + math.degrees(north_offset_m / earth_radius_m)
		longitude_scale = math.cos(math.radians(lat))
		if abs(longitude_scale) < 1e-12:
			longitude = lon
		else:
			longitude = lon + math.degrees(east_offset_m / (earth_radius_m * longitude_scale))
		positions.append((float(latitude), float(longitude)))

	return positions
