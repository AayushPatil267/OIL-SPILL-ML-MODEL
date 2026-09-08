"""Provide weather data retrieval and normalization helpers."""

import requests


OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_wind_data(lat, lon, api_key):
	"""Fetch current wind speed and direction from OpenWeatherMap."""
	try:
		response = requests.get(
			OPENWEATHER_CURRENT_URL,
			params={
				"lat": lat,
				"lon": lon,
				"appid": api_key,
				"units": "metric",
			},
			timeout=10,
		)
		response.raise_for_status()
		wind_data = response.json()["wind"]
		return {
			"wind_speed_mps": float(wind_data["speed"]),
			"wind_direction_degrees": float(wind_data["deg"]),
		}
	except (requests.RequestException, KeyError, TypeError, ValueError) as error:
		print(f"Warning: unable to fetch wind data: {error}")
		return None
