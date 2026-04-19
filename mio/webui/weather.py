"""Weather skill powered by Open-Meteo (no API key required).

The skill takes a location string, geocodes it via Open-Meteo's geocoding API,
fetches current + hourly + 7-day forecasts, and returns structured JSON that
the model wraps in an HTML artifact for the webui to render as an animated
widget (Meteocons-based).
"""

from __future__ import annotations

import json as _json
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def _http_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
        return _json.loads(r.read().decode("utf-8", errors="replace"))


@dataclass
class GeoHit:
    name: str
    country: str
    lat: float
    lon: float
    admin1: str = ""
    timezone: str = ""


def _geocode(location: str) -> GeoHit | None:
    q = urllib.parse.quote(location)
    data = _http_json(
        f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
    )
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    return GeoHit(
        name=r.get("name", location),
        country=r.get("country", ""),
        lat=float(r["latitude"]),
        lon=float(r["longitude"]),
        admin1=r.get("admin1", "") or "",
        timezone=r.get("timezone", "") or "auto",
    )


# WMO weather-code mapping → (short label, Meteocons icon key).
# Icon keys match github.com/basmilius/weather-icons (MIT) filenames without extension.
WMO: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "clear-day"),
    1: ("Mainly clear", "partly-cloudy-day"),
    2: ("Partly cloudy", "partly-cloudy-day"),
    3: ("Overcast", "overcast"),
    45: ("Fog", "fog"),
    48: ("Rime fog", "fog"),
    51: ("Light drizzle", "drizzle"),
    53: ("Drizzle", "drizzle"),
    55: ("Dense drizzle", "drizzle"),
    56: ("Freezing drizzle", "sleet"),
    57: ("Freezing drizzle", "sleet"),
    61: ("Light rain", "rain"),
    63: ("Rain", "rain"),
    65: ("Heavy rain", "rain"),
    66: ("Freezing rain", "sleet"),
    67: ("Freezing rain", "sleet"),
    71: ("Light snow", "snow"),
    73: ("Snow", "snow"),
    75: ("Heavy snow", "snow"),
    77: ("Snow grains", "snow"),
    80: ("Rain showers", "rain"),
    81: ("Rain showers", "rain"),
    82: ("Violent showers", "thunderstorms-rain"),
    85: ("Snow showers", "snow"),
    86: ("Heavy snow showers", "snow"),
    95: ("Thunderstorm", "thunderstorms"),
    96: ("Thunderstorm w/ hail", "thunderstorms-rain"),
    99: ("Thunderstorm w/ hail", "thunderstorms-rain"),
}


def _describe(code: int) -> tuple[str, str]:
    return WMO.get(int(code), ("Unknown", "not-available"))


def get_weather(location: str, units: str = "metric") -> dict:
    """Fetch current weather, 24h hourly, and 7-day forecast for a location."""
    geo = _geocode(location)
    if geo is None:
        return {"skill": "get_weather", "error": f"No geocoding match for '{location}'"}

    temp_unit = "celsius" if units == "metric" else "fahrenheit"
    wind_unit = "kmh" if units == "metric" else "mph"

    forecast = _http_json(
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={geo.lat}&longitude={geo.lon}"
        "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
        "precipitation,weather_code,wind_speed_10m,wind_direction_10m,is_day"
        "&hourly=temperature_2m,weather_code,precipitation_probability"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
        "precipitation_sum,sunrise,sunset,uv_index_max"
        f"&timezone=auto&forecast_days=7&temperature_unit={temp_unit}"
        f"&wind_speed_unit={wind_unit}"
    )

    current = forecast.get("current", {})
    code = int(current.get("weather_code", 0))
    label, icon = _describe(code)

    hourly = forecast.get("hourly", {})
    hours = []
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    codes = hourly.get("weather_code") or []
    pprob = hourly.get("precipitation_probability") or []
    for i in range(min(24, len(times))):
        h_code = int(codes[i]) if i < len(codes) else 0
        _, h_icon = _describe(h_code)
        hours.append({
            "time": times[i],
            "temp": temps[i] if i < len(temps) else None,
            "code": h_code,
            "icon": h_icon,
            "precip_prob": pprob[i] if i < len(pprob) else None,
        })

    daily = forecast.get("daily", {})
    days = []
    d_times = daily.get("time") or []
    d_tmax = daily.get("temperature_2m_max") or []
    d_tmin = daily.get("temperature_2m_min") or []
    d_codes = daily.get("weather_code") or []
    d_precip = daily.get("precipitation_sum") or []
    d_uv = daily.get("uv_index_max") or []
    d_sunrise = daily.get("sunrise") or []
    d_sunset = daily.get("sunset") or []
    for i in range(len(d_times)):
        d_code = int(d_codes[i]) if i < len(d_codes) else 0
        d_label, d_icon = _describe(d_code)
        days.append({
            "date": d_times[i],
            "tmax": d_tmax[i] if i < len(d_tmax) else None,
            "tmin": d_tmin[i] if i < len(d_tmin) else None,
            "code": d_code,
            "label": d_label,
            "icon": d_icon,
            "precip_mm": d_precip[i] if i < len(d_precip) else None,
            "uv_max": d_uv[i] if i < len(d_uv) else None,
            "sunrise": d_sunrise[i] if i < len(d_sunrise) else None,
            "sunset": d_sunset[i] if i < len(d_sunset) else None,
        })

    return {
        "skill": "get_weather",
        "location": {
            "name": geo.name,
            "country": geo.country,
            "admin1": geo.admin1,
            "lat": geo.lat,
            "lon": geo.lon,
            "timezone": geo.timezone,
        },
        "units": {
            "temperature": "°C" if units == "metric" else "°F",
            "wind": "km/h" if units == "metric" else "mph",
        },
        "current": {
            "temp": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_dir": current.get("wind_direction_10m"),
            "is_day": bool(current.get("is_day", 1)),
            "code": code,
            "label": label,
            "icon": icon,
        },
        "hourly": hours,
        "daily": days,
    }
