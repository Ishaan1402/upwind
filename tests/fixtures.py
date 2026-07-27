"""
Shared test fixtures for AQI scenarios (Wildfire Smoke, Ozone, Windblown Dust, Winter Stagnation, Urban PM).
"""

SMOKE_LOCATION = {"lat": 45.3199, "lon": -117.8147, "name": "Cove", "zip_code": "97824", "state": "OR", "city": "Cove"}
SMOKE_OBSERVATION = {"aqi": 141, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
SMOKE_SIGNALS = [
    {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.65},
    {"id": "firms_upwind", "status": "present", "count": 3, "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}, "incident_name": "Hay Creek Fire"},
    {"id": "wind", "status": "present", "speed_mph": 6.0, "direction_deg": 315.0, "boundary_layer_height_m": 850.0},
    {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
    {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 61.4}
]

OZONE_LOCATION = {"lat": 34.0522, "lon": -118.2437, "name": "Los Angeles", "zip_code": "90012", "state": "CA", "city": "Los Angeles"}
OZONE_OBSERVATION = {"aqi": 125, "primary_pollutant": "O3", "category": "Unhealthy for Sensitive Groups"}
OZONE_SIGNALS = [
    {"id": "aerosol_plume", "status": "absent", "aod_value": 0.08},
    {"id": "firms_upwind", "status": "absent", "count": 0},
    {"id": "wind", "status": "present", "speed_mph": 4.0, "direction_deg": 90.0, "boundary_layer_height_m": 1200.0},
    {"id": "surface_pm_level", "status": "absent", "primary": False, "pm10_primary": False, "pm25_primary": False, "elevated": False},
    {"id": "ozone_heat", "status": "present", "primary": True, "hot_day": True, "temperature_f": 92.0}
]

DUST_LOCATION = {"lat": 31.7619, "lon": -106.4850, "name": "El Paso", "zip_code": "79901", "state": "TX", "city": "El Paso"}
DUST_OBSERVATION = {"aqi": 155, "primary_pollutant": "PM10", "category": "Unhealthy"}
DUST_SIGNALS = [
    {"id": "aerosol_plume", "status": "absent", "aod_value": 0.12},
    {"id": "firms_upwind", "status": "absent", "count": 0},
    {"id": "wind", "status": "present", "speed_mph": 22.0, "direction_deg": 260.0, "boundary_layer_height_m": 1500.0},
    {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
    {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 78.0}
]

STAGNATION_LOCATION = {"lat": 40.7608, "lon": -111.8910, "name": "Salt Lake City", "zip_code": "84101", "state": "UT", "city": "Salt Lake City"}
STAGNATION_OBSERVATION = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
STAGNATION_SIGNALS = [
    {"id": "aerosol_plume", "status": "absent", "aod_value": 0.10},
    {"id": "firms_upwind", "status": "absent", "count": 0},
    {"id": "wind", "status": "present", "speed_mph": 2.0, "direction_deg": 0.0, "boundary_layer_height_m": 350.0},
    {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
    {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 28.0}
]

URBAN_LOCATION = {"lat": 41.8781, "lon": -87.6298, "name": "Chicago", "zip_code": "60601", "state": "IL", "city": "Chicago"}
URBAN_OBSERVATION = {"aqi": 85, "primary_pollutant": "PM2.5", "category": "Moderate"}
URBAN_SIGNALS = [
    {"id": "aerosol_plume", "status": "absent", "aod_value": 0.09},
    {"id": "firms_upwind", "status": "absent", "count": 0},
    {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 180.0, "boundary_layer_height_m": 900.0},
    {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
    {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 68.0}
]
