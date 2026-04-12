This repository contains two highly accurate Python toolkits based on the `skyfield` library and the NASA JPL DE440/DE421 ephemeris. Together, they provide precise calculations for solar and lunar eclipses, solar terms, lunar months, sunrise/sunset times, twilight, and real-time solar/lunar parameters.

## Files Included

1. **`Ecl.py`** (contains the `Ecl` class)
   - Calculates global solar and lunar eclipses within a given time frame.
   - Computes Besselian elements and exact theoretical magnitudes.
   - Calculates **local** eclipse details for a specific latitude, longitude, and elevation.
   - Uses rigorous Meeus corrections for precise contact times (P1, U1, Max, U2, P4, etc.).

2. **`SkyAstroKit.py`** (contains the `SkyAstroKit` class)
   - Calculates precise times for the 24 Traditional Chinese Solar Terms.
   - Calculates exact lunar months and major lunar phases (New, First Quarter, Full, Third Quarter).
   - Computes daily sunrise, sunset, and twilight (Civil, Nautical, Astronomical) adjusting for atmospheric refraction, pressure, temperature, and elevation.
   - Queries real-time parameters for the Sun and Moon (Altitude, Azimuth, Distance, Phase).
   - "Reverse-derives" times based on specific parameters (e.g., "At what time exactly will the sun hit an altitude of 30 degrees?").

---

## Installation & Prerequisites

These scripts require Python 3.7+ and a few scientific libraries. 

1. Install the required packages using `pip`:
   ```bash
   pip install numpy scipy skyfield
   ```

2. **Important Note on Ephemeris Data (`.bsp` files):**
   Both scripts require a JPL Planetary Ephemeris file (like `de440.bsp` or `de421.bsp`). 
   - The script will automatically attempt to download this file from the internet on its first run.
   - This file is large (~17 MB for DE421, ~115 MB for DE440). **The first run may take several minutes** depending on your internet connection. 
   - Once downloaded, it is cached locally and subsequent runs will be instantaneous.

---

## How to Use

### 1. Using the Eclipse Calculator (`Ecl.py`)

You can run the script directly from the terminal to see a demonstration:
```bash
python eclipse_calculator.py
```

**Using it in your own code:**
```python
from datetime import datetime, timezone
from Ecl import Ecl

# Initialize the engine (Default timezone is UTC+8, change tz_hours if needed)
ecl = Ecl(tz_hours=8)

# Define a time window
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
end = datetime(2028, 1, 1, tzinfo=timezone.utc)

# 1. Get all global eclipses
global_eclipses = ecl.get_global_eclipses(start, end)
for eclipse in global_eclipses:
    print(eclipse['Event'], eclipse['Type'], eclipse['Peak_Time_Str'])

# 2. Get local eclipse visibility for a specific location
lat, lon, elevation = 31.0, 121.0, 0
target_eclipse_time = global_eclipses[0]['Peak_Time'] # Pass the Skyfield time object

local_details = ecl.get_local_eclipse_details(lat, lon, elevation, target_eclipse_time, is_solar=True)
print("Visible?", local_details['Visible'])
if local_details['Visible']:
    print("Magnitude:", local_details['Magnitude'])
    print("Contact Times:", local_details['Contacts'])
```

### 2. Using the Astronomical Kit (`SkyAstroKit.py`)

You can run the script directly to see the console output of various calculations:
```bash
python astro_kit.py
```

**Using it in your own code:**
```python
from datetime import datetime, timedelta, timezone
from SkyAstroKit import SkyAstroKit

# Initialize with an observer's exact location
# Example: Beijing (lat=39.9, lon=116.4, elevation=50 meters)
kit = SkyAstroKit(lat=39.9, lon=116.4, elevation=50)

start_time = datetime(2026, 4, 11, tzinfo=kit.tz)
end_time = datetime(2026, 4, 30, tzinfo=kit.tz)

# --- Find Solar Terms ---
terms = kit.get_solar_terms(start_time, end_time)
print("Solar Terms:", terms)

# --- Get Daily Sunrise/Sunset ---
# You can input exact temp (Celsius) and pressure (mbar) for extreme accuracy
daily_events = kit.get_daily_sun_events(start_time, start_time + timedelta(days=2), temp_C=20, pressure_mbar=1013)
for day in daily_events:
    print(day)

# --- Reverse Time Derivation ---
# Example: Find exactly when the Moon's phase (illumination) hits 50% (0.5)
half_moon_times = kit.find_time_by_param('moon', 'Phase', 0.5, start_time, end_time)
print("Half Moon occurs at:", half_moon_times)
```

## Advanced Configuration
* **Timezones:** Both scripts default to UTC+8 (Beijing time) for output strings. You can modify `tz_hours=8` in `eclipse_calculator.py` or change the `timedelta(hours=8)` in `astro_kit.py` to match your local timezone.
* **Accuracy:[But note that there is no absolute accuracy]** The algorithms utilize `scipy.optimize.brentq` to find roots (exact moments of contact or altitude thresholds) down to the millisecond. Adjusting parameters like temperature and pressure in `get_daily_sun_events` dynamically recalculates atmospheric refraction, providing ultimate precision.
