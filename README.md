

# High-Precision Eclipse Calculator

A high-precision solar and lunar eclipse calculation engine built on Python's `skyfield`, `numpy`, and `scipy`. This engine not only retrieves global eclipse events and rigorously classifies their types but also calculates detailed local observation data for specific geographical coordinates (including precise contact times, magnitude, obscuration, and celestial altitude).

## ✨ Core Features

* **Global Eclipse Retrieval**: Quickly search for all solar and lunar eclipse events within a specified time range, utilizing Meeus algorithm rules to precisely subdivide solar eclipse types (central/non-central total, annular, hybrid, partial, etc.).
* **Local Observation Calculation**: Calculate the local visibility of an eclipse based on the observer's longitude, latitude, and elevation.
* **High-Precision Contact Times**: Utilizes `scipy.optimize.brentq` for root-finding to accurately calculate the exact times of First Contact (P1), Second Contact (U1), Maximum Eclipse (Max), Third Contact (U2), and Fourth Contact (P4).
* **Rigorous Astronomical Corrections**:
    * Supports JPL's latest `DE440` ephemeris (automatically falls back to `DE421`).
    * Introduces Meeus correction algorithms for lunar eclipses (accounting for Earth's umbral oblateness of 1/214).
    * Applies Danjon's French rule (parallax enlargement effect) instead of the traditional 1/50 simple enlargement.
    * Built-in atmospheric refraction correction algorithm.

---

## 🛠️ Dependencies & Installation

Before running this program, please ensure the following Python libraries are installed in your environment:

```bash
pip install skyfield numpy scipy
```

> **Note**: Upon first run, `skyfield` will automatically attempt to download `de440.bsp` (JPL ephemeris file, roughly 100MB). If the download fails or the file is not found, the program will automatically downgrade to use `de421.bsp`.

---

## 🚀 Quick Start

### 1. Initialize the Engine

The engine defaults to UTC+8. You can change the default time zone offset by passing the `tz_hours` parameter.

```python
from datetime import datetime, timezone
from eclipse_engine import Ecl # Assuming your file is named eclipse_engine.py

# Initialize the calculation engine
ecl = Ecl(tz_hours=8)
```

### 2. Retrieve Global Eclipses

Specify the start and end UTC times to retrieve all solar and lunar eclipses occurring within this period.

```python
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
end = datetime(2028, 1, 1, tzinfo=timezone.utc)

global_eclipses = ecl.get_global_eclipses(start, end)

for ge in global_eclipses:
    print(f"[{ge['Event']}] {ge['Type']} | Time: {ge['Peak_Time_Str']}")
```

---

## 🌗 Solar vs. Lunar: Code Differences

Because solar and lunar eclipses behave differently geometrically, the local observation function (`get_local_eclipse_details`) processes and returns data differently based on the `is_solar` boolean flag.

### Function 1: Local Solar Eclipse (`is_solar=True`)
A solar eclipse happens when the **Moon blocks the Sun**. It is highly dependent on your exact location on Earth. The solar calculation returns specific data like **Obscuration** (percentage of the Sun's surface covered).

```python
# Assuming you have a solar eclipse peak_time from get_global_eclipses()
solar_details = ecl.get_local_eclipse_details(
    lat=31.0, lon=121.0, elevation=0, 
    eclipse_peak_time=solar_peak_time, 
    is_solar=True  # Trigger Solar Logic
)

print(f"Eclipse Type: {solar_details['Type']}")
print(f"Obscuration: {solar_details['Obscuration']}") # e.g., "85.20 %"
print(solar_details['Contacts']) # Returns P1, U1, Max, U2, P4 based on solar transit
```

### Function 2: Local Lunar Eclipse (`is_solar=False`)
A lunar eclipse happens when the **Earth casts its shadow on the Moon**. It is visible to the entire night side of the Earth simultaneously. The lunar calculation uses Earth's umbra/penumbra physics and returns **Umbral Magnitude** instead of obscuration.

```python
# Assuming you have a lunar eclipse peak_time from get_global_eclipses()
lunar_details = ecl.get_local_eclipse_details(
    lat=31.0, lon=121.0, elevation=0, 
    eclipse_peak_time=lunar_peak_time, 
    is_solar=False  # Trigger Lunar Logic
)

print(f"Eclipse Type: {lunar_details['Type']}")
print(f"Umbral Magnitude: {lunar_details['Umbral_Magnitude']}") # e.g., 1.054
print(lunar_details['Contacts']) # Returns Penumbral (P1/P4) and Umbral (U1/U2/U3/U4) times
```

---

## 📚 API Reference

### `Ecl(tz_hours=8)`
Initializes the class.
* `tz_hours` *(int/float)*: Time zone offset used for formatting time output. Default is `8`.

### `get_global_eclipses(start_dt, end_dt)`
Retrieves global solar and lunar eclipse events.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `start_dt` | `datetime` | Start time (must be timezone-aware, e.g., `timezone.utc`) |
| `end_dt` | `datetime` | End time (must be timezone-aware, e.g., `timezone.utc`) |

**Returns**: A list of dictionaries. Depending on the event, the dictionary may contain: event type (`Event`), subdivided type (`Type`), peak time (`Peak_Time`), Besselian elements (`Besselian_X/Y`), approximate magnitude (`Magnitude_Approx`), etc.

### `get_local_eclipse_details(lat, lon, elevation, eclipse_peak_time, is_solar)`
Calculates eclipse details for a specific observation location.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `lat` | `float` | Latitude of the observation point (positive for North, negative for South) |
| `lon` | `float` | Longitude of the observation point (positive for East, negative for West) |
| `elevation` | `float` | Elevation of the observation point (in meters) |
| `eclipse_peak_time` | `Time` | The `skyfield` time object returned by `get_global_eclipses` |
| `is_solar` | `bool` | `True` to calculate a solar eclipse, `False` for a lunar eclipse |

**Returns**: A dictionary containing observation details with the following core fields:
* `Visible`: Whether the eclipse is visible locally (`bool`).
* `Type`: The local classification of the eclipse (e.g., Partial Solar Eclipse, Total Lunar Eclipse).
* `Magnitude` / `Umbral_Magnitude`: The local maximum magnitude.
* `Obscuration`: (Solar eclipses only) The percentage of the Sun's apparent area obscured by the Moon.
* `Contacts`: A dictionary containing the exact times of the respective contact points.

---

## 🧮 Algorithms & Principles

1.  **Besselian Elements**: To determine the type of solar eclipse, the program transforms the coordinate system to the **Fundamental Plane**, which is centered on the Earth and has its Z-axis parallel to the line connecting the Sun and Moon. It calculates the `X` and `Y` elements and the radius of the umbral cone `u` to determine the centrality of the eclipse.
2.  **Meeus Classification**: Adopts the standards from Jean Meeus's *Astronomical Algorithms*, rigorously subdividing non-central total, non-central annular, and hybrid eclipses based on the umbral cone radius $u$ and the distance from the umbral axis to the geocenter $\gamma$.
3.  **Root-Finding Optimization**: Utilizes `scipy.optimize.brentq` (Brent's method) to find the exact moment when the apparent disks of the Sun and Moon are tangent (for solar) or when the Moon enters Earth's shadow phases (for lunar). This provides significantly higher precision than simple linear interpolation.
