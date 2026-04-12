import math
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.optimize import brentq

from skyfield import almanac
from skyfield.api import load, wgs84
from skyfield.framelib import galactic_frame


class SkyAstroKit:
    def __init__(self, lat=39.9, lon=116.4, elevation=0):
        """
        Initialize the high-precision astronomical kit
        :param lat: Latitude (Default is Beijing)
        :param lon: Longitude
        :param elevation: Elevation (meters)
        """
        self.ts = load.timescale(builtin=False)
        try:
            self.eph = load('de440.bsp')
        except:
            print("de440.bsp not found, downgrading to de421.bsp...")
            self.eph = load('de421.bsp')

        self.sun = self.eph['sun']
        self.earth = self.eph['earth']
        self.moon = self.eph['moon']

        # Separate geographic location and observer vector
        self.location = wgs84.latlon(lat, lon, elevation_m=elevation)
        self.observer = self.earth + self.location

        self.tz = timezone(timedelta(hours=8))  # Default Beijing Time

        # Physical constants
        self.R_SUN = 695700.0  # Solar radius in km
        self.R_MOON = 1737.4  # Lunar radius in km

        self.SOLAR_TERMS = ["Vernal Equinox", "Clear and Bright", "Grain Rain", "Start of Summer", 
                            "Grain Buds", "Grain in Ear", "Summer Solstice", "Minor Heat", 
                            "Major Heat", "Start of Autumn", "End of Heat", "White Dew",
                            "Autumnal Equinox", "Cold Dew", "Frost Descent", "Start of Winter", 
                            "Minor Snow", "Major Snow", "Winter Solstice", "Minor Cold", 
                            "Major Cold", "Start of Spring", "Rain Water", "Awakening of Insects"]
                            
        self.MONTH_NAMES = ["First Month", "Second Month", "Third Month", "Fourth Month", 
                            "Fifth Month", "Sixth Month", "Seventh Month", "Eighth Month", 
                            "Ninth Month", "Tenth Month", "Eleventh Month", "Twelfth Month"]

    # --- Auxiliary Formatting Functions ---
    def _fmt_time(self, t):
        """Format as YYYY.MM.DD HH:MM:SS.mmm"""
        if t is None: return "---"
        return t.astimezone(self.tz).strftime('%Y.%m.%d %H:%M:%S.%f')[:-3]

    def _fmt_delta(self, t1, t2):
        """Calculate time difference and format as XXd XXh XXm XXs.mmm"""
        if t1 is None or t2 is None: return "---"
        dt = abs((t2.tt - t1.tt) * 86400)
        d, rem = divmod(dt, 86400)
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        return f"{int(d)}d {int(h):02d}h {int(m):02d}m {s:06.3f}s"

    # ================= 1. Solar Terms =================
    def get_solar_terms(self, start_dt, end_dt):
        t0 = self.ts.from_datetime(start_dt.astimezone(self.tz))
        t1 = self.ts.from_datetime(end_dt.astimezone(self.tz))

        def solar_longitude(t):
            _, lon, _ = self.earth.at(t).observe(self.sun).apparent().ecliptic_latlon(epoch='date')
            return (lon.degrees // 15).astype(int) % 24

        solar_longitude.step_days = 12.0
        t_terms, y_terms = almanac.find_discrete(t0, t1, solar_longitude)
        results = []
        for i in range(len(t_terms)):
            name = self.SOLAR_TERMS[y_terms[i]]
            start_t = t_terms[i]
            # Estimate the duration to the next solar term
            t_next, _ = almanac.find_discrete(start_t, self.ts.tt_jd(start_t.tt + 16), solar_longitude)
            duration = self._fmt_delta(start_t, t_next[0]) if t_next else "---"
            results.append([name, self._fmt_time(start_t), duration])
        return results

    # ================= 2. Lunar Months =================
    def get_lunar_months(self, start_dt, end_dt):
        t0 = self.ts.from_datetime(start_dt.astimezone(self.tz) - timedelta(days=32))
        t1 = self.ts.from_datetime(end_dt.astimezone(self.tz) + timedelta(days=32))

        f_phases = almanac.moon_phases(self.eph)
        t_phases, y_phases = almanac.find_discrete(t0, t1, f_phases)

        months = []
        cur_m = {}
        for t, y in zip(t_phases, y_phases):
            if y == 0:
                if cur_m: months.append(cur_m)
                cur_m = {0: t}
            elif cur_m is not None:
                cur_m[y] = t
        if cur_m: months.append(cur_m)

        results = []
        for i in range(len(months) - 1):
            m_start = months[i].get(0)
            m_end = months[i + 1].get(0)

            if m_start is None or m_end is None: continue

            st_utc = m_start.astimezone(self.tz)
            if st_utc > end_dt.astimezone(self.tz) or m_end.astimezone(self.tz) < start_dt.astimezone(self.tz):
                continue

            q1 = months[i].get(1)
            full = months[i].get(2)
            q3 = months[i].get(3)

            results.append([
                "Lunar Month",
                self._fmt_time(m_start),
                self._fmt_time(m_end),
                f"New:{self._fmt_time(m_start)} | 1/4:{self._fmt_time(q1)} | Full:{self._fmt_time(full)} | 3/4:{self._fmt_time(q3)}",
                self._fmt_delta(m_start, m_end)
            ])
        return results

    # ================= 3. Daily Sun and Twilight Events (Meeus Strict Correction Version) =================
    def get_daily_sun_events(self, start_dt, end_dt, temp_C=15.0, pressure_mbar=1010.0):
        results = []
        curr_dt = start_dt.astimezone(self.tz).replace(hour=0, minute=0, second=0, microsecond=0)

        # --- High-Precision Core Corrections ---
        # 1. Extract elevation, calculate the dip of the horizon
        elevation_m = max(0, self.location.elevation.m)
        dip_deg = 0.0347 * math.sqrt(elevation_m)

        # 2. Construct a high-precision sunrise determination function supporting numpy vectorization
        def precise_is_sun_up(t):
            # Get the apparent coordinates of the sun at the current time
            app = self.observer.at(t).observe(self.sun).apparent()
            # Introduce real-time pressure and temperature to calculate the true apparent altitude angle after refraction
            alt, az, dist = app.altaz(temperature_C=temp_C, pressure_mbar=pressure_mbar)
            # Calculate the strict apparent radius of the sun based on the current earth-sun distance
            semi_diameter = np.degrees(np.arcsin(self.R_SUN / dist.km))
            # Meeus sunrise definition: The upper limb of the sun touches the apparent horizon lowered by elevation
            upper_limb_alt = alt.degrees + semi_diameter
            return upper_limb_alt > -dip_deg

        # Step size setting for Skyfield's find_discrete (about 1.2 hours, ensuring rises/sets are not missed)
        precise_is_sun_up.step_days = 0.05

        while curr_dt <= end_dt.astimezone(self.tz):
            t0 = self.ts.from_datetime(curr_dt)
            t1 = self.ts.from_datetime(curr_dt + timedelta(days=1))

            # --- Use high-precision custom function to replace the original almanac.sunrise_sunset ---
            t_srss, y_srss = almanac.find_discrete(t0, t1, precise_is_sun_up)
            sr = next((t for t, y in zip(t_srss, y_srss) if y == 1), None)
            ss = next((t for t, y in zip(t_srss, y_srss) if y == 0), None)

            # Twilight is not affected by elevation occlusion, still using the astronomical center's geometric definition (-6, -12, -18)
            t_tw, y_tw = almanac.find_discrete(t0, t1, almanac.dark_twilight_day(self.eph, self.location))
            tw_data = {"Astronomical": [], "Nautical": [], "Civil": []}
            for t, y in zip(t_tw, y_tw):
                if y == 1:
                    tw_data["Astronomical"].append(self._fmt_time(t))
                elif y == 2:
                    tw_data["Nautical"].append(self._fmt_time(t))
                elif y == 3:
                    tw_data["Civil"].append(self._fmt_time(t))

            daylight = self._fmt_delta(sr, ss) if sr is not None and ss is not None else "---"
            transit = self._fmt_time(self.ts.tt_jd((sr.tt + ss.tt) / 2)) if sr is not None and ss is not None else "---"

            results.append([
                curr_dt.strftime('%Y.%m.%d'),
                f"Transit: {transit}",
                f"Sunrise: {self._fmt_time(sr)}",
                f"Sunset: {self._fmt_time(ss)}",
                f"Civil: {tw_data['Civil']}",
                f"Nautical: {tw_data['Nautical']}",
                f"Astronomical: {tw_data['Astronomical']}",
                f"Daylight: {daylight}"
            ])
            curr_dt += timedelta(days=1)
        return results

    # ================= 4 & 5. Query Parameters at Any Given Time =================
    def get_body_params(self, body_name, dt):
        t = self.ts.from_datetime(dt.astimezone(self.tz))
        target = self.sun if body_name == 'sun' else self.moon
        app = self.observer.at(t).observe(target).apparent()

        e_lat, e_lon, _ = app.ecliptic_latlon(epoch='date')
        ra, dec, distance = app.radec(epoch='date')
        alt, az, _ = app.altaz()

        dist_km = distance.km
        dist_au = distance.au
        R = self.R_SUN if body_name == 'sun' else self.R_MOON
        dia_arcsec = math.degrees(2 * math.atan(R / dist_km)) * 3600

        res = {
            "Time": self._fmt_time(t),
            "Ecl_Lon": e_lon.degrees, "Ecl_Lat": e_lat.degrees,
            "RA": ra.hours, "Dec": dec.degrees,
            "Alt": alt.degrees, "Az": az.degrees,
            "Dist_km": dist_km, "Dist_AU": dist_au,
            "Dia_arcsec": dia_arcsec
        }

        if body_name == 'moon':
            # Use the correct galactic_frame object
            g_lat, g_lon, _ = app.frame_latlon(galactic_frame)
            res["Gal_Lon"] = g_lon.degrees
            res["Gal_Lat"] = g_lat.degrees
            res["Phase"] = almanac.fraction_illuminated(self.eph, 'moon', t)

        return [list(res.values())]

    # ================= 6 & 7. Reverse Derivation of Time for Any Parameter =================
    def find_time_by_param(self, body_name, param, target_val, start_dt, end_dt):
        t0 = self.ts.from_datetime(start_dt.astimezone(self.tz))
        t1 = self.ts.from_datetime(end_dt.astimezone(self.tz))
        target = self.sun if body_name == 'sun' else self.moon

        def objective(jd):
            t = self.ts.tt_jd(jd)
            app = self.observer.at(t).observe(target).apparent()
            if param == 'Alt':
                val = app.altaz()[0].degrees
            elif param == 'Az':
                val = app.altaz()[1].degrees
            elif param == 'Dist_km':
                val = app.distance().km
            elif param == 'Phase':
                val = almanac.fraction_illuminated(self.eph, 'moon', t)
            elif param == 'Ecl_Lon':
                val = app.ecliptic_latlon(epoch='date')[1].degrees
            else:
                return 0
            return val - target_val

        # Set step size to 1 hour to scan the sign-change interval
        jd_arr = np.linspace(t0.tt, t1.tt, int((t1.tt - t0.tt) * 24) + 2)
        results = []
        for i in range(len(jd_arr) - 1):
            try:
                if objective(jd_arr[i]) * objective(jd_arr[i + 1]) <= 0:
                    root_jd = brentq(objective, jd_arr[i], jd_arr[i + 1])
                    results.append([f"{body_name}_{param}={target_val}", self._fmt_time(self.ts.tt_jd(root_jd))])
            except ValueError:
                pass
        return results

# ================= Test Run =================
if __name__ == "__main__":
    # Use Beijing coordinates for testing (add elevation parameter, e.g., 50 meters)
    kit = SkyAstroKit(lat=31.1, lon=121.4, elevation=0)
    tz = kit.tz

    # Define test time range: April 11, 2026 to April 30, 2026
    start_time = datetime(2026, 4, 11, tzinfo=tz)
    end_time = datetime(2026, 4, 30, tzinfo=tz)

    print("【1. Solar Term Calculation】")
    for row in kit.get_solar_terms(start_time, end_time): print(row)

    print("\n【2. Lunar Month Calculation】")
    for row in kit.get_lunar_months(start_time, end_time): print(row)

    print("\n【3. Daily Sunlight and Twilight Events (Showing the first day)】")
    # You can input the actual local temperature and pressure here to approach second-level precision limits
    for row in kit.get_daily_sun_events(start_time, end_time, temp_C=25.0, pressure_mbar=1010.0): print(row)

    print("\n【4. Solar Parameters (Current Time)】")
    print(kit.get_body_params('sun', start_time)[0])

    print("\n【5. Lunar Parameters (Current Time)】")
    print(kit.get_body_params('moon', start_time)[0])

    print("\n【6. Reverse Time Derivation: Find the time when the sun's altitude angle exactly reaches 30°】")
    for row in kit.find_time_by_param('sun', 'Alt', 30.0, start_time, start_time + timedelta(days=1)): print(row)

    print("\n【7. Reverse Time Derivation: Find the time when the lunar phase is exactly 0.5 (Quarter/Half Moon)】")
    for row in kit.find_time_by_param('moon', 'Phase', 0.5, start_time, end_time): print(row)
