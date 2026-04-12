import math
import numpy as np
from datetime import datetime, timedelta, timezone
import scipy.optimize as optimize

from skyfield.api import load, wgs84
from skyfield import almanac, eclipselib


class Ecl:
    def __init__(self, tz_hours=8):
        """
        Initialize the high-precision solar and lunar eclipse calculation engine
        :param tz_hours: Default time zone offset (default +8 for Beijing Time)
        """

        # 2. Load the latest timescale data and leap seconds

        self.ts = load.timescale(builtin=False)
        try:
            self.eph = load('de440.bsp')
        except:
            print("de440.bsp not found, downgrading to de421.bsp...")
            self.eph = load('de421.bsp')

        self.sun = self.eph['sun']
        self.earth = self.eph['earth']
        self.moon = self.eph['moon']

        self.R_earth = 6378.137
        self.R_sun = 695700.0
        self.R_moon = 1737.4

        self.tz = timezone(timedelta(hours=tz_hours))

    def _fmt_time(self, t):
        if t is None:
            return "N/A"
        return t.astimezone(self.tz).strftime('%Y-%m-%d %H:%M:%S')

    def _refraction_correction(self, h_true):
        if h_true < -5.0:
            return 0.0
        R = 1.02 / math.tan(math.radians(h_true + 10.3 / (h_true + 5.11)))
        return R / 60.0

    def get_besselian_elements(self, t):
        earth_obs = self.earth.at(t)
        sun_vec = earth_obs.observe(self.sun).position.km
        moon_vec = earth_obs.observe(self.moon).position.km

        z_vec = moon_vec - sun_vec
        z_hat = z_vec / np.linalg.norm(z_vec)

        pole = np.array([0.0, 0.0, 1.0])
        x_vec = np.cross(pole, z_hat)
        x_hat = x_vec / np.linalg.norm(x_vec)
        y_hat = np.cross(z_hat, x_hat)

        x = np.dot(moon_vec, x_hat) / self.R_earth
        y = np.dot(moon_vec, y_hat) / self.R_earth
        d = math.degrees(math.asin(z_hat[2]))

        return {"X": x, "Y": y, "d": d}

    def get_global_eclipses(self, start_dt, end_dt):
        t0 = self.ts.from_datetime(start_dt)
        t1 = self.ts.from_datetime(end_dt)
        results = []

        # 1. Find lunar eclipses
        t_lunar, y_lunar, details_lunar = eclipselib.lunar_eclipses(t0, t1, self.eph)
        for t, y in zip(t_lunar, y_lunar):
            e_type = ["Penumbral Lunar Eclipse", "Partial Lunar Eclipse", "Total Lunar Eclipse"][y]
            results.append({
                "Event": "Lunar Eclipse",
                "Type": e_type,
                "Peak_Time": t,
                "Peak_Time_Str": self._fmt_time(t),
                "Details": "Lunar eclipses are global events, visible from the entire hemisphere facing away from the sun."
            })

        # 2. Find solar eclipses
        t_phases, y_phases = almanac.find_discrete(t0, t1, almanac.moon_phases(self.eph))
        for t_phase, phase in zip(t_phases, y_phases):
            if phase == 0:
                earth_obs = self.earth.at(t_phase)
                app_sun = earth_obs.observe(self.sun).apparent()
                app_moon = earth_obs.observe(self.moon).apparent()
                sep = app_moon.separation_from(app_sun).degrees

                if sep < 1.6:
                    bess = self.get_besselian_elements(t_phase)
                    gamma = math.sqrt(bess["X"] ** 2 + bess["Y"] ** 2)

                    d_s = app_sun.distance().km
                    d_m = app_moon.distance().km

                    # Accurately calculate u: the radius of the moon's umbral cone on the fundamental plane (in units of Earth's equatorial radius)
                    sin_f2 = (self.R_sun - self.R_moon) / d_s
                    L2 = self.R_moon - d_m * sin_f2
                    u = L2 / self.R_earth

                    # Meeus algorithm rules for subdividing central and non-central solar eclipses
                    mag_approx = None
                    if gamma < 0.9972:
                        if u < 0:
                            e_type = "Central Total Solar Eclipse (Total)"
                        elif u > 0.0047:
                            e_type = "Central Annular Solar Eclipse (Annular)"
                        else:
                            omega = 0.00464 * math.sqrt(max(0.0, 1.0 - gamma ** 2))
                            if u < omega:
                                e_type = "Hybrid Eclipse (Annular-Total)"
                            else:
                                e_type = "Central Annular Solar Eclipse (Annular)"
                    elif 0.9972 <= gamma < 0.9972 + abs(u):
                        if u < 0:
                            e_type = "Non-central Total Solar Eclipse (Non-central Total)"
                        else:
                            e_type = "Non-central Annular Solar Eclipse (Non-central Annular)"
                    elif gamma < 1.5433 + u:
                        e_type = "Partial Solar Eclipse (Partial)"
                        mag_approx = max(0.0, (1.5433 + u - gamma) / (0.5461 + 2 * u))
                    else:
                        continue

                    res_dict = {
                        "Event": "Solar Eclipse",
                        "Type": e_type,
                        "Peak_Time": t_phase,
                        "Peak_Time_Str": self._fmt_time(t_phase),
                        "Gamma_Approx": round(gamma, 4),
                        "U_cone_radius": round(u, 6),
                        "Besselian_X": round(bess["X"], 4),
                        "Besselian_Y": round(bess["Y"], 4)
                    }
                    if mag_approx is not None:
                        res_dict["Magnitude_Approx"] = round(mag_approx, 4)

                    results.append(res_dict)

        results.sort(key=lambda x: x["Peak_Time"].tt)
        return results

    def _find_exact_contact(self, obs, jd_start, jd_end, contact_type):
        def objective(jd):
            t = self.ts.tt_jd(jd)
            app_sun = obs.at(t).observe(self.sun).apparent()
            app_moon = obs.at(t).observe(self.moon).apparent()
            sep = app_moon.separation_from(app_sun).degrees

            rs = math.degrees(math.asin(self.R_sun / app_sun.distance().km))
            rm = math.degrees(math.asin(self.R_moon / app_moon.distance().km))

            if contact_type == 'P':
                return sep - (rs + rm)
            elif contact_type == 'U':
                return sep - abs(rs - rm)
            return 0

        try:
            val_a = objective(jd_start)
            val_b = objective(jd_end)
            if np.sign(val_a) == np.sign(val_b):
                return None
            root = optimize.brentq(objective, jd_start, jd_end)
            return self.ts.tt_jd(root)
        except ValueError:
            return None

    def get_local_eclipse_details(self, lat, lon, elevation, eclipse_peak_time, is_solar=True):
        obs_loc = wgs84.latlon(lat, lon, elevation_m=elevation)
        obs = self.earth + obs_loc

        # ==========================================
        # --- Lunar Eclipse Refined Processing Logic (Meeus Corrections) ---
        # ==========================================
        if not is_solar:
            time_arr = self.ts.utc(eclipse_peak_time.utc_datetime() + timedelta(minutes=1) * np.arange(-300, 301))

            e_geo = self.earth.at(time_arr)
            app_sun_geo = e_geo.observe(self.sun).apparent()
            app_moon_geo = e_geo.observe(self.moon).apparent()

            sep_shadow = np.abs(180.0 - app_moon_geo.separation_from(app_sun_geo).degrees)

            ds = app_sun_geo.distance().km
            dm = app_moon_geo.distance().km

            s_s = np.degrees(np.arcsin(self.R_sun / ds))
            s_m = np.degrees(np.arcsin(self.R_moon / dm))
            pi_s = np.degrees(np.arcsin(self.R_earth / ds))

            # Meeus Correction: Earth's umbra oblateness (using 1/214)
            dec_m = app_moon_geo.radec()[1].radians
            f_shadow = 1.0 / 214.0
            r_eff_earth = self.R_earth * (1.0 - f_shadow * np.sin(dec_m) ** 2)
            pi_m_eff = np.degrees(np.arcsin(r_eff_earth / dm))

            # Meeus Correction: Using Danjon's French rule instead of traditional 1/50 enlargement
            # Equivalent to an expansion of parallax by about 1.01 times
            rho_u = pi_m_eff * 1.01 + pi_s - s_s
            rho_p = pi_m_eff * 1.01 + pi_s + s_s

            penumbra = sep_shadow <= (rho_p + s_m)
            umbra = sep_shadow <= (rho_u + s_m)
            total = sep_shadow <= (rho_u - s_m)

            min_idx = np.argmin(sep_shadow)
            max_t = time_arr[min_idx]
            max_sep = sep_shadow[min_idx]

            magnitude = (rho_u[min_idx] + s_m[min_idx] - max_sep) / (2 * s_m[min_idx])

            if magnitude >= 1.0:
                ecl_type = "Total Lunar Eclipse (Total)"
            elif magnitude > 0.0:
                ecl_type = "Partial Lunar Eclipse (Partial)"
            else:
                ecl_type = "Penumbral Lunar Eclipse (Penumbral)"

            app_moon_local = obs.at(max_t).observe(self.moon).apparent()
            alt, az, _ = app_moon_local.altaz()

            app_alt = alt.degrees + self._refraction_correction(alt.degrees)
            is_visible = app_alt > 0.0

            def find_lunar_contact(t_start, t_end, contact_type):
                def objective(jd):
                    t = self.ts.tt_jd(jd)
                    e = self.earth.at(t)
                    s_geo = e.observe(self.sun).apparent()
                    m_geo = e.observe(self.moon).apparent()
                    sep = np.abs(180.0 - m_geo.separation_from(s_geo).degrees)

                    d_s = s_geo.distance().km
                    d_m = m_geo.distance().km
                    _ss = math.degrees(math.asin(self.R_sun / d_s))
                    _sm = math.degrees(math.asin(self.R_moon / d_m))
                    _pis = math.degrees(math.asin(self.R_earth / d_s))

                    _dec_m = m_geo.radec()[1].radians
                    _r_eff = self.R_earth * (1.0 - f_shadow * np.sin(_dec_m) ** 2)
                    _pim_eff = math.degrees(math.asin(_r_eff / d_m))

                    _ru = _pim_eff * 1.01 + _pis - _ss
                    _rp = _pim_eff * 1.01 + _pis + _ss

                    if contact_type == 'P1_P4':
                        return sep - (_rp + _sm)
                    elif contact_type == 'U1_U4':
                        return sep - (_ru + _sm)
                    elif contact_type == 'U2_U3':
                        return sep - (_ru - _sm)
                    return 0

                try:
                    val_a = objective(t_start)
                    val_b = objective(t_end)
                    if np.sign(val_a) == np.sign(val_b): return None
                    root = optimize.brentq(objective, t_start, t_end)
                    return self.ts.tt_jd(root)
                except ValueError:
                    return None

            pen_diff = np.diff(penumbra.astype(int))
            umb_diff = np.diff(umbra.astype(int))
            tot_diff = np.diff(total.astype(int))

            p1_idx = np.where(pen_diff == 1)[0]
            p4_idx = np.where(pen_diff == -1)[0]
            u1_idx = np.where(umb_diff == 1)[0]
            u4_idx = np.where(umb_diff == -1)[0]
            u2_idx = np.where(tot_diff == 1)[0]
            u3_idx = np.where(tot_diff == -1)[0]

            p1_t = find_lunar_contact(time_arr[p1_idx[0]].tt, time_arr[p1_idx[0] + 1].tt, 'P1_P4') if len(
                p1_idx) > 0 else None
            p4_t = find_lunar_contact(time_arr[p4_idx[-1]].tt, time_arr[p4_idx[-1] + 1].tt, 'P1_P4') if len(
                p4_idx) > 0 else None

            u1_t = find_lunar_contact(time_arr[u1_idx[0]].tt, time_arr[u1_idx[0] + 1].tt, 'U1_U4') if len(
                u1_idx) > 0 else None
            u4_t = find_lunar_contact(time_arr[u4_idx[-1]].tt, time_arr[u4_idx[-1] + 1].tt, 'U1_U4') if len(
                u4_idx) > 0 else None

            u2_t = find_lunar_contact(time_arr[u2_idx[0]].tt, time_arr[u2_idx[0] + 1].tt, 'U2_U3') if len(
                u2_idx) > 0 else None
            u3_t = find_lunar_contact(time_arr[u3_idx[-1]].tt, time_arr[u3_idx[-1] + 1].tt, 'U2_U3') if len(
                u3_idx) > 0 else None

            return {
                "Event": "Lunar Eclipse Local",
                "Visible": bool(is_visible),
                "Type": ecl_type,
                "Umbral_Magnitude": round(magnitude, 4),
                "Max_Eclipse_Time": self._fmt_time(max_t),
                "Moon_Apparent_Altitude_at_Max": round(app_alt, 2),
                "Moon_Azimuth_at_Max": round(az.degrees, 2),
                "Contacts": {
                    "P1 (Penumbral Eclipse Begins)": self._fmt_time(p1_t),
                    "U1 (Partial Eclipse Begins)": self._fmt_time(u1_t),
                    "U2 (Total Eclipse Begins)": self._fmt_time(u2_t),
                    "Max (Maximum Eclipse)": self._fmt_time(max_t),
                    "U3 (Total Eclipse Ends)": self._fmt_time(u3_t),
                    "U4 (Partial Eclipse Ends)": self._fmt_time(u4_t),
                    "P4 (Penumbral Eclipse Ends)": self._fmt_time(p4_t),
                }
            }

        # ==========================================
        # --- Solar Eclipse Refined Processing Logic ---
        # ==========================================
        time_arr = self.ts.utc(eclipse_peak_time.utc_datetime() + timedelta(minutes=1) * np.arange(-240, 241))

        app_sun = obs.at(time_arr).observe(self.sun).apparent()
        app_moon = obs.at(time_arr).observe(self.moon).apparent()

        sep_arr = app_moon.separation_from(app_sun).degrees
        rs_arr = np.degrees(np.arcsin(self.R_sun / app_sun.distance().km))
        rm_arr = np.degrees(np.arcsin(self.R_moon / app_moon.distance().km))

        penumbra = sep_arr <= (rs_arr + rm_arr)
        umbra = sep_arr <= np.abs(rs_arr - rm_arr)

        if not np.any(penumbra):
            return {"Event": "Solar Eclipse Local", "Visible": False, "Msg": "Did not enter the partial eclipse zone."}

        min_idx = np.argmin(sep_arr)
        max_t = time_arr[min_idx]
        max_sep = sep_arr[min_idx]
        rs_max, rm_max = rs_arr[min_idx], rm_arr[min_idx]

        magnitude = max(0.0, (rs_max + rm_max - max_sep) / (2 * rs_max))

        obscuration = 0.0
        if magnitude > 0:
            if max_sep <= abs(rs_max - rm_max):
                obscuration = (rm_max ** 2 / rs_max ** 2) if rm_max < rs_max else 1.0
            else:
                arg1 = max(-1.0, min(1.0, (max_sep ** 2 + rs_max ** 2 - rm_max ** 2) / (2 * max_sep * rs_max)))
                arg2 = max(-1.0, min(1.0, (max_sep ** 2 + rm_max ** 2 - rs_max ** 2) / (2 * max_sep * rm_max)))
                part1 = rs_max ** 2 * math.acos(arg1)
                part2 = rm_max ** 2 * math.acos(arg2)
                part3 = 0.5 * math.sqrt(max(0.0, (-max_sep + rs_max + rm_max) * (max_sep + rs_max - rm_max) * (
                            max_sep - rs_max + rm_max) * (max_sep + rs_max + rm_max)))
                obscuration = (part1 + part2 - part3) / (math.pi * rs_max ** 2)

        if max_sep < abs(rs_max - rm_max):
            ecl_type = "Total Solar Eclipse (Total)" if rm_max >= rs_max else "Annular Solar Eclipse (Annular)"
        else:
            ecl_type = "Partial Solar Eclipse (Partial)"

        alt, _, _ = obs.at(max_t).observe(self.sun).apparent().altaz()
        app_alt = alt.degrees + self._refraction_correction(alt.degrees)
        if app_alt <= 0.0:
            return {"Event": "Solar Eclipse Local", "Visible": False, "Msg": "The sun is below the horizon when it occurs."}

        pen_diff = np.diff(penumbra.astype(int))
        umb_diff = np.diff(umbra.astype(int))

        p1_idx = np.where(pen_diff == 1)[0]
        p4_idx = np.where(pen_diff == -1)[0]
        u1_idx = np.where(umb_diff == 1)[0]
        u2_idx = np.where(umb_diff == -1)[0]

        p1_t = self._find_exact_contact(obs, time_arr[p1_idx[0]].tt, time_arr[p1_idx[0] + 1].tt, 'P') if len(
            p1_idx) > 0 else None
        p4_t = self._find_exact_contact(obs, time_arr[p4_idx[-1]].tt, time_arr[p4_idx[-1] + 1].tt, 'P') if len(
            p4_idx) > 0 else None

        u1_t, u2_t = None, None
        if len(u1_idx) > 0 and len(u2_idx) > 0:
            u1_t = self._find_exact_contact(obs, time_arr[u1_idx[0]].tt, time_arr[u1_idx[0] + 1].tt, 'U')
            u2_t = self._find_exact_contact(obs, time_arr[u2_idx[-1]].tt, time_arr[u2_idx[-1] + 1].tt, 'U')

        return {
            "Event": "Solar Eclipse Local",
            "Visible": True,
            "Type": ecl_type,
            "Magnitude": round(magnitude, 4),
            "Obscuration": f"{round(obscuration * 100, 2)} %",
            "Max_Eclipse_Time": self._fmt_time(max_t),
            "Sun_Apparent_Altitude_at_Max": round(app_alt, 2),
            "Contacts": {
                "P1 (First Contact / Partial Begins)": self._fmt_time(p1_t),
                "U1 (Second Contact / Total/Annular Begins)": self._fmt_time(u1_t),
                "Max (Maximum Eclipse)": self._fmt_time(max_t),
                "U2 (Third Contact / Total/Annular Ends)": self._fmt_time(u2_t),
                "P4 (Fourth Contact / Partial Ends)": self._fmt_time(p4_t)
            }
        }


if __name__ == "__main__":
    ecl = Ecl()

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2028, 1, 1, tzinfo=timezone.utc)

    # Below are test examples

    print(f"========== Retrieving {start.year} - {end.year} Global Solar and Lunar Eclipses ==========")
    global_eclipses = ecl.get_global_eclipses(start, end)

    test_solar, test_lunar = None, None

    for ge in global_eclipses:
        print(f"[{ge['Event']}] {ge['Type']} | Time: {ge['Peak_Time_Str']}")
        if ge['Event'] == 'Solar Eclipse':
            print(
                f"  -> Gamma Value: {ge.get('Gamma_Approx')} | Umbral cone radius u: {ge.get('U_cone_radius')} | (X,Y): ({ge.get('Besselian_X')}, {ge.get('Besselian_Y')})")
            if 'Magnitude_Approx' in ge:
                print(f"  -> Maximum Magnitude (Theoretical): {ge.get('Magnitude_Approx')}")
            if test_solar is None: test_solar = ge['Peak_Time']
        elif ge['Event'] == 'Lunar Eclipse':
            if test_lunar is None: test_lunar = ge['Peak_Time']

    lat, lon, ele = 31, 121, 0
    print(f"\n========== Local Observation Details Analysis (Test Point: Latitude {lat}, Longitude {lon}) ==========")

    if test_solar is not None:
        print(f"\n[Solar Eclipse Analysis Batch]: {ecl._fmt_time(test_solar)}")
        local_solar = ecl.get_local_eclipse_details(lat, lon, ele, test_solar, is_solar=True)
        for key, val in local_solar.items():
            if key == 'Contacts' and isinstance(val, dict):
                print(f"  Exact Contact Times (Contacts):")
                for c_k, c_v in val.items(): print(f"    * {c_k}: {c_v}")
            else:
                print(f"  {key}: {val}")

    if test_lunar is not None:
        print(f"\n[Lunar Eclipse Analysis Batch]: {ecl._fmt_time(test_lunar)}")
        local_lunar = ecl.get_local_eclipse_details(lat, lon, ele, test_lunar, is_solar=False)
        for key, val in local_lunar.items():
            if key == 'Contacts' and isinstance(val, dict):
                print(f"  Exact Contact Times (Contacts):")
                for c_k, c_v in val.items(): print(f"    * {c_k}: {c_v}")
            else:
                print(f"  {key}: {val}")
