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
        初始化高精度天文套件
        :param lat: 纬度 (默认北京)
        :param lon: 经度
        :param elevation: 海拔高度(米)
        """
        self.ts = load.timescale(builtin=False)
        try:
            self.eph = load('de440.bsp')
        except:
            print("未找到 de440.bsp，降级使用 de421.bsp...")
            self.eph = load('de421.bsp')

        self.sun = self.eph['sun']
        self.earth = self.eph['earth']
        self.moon = self.eph['moon']

        # 将地理位置与观测者向量分离
        self.location = wgs84.latlon(lat, lon, elevation_m=elevation)
        self.observer = self.earth + self.location

        self.tz = timezone(timedelta(hours=8))  # 默认北京时间

        # 物理常数
        self.R_SUN = 695700.0  # 太阳半径 km
        self.R_MOON = 1737.4  # 月球半径 km

        self.JIEQI_NAMES = ["春分", "清明", "谷雨", "立夏", "小满", "芒种", "夏至", "小暑", "大暑", "立秋", "处暑",
                            "白露",
                            "秋分", "寒露", "霜降", "立冬", "小雪", "大雪", "冬至", "小寒", "大寒", "立春", "雨水",
                            "惊蛰"]
        self.MONTH_NAMES = ["正月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月",
                            "十二月"]

    # --- 辅助格式化函数 ---
    def _fmt_time(self, t):
        """格式化为 YYYY.MM.DD HH:MM:SS.mmm"""
        if t is None: return "---"
        return t.astimezone(self.tz).strftime('%Y.%m.%d %H:%M:%S.%f')[:-3]

    def _fmt_delta(self, t1, t2):
        """计算时间差并格式化为 XXd XXh XXm XXs.mmm"""
        if t1 is None or t2 is None: return "---"
        dt = abs((t2.tt - t1.tt) * 86400)
        d, rem = divmod(dt, 86400)
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        return f"{int(d)}d {int(h):02d}h {int(m):02d}m {s:06.3f}s"

    # ================= 1. 节气 =================
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
            name = self.JIEQI_NAMES[y_terms[i]]
            start_t = t_terms[i]
            # 预估下一个节气计算时长
            t_next, _ = almanac.find_discrete(start_t, self.ts.tt_jd(start_t.tt + 16), solar_longitude)
            duration = self._fmt_delta(start_t, t_next[0]) if t_next else "---"
            results.append([name, self._fmt_time(start_t), duration])
        return results

    # ================= 2. 农历月 =================
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
                "农历月",
                self._fmt_time(m_start),
                self._fmt_time(m_end),
                f"New:{self._fmt_time(m_start)} | 1/4:{self._fmt_time(q1)} | Full:{self._fmt_time(full)} | 3/4:{self._fmt_time(q3)}",
                self._fmt_delta(m_start, m_end)
            ])
        return results

    # ================= 3. 每日太阳与晨昏蒙影 (Meeus 严密修正版) =================
    def get_daily_sun_events(self, start_dt, end_dt, temp_C=15.0, pressure_mbar=1010.0):
        results = []
        curr_dt = start_dt.astimezone(self.tz).replace(hour=0, minute=0, second=0, microsecond=0)

        # --- 高精度核心修正 ---
        # 1. 提取海拔高度，计算地平线下降角 (Dip of the horizon)
        elevation_m = max(0, self.location.elevation.m)
        dip_deg = 0.0347 * math.sqrt(elevation_m)

        # 2. 构造支持 numpy 向量化的高精度日出判定函数
        def precise_is_sun_up(t):
            # 获取当前时刻太阳的视坐标
            app = self.observer.at(t).observe(self.sun).apparent()
            # 引入实时气压与温度计算折射后的真实视高度角
            alt, az, dist = app.altaz(temperature_C=temp_C, pressure_mbar=pressure_mbar)
            # 通过当前日月距离计算太阳严格的视半径
            semi_diameter = np.degrees(np.arcsin(self.R_SUN / dist.km))
            # Meeus 日出定义：太阳上边缘触碰到了受海拔影响下降的视地平线
            upper_limb_alt = alt.degrees + semi_diameter
            return upper_limb_alt > -dip_deg

        # Skyfield find_discrete 的步长设置 (约1.2小时，确保不会错过起落)
        precise_is_sun_up.step_days = 0.05

        while curr_dt <= end_dt.astimezone(self.tz):
            t0 = self.ts.from_datetime(curr_dt)
            t1 = self.ts.from_datetime(curr_dt + timedelta(days=1))

            # --- 使用高精度自定义函数替代原有的 almanac.sunrise_sunset ---
            t_srss, y_srss = almanac.find_discrete(t0, t1, precise_is_sun_up)
            sr = next((t for t, y in zip(t_srss, y_srss) if y == 1), None)
            ss = next((t for t, y in zip(t_srss, y_srss) if y == 0), None)

            # 晨昏蒙影不受海拔遮挡影响，依旧采用天文学中心的几何定义 (-6, -12, -18)
            t_tw, y_tw = almanac.find_discrete(t0, t1, almanac.dark_twilight_day(self.eph, self.location))
            tw_data = {"天文": [], "航海": [], "民用": []}
            for t, y in zip(t_tw, y_tw):
                if y == 1:
                    tw_data["天文"].append(self._fmt_time(t))
                elif y == 2:
                    tw_data["航海"].append(self._fmt_time(t))
                elif y == 3:
                    tw_data["民用"].append(self._fmt_time(t))

            daylight = self._fmt_delta(sr, ss) if sr is not None and ss is not None else "---"
            transit = self._fmt_time(self.ts.tt_jd((sr.tt + ss.tt) / 2)) if sr is not None and ss is not None else "---"

            results.append([
                curr_dt.strftime('%Y.%m.%d'),
                f"中天: {transit}",
                f"日出: {self._fmt_time(sr)}",
                f"日落: {self._fmt_time(ss)}",
                f"民用: {tw_data['民用']}",
                f"航海: {tw_data['航海']}",
                f"天文: {tw_data['天文']}",
                f"日照: {daylight}"
            ])
            curr_dt += timedelta(days=1)
        return results

    # ================= 4 & 5. 查询任意时刻参数 =================
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
            # 使用正确的 galactic_frame 对象
            g_lat, g_lon, _ = app.frame_latlon(galactic_frame)
            res["Gal_Lon"] = g_lon.degrees
            res["Gal_Lat"] = g_lat.degrees
            res["Phase"] = almanac.fraction_illuminated(self.eph, 'moon', t)

        return [list(res.values())]

    # ================= 6 & 7. 逆向推导任意参数的时刻 =================
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

        # 步长设为 1 小时扫描变号区间
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

# ================= 测试运行 =================
if __name__ == "__main__":
    # 使用北京坐标测试 (加入海拔参数，比如 50米)
    kit = SkyAstroKit(lat=31.1, lon=121.4, elevation=0)
    tz = kit.tz

    # 定义测试时间范围：2026年3月21日 到 2026年4月25日
    start_time = datetime(2026, 4, 11, tzinfo=tz)
    end_time = datetime(2026, 4, 30, tzinfo=tz)

    print("【1. 节气计算】")
    for row in kit.get_solar_terms(start_time, end_time): print(row)

    print("\n【2. 农历月计算】")
    for row in kit.get_lunar_months(start_time, end_time): print(row)

    print("\n【3. 每日日照与晨昏蒙影 (截取第一天)】")
    # 这里可以填入当地真实的温度和气压来逼近秒级精度极限
    for row in kit.get_daily_sun_events(start_time, end_time, temp_C=25.0, pressure_mbar=1010.0): print(row)

    print("\n【4. 太阳参数 (当前时刻)】")
    print(kit.get_body_params('sun', start_time)[0])

    print("\n【5. 月亮参数 (当前时刻)】")
    print(kit.get_body_params('moon', start_time)[0])

    print("\n【6. 逆向推导时刻：寻找太阳高度角正好达到 30° 的时间】")
    for row in kit.find_time_by_param('sun', 'Alt', 30.0, start_time, start_time + timedelta(days=1)): print(row)

    print("\n【7. 逆向推导时刻：寻找月相(相角)刚好为 0.5 (满月) 的时间】")
    for row in kit.find_time_by_param('moon', 'Phase', 0.5, start_time, end_time): print(row)
