"""
NASA/JPL Horizons API client  (https://ssd.jpl.nasa.gov/api/horizons.api)

Used for:
  * Earth (399) and Sun (10) positions relative to the Moon's centre -> third-body
    perturbations and real sunlight direction
  * LRO (-85) reconstructed trajectory -> initial conditions + truth for validation
  * sub-Earth / sub-solar points on the Moon -> check of our lunar orientation model

Vectors are Moon-centred (CENTER=500@301), ICRF axes (REF_PLANE=FRAME), km & km/s,
time scale TDB. Results are cached in data/horizons/.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline

import nasa_http

API = "https://ssd.jpl.nasa.gov/api/horizons.api?"
TDB_MINUS_UTC = 69.184          # s, valid since 2017-01-01 (37 leap s + 32.184 s)
JD_J2000 = 2451545.0

BODIES = {"earth": "399", "sun": "10", "lro": "-85", "moon": "301"}


def utc_iso_to_jd_tdb(iso: str) -> float:
    """'2026-11-05T12:00:00' (UTC) -> Julian date (TDB)."""
    t = np.datetime64(iso.replace("Z", ""), "ms")
    jd_utc = (t - np.datetime64("2000-01-01T12:00:00", "ms")) / np.timedelta64(1, "D") + JD_J2000
    return float(jd_utc + TDB_MINUS_UTC / 86400.0)


def jd_tdb_to_utc_iso(jd: float) -> str:
    ms = (jd - JD_J2000 - TDB_MINUS_UTC / 86400.0) * 86400e3
    return str(np.datetime64("2000-01-01T12:00:00", "ms") + np.timedelta64(int(round(ms)), "ms"))


def _call(params: dict) -> str:
    q = {"format": "json", "MAKE_EPHEM": "YES", "OBJ_DATA": "NO", **params}
    q = {k: (f"'{v}'" if k not in ("format",) else v) for k, v in q.items()}
    url = nasa_http.build_url(API, q)
    res = nasa_http.get_json(url, nasa_http.cache_path("horizons", url, ".json"))
    text = res.get("result", "")
    if "$$SOE" not in text:
        raise RuntimeError("Horizons returned no ephemeris:\n" + text[-1500:])
    return text


def _rows(text: str) -> list[list[str]]:
    body = text.split("$$SOE")[1].split("$$EOE")[0]
    return [[c.strip() for c in ln.split(",")] for ln in body.strip().splitlines() if ln.strip()]


@dataclass
class Ephemeris:
    """Tabulated state of a body relative to the Moon, ICRF, SI units, with Hermite interpolation."""
    name: str
    jd_tdb: np.ndarray          # (N,)
    pos: np.ndarray             # (N,3) m
    vel: np.ndarray             # (N,3) m/s

    def __post_init__(self):
        t = (self.jd_tdb - self.jd_tdb[0]) * 86400.0
        self._t0 = self.jd_tdb[0]
        self._spl = CubicHermiteSpline(t, self.pos, self.vel, axis=0)

    def position(self, jd_tdb):
        """Interpolated position [m]; jd_tdb scalar or array -> (3,) or (N,3)."""
        return self._spl((np.asarray(jd_tdb) - self._t0) * 86400.0)

    def position_at_seconds(self, t_rel, jd0):
        return self._spl((jd0 - self._t0) * 86400.0 + np.asarray(t_rel))

    def velocity(self, jd_tdb):
        return self._spl((np.asarray(jd_tdb) - self._t0) * 86400.0, 1)


def vectors(body: str, jd_start: float, jd_stop: float, step: str = "10m",
            center: str = "500@301") -> Ephemeris:
    """State vectors of `body` (name or Horizons ID) between two TDB Julian dates."""
    cmd = BODIES.get(body.lower(), body)
    text = _call(dict(COMMAND=cmd, CENTER=center, EPHEM_TYPE="VECTORS",
                      START_TIME=f"JD {jd_start:.9f}", STOP_TIME=f"JD {jd_stop:.9f}",
                      STEP_SIZE=step, VEC_TABLE="2", REF_PLANE="FRAME", REF_SYSTEM="ICRF",
                      OUT_UNITS="KM-S", CSV_FORMAT="YES", VEC_LABELS="NO", TIME_TYPE="TDB"))
    rows = _rows(text)
    jd = np.array([float(r[0]) for r in rows])
    st = np.array([[float(x) for x in r[2:8]] for r in rows]) * 1e3
    return Ephemeris(body, jd, st[:, :3], st[:, 3:])


def sub_points(jd_start: float, jd_stop: float, step: str = "6h"):
    """Sub-Earth and sub-solar lon/lat on the Moon as seen from Earth's centre (Horizons
    observer quantities 14 & 15). Returns dict of arrays; times in UTC Julian date."""
    text = _call(dict(COMMAND="301", CENTER="500@399", EPHEM_TYPE="OBSERVER",
                      START_TIME=f"JD {jd_start - TDB_MINUS_UTC / 86400:.9f}",
                      STOP_TIME=f"JD {jd_stop - TDB_MINUS_UTC / 86400:.9f}",
                      STEP_SIZE=step, QUANTITIES="14,15", CSV_FORMAT="YES", CAL_FORMAT="JD",
                      ANG_FORMAT="DEG", APPARENT="AIRLESS"))
    rows = _rows(text)
    # columns: JDUT, flag, flag, ObsSub-LON, ObsSub-LAT, SunSub-LON, SunSub-LAT
    arr = np.array([[float(r[0])] + [float(x) for x in r[3:7]] for r in rows])
    return dict(jd_utc=arr[:, 0], earth_lon=arr[:, 1], earth_lat=arr[:, 2],
                sun_lon=arr[:, 3], sun_lat=arr[:, 4])


def perturber_ephemerides(jd_start: float, jd_stop: float, pad_days: float = 0.5):
    """Earth and Sun around the Moon, 1 h sampling, padded for interpolation."""
    a, b = jd_start - pad_days, jd_stop + pad_days
    return {"earth": vectors("earth", a, b, "1h"), "sun": vectors("sun", a, b, "1h")}


if __name__ == "__main__":
    jd = utc_iso_to_jd_tdb("2026-11-05T00:00:00")
    e = vectors("earth", jd, jd + 1, "6h")
    print("Earth from Moon [km]:", e.pos[0] / 1e3, " |r| =", np.linalg.norm(e.pos[0]) / 1e3)
    s = sub_points(jd, jd + 1, "12h")
    print("Sub-Earth lon/lat:", s["earth_lon"], s["earth_lat"])
