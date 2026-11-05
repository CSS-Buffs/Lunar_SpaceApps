"""
Lunar reference frames.

ICRF -> Moon Mean-Earth/polar-axis (ME) frame: IAU WGCCRE 2015 rotation model
(Archinal et al. 2018, Celest. Mech. Dyn. Astr. 130:22), which includes the main
physical-libration terms. LOLA topography is defined in the ME frame.

ME -> Principal-Axis (PA) frame: constant rotation for DE440 from the NASA NAIF lunar
frame kernel moon_de440_250416.tf (0.0289 deg, ~875 m on the surface). GRAIL gravity
models are defined in the PA frame.

All rotation matrices R here map vectors: r_target = R @ r_source.
"""
from __future__ import annotations

import numpy as np

JD_J2000 = 2451545.0
ARCSEC = np.pi / (180 * 3600)

# DE440 ME<->PA angles from the NAIF frame kernel (TKFRAME_31009, axes 3,2,1, arcsec).
# Following the SPICE TK convention, r_PA = [67.8526"]_3 [78.6944"]_2 [0.2785"]_1 r_ME with
# passive (frame) rotations. The sign was confirmed empirically: propagating real LRO
# states from JPL Horizons, this convention reduces 1-day prediction errors by 2-5x
# compared with the opposite sign (see validate.py).
_ME_PA_ANGLES = (67.8526, 78.6944, 0.2785)


def _R1(a):  # passive (frame) rotations
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _R2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


R_ME_TO_PA = (_R3(_ME_PA_ANGLES[0] * ARCSEC) @ _R2(_ME_PA_ANGLES[1] * ARCSEC)
              @ _R1(_ME_PA_ANGLES[2] * ARCSEC))
R_PA_TO_ME = R_ME_TO_PA.T


def iau_moon_angles(jd_tdb):
    """Pole right ascension a0, declination d0 and prime meridian W [rad] (IAU 2015)."""
    d = np.asarray(jd_tdb, dtype=float) - JD_J2000
    T = d / 36525.0
    E = np.radians(np.array([
        125.045 - 0.0529921 * d, 250.089 - 0.1059842 * d, 260.008 + 13.0120009 * d,
        176.625 + 13.3407154 * d, 357.529 + 0.9856003 * d, 311.589 + 26.4057084 * d,
        134.963 + 13.0649930 * d, 276.617 + 0.3287146 * d, 34.226 + 1.7484877 * d,
        15.134 - 0.1589763 * d, 119.743 + 0.0036096 * d, 239.961 + 0.1643573 * d,
        25.053 + 12.9590088 * d]))
    s, c = np.sin(E), np.cos(E)
    a0 = (269.9949 + 0.0031 * T - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2]
          - 0.0172 * s[3] + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12])
    d0 = (66.5392 + 0.0130 * T + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2]
          + 0.0068 * c[3] - 0.0029 * c[5] + 0.0009 * c[6] + 0.0008 * c[9] - 0.0009 * c[12])
    W = (38.3213 + 13.17635815 * d - 1.4e-12 * d * d + 3.5610 * s[0] + 0.1208 * s[1]
         - 0.0642 * s[2] + 0.0158 * s[3] + 0.0252 * s[4] - 0.0066 * s[5] - 0.0047 * s[6]
         - 0.0046 * s[7] + 0.0028 * s[8] + 0.0052 * s[9] + 0.0040 * s[10] + 0.0019 * s[11]
         - 0.0044 * s[12])
    return np.radians(a0), np.radians(d0), np.radians(W)


def icrf_to_me(jd_tdb) -> np.ndarray:
    """3x3 rotation (or (N,3,3) for array input) from ICRF to the Moon ME frame."""
    a0, d0, W = iau_moon_angles(jd_tdb)
    if np.ndim(jd_tdb) == 0:
        return _R3(W) @ _R1(np.pi / 2 - d0) @ _R3(np.pi / 2 + a0)
    return np.array([_R3(w) @ _R1(np.pi / 2 - d) @ _R3(np.pi / 2 + a) for a, d, w in zip(a0, d0, W)])


def icrf_to_pa(jd_tdb) -> np.ndarray:
    return R_ME_TO_PA @ icrf_to_me(jd_tdb)


def latlon(r_bf):
    """Body-fixed (...,3) or (3,) -> lat [deg], east lon [deg, 0..360), radius."""
    r_bf = np.asarray(r_bf)
    x, y, z = r_bf[..., 0], r_bf[..., 1], r_bf[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    return np.degrees(np.arcsin(z / r)), np.mod(np.degrees(np.arctan2(y, x)), 360.0), r


def direction_from_latlon(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


if __name__ == "__main__":
    print("ME->PA rotation angle: %.5f deg" % np.degrees(np.arccos((np.trace(R_ME_TO_PA) - 1) / 2)))
