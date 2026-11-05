"""
High-fidelity lunar orbit dynamics.

Force model (Moon-centred ICRF frame, which is what JPL Horizons uses):
  * GRAIL GRGM1200A spherical-harmonic gravity to a chosen degree/order, evaluated in
    the lunar Principal-Axis frame (orientation: IAU 2015 model + DE440 ME->PA rotation)
  * Earth and Sun third-body perturbations, with their positions taken from the
    JPL Horizons API (Hermite-interpolated)
  * optional cannonball solar radiation pressure with lunar shadow
Terrain: altitude is measured against NASA LOLA topography; the integration stops on impact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import grail_gravity
import horizons
import lunar_frames as lf

R_LOLA = 1737.4e3                       # m, LOLA reference sphere
GM_EARTH = 3.986004418e14               # m^3/s^2
GM_SUN = 1.32712440041279419e20         # m^3/s^2 (DE440)
AU = 1.495978707e11                     # m
P_SUN_1AU = 4.56e-6                     # N/m^2 solar radiation pressure at 1 AU


class Orientation:
    """ICRF -> PA (and ME) rotation, tabulated every 30 min and spline-interpolated (error < 1e-11 rad)."""

    def __init__(self, jd0: float, duration_s: float):
        self.jd0 = jd0
        n = max(8, int(duration_s / 1800) + 8)
        self.t = np.linspace(-3600.0, duration_s + 3600.0, n)
        mats = lf.icrf_to_me(jd0 + self.t / 86400.0)                  # (n,3,3)
        self._me = CubicSpline(self.t, mats.reshape(n, 9), axis=0)

    def icrf_to_me(self, t):
        m = self._me(t)
        return m.reshape(3, 3) if np.ndim(t) == 0 else m.reshape(-1, 3, 3)

    def icrf_to_pa(self, t):
        return lf.R_ME_TO_PA @ self.icrf_to_me(t)


@dataclass
class ForceModel:
    gravity: grail_gravity.GravityModel
    jd0: float                           # epoch (TDB Julian date) for t = 0
    duration_s: float
    third_bodies: bool = True
    srp_area_mass: float = 0.0           # m^2/kg (0 disables SRP)
    srp_cr: float = 1.3

    def __post_init__(self):
        self.orient = Orientation(self.jd0, self.duration_s)
        self.ephem = None
        if self.third_bodies or self.srp_area_mass > 0:
            self.ephem = horizons.perturber_ephemerides(self.jd0, self.jd0 + self.duration_s / 86400)

    def _body(self, name, t):
        return self.ephem[name].position_at_seconds(t, self.jd0)

    def accel(self, t: float, r: np.ndarray) -> np.ndarray:
        Rpa = self.orient.icrf_to_pa(t)
        a = Rpa.T @ self.gravity.accel(Rpa @ r)
        if self.third_bodies:
            for name, gm in (("earth", GM_EARTH), ("sun", GM_SUN)):
                rb = self._body(name, t)
                d = rb - r
                a += gm * (d / np.linalg.norm(d) ** 3 - rb / np.linalg.norm(rb) ** 3)
        if self.srp_area_mass > 0:
            rs = self._body("sun", t)
            shat = rs / np.linalg.norm(rs)
            along = r @ shat
            in_shadow = along < 0 and np.linalg.norm(r - along * shat) < R_LOLA
            if not in_shadow:
                d = r - rs
                dn = np.linalg.norm(d)
                a += self.srp_cr * P_SUN_1AU * (AU / dn) ** 2 * self.srp_area_mass * d / dn
        return a


@dataclass
class OrbitResult:
    t: np.ndarray            # s since epoch
    jd_tdb: np.ndarray
    r: np.ndarray            # (N,3) ICRF position, m
    v: np.ndarray            # (N,3) ICRF velocity, m/s
    r_me: np.ndarray         # (N,3) Moon-fixed (ME) position, m
    lat: np.ndarray
    lon: np.ndarray
    alt_terrain: np.ndarray  # m above LOLA surface (or above 1737.4 km sphere if no DEM)
    impacted: bool
    impact_time: float | None


def propagate(r0, v0, force: ForceModel, duration_s: float, dem=None, n_out: int = 4000,
              rtol: float = 1e-11, atol: float = 1e-5, max_step: float | None = None) -> OrbitResult:
    """Integrate with DOP853 (8th order). Stops at terrain impact if a LOLA DEM is given."""

    def rhs(t, s):
        return np.concatenate([s[3:], force.accel(t, s[:3])])

    events = None
    if dem is not None:
        def hit_ground(t, s):
            lat, lon, rad = lf.latlon(force.orient.icrf_to_me(t) @ s[:3])
            return rad - (R_LOLA + dem.interp_height(lat, lon))
        hit_ground.terminal, hit_ground.direction = True, -1
        events = [hit_ground]
        if max_step is None:   # keep steps short enough not to skip over terrain when low
            peri_alt = np.linalg.norm(r0) - R_LOLA
            max_step = 10.0 if peri_alt < 40e3 else 60.0

    t_eval = np.linspace(0.0, duration_s, n_out)
    sol = solve_ivp(rhs, (0.0, duration_s), np.concatenate([r0, v0]), method="DOP853",
                    t_eval=t_eval, events=events, rtol=rtol, atol=atol,
                    max_step=max_step if max_step else np.inf)
    if not sol.success:
        raise RuntimeError(sol.message)
    t, y = sol.t, sol.y
    impacted = bool(events and sol.t_events[0].size)
    if impacted:
        t = np.append(t, sol.t_events[0][0])
        y = np.hstack([y, sol.y_events[0][0][:, None]])
    r, v = y[:3].T, y[3:].T
    r_me = np.einsum("nij,nj->ni", force.orient.icrf_to_me(t), r)
    lat, lon, rad = lf.latlon(r_me)
    ground = R_LOLA + (dem.interp_height(lat, lon) if dem is not None else 0.0)
    return OrbitResult(t, force.jd0 + t / 86400.0, r, v, r_me, lat, lon, rad - ground,
                       impacted, float(sol.t_events[0][0]) if impacted else None)


# ---- initial conditions -------------------------------------------------------
def kepler_to_state(a, e, inc_deg, raan_deg, argp_deg, nu_deg, mu):
    """Classical elements -> position/velocity in the frame the elements refer to."""
    i, O, w, nu = np.radians([inc_deg, raan_deg, argp_deg, nu_deg])
    p = a * (1 - e * e)
    r_pf = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO, ci, si, cw, sw = np.cos(O), np.sin(O), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    Q = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return Q @ r_pf, Q @ v_pf


def state_from_lunar_elements(jd_tdb, alt_peri_m, ecc, inc, raan, argp, nu, gm):
    """Elements referred to the lunar equator / ME prime meridian at epoch -> ICRF state."""
    a = (R_LOLA + alt_peri_m) / (1 - ecc)
    r_me, v_me = kepler_to_state(a, ecc, inc, raan, argp, nu, gm)
    R = lf.icrf_to_me(jd_tdb)
    return R.T @ r_me, R.T @ v_me


def lro_state(jd_tdb):
    """LRO (-85) state relative to the Moon from JPL Horizons (ICRF, m and m/s)."""
    e = horizons.vectors("lro", jd_tdb, jd_tdb + 1 / 1440, "1m")
    return e.pos[0], e.vel[0]


def rtn_errors(r_sim, v_sim, r_true):
    """Position error resolved into radial / along-track / cross-track [m]."""
    d = r_sim - r_true
    rh = r_sim / np.linalg.norm(r_sim, axis=1, keepdims=True)
    h = np.cross(r_sim, v_sim)
    nh = h / np.linalg.norm(h, axis=1, keepdims=True)
    th = np.cross(nh, rh)
    return np.einsum("ij,ij->i", d, rh), np.einsum("ij,ij->i", d, th), np.einsum("ij,ij->i", d, nh)


if __name__ == "__main__":
    g = grail_gravity.load(150)
    print(f"Surface gravity (GM/R^2): {g.GM / R_LOLA**2:.4f} m/s^2,  "
          f"escape velocity {np.sqrt(2 * g.GM / R_LOLA):.1f} m/s")
