"""
Lunar physics: constants, gravity model and orbit propagation.

Gravity model (body-fixed, Mean Earth / Polar Axis frame):
    * central term                 GM / r
    * J2 (polar flattening)        from GRAIL GRGM1200A
    * C22 (equatorial ellipticity) from GRAIL GRGM1200A
    * optional Earth third-body tide (Earth fixed near +X in the body frame
      because the Moon is in synchronous rotation)

The orbit is integrated in a Moon-centred inertial frame whose axes coincide
with the body-fixed frame at t = 0; the Moon spins about +Z at its sidereal
rate. Altitude is measured against the real LOLA terrain, so the integrator
can detect impacts with crater rims and mountains.

Limitations: degree-2 gravity only. Real low lunar orbits are strongly
perturbed by mascons (higher-degree terms); add a full spherical-harmonic
model (e.g. GRGM1200A from PDS) for mission-grade accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

# --- Constants (SI) --------------------------------------------------------
GM_MOON = 4.9028001218467e12          # m^3/s^2   GRGM1200A
R_GRAV = 1738.0e3                     # m         gravity-field reference radius
R_LOLA = 1737.4e3                     # m         LOLA reference sphere
J2 = 2.0321568e-4                     # unnormalised (= -C20 * sqrt(5))
C22 = 2.2382e-5                       # unnormalised
OMEGA_MOON = 2 * np.pi / (27.321661 * 86400.0)   # rad/s sidereal rotation
GM_EARTH = 3.986004418e14             # m^3/s^2
EARTH_DIST = 384_400e3                # m mean Earth-Moon distance


def gravity_body(r_bf: np.ndarray, earth: bool = False, t: float = 0.0) -> np.ndarray:
    """Gravitational acceleration [m/s^2] at body-fixed position(s) r_bf (3,) or (3,N)."""
    x, y, z = r_bf
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    r3, r5, r7 = r2 * r, r2 * r2 * r, r2 * r2 * r2 * r

    # central
    ax = -GM_MOON * x / r3
    ay = -GM_MOON * y / r3
    az = -GM_MOON * z / r3

    # J2
    k = -1.5 * J2 * GM_MOON * R_GRAV**2 / r5
    zz = 5 * z * z / r2
    ax += k * x * (1 - zz)
    ay += k * y * (1 - zz)
    az += k * z * (3 - zz)

    # C22 : U = 3 GM R^2 C22 (x^2 - y^2) / r^5
    c = 3 * GM_MOON * R_GRAV**2 * C22
    d = x * x - y * y
    ax += c * (2 * x / r5 - 5 * x * d / r7)
    ay += c * (-2 * y / r5 - 5 * y * d / r7)
    az += c * (-5 * z * d / r7)

    acc = np.array([ax, ay, az])
    if earth:
        acc = acc + earth_tide_body(r_bf)
    return acc


def earth_tide_body(r_bf: np.ndarray) -> np.ndarray:
    """Third-body (tidal) acceleration from Earth, Earth on body-fixed +X axis."""
    re = np.array([EARTH_DIST, 0.0, 0.0]).reshape(3, *([1] * (np.ndim(r_bf) - 1)))
    d = re - r_bf
    dn = np.linalg.norm(d, axis=0)
    return GM_EARTH * (d / dn**3 - re / EARTH_DIST**3)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def inertial_to_body(r_in: np.ndarray, t) -> np.ndarray:
    """Rotate inertial (3,) or (3,N) positions into body-fixed coordinates at time(s) t."""
    t = np.asarray(t, dtype=float)
    th = -OMEGA_MOON * t
    c, s = np.cos(th), np.sin(th)
    x, y, z = r_in
    return np.array([c * x - s * y, s * x + c * y, z])


def potential_body(r_bf: np.ndarray, earth: bool = False) -> np.ndarray:
    """Gravitational potential U [m^2/s^2] (sign convention a = +grad U)."""
    x, y, z = r_bf
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    U = GM_MOON / r
    U -= GM_MOON * J2 * R_GRAV**2 / (2 * r**3) * (3 * z * z / r2 - 1)
    U += 3 * GM_MOON * R_GRAV**2 * C22 * (x * x - y * y) / r**5
    if earth:
        d = np.sqrt((EARTH_DIST - x) ** 2 + y * y + z * z)
        U += GM_EARTH * (1 / d - x / EARTH_DIST**2)
    return U


def jacobi_constant(r_in, v_in, t, earth: bool = False):
    """Energy in the Moon-fixed rotating frame. Conserved exactly by this force model,
    so its drift measures numerical integration error."""
    r_bf = inertial_to_body(r_in, t)
    x, y, _ = r_in
    # velocity relative to rotating frame: v - omega x r  (expressed in inertial axes)
    v_rel = np.array([v_in[0] + OMEGA_MOON * y, v_in[1] - OMEGA_MOON * x, v_in[2]])
    rho2 = r_bf[0] ** 2 + r_bf[1] ** 2
    return 0.5 * np.sum(v_rel**2, axis=0) - potential_body(r_bf, earth) - 0.5 * OMEGA_MOON**2 * rho2


def body_to_latlon(r_bf: np.ndarray):
    x, y, z = r_bf
    r = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    return lat, lon, r


def surface_gravity(dem) -> np.ndarray:
    """|g| [m/s^2] (gravity + centrifugal) on the LOLA surface for every DEM pixel."""
    lat = np.radians(dem.lat_deg)[:, None]
    lon = np.radians(dem.lon_deg)[None, :]
    rad = dem.radius_m
    r_bf = np.array([rad * np.cos(lat) * np.cos(lon),
                     rad * np.cos(lat) * np.sin(lon),
                     rad * np.sin(lat) * np.ones_like(lon)])
    g = gravity_body(r_bf)
    g[0] += OMEGA_MOON**2 * r_bf[0]
    g[1] += OMEGA_MOON**2 * r_bf[1]
    return np.linalg.norm(g, axis=0)


# --- Orbits ----------------------------------------------------------------
def kepler_to_state(a, e, inc_deg, raan_deg, argp_deg, nu_deg, mu=GM_MOON):
    """Classical orbital elements -> inertial position/velocity [m, m/s]."""
    i, O, w, nu = np.radians([inc_deg, raan_deg, argp_deg, nu_deg])
    p = a * (1 - e * e)
    r_pf = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO, ci, si, cw, sw = np.cos(O), np.sin(O), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    Q = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return Q @ r_pf, Q @ v_pf


@dataclass
class OrbitResult:
    t: np.ndarray                 # s
    r_in: np.ndarray              # (3,N) inertial position, m
    v_in: np.ndarray              # (3,N) inertial velocity, m/s
    r_bf: np.ndarray              # (3,N) body-fixed position, m
    lat: np.ndarray
    lon: np.ndarray
    alt_terrain: np.ndarray       # m above LOLA surface
    impacted: bool
    impact_time: float | None = None
    jacobi: np.ndarray = field(default=None)   # rotating-frame energy (should stay constant)


def propagate(r0, v0, duration_s, dem=None, earth=True, n_out=4000,
              rtol=1e-10, atol=1e-6, max_step=None) -> OrbitResult:
    """Integrate an orbit with DOP853; stops on impact with LOLA terrain if dem is given."""

    def rhs(t, s):
        r_in = s[:3]
        th = OMEGA_MOON * t
        r_bf = rot_z(-th) @ r_in
        a_bf = gravity_body(r_bf, earth=earth)
        return np.concatenate([s[3:], rot_z(th) @ a_bf])

    events = None
    if dem is not None:
        def hit_ground(t, s):
            lat, lon, r = body_to_latlon(rot_z(-OMEGA_MOON * t) @ s[:3])
            return r - (R_LOLA + dem.interp_height(lat, lon))
        hit_ground.terminal = True
        hit_ground.direction = -1
        events = [hit_ground]

    t_eval = np.linspace(0, duration_s, n_out)
    sol = solve_ivp(rhs, (0, duration_s), np.concatenate([r0, v0]), method="DOP853",
                    t_eval=t_eval, events=events, rtol=rtol, atol=atol,
                    max_step=max_step or (10.0 if dem is not None else np.inf))
    if not sol.success:
        raise RuntimeError(sol.message)

    impacted = bool(events and sol.t_events[0].size)
    t, y = sol.t, sol.y
    if impacted:                       # append the exact impact point
        t = np.append(t, sol.t_events[0][0])
        y = np.hstack([y, sol.y_events[0][0][:, None]])

    r_in, v_in = y[:3], y[3:]
    r_bf = inertial_to_body(r_in, t)
    lat, lon, r = body_to_latlon(r_bf)
    ground = R_LOLA + (dem.interp_height(lat, lon) if dem is not None else 0.0)
    jacobi = jacobi_constant(r_in, v_in, t, earth)
    return OrbitResult(t, r_in, v_in, r_bf, lat, lon, r - ground, impacted,
                       float(sol.t_events[0][0]) if impacted else None, jacobi)


if __name__ == "__main__":
    g_eq = GM_MOON / R_LOLA**2
    v_esc = np.sqrt(2 * GM_MOON / R_LOLA)
    print(f"Surface gravity (point mass) : {g_eq:.4f} m/s^2")
    print(f"Escape velocity              : {v_esc:.1f} m/s")
    a = R_LOLA + 100e3
    print(f"Circular orbit 100 km        : v = {np.sqrt(GM_MOON / a):.1f} m/s, "
          f"T = {2 * np.pi * np.sqrt(a**3 / GM_MOON) / 60:.1f} min")
