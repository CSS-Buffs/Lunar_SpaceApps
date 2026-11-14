"""
GRAIL spherical-harmonic lunar gravity (NASA PDS Geosciences Node).

Model: GRGM1200A (Goddard, full GRAIL mission), fully-normalised coefficients,
Principal-Axis frame, R_ref = 1738.0 km, GM = 4902.80011526323 km^3/s^2.
Only the first `degree` degrees are downloaded: the file is ordered by degree, so
the stream is closed as soon as enough lines have been read.

Acceleration uses the Cunningham/Montenbruck-Gill V/W recursion with fully-normalised
V and W terms (numerically stable to high degree). It is JIT-compiled with numba when
available.
"""
from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass

import numpy as np

import nasa_http

PDS_SHADR = "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/"
DEFAULT_MODEL = "gggrx_1200a_sha.tab"

try:
    from numba import njit
except ImportError:                                    # pure-python fallback
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@dataclass
class GravityModel:
    GM: float          # m^3/s^2
    R: float           # m
    degree: int
    C: np.ndarray      # (N+1, N+1) normalised
    S: np.ndarray
    name: str = ""

    def __post_init__(self):
        k_up, k_dn, k_z, a, b, c = _ratio_tables(self.degree)
        T = lambda x: np.ascontiguousarray(x.T)
        self._tables = (T(self.C), T(self.S), T(k_up), T(k_dn), T(k_z), T(a), T(b), c)

    def accel(self, r_pa: np.ndarray) -> np.ndarray:
        """Acceleration [m/s^2] at a PA-frame position [m] (3,)."""
        return _accel(np.ascontiguousarray(r_pa, dtype=np.float64), self.GM, self.R, self.degree,
                      *self._tables)

    def truncated(self, degree: int) -> "GravityModel":
        return GravityModel(self.GM, self.R, degree, self.C[:degree + 1, :degree + 1].copy(),
                            self.S[:degree + 1, :degree + 1].copy(), self.name)

    @property
    def J2(self):
        return -self.C[2, 0] * math.sqrt(5)


def load(degree: int = 150, model: str = DEFAULT_MODEL) -> GravityModel:
    """Download (cached) and parse the first `degree` degrees of a GRAIL SHADR model."""
    cache = nasa_http.DATA_DIR / "grail" / f"{model.replace('.tab', '')}_d{degree}.npz"
    if cache.exists():
        z = np.load(cache)
        return GravityModel(float(z["GM"]), float(z["R"]), degree, z["C"], z["S"], model)

    print(f"Downloading GRAIL {model} up to degree {degree} from NASA PDS ...")
    C = np.zeros((degree + 1, degree + 1))
    S = np.zeros((degree + 1, degree + 1))
    req = urllib.request.Request(PDS_SHADR + model, headers={"User-Agent": nasa_http.USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        head = resp.readline().decode().split(",")
        R, GM = float(head[0]) * 1e3, float(head[1]) * 1e9
        for raw in resp:
            parts = raw.decode().split(",")
            n, m = int(parts[0]), int(parts[1])
            if n > degree:
                break
            C[n, m], S[n, m] = float(parts[2]), float(parts[3])
    C[0, 0] = 1.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, GM=GM, R=R, C=C, S=S)
    return GravityModel(GM, R, degree, C, S, model)


def _lognorm(n, m):
    return 0.5 * (math.log(2.0 - (m == 0)) + math.log(2 * n + 1)
                  + math.lgamma(n - m + 1) - math.lgamma(n + m + 1))


def _ratio_tables(N):
    """Normalisation ratios N_nm / N_{n+1,k} used to convert normalised V,W back."""
    k_up = np.zeros((N + 1, N + 1))     # N_nm / N_{n+1,m+1}
    k_dn = np.zeros((N + 1, N + 1))     # N_nm / N_{n+1,m-1}
    k_z = np.zeros((N + 1, N + 1))      # N_nm / N_{n+1,m}
    for n in range(N + 1):
        for m in range(n + 1):
            ln = _lognorm(n, m)
            k_up[n, m] = math.exp(ln - _lognorm(n + 1, m + 1))
            k_z[n, m] = math.exp(ln - _lognorm(n + 1, m))
            if m > 0:
                k_dn[n, m] = math.exp(ln - _lognorm(n + 1, m - 1))
    # recursion coefficients for normalised V/W up to degree N+1
    M = N + 2
    a = np.zeros((M, M))
    b = np.zeros((M, M))
    for n in range(1, M):
        for m in range(n):
            a[n, m] = math.sqrt((2 * n + 1) * (2 * n - 1) / ((n - m) * (n + m)))
            if n >= 2:
                b[n, m] = math.sqrt((2 * n + 1) * (n + m - 1) * (n - m - 1)
                                    / ((2 * n - 3) * (n + m) * (n - m)))
    c = np.zeros(M)
    for m in range(1, M):
        c[m] = math.sqrt(3.0) if m == 1 else math.sqrt((2 * m + 1) / (2 * m))
    return k_up, k_dn, k_z, a, b, c


@njit(cache=True, fastmath=False)
def _accel(r, GM, R, N, C, S, k_up, k_dn, k_z, a, b, c):
    # All tables are passed transposed ([m, n]) so inner loops over n are contiguous
    x, y, z = r[0], r[1], r[2]
    r2 = x * x + y * y + z * z
    rho = R / r2
    x0, y0, z0, rr = x * rho, y * rho, z * rho, R * rho
    M = N + 2
    V = np.zeros((M, M))
    W = np.zeros((M, M))
    V[0, 0] = R / math.sqrt(r2)
    for m in range(M):
        if m > 0:
            V[m, m] = c[m] * (x0 * V[m - 1, m - 1] - y0 * W[m - 1, m - 1])
            W[m, m] = c[m] * (x0 * W[m - 1, m - 1] + y0 * V[m - 1, m - 1])
        for n in range(m + 1, M):
            V[m, n] = a[m, n] * z0 * V[m, n - 1]
            W[m, n] = a[m, n] * z0 * W[m, n - 1]
            if n >= m + 2:
                V[m, n] -= b[m, n] * rr * V[m, n - 2]
                W[m, n] -= b[m, n] * rr * W[m, n - 2]
    ax = ay = az = 0.0
    for m in range(N + 1):
        for n in range(m, N + 1):
            Cn, Sn = C[m, n], S[m, n]
            if m == 0:
                ax -= Cn * k_up[0, n] * V[1, n + 1]
                ay -= Cn * k_up[0, n] * W[1, n + 1]
            else:
                f = (n - m + 2) * (n - m + 1)
                ax += 0.5 * (k_up[m, n] * (-Cn * V[m + 1, n + 1] - Sn * W[m + 1, n + 1])
                             + f * k_dn[m, n] * (Cn * V[m - 1, n + 1] + Sn * W[m - 1, n + 1]))
                ay += 0.5 * (k_up[m, n] * (-Cn * W[m + 1, n + 1] + Sn * V[m + 1, n + 1])
                             + f * k_dn[m, n] * (-Cn * W[m - 1, n + 1] + Sn * V[m - 1, n + 1]))
            az += (n - m + 1) * k_z[m, n] * (-Cn * V[m, n + 1] - Sn * W[m, n + 1])
    s = GM / (R * R)
    out = np.empty(3)
    out[0], out[1], out[2] = s * ax, s * ay, s * az
    return out


# ---- gridded quantities (maps) -------------------------------------------------
def _legendre_rows(N, sinlat):
    """Fully normalised associated Legendre P̄nm(sinlat): array (N+1, N+1, L)."""
    t = np.asarray(sinlat, dtype=float)
    u = np.sqrt(1 - t * t)
    P = np.zeros((N + 1, N + 1, t.size))
    P[0, 0] = 1.0
    if N >= 1:
        P[1, 1] = math.sqrt(3) * u
    for m in range(2, N + 1):
        P[m, m] = math.sqrt((2 * m + 1) / (2 * m)) * u * P[m - 1, m - 1]
    for m in range(0, N + 1):
        for n in range(m + 1, N + 1):
            an = math.sqrt((2 * n + 1) * (2 * n - 1) / ((n - m) * (n + m)))
            P[n, m] = an * t * P[n - 1, m]
            if n >= m + 2:
                bn = math.sqrt((2 * n + 1) * (n + m - 1) * (n - m - 1) / ((2 * n - 3) * (n + m) * (n - m)))
                P[n, m] -= bn * P[n - 2, m]
    return P


def free_air_anomaly_grid(model: GravityModel, lat_deg, lon_deg, r=None, nmin=2):
    """Free-air gravity anomaly [mGal] on a lat/lon grid at radius r (default R_ref):
    dg = GM/r^2 * sum_n (n-1)(R/r)^n sum_m P̄nm (C̄nm cos mλ + S̄nm sin mλ)."""
    N = model.degree
    r = model.R if r is None else r
    P = _legendre_rows(N, np.sin(np.radians(lat_deg)))           # (n, m, L)
    n = np.arange(N + 1)
    wn = np.where(n >= nmin, (n - 1) * (model.R / r) ** n, 0.0)
    A = np.einsum("n,nm,nml->ml", wn, model.C, P)                # (m, L)
    B = np.einsum("n,nm,nml->ml", wn, model.S, P)
    lam = np.radians(lon_deg)
    mm = np.arange(N + 1)[:, None] * lam[None, :]
    g = A.T @ np.cos(mm) + B.T @ np.sin(mm)                       # (L, K)
    return model.GM / r**2 * g * 1e5                              # m/s^2 -> mGal


if __name__ == "__main__":
    import time
    g = load(150)
    print(f"{g.name}: GM={g.GM:.6e}, R={g.R:.0f}, J2={g.J2:.7e}")
    r = np.array([1838e3, 300e3, 400e3])
    a = g.accel(r)                      # compile
    t = time.perf_counter()
    for _ in range(2000):
        g.accel(r)
    print("accel", a, f"{(time.perf_counter() - t) / 2000 * 1e6:.1f} us/eval")
    # check against finite differences of point-mass + J2 analytic
    g2 = g.truncated(2)
    x, y, z = r
    rn = np.linalg.norm(r)
    J2, R, GM = g2.J2, g2.R, g2.GM
    aJ2 = -GM * r / rn**3 - 1.5 * J2 * GM * R**2 / rn**5 * np.array(
        [x * (1 - 5 * z * z / rn**2), y * (1 - 5 * z * z / rn**2), z * (3 - 5 * z * z / rn**2)])
    g20 = GravityModel(g2.GM, g2.R, 2, np.diag([1.0, 0, 0]) + np.array([[0, 0, 0], [0, 0, 0], [g2.C[2, 0], 0, 0]]),
                       np.zeros((3, 3)))
    print("J2-only check |diff| =", np.linalg.norm(g20.accel(r) - aJ2), "m/s^2")
