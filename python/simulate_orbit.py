"""
Lunar orbit simulation driven by the JPL Horizons API.

  * Earth & Sun positions (third-body forces, sunlight) come from Horizons for the epoch
  * Gravity: GRAIL GRGM1200A spherical harmonics (default degree 100)
  * Terrain: LOLA; the run stops if the spacecraft hits the ground

Modes
  --lro            start from the real LRO state (Horizons body -85) at --epoch and
                   compare the prediction with LRO's actual reconstructed trajectory
  (default)        custom orbit from elements relative to the lunar equator

Examples
    python simulate_orbit.py --lro --epoch 2025-06-01T00:00 --days 1
    python simulate_orbit.py --lro --epoch 2025-06-01T00:00 --days 1 --compare-degrees 2,20,60,100,150
    python simulate_orbit.py --alt 50 --inc 90 --days 2 --epoch 2026-11-20T00:00
    python simulate_orbit.py --alt 8 --ecc 0.002 --inc 85 --days 3          # decays into terrain
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grail_gravity
import horizons
import lola_data
import lunar_physics as ph
from moon_viewer import OUT_DIR, add_orbit, moon_figure, write


def run(r0, v0, jd0, dur, degree, dem, third_bodies=True, srp=0.0, n_out=3000):
    g = grail_gravity.load(max(degree, 2)).truncated(degree)
    f = ph.ForceModel(g, jd0, dur, third_bodies=third_bodies, srp_area_mass=srp)
    t = time.perf_counter()
    o = ph.propagate(r0, v0, f, dur, dem=dem, n_out=n_out)
    return o, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", default="2025-06-01T00:00", help="start time, UTC ISO")
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--degree", type=int, default=100, help="GRAIL gravity degree/order")
    ap.add_argument("--lro", action="store_true", help="use LRO's real state from Horizons and validate")
    ap.add_argument("--compare-degrees", default="", help="e.g. 2,20,60,100 (with --lro)")
    ap.add_argument("--alt", type=float, default=50.0, help="periapsis altitude [km] (custom orbit)")
    ap.add_argument("--ecc", type=float, default=0.0)
    ap.add_argument("--inc", type=float, default=90.0, help="inclination to lunar equator [deg]")
    ap.add_argument("--raan", type=float, default=0.0)
    ap.add_argument("--argp", type=float, default=0.0)
    ap.add_argument("--srp", type=float, default=0.0, help="area/mass [m^2/kg] for solar pressure")
    ap.add_argument("--no-third-bodies", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    jd0 = horizons.utc_iso_to_jd_tdb(a.epoch)
    dur = a.days * 86400.0
    dem = lola_data.load(4)
    n_out = int(min(20000, max(1500, dur / 60)))
    OUT_DIR.mkdir(exist_ok=True)

    if a.lro:
        truth = horizons.vectors("lro", jd0, jd0 + a.days, "2m")
        jd0 = truth.jd_tdb[0]                       # use Horizons' exact epoch
        r0, v0 = truth.pos[0], truth.vel[0]
        print(f"LRO state from JPL Horizons at {a.epoch} UTC: |r| = {np.linalg.norm(r0) / 1e3:.2f} km, "
              f"|v| = {np.linalg.norm(v0):.2f} m/s")
    else:
        g = grail_gravity.load(max(a.degree, 2))
        r0, v0 = ph.state_from_lunar_elements(jd0, a.alt * 1e3, a.ecc, a.inc, a.raan, a.argp, 0.0, g.GM)

    degrees = [int(x) for x in a.compare_degrees.split(",") if x] or [a.degree]
    results = {}
    for deg in degrees:
        o, secs = run(r0, v0, jd0, dur, deg, dem, not a.no_third_bodies, a.srp, n_out)
        results[deg] = o
        msg = f"degree {deg:4d}: {secs:6.1f} s, altitude above terrain {o.alt_terrain.min() / 1e3:.2f}-" \
              f"{o.alt_terrain.max() / 1e3:.2f} km"
        if a.lro:
            tr = truth.position(o.jd_tdb)
            err = np.linalg.norm(o.r - tr, axis=1)
            R, T, N = ph.rtn_errors(o.r, o.v, tr)
            msg += (f" | error vs real LRO: max {err.max():8.1f} m, final {err[-1]:8.1f} m "
                    f"(radial RMS {np.sqrt(np.mean(R ** 2)):.1f} m)")
        print(msg)
        if o.impacted:
            print(f"   IMPACT after {o.impact_time / 3600:.2f} h at lat {o.lat[-1]:.3f}, lon {o.lon[-1]:.3f} E")

    main_deg = degrees[-1]
    o = results[main_deg]

    # --- plots
    nrows = 2 if a.lro else 1
    fig, ax = plt.subplots(nrows, 1, figsize=(10, 3.6 * nrows), sharex=True, squeeze=False)
    ax = ax[:, 0]
    ax[0].plot(o.t / 3600, o.alt_terrain / 1e3, lw=0.7, label=f"simulated (deg {main_deg})")
    if a.lro:
        tr = truth.position(o.jd_tdb)
        ax[0].plot(o.t / 3600, (np.linalg.norm(tr, axis=1) - ph.R_LOLA
                                - (o.alt_terrain - (np.linalg.norm(o.r, axis=1) - ph.R_LOLA))) / 1e3,
                   lw=0.7, ls="--", label="real LRO (Horizons)")
        for deg, od in results.items():
            ax[1].semilogy(od.t / 3600, np.linalg.norm(od.r - truth.position(od.jd_tdb), axis=1) + 1e-3,
                           lw=1, label=f"GRAIL degree {deg}")
        ax[1].set_ylabel("Position error vs real LRO [m]")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
    ax[0].set_ylabel("Altitude above LOLA terrain [km]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[-1].set_xlabel(f"Hours after {a.epoch} UTC")
    fig.suptitle("Lunar orbit: GRAIL gravity + Earth/Sun from JPL Horizons + LOLA terrain")
    fig.tight_layout()
    png = OUT_DIR / ("lro_validation.png" if a.lro else "orbit_altitude.png")
    fig.savefig(png, dpi=130)
    print(f"Wrote {png}")

    # --- 3D: Moon lit by the real Sun at the epoch, orbit in Moon-fixed frame
    f3 = moon_figure(stride=2, exaggeration=1.0, color="wac", date=a.epoch)
    add_orbit(f3, o.r_me, o.t, o.alt_terrain, name=f"Simulated orbit (deg {main_deg})", impact=o.impacted)
    if a.lro:
        tr_me = np.einsum("nij,nj->ni", ph.Orientation(jd0, dur).icrf_to_me(o.t), truth.position(o.jd_tdb))
        add_orbit(f3, tr_me, name="Real LRO (JPL Horizons)", color="#ff4fd8")
    write(f3, OUT_DIR / ("lro_orbit.html" if a.lro else "moon_orbit.html"), not a.no_open)


if __name__ == "__main__":
    main()
