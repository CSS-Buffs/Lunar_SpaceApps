"""
Propagate a spacecraft around the Moon using a GRAIL degree-2 gravity field
(+ Earth tide) and check its altitude against real LOLA terrain.

Example (LRO-like 50 km polar orbit for 1 day):
    python simulate_orbit.py --alt 50 --inc 90 --days 1
Low, eccentric orbit that eventually hits the surface:
    python simulate_orbit.py --alt 15 --ecc 0.004 --inc 85 --days 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lola_data
import lunar_physics as phys
from moon_viewer import OUT_DIR, add_orbit, moon_figure


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alt", type=float, default=50.0, help="periapsis altitude above 1737.4 km sphere [km]")
    ap.add_argument("--ecc", type=float, default=0.0)
    ap.add_argument("--inc", type=float, default=90.0, help="inclination [deg]")
    ap.add_argument("--raan", type=float, default=0.0)
    ap.add_argument("--argp", type=float, default=0.0)
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--no-earth", action="store_true", help="disable Earth tidal perturbation")
    ap.add_argument("--ppd", type=int, default=4, choices=lola_data.SUPPORTED_RES)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    dem = lola_data.load(args.ppd)
    rp = phys.R_LOLA + args.alt * 1e3
    a = rp / (1 - args.ecc)
    r0, v0 = phys.kepler_to_state(a, args.ecc, args.inc, args.raan, args.argp, 0.0)
    period = 2 * np.pi * np.sqrt(a**3 / phys.GM_MOON)
    print(f"Initial orbit: a = {a / 1e3:.1f} km, e = {args.ecc}, i = {args.inc} deg, "
          f"period = {period / 60:.1f} min, |v| = {np.linalg.norm(v0):.1f} m/s")

    n_out = int(min(20000, max(2000, args.days * 86400 / 30)))
    orb = phys.propagate(r0, v0, args.days * 86400, dem=dem, earth=not args.no_earth, n_out=n_out)

    drift = np.max(np.abs(orb.jacobi - orb.jacobi[0]) / abs(orb.jacobi[0]))
    print(f"Max relative Jacobi-constant drift (integration error): {drift:.2e}")
    print(f"Altitude above terrain: min {orb.alt_terrain.min() / 1e3:.2f} km, "
          f"max {orb.alt_terrain.max() / 1e3:.2f} km")
    if orb.impacted:
        print(f"IMPACT after {orb.impact_time / 3600:.2f} h at lat {orb.lat[-1]:.2f}, "
              f"lon {orb.lon[-1]:.2f} E")
    else:
        print("No impact during simulation.")

    OUT_DIR.mkdir(exist_ok=True)
    # 2D diagnostics
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    th = orb.t / 3600
    ax[0].plot(th, orb.alt_terrain / 1e3, lw=0.8)
    ax[0].set_ylabel("Altitude above LOLA terrain [km]")
    ax[0].grid(alpha=0.3)
    ax[1].plot(th, (orb.jacobi - orb.jacobi[0]) / abs(orb.jacobi[0]), lw=0.8)
    ax[1].set_ylabel("Relative Jacobi-constant drift\n(numerical error)")
    ax[1].set_xlabel("Time [h]")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Lunar orbit with GRAIL J2+C22 gravity and LOLA terrain")
    fig.tight_layout()
    png = OUT_DIR / "orbit_altitude.png"
    fig.savefig(png, dpi=130)
    print(f"Wrote {png}")

    # 3D interactive
    f3 = add_orbit(moon_figure(dem, stride=2, exaggeration=1.0), orb)   # true relief
    html = OUT_DIR / "moon_orbit.html"
    f3.write_html(html, include_plotlyjs="cdn", auto_open=not args.no_open,
                  config=dict(scrollZoom=True, displaylogo=False))
    print(f"Wrote {html}")


if __name__ == "__main__":
    main()
