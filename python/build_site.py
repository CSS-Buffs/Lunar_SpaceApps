"""
Build a static copy of the lunar site planner that any static host (GitHub Pages, Netlify, S3...)
can serve. No Python server is needed at runtime.

    python build_site.py                          # -> site/  (2024-01 .. 2028-12)
    python build_site.py --first 2025-01 --last 2030-12 --out ../site
    python -m http.server -d ../site 8000         # preview at http://localhost:8000

The page normally asks the local server (moon_app.py) for Sun/Earth positions from JPL Horizons.
Browsers cannot call the Horizons API directly, so the build pre-computes the positions (hourly,
Moon-fixed ME frame) and stores them as one small JSON file per month. The site then works for
any start date in the built range.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import horizons
import lunar_frames as lf
from moon_app import CACHE, WEB, build_static

ROOT = Path(__file__).resolve().parent.parent


def month_starts(first: str, last: str):
    y, m = map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    while (y, m) <= (ly, lm):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def iso(y, m):
    return f"{y:04d}-{m:02d}-01T00:00"


def build_tracks(first: str, last: str, out: Path):
    """Hourly Sun/Earth (ME frame, km) per month. Each file runs from the first hour of its month to
    the first hour of the next month inclusive, so consecutive months join without gaps."""
    out.mkdir(parents=True, exist_ok=True)
    months = list(month_starts(first, last))
    years = sorted({y for y, _ in months})
    for year in years:                                   # one Horizons query per body per year
        ym = [(y, m) for y, m in months if y == year]
        jd_a = horizons.utc_iso_to_jd_tdb(iso(*ym[0]))
        ny, nm = (ym[-1][0] + 1, 1) if ym[-1][1] == 12 else (ym[-1][0], ym[-1][1] + 1)
        jd_b = horizons.utc_iso_to_jd_tdb(iso(ny, nm))
        eph = horizons.perturber_ephemerides(jd_a, jd_b, pad_days=0.2)
        for y, m in ym:
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            t0 = np.datetime64(iso(y, m), "h"); t1 = np.datetime64(iso(ny, nm), "h")
            hours = int((t1 - t0) / np.timedelta64(1, "h"))
            jd0 = horizons.utc_iso_to_jd_tdb(iso(y, m))
            jds = jd0 + np.arange(hours + 1) / 24.0
            R = lf.icrf_to_me(jds)
            sun = np.einsum("nij,nj->ni", R, eph["sun"].position(jds)) / 1e3
            earth = np.einsum("nij,nj->ni", R, eph["earth"].position(jds)) / 1e3
            t0_ms = int((t0 - np.datetime64("1970-01-01T00", "h")) / np.timedelta64(1, "ms"))
            (out / f"{y:04d}-{m:02d}.json").write_text(json.dumps(dict(
                t0_ms=t0_ms, dt_ms=3600000, sun=np.round(sun).astype(np.int64).ravel().tolist(),
                earth=np.round(earth, 1).ravel().tolist()), separators=(",", ":")))
        print(f"  {year}: {len(ym)} months")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", default="2024-01", help="first month with Sun/Earth data (YYYY-MM)")
    ap.add_argument("--last", default="2028-12", help="last month with Sun/Earth data (YYYY-MM)")
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    a = ap.parse_args()

    out = a.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "track").mkdir(parents=True)

    print("Moon data (LOLA, Moon Trek WAC)...")
    static = build_static()
    shutil.copy(static, out / "data" / "static.json")
    shutil.copy(CACHE / "wac_texture.png", out / "data" / "texture.png")

    print(f"Sun and Earth from JPL Horizons, {a.first} to {a.last}...")
    build_tracks(a.first, a.last, out / "data" / "track")

    for name in ("app.js", "app.css"):
        shutil.copy(WEB / name, out / name)
    cfg = dict(first=a.first, last=a.last, built=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    html = (WEB / "index.html").read_text(encoding="utf-8").replace(
        '<script src="app.js"></script>',
        f'<script>window.LUNAR_STATIC = {json.dumps(cfg)};</script>\n<script src="app.js"></script>')
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Wrote {out} ({size:.1f} MB). Preview: python -m http.server -d \"{out}\" 8000")


if __name__ == "__main__":
    main()
