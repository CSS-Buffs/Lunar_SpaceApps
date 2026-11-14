"""
Lunar site planner: the interactive app laid out like wireframe/lunar_wireframe.png.

    python moon_app.py                 # builds the NASA data once, serves http://localhost:8050, opens it
    python moon_app.py --port 8060 --no-open

What runs where
  * This script prepares the Moon (LOLA shape, WAC texture from Moon Trek, terrain normals,
    LOLA heights for terrain horizons), caches it in data/app/, and serves the web page in
    python/web/.
  * When the start date or span changes, the page asks /api/track. That calls the JPL Horizons
    API for the Sun and Earth around the Moon and rotates them into the Moon-fixed (ME) frame.
  * Everything else (lighting, terrain horizons, visibility windows, comparisons) is computed
    in the browser from those two inputs.
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

import horizons
import lola_data
import lunar_frames as lf
import moontrek
from moon_viewer import R_LOLA, color_layer, illumination, load_geometry

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
CACHE = ROOT.parent / "data" / "app"
MAX_DAYS = 90


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def build_static(stride=2, exaggeration=3.0, horizon_ppd=4, force=False) -> Path:
    """Moon data that does not depend on time. Cached as JSON + PNG in data/app/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"static_s{stride}_x{exaggeration:g}_h{horizon_ppd}.json"
    tex = CACHE / "wac_texture.png"
    if out.exists() and tex.exists() and not force:
        return out
    print("Preparing Moon data (first run only)...")
    lat, lon, h, geo_label = load_geometry(4, stride)
    refl, *_ = color_layer("wac", lat, lon, h, None)
    refl = np.clip(refl / max(np.percentile(refl, 99.5), 1e-6), 0, 1)
    # cos(incidence) is linear in the Sun vector: three basis shadings give any Sun direction
    basis = [illumination(h, lat, lon, e, exaggeration) for e in np.eye(3)]
    normals = [np.clip(np.round(b * 127), -127, 127).astype(np.int8) for b in basis]
    dem = lola_data.load(horizon_ppd)
    payload = dict(
        mesh=dict(lat=np.round(lat, 4).tolist(), lon=np.round(lon, 4).tolist(),
                  h=_b64(np.round(h).astype("<i2")), shape=list(h.shape), exag=exaggeration,
                  nx=_b64(normals[0]), ny=_b64(normals[1]), nz=_b64(normals[2]),
                  refl=_b64(np.round(refl * 255).astype(np.uint8)), label=geo_label),
        dem=dict(h=_b64(np.clip(np.round(dem.height_m), -32767, 32767).astype("<i2")),
                 shape=list(dem.height_m.shape), ppd=horizon_ppd),
        r_moon_km=R_LOLA / 1e3,
    )
    out.write_text(json.dumps(payload))

    img, ilat, ilon = moontrek.wac_mosaic(3)          # 4096 x 2048, lon -180..180, lat N->S
    img = np.clip(img / max(np.percentile(img, 99.7), 1e-6), 0, 1)
    Image.fromarray(np.round(img * 255).astype(np.uint8), "L").save(tex, optimize=True)
    print(f"Cached {out.name} ({out.stat().st_size / 1e6:.1f} MB) and {tex.name}")
    return out


def track(start_iso: str, days: float) -> dict:
    """Sun and Earth positions around the Moon (ME frame, km) from JPL Horizons."""
    days = float(np.clip(days, 0.25, MAX_DAYS))
    jd0 = horizons.utc_iso_to_jd_tdb(start_iso)
    step_h = 0.5 if days <= 14 else 1.0
    jds = jd0 + np.arange(0.0, days * 24.0 + 1e-9, step_h) / 24.0
    eph = horizons.perturber_ephemerides(jds[0], jds[-1], pad_days=0.2)
    R = lf.icrf_to_me(jds)
    sun = np.einsum("nij,nj->ni", R, eph["sun"].position(jds)) / 1e3
    earth = np.einsum("nij,nj->ni", R, eph["earth"].position(jds)) / 1e3
    t0_ms = (np.datetime64(start_iso.replace("Z", ""), "ms") - np.datetime64("1970-01-01T00:00", "ms")) \
        / np.timedelta64(1, "ms")
    return dict(start=start_iso, days=days, t0_ms=int(t0_ms), dt_ms=int(step_h * 3600e3),
                sun=np.round(sun).astype(np.int64).ravel().tolist(),
                earth=np.round(earth, 1).ravel().tolist(), source="JPL Horizons API")


class Handler(SimpleHTTPRequestHandler):
    static_path: Path = None

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):  # keep the console quiet apart from errors
        if len(args) > 1 and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)

    def _send(self, body: bytes, ctype: str, code=200, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/static":
            return self._send(self.static_path.read_bytes(), "application/json", cache=True)
        if url.path == "/api/texture.png":
            return self._send((CACHE / "wac_texture.png").read_bytes(), "image/png", cache=True)
        if url.path == "/api/track":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                start = q.get("start") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
                np.datetime64(start.replace("Z", ""))      # validates the date
                body = json.dumps(track(start, float(q.get("days", 30)))).encode()
                return self._send(body, "application/json")
            except Exception as exc:                      # surface Horizons/date errors in the UI
                return self._send(json.dumps({"error": str(exc)[:500]}).encode(), "application/json", 502)
        if url.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--stride", type=int, default=2, help="3D mesh sub-sampling of LOLA LDEM_4")
    ap.add_argument("--exaggeration", type=float, default=3.0, help="3D relief exaggeration")
    ap.add_argument("--horizon-ppd", type=int, default=4, choices=lola_data.SUPPORTED_RES,
                    help="LOLA grid for terrain horizons (4 = 7.6 km; 16 = 1.9 km, ~45 MB download)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached Moon data")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    Handler.static_path = build_static(a.stride, a.exaggeration, a.horizon_ppd, a.rebuild)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/"
    print(f"Lunar site planner running at {url}  (Ctrl+C to stop)")
    if not a.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
