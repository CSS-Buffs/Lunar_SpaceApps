"""
Download and read NASA LRO / LOLA Gridded Data Records (GDR).

Data source (NASA PDS Geosciences Node):
    https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/

Products used: LDEM_<res>.IMG  (simple-cylindrical, pixel registered)
    res = 4   ->  1440 x  720  (~7.6 km/pix,   2 MB)
    res = 16  ->  5760 x 2880  (~1.9 km/pix,  33 MB)
    res = 64  -> 23040 x 11520 (~474 m/pix,  531 MB)

Encoding (from the PDS label):
    16-bit little-endian signed integers (LSB_INTEGER)
    HEIGHT [m]           = DN * 0.5
    PLANETARY_RADIUS [m] = DN * 0.5 + 1737400
    Row 0 is the northern-most line (90 N), column 0 starts at 0 E,
    longitude is positive EAST, frame = Mean Earth / Polar Axis (DE421).
"""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PDS_BASE = ("https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
            "lrolol_1xxx/data/lola_gdr/cylindrical/img/")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUPPORTED_RES = (4, 16, 64)


@dataclass
class LolaDEM:
    """A global LOLA elevation grid."""
    height_m: np.ndarray        # (lines, samples) float32, height above 1737.4 km sphere
    lat_deg: np.ndarray         # (lines,)   pixel-centre latitudes, north -> south
    lon_deg: np.ndarray         # (samples,) pixel-centre east longitudes, 0 -> 360
    ref_radius_m: float         # 1737400 m
    ppd: int                    # pixels per degree

    @property
    def radius_m(self) -> np.ndarray:
        return self.height_m + self.ref_radius_m

    def interp_height(self, lat_deg, lon_deg):
        """Bilinear interpolation of height [m] at arbitrary lat/lon (vectorised)."""
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.mod(np.asarray(lon_deg, dtype=float), 360.0)
        # fractional pixel coordinates (pixel centres at (i + 0.5) / ppd)
        row = (90.0 - lat) * self.ppd - 0.5
        col = lon * self.ppd - 0.5
        nrow, ncol = self.height_m.shape
        row = np.clip(row, 0, nrow - 1)
        r0 = np.floor(row).astype(int)
        r1 = np.minimum(r0 + 1, nrow - 1)
        fr = row - r0
        c0 = np.floor(col).astype(int)
        fc = col - c0
        c0 %= ncol                      # wrap in longitude
        c1 = (c0 + 1) % ncol
        h = self.height_m
        return ((1 - fr) * (1 - fc) * h[r0, c0] + (1 - fr) * fc * h[r0, c1]
                + fr * (1 - fc) * h[r1, c0] + fr * fc * h[r1, c1])


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "lunar-lola-viewer"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def fetch(ppd: int = 4, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Ensure LDEM_<ppd>.IMG and .LBL exist locally; return their paths."""
    if ppd not in SUPPORTED_RES:
        raise ValueError(f"ppd must be one of {SUPPORTED_RES}")
    img = data_dir / f"ldem_{ppd}.img"
    lbl = data_dir / f"ldem_{ppd}.lbl"
    for path in (lbl, img):
        if not path.exists():
            _download(PDS_BASE + path.name, path)
    return img, lbl


def _parse_label(lbl: Path) -> dict:
    text = lbl.read_text(errors="replace")

    def num(key):
        m = re.search(rf"^\s*{key}\s*=\s*([-+0-9.Ee]+)", text, re.MULTILINE)
        if not m:
            raise ValueError(f"{key} not found in {lbl}")
        return float(m.group(1))

    return dict(lines=int(num("LINES")), samples=int(num("LINE_SAMPLES")),
                scale=num("SCALING_FACTOR"), offset=num("OFFSET"),
                ppd=int(num("MAP_RESOLUTION")))


def load(ppd: int = 4, data_dir: Path = DATA_DIR) -> LolaDEM:
    """Download (if needed) and load a global LOLA DEM."""
    img, lbl = fetch(ppd, data_dir)
    meta = _parse_label(lbl)
    dn = np.fromfile(img, dtype="<i2")
    expected = meta["lines"] * meta["samples"]
    if dn.size != expected:
        raise ValueError(f"{img} has {dn.size} samples, expected {expected}; delete it and re-run")
    height = dn.reshape(meta["lines"], meta["samples"]).astype(np.float32) * meta["scale"]
    p = meta["ppd"]
    lat = 90.0 - (np.arange(meta["lines"]) + 0.5) / p
    lon = (np.arange(meta["samples"]) + 0.5) / p
    return LolaDEM(height, lat, lon, meta["offset"], p)


if __name__ == "__main__":
    dem = load(4)
    h = dem.height_m
    print(f"LDEM_{dem.ppd}: {h.shape}, height range {h.min():.0f} .. {h.max():.0f} m")
    print(f"Mean radius: {dem.radius_m.mean() / 1000:.3f} km")
