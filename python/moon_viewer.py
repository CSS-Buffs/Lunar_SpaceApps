"""
Interactive 3D Moon built from NASA data:

  geometry      LRO LOLA laser-altimeter DEM (PDS)  - global
                LROC GLD100 stereo DTM via QuickMap - high-resolution regional close-ups
  appearance    LRO WAC global mosaic via NASA Moon Trek WMTS tiles
  lighting      true Sun direction for any date from the JPL Horizons API
  overlays      GRAIL gravity (computed from the GRGM1200A spherical harmonics, or the
                QuickMap free-air map), TiO2 abundance (QuickMap), slope, elevation
  labels        IAU nomenclature (maria, major craters, Apollo sites) from Moon Trek

Opens an HTML viewer: drag = rotate, scroll = zoom, right-drag = pan, double-click = reset.

Examples
    python moon_viewer.py                                    # photo-textured Moon, lit as of now
    python moon_viewer.py --date 2026-11-20T00:00 --labels   # phase on a given date + names
    python moon_viewer.py --color gravity                    # GRAIL free-air anomaly
    python moon_viewer.py --color tio2 --labels
    python moon_viewer.py --region -45 -41.5 -13.5 -9 --source gld100 --exaggeration 2  # Tycho
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, map_coordinates

import grail_gravity
import horizons
import lola_data
import lunar_frames as lf
import moontrek
import quickmap

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
R_LOLA = 1737.4e3

GREY = [[0.0, "#000000"], [1.0, "#ffffff"]]
LUNAR_GREY = [[0.0, "#1b1b1f"], [0.25, "#4a4a52"], [0.5, "#8a8a90"], [0.75, "#c4c2bc"], [1.0, "#f4f1e8"]]


# ---------------------------------------------------------------------------- sampling
def sample_grid(img, lat, lon, LAT, LON, wrap=True):
    """Bilinear sample of a regular grid (lat descending, lon ascending, either -180..180
    or 0..360) at target LAT/LON, anti-aliased when down-sampling. NaNs are filled."""
    img = np.asarray(img, dtype=np.float64)
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    if np.isnan(img).any():
        fill = np.nanmean(img)
        img = np.where(np.isnan(img), fill, img)
    if LAT.ndim == 2 and LAT.shape[0] > 1 and LON.shape[1] > 1:
        step = min(abs(LAT[1, 0] - LAT[0, 0]) / dlat, abs(LON[0, 1] - LON[0, 0]) / dlon)
        if step > 1.5:
            img = gaussian_filter(img, 0.5 * step, mode=("nearest", "wrap" if wrap else "nearest"))
    row = (lat[0] - LAT) / dlat
    lon_t = LON if lon[0] >= 0 else np.where(LON > 180, LON - 360, LON)
    if wrap:
        lon_t = np.mod(lon_t - lon[0], 360.0) + lon[0]
    col = (lon_t - lon[0]) / dlon
    if wrap:   # pad one column each side so interpolation wraps across the seam
        img = np.hstack([img[:, -1:], img, img[:, :1]])
        col = col + 1
    return map_coordinates(img, [row, col], order=1, mode="nearest")


def slope_components(h, lat, lon, R=R_LOLA):
    """dh/d(north), dh/d(east) [m/m] on the sphere."""
    dlat = np.gradient(np.radians(lat))
    dlon = np.gradient(np.radians(lon))
    dn = np.gradient(h, axis=0) / dlat[:, None] / R
    coslat = np.clip(np.cos(np.radians(lat)), 1e-3, None)[:, None]
    de = np.gradient(h, axis=1) / dlon[None, :] / (R * coslat)
    return dn, de


def sun_and_earth_me(jd_tdb):
    """Unit vectors to the Sun and Earth in the Moon ME frame (JPL Horizons)."""
    R = lf.icrf_to_me(jd_tdb)
    eph = horizons.perturber_ephemerides(jd_tdb, jd_tdb, pad_days=0.2)
    s = R @ eph["sun"].position(jd_tdb)
    e = R @ eph["earth"].position(jd_tdb)
    return s / np.linalg.norm(s), e / np.linalg.norm(e)


def illumination(h, lat, lon, sun_me, exaggeration=1.0):
    """cos(solar incidence) on the (exaggerated) terrain, from the real Sun direction."""
    dn, de = slope_components(h * exaggeration, lat, lon)
    la, lo = np.radians(lat)[:, None], np.radians(lon)[None, :]
    s_e = -np.sin(lo) * sun_me[0] + np.cos(lo) * sun_me[1]
    s_n = (-np.sin(la) * np.cos(lo) * sun_me[0] - np.sin(la) * np.sin(lo) * sun_me[1]
           + np.cos(la) * sun_me[2])
    s_u = (np.cos(la) * np.cos(lo) * sun_me[0] + np.cos(la) * np.sin(lo) * sun_me[1]
           + np.sin(la) * sun_me[2])
    return (s_u - de * s_e - dn * s_n) / np.sqrt(1 + de**2 + dn**2)


# ---------------------------------------------------------------------------- geometry
def load_geometry(ppd=4, stride=2, region=None, source="lola", gld_ppd=128):
    """Return lat (deg, N->S), lon (deg east), h (m above 1737.4 km sphere), label."""
    if region and source == "gld100":
        la0, la1, lo0, lo1 = region
        lo0w, lo1w = [((x + 180) % 360) - 180 for x in (lo0, lo1)]
        w = int(min(2000, max(64, (lo1w - lo0w) * gld_ppd)))
        hgt = int(min(2000, max(64, (la1 - la0) * gld_ppd)))
        r, lat, lon = quickmap.numeric("gld100", (lo0w, la0, lo1w, la1), w, hgt)
        r = r[::stride, ::stride]
        return lat[::stride], np.mod(lon[::stride], 360), r - R_LOLA, \
            f"LROC GLD100 via QuickMap, {w // stride}x{hgt // stride}"

    dem = lola_data.load(ppd)
    h, lat, lon = dem.height_m, dem.lat_deg, dem.lon_deg
    if region:
        la0, la1, lo0, lo1 = region
        lo0, lo1 = lo0 % 360, lo1 % 360
        rows = np.where((lat >= la0) & (lat <= la1))[0][::stride]
        cols = (np.where((lon >= lo0) & (lon <= lo1))[0] if lo0 <= lo1 else
                np.concatenate([np.where(lon >= lo0)[0], np.where(lon <= lo1)[0]]))[::stride]
        return lat[rows], lon[cols], h[np.ix_(rows, cols)].astype(float), f"LOLA LDEM_{ppd}"
    h, lat, lon = h[::stride, ::stride].astype(float), lat[::stride], lon[::stride]
    # close the seam and cap the poles -> watertight sphere
    h = np.hstack([h, h[:, :1]])
    lon = np.append(lon, lon[0] + 360.0)
    h = np.vstack([np.full((1, h.shape[1]), h[0].mean()), h, np.full((1, h.shape[1]), h[-1].mean())])
    lat = np.concatenate([[90.0], lat, [-90.0]])
    return lat, lon, h, f"LOLA LDEM_{ppd}"


def color_layer(kind, lat, lon, h, region, gravity_degree=150, trek_level=None):
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    bbox = None
    if region:
        la0, la1, lo0, lo1 = region
        lo0w, lo1w = [((x + 180) % 360) - 180 for x in (lo0, lo1)]
        bbox = (lo0w - 0.2, la0 - 0.2, lo1w + 0.2, la1 + 0.2)
    if kind == "wac":
        if trek_level is None:
            trek_level = 3 if not region else int(np.clip(np.ceil(np.log2(
                (len(lon) / max(1e-6, abs(lon[-1] - lon[0]))) * 180 / 256)), 3, 7))
        img, ilat, ilon = moontrek.wac_mosaic(trek_level, bbox=bbox)
        v = sample_grid(img, ilat, ilon, LAT, LON, wrap=not region)
        return v, GREY, "WAC reflectance (Moon Trek)", ""
    if kind == "height":
        return h / 1000, LUNAR_GREY, "Elevation", "km"
    if kind == "slope":
        dn, de = slope_components(h, lat, lon)
        return np.degrees(np.arctan(np.hypot(dn, de))), "Inferno", "Slope", "deg"
    if kind == "gravity":
        g = grail_gravity.load(gravity_degree)
        lon_c = np.mod(lon, 360)
        fa = grail_gravity.free_air_anomaly_grid(g, lat, lon_c)
        return fa, "RdBu_r", f"Free-air anomaly, GRAIL deg {gravity_degree}", "mGal"
    if kind in ("freeair", "tio2"):
        qb = bbox or (-180, -90, 180, 90)
        w = 1440 if not region else 800
        v, qlat, qlon = quickmap.numeric(kind, qb, w, w // 2 if not region else w)
        out = sample_grid(v, qlat, qlon, LAT, LON, wrap=not region)
        if kind == "tio2":
            out = np.clip(out, 0, 15)
            return out, "Turbo", "TiO2 (QuickMap/LROC)", "wt%"
        return out, "RdBu_r", "Free-air anomaly (QuickMap/GRAIL)", "mGal"
    raise ValueError(kind)


# ---------------------------------------------------------------------------- figure
def moon_figure(ppd=4, stride=2, region=None, exaggeration=3.0, color="wac", date=None,
                labels=False, source="lola", gravity_degree=150):
    lat, lon, h, geo_label = load_geometry(ppd, stride, region, source)
    c, cs, ctitle, unit = color_layer(color, lat, lon, h, region, gravity_degree)

    sun_me = earth_me = None
    lit = date is not None
    if lit:
        jd = horizons.utc_iso_to_jd_tdb(date)
        sun_me, earth_me = sun_and_earth_me(jd)
        cosi = np.clip(illumination(h, lat, lon, sun_me, exaggeration), 0, None)
        if color == "wac":     # reflectance x Lambert shading + faint earthshine
            c = c * (0.03 + 0.97 * cosi)
            c = np.clip(c / max(np.percentile(c, 99.5), 1e-6), 0, 1)
            ctitle = f"Sunlit WAC reflectance, {date} UTC"

    r = (R_LOLA + exaggeration * h) / 1000.0
    la, lo = np.radians(lat)[:, None], np.radians(lon)[None, :]
    x = np.round(r * np.cos(la) * np.cos(lo), 3)
    y = np.round(r * np.cos(la) * np.sin(lo), 3)
    z = np.round(r * np.sin(la) * np.ones_like(lo), 3)
    LAT, LON = np.meshgrid(lat, np.where(np.mod(lon, 360) > 180, np.mod(lon, 360) - 360, np.mod(lon, 360)),
                           indexing="ij")
    custom = np.round(np.dstack([LAT, LON, h]), 2)

    surface = go.Surface(
        x=x, y=y, z=z, surfacecolor=np.round(c, 4), colorscale=cs,
        showscale=color != "wac",
        colorbar=dict(title=f"{ctitle}" + (f" [{unit}]" if unit else ""), thickness=15, len=0.6),
        customdata=custom,
        hovertemplate=("lat %{customdata[0]:.3f}°, lon %{customdata[1]:.3f}°<br>"
                       "elevation %{customdata[2]:.0f} m<extra></extra>"),
        # with --date the shading is computed physically, so plotly's own light is turned off
        lighting=(dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0) if lit and color == "wac"
                  else dict(ambient=0.45, diffuse=0.8, specular=0.03, roughness=0.95, fresnel=0.0)),
        lightposition=dict(x=-0.5e5, y=-1e5, z=2e4),
        name="Moon",
    )
    fig = go.Figure(surface)

    if labels:
        feats = moontrek.features()
        if region:
            la0, la1, lo0, lo1 = region
            feats = [f for f in feats if la0 <= f["lat"] <= la1 and
                     ((f["lon"] - lo0) % 360) <= ((lo1 - lo0) % 360)]
        if feats:
            fl = np.array([[f["lat"], f["lon"]] for f in feats])
            top = np.nanmax(h) * exaggeration
            rr = (R_LOLA + top + (3e3 if region else 25e3 * max(1, exaggeration / 3))) / 1000
            d = lf.direction_from_latlon(fl[:, 0], fl[:, 1]) * rr
            colors = {"Mare": "#9ad0ff", "Oceanus": "#9ad0ff", "Crewed Landing": "#ffd166", "Crater": "#ffffff"}
            fig.add_trace(go.Scatter3d(
                x=d[:, 0], y=d[:, 1], z=d[:, 2], mode="markers+text",
                text=[f["name"] for f in feats], textposition="top center",
                textfont=dict(size=10, color=[colors.get(f["type"], "white") for f in feats]),
                marker=dict(size=2, color=[colors.get(f["type"], "white") for f in feats]),
                hoverinfo="text", name="IAU names (Moon Trek)"))

    if lit and not region:
        for vec, name, col in ((sun_me, "Sun direction", "#ffcc33"), (earth_me, "Earth direction", "#4da6ff")):
            p = np.outer([1.02, 1.6], vec) * R_LOLA / 1000
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+text",
                                       text=["", name], line=dict(color=col, width=6),
                                       textfont=dict(color=col), name=name))

    title = f"The Moon from NASA data: {geo_label}, relief ×{exaggeration:g}"
    if region:
        title += f", lat {region[0]}..{region[1]}, lon {region[2]}..{region[3]}"
    eye = dict(x=0.6, y=0.6, z=1.2)
    if not region:
        v = earth_me if earth_me is not None else np.array([1.0, 0.15, 0.2])
        eye = dict(x=1.7 * v[0], y=1.7 * v[1], z=1.7 * v[2])   # look from Earth by default
    fig.update_layout(
        title=dict(text=title + f"<br><sup>Colour: {ctitle}. Sources: LOLA (PDS), Moon Trek, "
                               f"LROC QuickMap, GRAIL (PDS), JPL Horizons</sup>", x=0.02),
        template="plotly_dark", paper_bgcolor="black",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="data", bgcolor="black", camera=dict(eye=eye), dragmode="orbit"),
        margin=dict(l=0, r=0, t=70, b=0), legend=dict(x=0.01, y=0.02),
    )
    return fig


def add_orbit(fig, r_me_m, t_s=None, alt_m=None, name="Orbit (Moon-fixed)", color="#00e5ff", impact=False):
    r = np.asarray(r_me_m) / 1000.0
    kw = {}
    if t_s is not None and alt_m is not None:
        kw = dict(customdata=np.column_stack([np.asarray(t_s) / 3600, np.asarray(alt_m) / 1000]),
                  hovertemplate="t = %{customdata[0]:.2f} h<br>altitude %{customdata[1]:.1f} km<extra></extra>")
    fig.add_trace(go.Scatter3d(x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="lines",
                               line=dict(color=color, width=3), name=name, **kw))
    if impact:
        fig.add_trace(go.Scatter3d(x=[r[-1, 0]], y=[r[-1, 1]], z=[r[-1, 2]], mode="markers",
                                   marker=dict(size=6, color="red"), name="Impact"))
    return fig


def write(fig, path: Path, open_browser: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", auto_open=open_browser,
                   config=dict(scrollZoom=True, displaylogo=False))
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppd", type=int, default=4, choices=lola_data.SUPPORTED_RES, help="LOLA grid pixels/degree")
    ap.add_argument("--stride", type=int, default=2, help="mesh sub-sampling (1 = full resolution)")
    ap.add_argument("--exaggeration", type=float, default=3.0, help="vertical relief exaggeration")
    ap.add_argument("--color", default="wac", choices=["wac", "height", "slope", "gravity", "freeair", "tio2"])
    ap.add_argument("--date", default="none",
                    help="UTC date/time (ISO) or 'now': real sunlight + phase from JPL Horizons")
    ap.add_argument("--labels", action="store_true", help="IAU feature names from Moon Trek")
    ap.add_argument("--region", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--source", default="lola", choices=["lola", "gld100"],
                    help="terrain for --region: LOLA grid or LROC GLD100 (QuickMap, ~100 m)")
    ap.add_argument("--gravity-degree", type=int, default=150)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    date = None if a.date.lower() == "none" else (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M") if a.date.lower() == "now" else a.date)
    fig = moon_figure(a.ppd, a.stride, a.region, a.exaggeration, a.color, date, a.labels, a.source,
                      a.gravity_degree)
    write(fig, a.out or OUT_DIR / f"moon_{a.color}{'_region' if a.region else ''}.html", not a.no_open)


if __name__ == "__main__":
    main()
