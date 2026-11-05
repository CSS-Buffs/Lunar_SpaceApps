"""
Interactive 3D Moon built from NASA LRO/LOLA laser-altimeter data.

Opens an HTML viewer in your browser: drag to rotate, scroll to zoom,
right-drag (or shift-drag) to pan, double-click to reset the view.

Examples
    python moon_viewer.py                               # global moon, 0.5 deg mesh
    python moon_viewer.py --stride 1 --exaggeration 5   # full LDEM_4 detail, 5x relief
    python moon_viewer.py --color gravity               # colour by surface gravity
    python moon_viewer.py --ppd 16 --region -50 -35 -20 -5 --exaggeration 2   # Tycho crater close-up
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import lola_data
import lunar_physics as phys

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

LUNAR_GREY = [[0.0, "#1b1b1f"], [0.25, "#4a4a52"], [0.5, "#8a8a90"],
              [0.75, "#c4c2bc"], [1.0, "#f4f1e8"]]


def build_grid(dem, stride=2, region=None, exaggeration=1.0):
    """Return lat, lon (deg) and x, y, z (km) for a (sub-sampled) DEM patch."""
    h = dem.height_m
    lat, lon = dem.lat_deg, dem.lon_deg
    if region:
        la0, la1, lo0, lo1 = region
        lo0, lo1 = lo0 % 360, lo1 % 360
        rows = np.where((lat >= la0) & (lat <= la1))[0]
        if lo0 <= lo1:
            cols = np.where((lon >= lo0) & (lon <= lo1))[0]
        else:                                  # region crosses 0/360 E
            cols = np.concatenate([np.where(lon >= lo0)[0], np.where(lon <= lo1)[0]])
        rows, cols = rows[::stride], cols[::stride]
        h = h[np.ix_(rows, cols)]
        lat, lon = lat[rows], lon[cols]
    else:
        h, lat, lon = h[::stride, ::stride], lat[::stride], lon[::stride]
        # close the longitude seam and cap the poles so the sphere is watertight
        h = np.hstack([h, h[:, :1]])
        lon = np.append(lon, lon[0] + 360.0)
        h = np.vstack([np.full((1, h.shape[1]), h[0].mean()), h,
                       np.full((1, h.shape[1]), h[-1].mean())])
        lat = np.concatenate([[90.0], lat, [-90.0]])

    r = (phys.R_LOLA + exaggeration * h) / 1000.0
    la, lo = np.radians(lat)[:, None], np.radians(lon)[None, :]
    x = r * np.cos(la) * np.cos(lo)
    y = r * np.cos(la) * np.sin(lo)
    z = r * np.sin(la) * np.ones_like(lo)
    return lat, lon, h, x, y, z


def slope_deg(h, lat, lon):
    """Surface slope [deg] from finite differences on the sphere."""
    dlat = np.gradient(np.radians(lat))
    dlon = np.gradient(np.radians(lon))
    dh_dlat = np.gradient(h, axis=0) / dlat[:, None] / phys.R_LOLA
    coslat = np.clip(np.cos(np.radians(lat)), 1e-3, None)[:, None]
    dh_dlon = np.gradient(h, axis=1) / dlon[None, :] / (phys.R_LOLA * coslat)
    return np.degrees(np.arctan(np.hypot(dh_dlat, dh_dlon)))


def moon_figure(dem, stride=2, region=None, exaggeration=1.0, color="height",
                sun_lon=-60.0):
    lat, lon, h, x, y, z = build_grid(dem, stride, region, exaggeration)

    if color == "height":
        c, cs, title, unit = h / 1000.0, LUNAR_GREY, "Elevation", "km"
    elif color == "terrain":
        c, cs, title, unit = h / 1000.0, "Earth", "Elevation", "km"
    elif color == "slope":
        c, cs, title, unit = slope_deg(h, lat, lon), "Inferno", "Slope", "deg"
    elif color == "gravity":
        sub = lola_data.LolaDEM(h.astype(np.float32), lat, lon, dem.ref_radius_m, dem.ppd)
        c, cs, title, unit = phys.surface_gravity(sub), "Viridis", "|g| surface", "m/s²"
    else:
        raise ValueError(color)

    # Plotly's hover text: show real coordinates and elevation
    LAT, LON = np.meshgrid(lat, np.mod(lon, 360), indexing="ij")
    custom = np.dstack([LAT, np.where(LON > 180, LON - 360, LON), h])

    sl = np.radians(sun_lon)
    surface = go.Surface(
        x=x, y=y, z=z, surfacecolor=c, colorscale=cs,
        colorbar=dict(title=f"{title} [{unit}]", thickness=15, len=0.6),
        customdata=custom,
        hovertemplate=("lat %{customdata[0]:.2f}°, lon %{customdata[1]:.2f}°<br>"
                       "elevation %{customdata[2]:.0f} m<extra></extra>"),
        lighting=dict(ambient=0.35, diffuse=0.9, specular=0.05, roughness=0.95, fresnel=0.0),
        lightposition=dict(x=1e5 * np.cos(sl), y=1e5 * np.sin(sl), z=2e4),
        name="Moon (LOLA)",
    )

    fig = go.Figure(surface)
    label = f"LOLA LDEM_{dem.ppd}, mesh {x.shape[1]}×{x.shape[0]}, relief ×{exaggeration:g}"
    if region:
        label += f", region lat {region[0]}..{region[1]}, lon {region[2]}..{region[3]}"
    fig.update_layout(
        title=dict(text=f"The Moon: NASA LRO Lunar Orbiter Laser Altimeter<br><sup>{label}</sup>",
                   x=0.02),
        template="plotly_dark",
        paper_bgcolor="black",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor="black",
            camera=dict(eye=dict(x=1.6, y=0.4, z=0.5) if not region else dict(x=0.6, y=0.6, z=1.2)),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def add_orbit(fig, orbit, name="Spacecraft orbit (body-fixed)", color="#00e5ff"):
    """Overlay an OrbitResult (from lunar_physics.propagate) in the Moon-fixed frame."""
    r = orbit.r_bf / 1000.0
    fig.add_trace(go.Scatter3d(
        x=r[0], y=r[1], z=r[2], mode="lines", line=dict(color=color, width=3), name=name,
        customdata=np.column_stack([orbit.t / 3600.0, orbit.alt_terrain / 1000.0]),
        hovertemplate="t = %{customdata[0]:.2f} h<br>alt above terrain %{customdata[1]:.1f} km"
                      "<extra></extra>"))
    if orbit.impacted:
        fig.add_trace(go.Scatter3d(x=[r[0, -1]], y=[r[1, -1]], z=[r[2, -1]], mode="markers",
                                   marker=dict(size=6, color="red"), name="Impact"))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppd", type=int, default=4, choices=lola_data.SUPPORTED_RES,
                    help="LOLA grid resolution in pixels/degree (4 = 2 MB, 16 = 33 MB, 64 = 531 MB)")
    ap.add_argument("--stride", type=int, default=2, help="sub-sampling step (1 = full resolution)")
    ap.add_argument("--exaggeration", type=float, default=3.0, help="vertical relief exaggeration")
    ap.add_argument("--color", default="height", choices=["height", "terrain", "slope", "gravity"])
    ap.add_argument("--region", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="render only this patch (east longitudes, -180..360 accepted)")
    ap.add_argument("--sun-lon", type=float, default=-60.0, help="light direction longitude [deg]")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "moon_lola.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    dem = lola_data.load(args.ppd)
    fig = moon_figure(dem, args.stride, args.region, args.exaggeration, args.color, args.sun_lon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn", auto_open=not args.no_open,
                   config=dict(scrollZoom=True, displaylogo=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
