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
    python moon_viewer.py --timeline 30                      # time slider: Sun + Earth over 30 days
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


def sun_earth_track(jd0, days, step_h):
    """Sun and Earth around the Moon over a time span, from JPL Horizons, in the ME frame.
    Returns times (TDB JD) and (N,3) Moon-centred positions in metres."""
    jds = jd0 + np.arange(0.0, days * 24.0 + 1e-9, step_h) / 24.0
    eph = horizons.perturber_ephemerides(jds[0], jds[-1], pad_days=0.2)
    R = lf.icrf_to_me(jds)
    sun = np.einsum("nij,nj->ni", R, eph["sun"].position(jds))
    earth = np.einsum("nij,nj->ni", R, eph["earth"].position(jds))
    return jds, sun, earth


def phase_name(frac, waxing):
    if frac < 0.03:
        return "New Moon"
    if frac > 0.97:
        return "Full Moon"
    if abs(frac - 0.5) < 0.04:
        return "First Quarter" if waxing else "Last Quarter"
    return ("Waxing " if waxing else "Waning ") + ("Crescent" if frac < 0.5 else "Gibbous")


def timeline_meta(jds, sun, earth, lat, lon, h, refl, exaggeration):
    """Everything the in-page time slider needs to re-light the Moon and move the Sun and Earth
    in the browser. Terrain normals and reflectance are packed as 8-bit base64 arrays."""
    import base64
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")
    # cos(incidence) is linear in the Sun vector, so three basis shadings give every Sun direction
    basis = [illumination(h, lat, lon, e, exaggeration) for e in np.eye(3)]
    normals = [np.clip(np.round(b * 127), -127, 127).astype(np.int8) for b in basis]
    refl = np.clip(refl / max(np.percentile(refl, 99.5), 1e-6), 0, 1)

    s_hat = sun / np.linalg.norm(sun, axis=1, keepdims=True)
    e_hat = earth / np.linalg.norm(earth, axis=1, keepdims=True)
    cos_i = np.sum(s_hat * e_hat, axis=1)                 # phase angle at the Moon
    frac = (1 + cos_i) / 2                                # illuminated fraction seen from Earth
    s_lat, s_lon, _ = lf.latlon(sun)
    e_lat, e_lon, _ = lf.latlon(earth)
    wrap = lambda x: (x + 180) % 360 - 180
    waxing = wrap(s_lon - e_lon) > 0                      # Sun east of the sub-Earth point
    return dict(
        shape=list(h.shape), nx=b64(normals[0]), ny=b64(normals[1]), nz=b64(normals[2]),
        refl=b64(np.round(refl * 255).astype(np.uint8)),
        t=[horizons.jd_tdb_to_utc_iso(j)[:16].replace("T", " ") for j in jds],
        sun=np.round(s_hat, 5).tolist(), earth=np.round(e_hat, 5).tolist(),
        sun_km=np.round(np.linalg.norm(sun, axis=1) / 1e3).tolist(),
        earth_km=np.round(np.linalg.norm(earth, axis=1) / 1e3).tolist(),
        frac=np.round(frac, 4).tolist(),
        phase=[phase_name(f, w) for f, w in zip(frac, waxing)],
        sub_sun=np.round(np.column_stack([s_lat, wrap(s_lon)]), 2).tolist(),
        sub_earth=np.round(np.column_stack([e_lat, wrap(e_lon)]), 2).tolist(),
        r_sun=SUN_DISPLAY_R * R_LOLA / 1e3, r_earth=EARTH_DISPLAY_R * R_LOLA / 1e3,
    )


SUN_DISPLAY_R, EARTH_DISPLAY_R = 3.4, 2.3   # marker distances in Moon radii (not to scale)

TIMELINE_JS = r"""
(function () {
  const gd = document.getElementById('{plot_id}');
  const T = gd.layout.meta.timeline;
  const bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const nx = new Int8Array(bytes(T.nx).buffer), ny = new Int8Array(bytes(T.ny).buffer),
        nz = new Int8Array(bytes(T.nz).buffer), refl = bytes(T.refl);
  const [nr, nc] = T.shape, N = T.t.length;
  const idx = name => gd.data.findIndex(d => d.name === name);
  const iSun = idx('Sun'), iEarth = idx('Earth');

  document.body.insertAdjacentHTML('beforeend', `
   <div id="tl" style="position:fixed;left:50%;bottom:14px;transform:translateX(-50%);
        width:min(760px,calc(100vw - 32px));background:rgba(20,20,28,.88);color:#eee;
        font:13px system-ui,sans-serif;padding:10px 14px;border-radius:10px;z-index:10;
        box-shadow:0 2px 12px rgba(0,0,0,.6)">
     <div style="display:flex;gap:10px;align-items:center">
       <button id="tl-play" style="width:64px;background:#333;color:#eee;border:1px solid #555;
               border-radius:6px;padding:4px 0;cursor:pointer">Play</button>
       <input id="tl-range" type="range" min="0" max="${N - 1}" value="0" step="1" style="flex:1">
       <b id="tl-date" style="min-width:128px;text-align:right"></b>
     </div>
     <div id="tl-info" style="margin-top:6px;opacity:.85;font-size:12px;line-height:1.5"></div>
   </div>`);
  const range = document.getElementById('tl-range'), play = document.getElementById('tl-play');

  function surfacecolor(k) {
    const [sx, sy, sz] = T.sun[k], rows = new Array(nr);
    for (let r = 0, i = 0; r < nr; r++) {
      const row = new Array(nc);
      for (let c = 0; c < nc; c++, i++) {
        const cosi = Math.max(0, (nx[i] * sx + ny[i] * sy + nz[i] * sz) / 127);
        row[c] = Math.min(1, refl[i] / 255 * (0.03 + 0.97 * cosi) * 1.35);
      }
      rows[r] = row;
    }
    return rows;
  }
  const seg = (v, r) => [[v[0] * 1782, v[0] * r], [v[1] * 1782, v[1] * r], [v[2] * 1782, v[2] * r]];

  let busy = false, pending = null;
  async function show(k) {
    if (busy) { pending = k; return; }
    busy = true;
    const s = seg(T.sun[k], T.r_sun), e = seg(T.earth[k], T.r_earth);
    document.getElementById('tl-date').textContent = T.t[k] + ' UTC';
    document.getElementById('tl-info').innerHTML =
      `<b>${T.phase[k]}</b>, ${(100 * T.frac[k]).toFixed(1)}% lit as seen from Earth · ` +
      `<span style="color:#4da6ff">Earth</span> ${T.earth_km[k].toLocaleString()} km, ` +
      `sub-Earth ${T.sub_earth[k][0]}°, ${T.sub_earth[k][1]}° · ` +
      `<span style="color:#ffcc33">Sun</span> ${(T.sun_km[k] / 1e6).toFixed(2)} M km, ` +
      `sub-solar ${T.sub_sun[k][0]}°, ${T.sub_sun[k][1]}°`;
    await Plotly.restyle(gd, {x: [s[0], e[0]], y: [s[1], e[1]], z: [s[2], e[2]],
      hovertext: [[`Sun: ${T.sun_km[k].toLocaleString()} km`, `Sun: ${T.sun_km[k].toLocaleString()} km`],
                  [`Earth: ${T.earth_km[k].toLocaleString()} km`, `Earth: ${T.earth_km[k].toLocaleString()} km`]]},
      [iSun, iEarth]);
    await Plotly.restyle(gd, {surfacecolor: [surfacecolor(k)]}, [0]);
    busy = false;
    if (pending !== null) { const p = pending; pending = null; show(p); }
  }
  range.addEventListener('input', () => show(+range.value));

  let timer = null;
  play.addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = 'Play'; return; }
    play.textContent = 'Pause';
    timer = setInterval(() => {
      if (busy) return;
      range.value = (+range.value + 1) % N;
      show(+range.value);
    }, 120);
  });
  show(0);
})();
"""


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
                labels=False, source="lola", gravity_degree=150, timeline=None):
    """timeline=(days, step_hours) adds a time slider that moves the Sun and Earth (JPL Horizons)
    and re-lights the Moon in the browser. Needs a global view with the WAC texture."""
    if timeline and (region or color != "wac"):
        raise ValueError("--timeline needs the global WAC view (no --region, --color wac)")
    if timeline and date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    lat, lon, h, geo_label = load_geometry(ppd, stride, region, source)
    c, cs, ctitle, unit = color_layer(color, lat, lon, h, region, gravity_degree)
    refl = c.copy()

    sun_me = earth_me = None
    lit = date is not None
    if lit:
        jd = horizons.utc_iso_to_jd_tdb(date)
        if timeline:
            jds, sun_track, earth_track = sun_earth_track(jd, *timeline)
            sun_me = sun_track[0] / np.linalg.norm(sun_track[0])
            earth_me = earth_track[0] / np.linalg.norm(earth_track[0])
        else:
            sun_me, earth_me = sun_and_earth_me(jd)
        cosi = np.clip(illumination(h, lat, lon, sun_me, exaggeration), 0, None)
        if color == "wac":     # reflectance x Lambert shading + faint earthshine
            c = c * (0.03 + 0.97 * cosi)
            c = np.clip(c / max(np.percentile(c, 99.5), 1e-6), 0, 1)
            ctitle = f"Sunlit WAC reflectance, {date} UTC"
            if timeline:
                ctitle = f"Sunlit WAC reflectance, {date} UTC + {timeline[0]:g} days (use the time slider)"

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
        **(dict(cmin=0, cmax=1) if timeline else {}),
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

    if timeline:
        R0 = R_LOLA / 1000
        for track, name, col, rd, size in ((sun_track, "Sun", "#ffcc33", SUN_DISPLAY_R, 22),
                                          (earth_track, "Earth", "#4da6ff", EARTH_DISPLAY_R, 14)):
            u = track / np.linalg.norm(track, axis=1, keepdims=True) * rd * R0
            fig.add_trace(go.Scatter3d(x=u[:, 0], y=u[:, 1], z=u[:, 2], mode="lines",
                                       line=dict(color=col, width=2, dash="dot"), opacity=0.5,
                                       hoverinfo="skip", name=f"{name} path (Moon-fixed)"))
            p = np.outer([1.025, rd], u[0] / np.linalg.norm(u[0]))
            fig.add_trace(go.Scatter3d(x=p[:, 0] * R0, y=p[:, 1] * R0, z=p[:, 2] * R0,
                                       mode="lines+markers+text", text=["", name], textposition="top center",
                                       line=dict(color=col, width=5), marker=dict(size=[0, size], color=col),
                                       textfont=dict(color=col, size=13), hoverinfo="text", name=name))
        fig.update_layout(meta=dict(timeline=timeline_meta(jds, sun_track, earth_track, lat, lon, h,
                                                           refl, exaggeration)))
    elif lit and not region:
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
    has_timeline = isinstance(fig.layout.meta, dict) and "timeline" in fig.layout.meta
    fig.write_html(path, include_plotlyjs="cdn", auto_open=open_browser,
                   config=dict(scrollZoom=True, displaylogo=False),
                   post_script=TIMELINE_JS if has_timeline else None)
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
    ap.add_argument("--timeline", type=float, metavar="DAYS",
                    help="add a time slider spanning DAYS from --date (default now): Sun and Earth "
                         "positions from JPL Horizons, Moon re-lit at every step")
    ap.add_argument("--step", type=float, default=6.0, help="time-slider step in hours (default 6)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    date = None if a.date.lower() == "none" else (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M") if a.date.lower() == "now" else a.date)
    fig = moon_figure(a.ppd, a.stride, a.region, a.exaggeration, a.color, date, a.labels, a.source,
                      a.gravity_degree, timeline=(a.timeline, a.step) if a.timeline else None)
    name = f"moon_{a.color}{'_region' if a.region else ''}{'_timeline' if a.timeline else ''}.html"
    write(fig, a.out or OUT_DIR / name, not a.no_open)


if __name__ == "__main__":
    main()
