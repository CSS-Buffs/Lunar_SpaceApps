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
    Returns dicts {jd (TDB), sun, earth (N,3) Moon-centred metres} at the slider step and at a
    finer step (<= 1 h) used for the site visibility windows."""
    eph = horizons.perturber_ephemerides(jd0, jd0 + days, pad_days=0.2)

    def at(step):
        jds = jd0 + np.arange(0.0, days * 24.0 + 1e-9, step) / 24.0
        R = lf.icrf_to_me(jds)
        return dict(jd=jds, sun=np.einsum("nij,nj->ni", R, eph["sun"].position(jds)),
                    earth=np.einsum("nij,nj->ni", R, eph["earth"].position(jds)))
    return at(step_h), at(min(1.0, step_h))


def jd_tdb_to_unix_ms(jd):
    return np.round((np.asarray(jd) - 2440587.5) * 86400e3 - horizons.TDB_MINUS_UTC * 1e3).astype(np.int64)


def site_meta(fine, horizon_ppd, exaggeration):
    """Data for the in-page site picker: a LOLA height grid for terrain horizons and a fine
    Sun/Earth track (Moon-centred ME, km) for visibility windows."""
    import base64
    dem = lola_data.load(horizon_ppd)
    hgt = np.clip(np.round(dem.height_m), -32767, 32767).astype("<i2")
    return dict(
        dem=base64.b64encode(hgt.tobytes()).decode("ascii"), dem_shape=list(hgt.shape),
        dem_ppd=horizon_ppd, r_moon_km=R_LOLA / 1e3, exag=exaggeration,
        t0_ms=int(jd_tdb_to_unix_ms(fine["jd"][0])),
        dt_ms=int(round((fine["jd"][1] - fine["jd"][0]) * 86400e3)) if len(fine["jd"]) > 1 else 3600000,
        sun=np.round(fine["sun"] / 1e3).astype(np.int64).ravel().tolist(),
        earth=np.round(fine["earth"] / 1e3, 1).ravel().tolist(),
    )


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
   <div id="tl" style="position:fixed;left:16px;bottom:14px;
        width:min(760px,calc(100vw - 32px - var(--site-w, 0px)));background:rgba(20,20,28,.88);color:#eee;
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
    window.dispatchEvent(new CustomEvent('tl-step', {detail: k}));
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

SITE_JS = r"""
(function () {
  const gd = document.getElementById('{plot_id}');
  const T = gd.layout.meta.timeline, S = T.site;
/*CORE*/
  const Core = (function (S) {
    const D2R = Math.PI / 180, R2D = 180 / Math.PI, RM = S.r_moon_km, N_AZ = 720;
    const dem = new Int16Array(Uint8Array.from(atob(S.dem), c => c.charCodeAt(0)).buffer);
    const [NR, NC] = S.dem_shape, ppd = S.dem_ppd, nF = S.sun.length / 3;
    const t1 = S.t0_ms + (nF - 1) * S.dt_ms;

    function height(lat, lon) {                       // LOLA height [m], bilinear
      let r = Math.min(Math.max((90 - lat) * ppd - 0.5, 0), NR - 1);
      const c = (((lon % 360) + 360) % 360) * ppd - 0.5;
      const r0 = Math.min(Math.floor(r), NR - 2), fr = r - r0, cf = Math.floor(c), fc = c - cf;
      const c0 = ((cf % NC) + NC) % NC, c1 = (c0 + 1) % NC, i0 = r0 * NC, i1 = i0 + NC;
      return (dem[i0 + c0] * (1 - fc) + dem[i0 + c1] * fc) * (1 - fr) +
             (dem[i1 + c0] * (1 - fc) + dem[i1 + c1] * fc) * fr;
    }

    // Terrain horizon: for each azimuth, march along the great circle and keep the highest
    // elevation angle of the terrain, including the curvature of the Moon.
    const DIST = []; for (let d = 0.5; d < 600; d += Math.max(0.5, d * 0.015)) DIST.push(d);
    function horizon(lat, lon, ro) {
      const p1 = lat * D2R, sp = Math.sin(p1), cp = Math.cos(p1), hz = new Float32Array(N_AZ);
      for (let k = 0; k < N_AZ; k++) {
        const az = k * 2 * Math.PI / N_AZ, ca = Math.cos(az), sa = Math.sin(az);
        let best = -90;
        for (const d of DIST) {
          const del = d / RM, cd = Math.cos(del), sd = Math.sin(del);
          const sp2 = Math.min(1, Math.max(-1, sp * cd + cp * sd * ca));
          const lon2 = lon + Math.atan2(sa * sd * cp, cd - sp * sp2) * R2D;
          const rt = RM + height(Math.asin(sp2) * R2D, lon2) / 1000;
          const el = Math.atan2(rt * cd - ro, rt * sd) * R2D;
          if (el > best) best = el;
        }
        hz[k] = best;
      }
      return hz;
    }

    function site(lat, lon, mast_m = 2) {
      lat = Math.max(-89.99, Math.min(89.99, lat));
      const h_m = height(lat, lon), ro = RM + (h_m + mast_m) / 1000;
      const la = lat * D2R, lo = lon * D2R;
      const u = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
      const e = [-Math.sin(lo), Math.cos(lo), 0];
      const n = [-Math.sin(la) * Math.cos(lo), -Math.sin(la) * Math.sin(lo), Math.cos(la)];
      return {lat, lon, h_m, u, e, n, p: u.map(x => x * ro), hz: horizon(lat, lon, ro)};
    }
    function hzAt(hz, az) {
      const x = az / 360 * N_AZ, k0 = Math.floor(x), f = x - k0;
      return hz[k0 % N_AZ] * (1 - f) + hz[(k0 + 1) % N_AZ] * f;
    }
    function bodyAt(arr, t) {                         // Moon-centred ME position [km]
      const x = Math.min(Math.max((t - S.t0_ms) / S.dt_ms, 0), nF - 1);
      const i = Math.min(Math.floor(x), Math.max(nF - 2, 0)), f = x - i, j = Math.min(i + 1, nF - 1);
      return [0, 1, 2].map(c => arr[3 * i + c] * (1 - f) + arr[3 * j + c] * f);
    }
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    function state(st, body, t) {
      const v = bodyAt(S[body], t), d = [v[0] - st.p[0], v[1] - st.p[1], v[2] - st.p[2]];
      const r = Math.hypot(d[0], d[1], d[2]);
      const el = Math.asin(dot(d, st.u) / r) * R2D;
      const az = ((Math.atan2(dot(d, st.e), dot(d, st.n)) * R2D) + 360) % 360;
      const hz = hzAt(st.hz, az), rad = Math.asin((body === 'sun' ? 695700 : 6371) / r) * R2D;
      return {el, az, r, hz, rad, margin: el - hz};
    }
    // Visibility (body centre above the terrain horizon), sampled every stepMin minutes with
    // rise/set times refined by linear interpolation of the margin.
    function intervals(ts, m) {
      const out = [];
      let a = m[0] > 0 ? ts[0] : null, openA = m[0] > 0;
      for (let i = 1; i < ts.length; i++) {
        if ((m[i - 1] > 0) === (m[i] > 0)) continue;
        const tc = ts[i - 1] + (ts[i] - ts[i - 1]) * m[i - 1] / (m[i - 1] - m[i]);
        if (m[i] > 0) { a = tc; openA = false; } else { out.push({a, b: tc, openA, openB: false}); a = null; }
      }
      if (a !== null) out.push({a, b: ts[ts.length - 1], openA, openB: true});
      return out;
    }
    function windows(st, stepMin = 10) {
      const ts = [], sun = [], earth = [], both = [];
      for (let t = S.t0_ms; t <= t1 + 1; t += stepMin * 60000) {
        const ms = state(st, 'sun', t).margin, me = state(st, 'earth', t).margin;
        ts.push(t); sun.push(ms); earth.push(me); both.push(Math.min(ms, me));
      }
      const frac = m => m.filter(x => x > 0).length / m.length;
      return {ts, sun: intervals(ts, sun), earth: intervals(ts, earth), both: intervals(ts, both),
              frac: {sun: frac(sun), earth: frac(earth), both: frac(both)}};
    }
    return {site, state, windows, height, hzAt, N_AZ, t0: S.t0_ms, t1};
  })(S);
/*ENDCORE*/

  const RM = S.r_moon_km, R2D = 180 / Math.PI;
  const iSite = gd.data.findIndex(d => d.name === 'Selected site');
  const range = document.getElementById('tl-range');
  let st = null, win = null, paths = null, curK = 0;

  document.head.insertAdjacentHTML('beforeend', `<style>
    :root { --site-w: 368px; }
    html, body { margin: 0; overflow: hidden; background: #000; }
    #site { position: fixed; top: 70px; right: 12px; width: 340px; max-height: calc(100vh - 90px);
      overflow: auto; background: rgba(20,20,28,.92); color: #eee; font: 13px system-ui, sans-serif;
      padding: 12px 14px; border-radius: 10px; z-index: 11; box-shadow: 0 2px 12px rgba(0,0,0,.6); }
    #site input { width: 76px; background: #111; color: #eee; border: 1px solid #555; border-radius: 5px; padding: 3px 5px; }
    #site button { background: #333; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 3px 10px; cursor: pointer; }
    #site .muted { opacity: .7; font-size: 12px; }
    #site .row { display: flex; gap: 8px; align-items: baseline; margin: 3px 0; }
    #site .tag { display: inline-block; min-width: 52px; font-weight: 600; }
    #site details { margin-top: 8px; } #site summary { cursor: pointer; font-weight: 600; }
    #site ol { margin: 4px 0 0; padding-left: 20px; font-size: 12px; line-height: 1.55; }
    #site svg text { font: 10px system-ui, sans-serif; fill: #bbb; }
    #site.collapsed #site-body { display: none; }
    @media (max-width: 900px) {
      :root { --site-w: 0px; }
      #site { top: auto; bottom: 118px; left: 16px; right: 16px; width: auto; max-height: 42vh; }
    }</style>`);
  document.body.insertAdjacentHTML('beforeend', `
    <div id="site">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b style="font-size:14px">Site visibility</b>
        <button id="site-hide" title="Collapse">–</button>
      </div>
      <div id="site-body">
        <div class="muted" style="margin:4px 0 8px">Click anywhere on the Moon, or type coordinates
          (°, east longitude positive).</div>
        <div class="row">
          <label>Lat <input id="site-lat" type="number" step="0.01" min="-90" max="90"></label>
          <label>Lon <input id="site-lon" type="number" step="0.01" min="-360" max="360"></label>
          <button id="site-go">Go</button>
        </div>
        <div id="site-out"></div>
      </div>
    </div>`);
  const $ = id => document.getElementById(id);
  $('site-hide').onclick = () => {
    const c = $('site').classList.toggle('collapsed');
    $('site-hide').textContent = c ? '+' : '–';
  };

  const fmtLat = v => `${Math.abs(v).toFixed(3)}° ${v >= 0 ? 'N' : 'S'}`;
  const fmtLon = v => { v = ((v + 540) % 360) - 180; return `${Math.abs(v).toFixed(3)}° ${v >= 0 ? 'E' : 'W'}`; };
  const fmtT = t => new Date(t).toISOString().slice(5, 16).replace('T', ' ');
  const fmtDur = ms => { const h = ms / 3.6e6; return h >= 48 ? `${Math.floor(h / 24)} d ${Math.round(h % 24)} h`
                                                              : `${Math.floor(h)} h ${Math.round((h % 1) * 60)} min`; };
  const COL = {sun: '#ffcc33', earth: '#4da6ff', both: '#7ee08a'};

  function select(lat, lon) {
    $('site-lat').value = lat.toFixed(3);
    $('site-lon').value = (((lon + 540) % 360) - 180).toFixed(3);
    $('site-out').innerHTML = '<div class="muted" style="margin-top:10px">Computing terrain horizon…</div>';
    setTimeout(() => {
      st = Core.site(lat, lon);
      win = Core.windows(st);
      paths = {sun: [], earth: []};
      for (let t = Core.t0; t <= Core.t1; t += 3600000)
        for (const b of ['sun', 'earth']) { const s = Core.state(st, b, t); paths[b].push([s.az, s.el]); }
      const R = (RM * 1000 + S.exag * st.h_m) / 1000 + 8;
      Plotly.restyle(gd, {x: [[st.u[0] * R]], y: [[st.u[1] * R]], z: [[st.u[2] * R]],
                          hovertext: [[`Site ${fmtLat(st.lat)}, ${fmtLon(st.lon)}`]]}, [iSite]);
      render();
    }, 20);
  }

  function statusLine(body, k) {
    const s = Core.state(st, body, T.t_ms[k]);
    let txt, col;
    if (s.margin > s.rad) { txt = `Visible, ${s.margin.toFixed(1)}° above the terrain`; col = '#7ee08a'; }
    else if (s.margin > -s.rad) { txt = 'Partly visible (disc on the horizon line)'; col = '#ffd166'; }
    else if (s.el > 0) { txt = `Blocked by terrain (${(-s.margin).toFixed(1)}° below the ridge)`; col = '#ff7b7b'; }
    else { txt = 'Below the horizon'; col = '#ff7b7b'; }
    return `<div class="row"><span class="tag" style="color:${COL[body]}">${body === 'sun' ? 'Sun' : 'Earth'}</span>
      <span><span style="color:${col}">${txt}</span><br><span class="muted">elevation ${s.el.toFixed(2)}°,
      azimuth ${s.az.toFixed(1)}°, terrain horizon ${s.hz.toFixed(2)}°</span></span></div>`;
  }

  function bars(k) {
    const W = 312, L = 44, H = 16, span = Core.t1 - Core.t0, x = t => L + (t - Core.t0) / span * (W - L);
    let g = '';
    [['sun', 'Sun'], ['earth', 'Earth'], ['both', 'Both']].forEach(([b, name], r) => {
      const y = 4 + r * (H + 6);
      g += `<text x="0" y="${y + 12}">${name}</text><rect x="${L}" y="${y}" width="${W - L}" height="${H}" fill="#2a2a33"/>`;
      for (const w of win[b]) g += `<rect x="${x(w.a)}" y="${y}" width="${Math.max(1, x(w.b) - x(w.a))}" height="${H}" fill="${COL[b]}"/>`;
    });
    const cx = x(T.t_ms[k]);
    g += `<line x1="${cx}" x2="${cx}" y1="0" y2="70" stroke="#fff" stroke-width="1.5"/>`;
    g += `<text x="${L}" y="82">${fmtT(Core.t0)}</text><text x="${W}" y="82" text-anchor="end">${fmtT(Core.t1)}</text>`;
    return `<svg id="site-bars" width="${W}" height="86" style="cursor:pointer;display:block;margin-top:6px">${g}</svg>`;
  }

  function panorama(k) {
    const W = 312, H = 160, pl = 26, pb = 16, hz = st.hz;
    let hmax = -90, hmin = 90; hz.forEach(v => { hmax = Math.max(hmax, v); hmin = Math.min(hmin, v); });
    const y0 = Math.floor(Math.min(-2, hmin - 1)), y1 = Math.ceil(Math.min(90, Math.max(10, hmax + 6)));
    const X = az => pl + az / 360 * (W - pl), Y = el => (H - pb) * (1 - (el - y0) / (y1 - y0));
    let g = `<rect x="${pl}" y="0" width="${W - pl}" height="${H - pb}" fill="#0b0d18"/>`;
    let pts = `${X(0)},${Y(y0)} `;
    for (let i = 0; i <= Core.N_AZ; i++) pts += `${X(i * 360 / Core.N_AZ)},${Y(Math.max(y0, hz[i % Core.N_AZ]))} `;
    g += `<polygon points="${pts}${X(360)},${Y(y0)}" fill="#4a4a52" stroke="#8a8a90" stroke-width="1"/>`;
    if (y0 < 0) g += `<line x1="${pl}" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="#666" stroke-dasharray="3 3"/>`;
    for (const b of ['sun', 'earth'])
      for (const [az, el] of paths[b]) if (el >= y0 && el <= y1)
        g += `<circle cx="${X(az)}" cy="${Y(el)}" r="1" fill="${COL[b]}" opacity=".45"/>`;
    for (const b of ['sun', 'earth']) {
      const s = Core.state(st, b, T.t_ms[k]), cx = X(s.az);
      if (s.el > y1) g += `<path d="M${cx - 5},10 L${cx + 5},10 L${cx},2 Z" fill="${COL[b]}"/>
                           <text x="${cx + 7}" y="10" style="fill:${COL[b]}">${s.el.toFixed(0)}°</text>`;
      else if (s.el >= y0) g += `<circle cx="${cx}" cy="${Y(s.el)}" r="${b === 'sun' ? 5 : 4}" fill="${COL[b]}" stroke="#000"/>`;
    }
    [[0, 'N'], [90, 'E'], [180, 'S'], [270, 'W'], [360, 'N']].forEach(([a, n]) =>
      g += `<text x="${X(a)}" y="${H - 3}" text-anchor="middle">${n}</text>`);
    [y0, 0, y1].filter((v, i, a) => a.indexOf(v) === i).forEach(v =>
      g += `<text x="${pl - 3}" y="${Math.min(H - pb, Math.max(9, Y(v) + 3))}" text-anchor="end">${v}°</text>`);
    return `<svg width="${W}" height="${H}" style="display:block;margin-top:4px">${g}</svg>`;
  }

  function list(b) {
    const w = win[b];
    if (!w.length) return '<div class="muted">Never visible in this time span.</div>';
    if (w.length === 1 && w[0].openA && w[0].openB) return '<div class="muted">Visible for the whole time span.</div>';
    return '<ol>' + w.map(x => `<li>${x.openA ? '… ' : ''}${fmtT(x.a)} → ${fmtT(x.b)}${x.openB ? ' …' : ''}
      <span class="muted">(${fmtDur(x.b - x.a)})</span></li>`).join('') + '</ol>';
  }

  function render() {
    if (!st) return;
    const k = curK, pct = v => `${(100 * v).toFixed(0)}%`;
    let hmax = -90, azmax = 0; st.hz.forEach((v, i) => { if (v > hmax) { hmax = v; azmax = i * 360 / Core.N_AZ; } });
    $('site-out').innerHTML = `
      <div style="margin-top:10px;font-size:15px"><b>${fmtLat(st.lat)}, ${fmtLon(st.lon)}</b></div>
      <div class="muted">Elevation ${Math.round(st.h_m).toLocaleString()} m (LOLA, vs 1737.4 km sphere) ·
        highest horizon ${hmax.toFixed(1)}° toward azimuth ${azmax.toFixed(0)}°</div>
      <div style="margin-top:10px"><b>${T.t[k]} UTC</b></div>
      ${statusLine('sun', k)}${statusLine('earth', k)}
      <div style="margin-top:10px"><b>Visibility windows</b> <span class="muted">(click a bar to jump in time)</span></div>
      ${bars(k)}
      <div class="muted">Sun ${pct(win.frac.sun)} · Earth ${pct(win.frac.earth)} · both together ${pct(win.frac.both)} of the time span</div>
      <div style="margin-top:10px"><b>Horizon panorama</b> <span class="muted">(terrain, Sun and Earth paths)</span></div>
      ${panorama(k)}
      <details open><summary style="color:${COL.sun}">Sunlight windows (${win.sun.length})</summary>${list('sun')}</details>
      <details open><summary style="color:${COL.earth}">Earth line-of-sight windows (${win.earth.length})</summary>${list('earth')}</details>
      <details><summary style="color:${COL.both}">Sun and Earth together (${win.both.length})</summary>${list('both')}</details>
      <div class="muted" style="margin-top:10px">Windows count the Sun or Earth as visible when the centre of its disc is above the
        terrain horizon. The horizon is traced to 600 km along 720 azimuths over LOLA LDEM_${S.dem_ppd}
        (${(30.32 / S.dem_ppd).toFixed(1)} km grid), including the Moon's curvature, for an antenna 2 m above the ground.
        Features smaller than the grid are not resolved. Earth visibility is the Moon–Earth line of sight; it
        does not check which ground station on Earth is facing the Moon. Times are UTC.</div>`;
    $('site-bars').onclick = ev => {
      const r = ev.currentTarget.getBoundingClientRect(), f = (ev.clientX - r.left - 44) / (r.width - 44);
      const t = Core.t0 + Math.min(1, Math.max(0, f)) * (Core.t1 - Core.t0);
      let best = 0; T.t_ms.forEach((v, i) => { if (Math.abs(v - t) < Math.abs(T.t_ms[best] - t)) best = i; });
      range.value = best; range.dispatchEvent(new Event('input'));
    };
  }

  window.addEventListener('tl-step', e => { curK = e.detail; render(); });
  gd.on('plotly_click', ev => {
    const p = ev.points && ev.points[0];
    if (!p || p.data.type !== 'surface') return;
    const r = Math.hypot(p.x, p.y, p.z);
    select(Math.asin(p.z / r) * R2D, Math.atan2(p.y, p.x) * R2D);
  });
  $('site-go').onclick = () => {
    const la = parseFloat($('site-lat').value), lo = parseFloat($('site-lon').value);
    if (isFinite(la) && isFinite(lo) && Math.abs(la) <= 90) select(la, lo);
  };
  ['site-lat', 'site-lon'].forEach(id => $(id).addEventListener('keydown', e => { if (e.key === 'Enter') $('site-go').click(); }));
  const m = location.hash.match(/site=(-?[\d.]+),(-?[\d.]+)/);   // e.g. page.html#site=-89.45,-137
  if (m) select(+m[1], +m[2]);
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
                labels=False, source="lola", gravity_degree=150, timeline=None,
                horizon_ppd=4):
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
            coarse, fine = sun_earth_track(jd, *timeline)
            jds, sun_track, earth_track = coarse["jd"], coarse["sun"], coarse["earth"]
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
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Selected site",
                                   marker=dict(size=7, color="#ff4d6d", line=dict(color="white", width=2)),
                                   hoverinfo="text"))
        tl = timeline_meta(jds, sun_track, earth_track, lat, lon, h, refl, exaggeration)
        tl["t_ms"] = jd_tdb_to_unix_ms(jds).tolist()
        tl["site"] = site_meta(fine, horizon_ppd, exaggeration)
        fig.update_layout(meta=dict(timeline=tl))
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
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.01, y=0.88 if timeline else 0.02, bgcolor="rgba(0,0,0,0)"),
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
                   post_script=[TIMELINE_JS, SITE_JS] if has_timeline else None)
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
    ap.add_argument("--horizon-ppd", type=int, default=4, choices=lola_data.SUPPORTED_RES,
                    help="LOLA grid used for site terrain horizons (4 = 7.6 km, 16 = 1.9 km but ~40 MB page)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    date = None if a.date.lower() == "none" else (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M") if a.date.lower() == "now" else a.date)
    fig = moon_figure(a.ppd, a.stride, a.region, a.exaggeration, a.color, date, a.labels, a.source,
                      a.gravity_degree, timeline=(a.timeline, a.step) if a.timeline else None, horizon_ppd=a.horizon_ppd)
    name = f"moon_{a.color}{'_region' if a.region else ''}{'_timeline' if a.timeline else ''}.html"
    write(fig, a.out or OUT_DIR / name, not a.no_open)


if __name__ == "__main__":
    main()
