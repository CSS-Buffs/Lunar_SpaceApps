/* Lunar site planner front end.
 * Data: /api/static (LOLA mesh, terrain normals, WAC reflectance, LOLA heights for horizons),
 *       /api/texture.png (Moon Trek WAC mosaic), /api/track (JPL Horizons Sun + Earth, Moon ME frame, km).
 * All geometry is in the Moon Mean-Earth frame: x toward 0°E on the equator, z north. */
'use strict';

const D2R = Math.PI / 180, R2D = 180 / Math.PI, MIN = 60000, HOUR = 3600000, DAY = 86400000;
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const wrap180 = v => ((v % 360) + 540) % 360 - 180;
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const dirOf = (lat, lon) => [Math.cos(lat * D2R) * Math.cos(lon * D2R), Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R)];
const latlonOf = v => { const r = Math.hypot(v[0], v[1], v[2]); return [Math.asin(v[2] / r) * R2D, Math.atan2(v[1], v[0]) * R2D]; };

/* ------------------------------------------------------------------ formatting */
const pad = n => String(n).padStart(2, '0');
const fmtUTC = t => new Date(t).toISOString().slice(0, 16).replace('T', ' ');
const fmtShort = t => new Date(t).toISOString().slice(5, 16).replace('T', ' ');
const fmtHM = ms => { const m = Math.round(ms / MIN); return `${pad(Math.floor(m / 60))}:${pad(m % 60)}`; };
const fmtDeg = (v, d = 2) => `${v.toFixed(d)}°`;
const fmtLat = v => `${Math.abs(v).toFixed(3)}° ${v >= 0 ? 'N' : 'S'}`;
const fmtLon = v => { v = wrap180(v); return `${Math.abs(v).toFixed(3)}° ${v >= 0 ? 'E' : 'W'}`; };
const fmtSite = s => `${s.lat.toFixed(2)}°, ${wrap180(s.lon).toFixed(2)}°`;

/* ------------------------------------------------------------------ state */
// [name, lat °N, lon °E, approximate?]  Landing coordinates from LROC / mission reports; "≈" marks
// region centres or published approximate positions.
const LIBRARY = [
  ['Apollo (crewed)', [
    ['Apollo 11', 0.6741, 23.4730], ['Apollo 12', -3.0124, -23.4216], ['Apollo 14', -3.6453, -17.4714],
    ['Apollo 15', 26.1322, 3.6339], ['Apollo 16', -8.9730, 15.5002], ['Apollo 17', 20.1908, 30.7717]]],
  ['Surveyor (NASA)', [
    ['Surveyor 1', -2.4745, -43.3394], ['Surveyor 3', -3.0161, -23.4153], ['Surveyor 5', 1.4550, 23.1943],
    ['Surveyor 6', 0.4734, -1.4269], ['Surveyor 7', -40.9813, -11.5100]]],
  ['Luna & Lunokhod (USSR)', [
    ['Luna 9', 7.08, -64.37, 1], ['Luna 13', 18.87, -62.05, 1], ['Luna 16', -0.5137, 56.3638],
    ['Luna 17 / Lunokhod 1', 38.2378, -35.0007], ['Luna 20', 3.7866, 56.6243],
    ['Luna 21 / Lunokhod 2', 25.85, 30.45, 1], ['Luna 24', 12.7145, 62.2129]]],
  ["Chang'e (China)", [
    ["Chang'e 3 / Yutu", 44.1214, -19.5116], ["Chang'e 4 / Yutu-2 (far side)", -45.4446, 177.5991],
    ["Chang'e 5", 43.0576, -51.9161], ["Chang'e 6 (far side)", -41.6385, -153.9852]]],
  ['Recent landers', [
    ['Chandrayaan-3 (Vikram)', -69.3733, 32.3191], ['SLIM (JAXA)', -13.316, 25.251],
    ['IM-1 Odysseus', -80.13, 1.44, 1], ['Blue Ghost Mission 1', 18.56, 61.81, 1], ['IM-2 Athena', -84.79, 29.20, 1]]],
  ['South polar sites', [
    ['Malapert Massif summit', -85.90, 2.30, 1], ['Shackleton connecting ridge', -89.45, -137.0, 1],
    ['Shackleton crater floor (permanent shadow)', -89.67, 129.78], ['de Gerlache crater', -88.5, -87.1, 1],
    ['Haworth crater', -87.45, -5.0, 1], ['Nobile crater', -85.2, 53.5, 1], ['Mons Mouton', -84.6, 31.0, 1],
    ['Cabeus (LCROSS impact)', -84.68, -48.70], ['Schrödinger basin', -75.0, 132.4, 1]]],
  ['North polar sites', [
    ['Peary crater', 88.6, 33.0, 1], ['Hermite crater', 86.0, -89.9, 1], ['Plaskett crater', 82.3, 174.3, 1]]],
  ['Craters', [
    ['Tycho', -43.31, -11.36], ['Copernicus', 9.62, -20.08], ['Aristarchus', 23.73, -47.49], ['Kepler', 8.12, -38.01],
    ['Plato', 51.62, -9.38], ['Clavius', -58.62, -14.73], ['Ptolemaeus', -9.16, -1.84], ['Gassendi', -17.55, -39.96],
    ['Theophilus', -11.4, 26.4], ['Langrenus', -8.86, 60.9], ['Petavius', -25.1, 60.4], ['Archimedes', 29.7, -4.0],
    ['Tsiolkovskiy (far side)', -20.4, 129.1], ['Von Kármán (far side)', -44.8, 175.9], ['Korolev (far side)', -4.0, -157.4]]],
  ['Maria & basins', [
    ['Mare Tranquillitatis', 8.35, 30.83, 1], ['Mare Imbrium', 32.8, -15.6, 1], ['Mare Serenitatis', 28.0, 17.5, 1],
    ['Mare Crisium', 16.18, 59.1, 1], ['Oceanus Procellarum', 18.4, -57.4, 1], ['Mare Orientale', -19.4, -92.8, 1],
    ['Mare Moscoviense (far side)', 27.3, 147.9, 1], ['South Pole–Aitken basin centre', -53.0, -169.0, 1]]],
  ['Science targets', [
    ['Marius Hills pit (lava tube)', 14.09, -56.77], ['Mare Tranquillitatis pit', 8.335, 33.222],
    ['Reiner Gamma swirl', 7.5, -59.0, 1], ['Ina irregular mare patch', 18.65, 5.30, 1], ['Gruithuisen domes', 36.3, -40.5, 1]]],
  ['Reference points', [
    ['Mean sub-Earth point', 0, 0], ['Far-side centre', 0, 180], ['East limb', 0, 90], ['West limb', 0, -90]]],
];
const DEFAULT_SITES = [
  {id: 'A', name: 'Apollo 11', note: 'Apollo (crewed)', lat: 0.6741, lon: 23.4730},
  {id: 'B', name: 'Malapert Massif summit', note: 'South polar sites', lat: -85.90, lon: 2.30},
  {id: 'C', name: 'Shackleton connecting ridge', note: 'South polar sites', lat: -89.45, lon: -137.0},
];

const S = {
  stat: null, dem: null, mesh: null, tex: null, track: null,
  sites: loadSites(), sel: null, t: 0, view: '3d', playing: false, drop: false,
  cmpMode: 'sites', snapshots: [], geoCache: new Map(), calc: null,
  v2: {lat0: 18, lon0: 30, zoom: 1},
};

function loadSites() {
  try {
    const v = JSON.parse(localStorage.getItem('lunar.sites'));
    if (Array.isArray(v)) return v.map(s => /^Site [A-Z]+$/.test(s.name)              // older "Site A" entries
      ? {...s, name: s.note || `Point ${s.id}`, note: s.note ? '' : s.note} : s);
  } catch (e) {}
  return DEFAULT_SITES.map(s => ({...s}));
}
function saveSites() { try { localStorage.setItem('lunar.sites', JSON.stringify(S.sites)); } catch (e) {} }
function toast(msg) {
  const el = $('toast'); el.textContent = msg; el.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (el.hidden = true), 2600);
}

/* ------------------------------------------------------------------ physics core */
function demHeight(lat, lon) {                          // LOLA height [m], bilinear
  const {h, NR, NC, ppd} = S.dem;
  const r = clamp((90 - lat) * ppd - 0.5, 0, NR - 1), c = (((lon % 360) + 360) % 360) * ppd - 0.5;
  const r0 = Math.min(Math.floor(r), NR - 2), fr = r - r0, cf = Math.floor(c), fc = c - cf;
  const c0 = ((cf % NC) + NC) % NC, c1 = (c0 + 1) % NC, i0 = r0 * NC, i1 = i0 + NC;
  return (h[i0 + c0] * (1 - fc) + h[i0 + c1] * fc) * (1 - fr) + (h[i1 + c0] * (1 - fc) + h[i1 + c1] * fc) * fr;
}

// Terrain horizon: for 720 azimuths, march along the great circle out to 600 km and keep the
// highest elevation angle of the terrain, including the Moon's curvature.
const N_AZ = 720, DIST = [];
for (let d = 0.5; d < 600; d += Math.max(0.5, d * 0.015)) DIST.push(d);
function siteGeom(lat, lon) {
  lat = clamp(lat, -89.99, 89.99); lon = wrap180(lon);
  const key = `${lat.toFixed(4)},${lon.toFixed(4)}`;
  if (S.geoCache.has(key)) return S.geoCache.get(key);
  const RM = S.stat.r_moon_km, h_m = demHeight(lat, lon), ro = RM + (h_m + 2) / 1000;   // 2 m antenna
  const p1 = lat * D2R, sp = Math.sin(p1), cp = Math.cos(p1), hz = new Float32Array(N_AZ);
  for (let k = 0; k < N_AZ; k++) {
    const az = k * 2 * Math.PI / N_AZ, ca = Math.cos(az), sa = Math.sin(az);
    let best = -90;
    for (const d of DIST) {
      const del = d / RM, cd = Math.cos(del), sd = Math.sin(del);
      const sp2 = clamp(sp * cd + cp * sd * ca, -1, 1);
      const lon2 = lon + Math.atan2(sa * sd * cp, cd - sp * sp2) * R2D;
      const rt = RM + demHeight(Math.asin(sp2) * R2D, lon2) / 1000;
      const el = Math.atan2(rt * cd - ro, rt * sd) * R2D;
      if (el > best) best = el;
    }
    hz[k] = best;
  }
  const la = lat * D2R, lo = lon * D2R, u = dirOf(lat, lon);
  const g = {lat, lon, h_m, hz, u, p: u.map(x => x * ro),
             e: [-Math.sin(lo), Math.cos(lo), 0],
             n: [-Math.sin(la) * Math.cos(lo), -Math.sin(la) * Math.sin(lo), Math.cos(la)]};
  S.geoCache.set(key, g);
  return g;
}
const hzAt = (hz, az) => { const x = az / 360 * N_AZ, k = Math.floor(x), f = x - k; return hz[k % N_AZ] * (1 - f) + hz[(k + 1) % N_AZ] * f; };

function bodyAt(body, t) {                              // Moon-centred ME position [km]
  const T = S.track, arr = T[body], n = arr.length / 3;
  const x = clamp((t - T.t0_ms) / T.dt_ms, 0, n - 1), i = Math.min(Math.floor(x), Math.max(n - 2, 0));
  const f = x - i, j = Math.min(i + 1, n - 1);
  return [0, 1, 2].map(c => arr[3 * i + c] * (1 - f) + arr[3 * j + c] * f);
}
function look(g, body, t) {
  const v = bodyAt(body, t), d = [v[0] - g.p[0], v[1] - g.p[1], v[2] - g.p[2]], r = Math.hypot(...d);
  const el = Math.asin(dot(d, g.u) / r) * R2D, az = (Math.atan2(dot(d, g.e), dot(d, g.n)) * R2D + 360) % 360;
  const hz = hzAt(g.hz, az), rad = Math.asin((body === 'sun' ? 695700 : 6371) / r) * R2D;
  return {el, az, r, hz, rad, margin: el - hz};
}
function intervals(ts, m) {                             // visible while margin > 0; crossings interpolated
  const out = []; let a = m[0] > 0 ? ts[0] : null, openA = m[0] > 0;
  for (let i = 1; i < ts.length; i++) {
    if ((m[i - 1] > 0) === (m[i] > 0)) continue;
    const tc = ts[i - 1] + (ts[i] - ts[i - 1]) * m[i - 1] / (m[i - 1] - m[i]);
    if (m[i] > 0) { a = tc; openA = false; } else { out.push({a, b: tc, openA, openB: false}); a = null; }
  }
  if (a !== null) out.push({a, b: ts[ts.length - 1], openA, openB: true});
  return out;
}
function analyse(lat, lon) {
  const g = siteGeom(lat, lon), T = S.track, t1 = spanEnd(), ts = [], ms = [], me = [], mb = [];
  let maxSun = -90, maxSunT = T.t0_ms;
  for (let t = T.t0_ms; t <= t1 + 1; t += 10 * MIN) {
    const s = look(g, 'sun', t), e = look(g, 'earth', t);
    ts.push(t); ms.push(s.margin); me.push(e.margin); mb.push(Math.min(s.margin, e.margin));
    if (s.el > maxSun) { maxSun = s.el; maxSunT = t; }
  }
  const total = w => w.reduce((acc, x) => acc + (x.b - x.a), 0);
  const win = {sun: intervals(ts, ms), earth: intervals(ts, me), both: intervals(ts, mb)};
  const span = t1 - T.t0_ms;
  return {g, win, span, maxSun, maxSunT,
          dur: {sun: total(win.sun), earth: total(win.earth), both: total(win.both)}};
}
function nextChange(w, t) {
  for (const x of w) {
    if (t >= x.a && t < x.b) return {vis: true, at: x.b, open: x.openB};
    if (x.a > t) return {vis: false, at: x.a, open: false};
  }
  return {vis: false, at: null};
}
const spanEnd = () => S.track.t0_ms + (S.track.sun.length / 3 - 1) * S.track.dt_ms;

function phaseInfo(t) {
  const s = bodyAt('sun', t), e = bodyAt('earth', t), sh = norm(s), eh = norm(e);
  const frac = (1 + dot(sh, eh)) / 2, [sLat, sLon] = latlonOf(s), [eLat, eLon] = latlonOf(e);
  const waxing = wrap180(sLon - eLon) > 0;
  const name = frac < 0.03 ? 'New Moon' : frac > 0.97 ? 'Full Moon' : Math.abs(frac - 0.5) < 0.04
    ? (waxing ? 'First Quarter' : 'Last Quarter') : (waxing ? 'Waxing ' : 'Waning ') + (frac < 0.5 ? 'Crescent' : 'Gibbous');
  return {frac, name, sh, eh, sKm: Math.hypot(...s), eKm: Math.hypot(...e), sLat, sLon, eLat, eLon};
}

/* ------------------------------------------------------------------ 2D: view from the site */
// What an observer standing at the selected site sees: the terrain skyline (LOLA), the Sun and Earth
// at their true azimuth / elevation (JPL Horizons), Earth's phase, and how much sunlight reaches the ground.
const G2 = {cv: $('globe2d'), drag: null};
const SKY = {az0: null, el0: 0, fov: 140};      // el0: elevation shown at the horizon anchor line
const SOLAR_CONST = 1361, AU_KM = 149597870.7;
const azDiff = (a, b) => ((a - b) % 360 + 540) % 360 - 180;
const compass = az => ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'][Math.round(((az % 360) + 360) % 360 / 22.5) % 16];
function discVisible(margin, rad) {                     // fraction of a disc of radius rad above the skyline
  if (margin >= rad) return 1; if (margin <= -rad) return 0;
  const x = margin / rad; return 1 - (Math.acos(x) - x * Math.sqrt(1 - x * x)) / Math.PI;
}
function sunlight(g, t) {
  const s = look(g, 'sun', t), f = discVisible(s.margin, s.rad), S0 = SOLAR_CONST * (AU_KM / s.r) ** 2;
  return {s, f, S0, flat: S0 * f * Math.max(0, Math.sin(s.el * D2R)), panel: S0 * f};
}
function earthPhase(g, t) {                             // Earth as seen from the site: lit fraction, cos(phase angle)
  const E = bodyAt('earth', t), Sn = bodyAt('sun', t);
  const toSun = norm([Sn[0] - E[0], Sn[1] - E[1], Sn[2] - E[2]]), toObs = norm([g.p[0] - E[0], g.p[1] - E[1], g.p[2] - E[2]]);
  const c = dot(toSun, toObs);
  return {k: (1 + c) / 2, c};
}
const enu = (az, el) => [Math.sin(az * D2R) * Math.cos(el * D2R), Math.cos(az * D2R) * Math.cos(el * D2R), Math.sin(el * D2R)];
function skyPaths() {
  const C = S.calc;
  if (!C.sky) {
    C.sky = {sun: [], earth: []};
    for (let t = S.track.t0_ms; t <= spanEnd(); t += HOUR)
      for (const b of ['sun', 'earth']) { const v = look(C.g, b, t); C.sky[b].push([v.az, v.el]); }
  }
  return C.sky;
}
function skyLayout() {
  const cv = G2.cv, dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(10, Math.round(cv.clientWidth * dpr)), h = Math.max(10, Math.round(cv.clientHeight * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  // frame the view in the gap between open side panels, so the Sun and Earth are not hidden behind them
  const vw = cv.clientWidth; let l = 0, r = vw;
  for (const win of document.querySelectorAll('.win:not([hidden]):not(.min)')) {
    const b = win.getBoundingClientRect();
    if (b.top > cv.clientHeight * 0.75 || b.width > vw * 0.45) continue;
    if (b.left + b.width / 2 < vw / 2) l = Math.max(l, b.right); else r = Math.min(r, b.left);
  }
  if (r - l < vw * 0.35) { l = 0; r = vw; }
  const free = (r - l) * dpr;
  return {w, h, dpr, ppd: free / SKY.fov, cx: (l + r) / 2 * dpr, cy: h * 0.74, free};
}
function aimSky() {                                    // frame the horizon plus whichever of Sun / Earth is up
  const g = S.calc.g, s = look(g, 'sun', S.t), e = look(g, 'earth', S.t), cv = G2.cv;
  const up = [s, e].filter(v => v.el > -3), sep = up.length === 2 ? Math.abs(azDiff(s.az, e.az)) : 0;
  const both = up.length === 2 && sep + 40 <= 320;
  SKY.az0 = both ? e.az + azDiff(s.az, e.az) / 2 : (e.el > -3 ? e.az : s.az);
  SKY.el0 = 0;
  const maxEl = Math.max(10, ...(both ? up : [up[0] || e]).map(v => v.el)) + 12;
  const aspect = (cv.clientHeight || 800) / (skyLayout().free / Math.min(window.devicePixelRatio || 1, 2) || 1400);
  SKY.fov = clamp(Math.max(both ? sep + 40 : 60, maxEl / (0.68 * aspect)), 60, 340);
}
function drawSky() {
  if (!S.track || !S.calc || S.view !== '2d') return;
  const L = skyLayout(), ctx = G2.cv.getContext('2d'), g = S.calc.g, sc = L.dpr;
  if (SKY.az0 === null) aimSky();
  const sun = sunlight(g, S.t), s = sun.s, e = look(g, 'earth', S.t), ep = earthPhase(g, S.t);
  const X = az => L.cx + azDiff(az, SKY.az0) * L.ppd, Y = el => L.cy - (el - SKY.el0) * L.ppd;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, L.w, L.h);

  // elevation grid
  ctx.font = `${11 * sc}px system-ui, sans-serif`; ctx.textAlign = 'left'; ctx.lineWidth = 1 * sc;
  for (let el = -20; el <= 90; el += 10) {
    const y = Y(el); if (y < 0 || y > L.h) continue;
    ctx.strokeStyle = el === 0 ? 'rgba(255,255,255,.3)' : 'rgba(255,255,255,.08)'; ctx.setLineDash(el === 0 ? [] : [4 * sc, 6 * sc]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(L.w, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.4)'; ctx.fillText(`${el}°`, 6 * sc, y - 3 * sc);
  }
  ctx.setLineDash([]);

  // Sun and Earth paths over the simulation span
  const P = skyPaths();
  for (const [b, col] of [['sun', 'rgba(255,204,51,.45)'], ['earth', 'rgba(77,166,255,.45)']]) {
    ctx.fillStyle = col;
    for (const [az, el] of P[b]) { const x = X(az), y = Y(el); if (x > -5 && x < L.w + 5 && y > -5 && y < L.h + 5) ctx.fillRect(x - sc, y - sc, 2 * sc, 2 * sc); }
  }

  // Sun: glow + disc (terrain drawn afterwards hides whatever is below the skyline)
  const sx = X(s.az), sy = Y(s.el), sr = Math.max(s.rad * L.ppd, 6 * sc);
  const glow = ctx.createRadialGradient(sx, sy, sr * 0.5, sx, sy, sr * 7);
  glow.addColorStop(0, 'rgba(255,230,150,.55)'); glow.addColorStop(1, 'rgba(255,204,51,0)');
  ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(sx, sy, sr * 7, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = '#fff6d8'; ctx.beginPath(); ctx.arc(sx, sy, sr, 0, 2 * Math.PI); ctx.fill();

  // Earth with its phase; the lit limb points toward the Sun
  const ex = X(e.az), ey = Y(e.el), er = Math.max(e.rad * L.ppd, 9 * sc), sv = enu(s.az, s.el);
  const right = [Math.cos(e.az * D2R), -Math.sin(e.az * D2R), 0];
  const up = [-Math.sin(e.az * D2R) * Math.sin(e.el * D2R), -Math.cos(e.az * D2R) * Math.sin(e.el * D2R), Math.cos(e.el * D2R)];
  ctx.save(); ctx.translate(ex, ey); ctx.rotate(Math.atan2(-dot(sv, up), dot(sv, right)));
  ctx.fillStyle = '#0d1a2b'; ctx.beginPath(); ctx.arc(0, 0, er, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = '#4da6ff'; ctx.beginPath(); ctx.arc(0, 0, er, -Math.PI / 2, Math.PI / 2, false);
  ctx.ellipse(0, 0, er * Math.abs(ep.c), er, 0, Math.PI / 2, 3 * Math.PI / 2, ep.c < 0); ctx.fill();
  ctx.strokeStyle = 'rgba(77,166,255,.6)'; ctx.lineWidth = 1 * sc; ctx.beginPath(); ctx.arc(0, 0, er, 0, 2 * Math.PI); ctx.stroke();
  ctx.restore();

  // terrain skyline: faces turned away from the Sun (i.e. lit for an observer looking at them) are brighter
  const sinEl = Math.max(0, Math.sin(s.el * D2R)), shine = e.margin > 0 ? 0.03 * ep.k : 0;
  const step = Math.max(1, Math.round(2 * sc));
  let yMin = L.h;
  for (let x = 0; x < L.w; x += step) {
    const az = ((SKY.az0 + (x - L.cx) / L.ppd) % 360 + 360) % 360, y = Y(hzAt(g.hz, az));
    const face = Math.max(0, -Math.cos((az - s.az) * D2R));
    const lum = clamp(0.03 + shine + sun.f * (0.08 + 0.45 * face * Math.min(1, 0.25 + 3 * sinEl) + 0.3 * sinEl), 0, 0.85);
    const v = Math.round(255 * lum);
    ctx.fillStyle = `rgb(${v},${v},${Math.round(v * 0.96)})`; ctx.fillRect(x, y, step, L.h - y);
    yMin = Math.min(yMin, y);
  }
  const fade = ctx.createLinearGradient(0, yMin, 0, L.h);
  fade.addColorStop(0, 'rgba(0,0,0,0)'); fade.addColorStop(1, 'rgba(0,0,0,.55)');
  ctx.fillStyle = fade; ctx.fillRect(0, yMin, L.w, L.h - yMin);

  // compass along the bottom (above the time bar)
  const cy = L.h - 112 * sc;
  ctx.textAlign = 'center';
  for (let a = 0; a < 360; a += 15) {
    const x = X(a); if (x < 0 || x > L.w) continue;
    const major = a % 45 === 0;
    ctx.fillStyle = major ? '#eee' : 'rgba(255,255,255,.5)';
    ctx.font = `${major ? 600 : 400} ${(major ? 13 : 10) * sc}px system-ui, sans-serif`;
    ctx.fillText(major ? compass(a) : `${a}°`, x, cy);
  }

  // labels and off-screen pointers
  ctx.font = `600 ${13 * sc}px system-ui, sans-serif`;
  const tag = (x, y, r, col, text, sub) => {
    const inside = x > 0 && x < L.w && y > 0 && y < L.h;
    const lo = L.cx - L.free / 2, hi = L.cx + L.free / 2, inside2 = x > lo && x < hi;
    const px = clamp(x, lo + 70 * sc, hi - 70 * sc), py = clamp(y, 150 * sc, L.h - 140 * sc);
    ctx.fillStyle = col; ctx.textAlign = 'center';
    if (!inside || !inside2) {
      const a = Math.atan2(y - py, x - px);
      ctx.beginPath(); ctx.moveTo(px + Math.cos(a) * 14 * sc, py + Math.sin(a) * 14 * sc);
      ctx.lineTo(px + Math.cos(a + 2.5) * 9 * sc, py + Math.sin(a + 2.5) * 9 * sc);
      ctx.lineTo(px + Math.cos(a - 2.5) * 9 * sc, py + Math.sin(a - 2.5) * 9 * sc); ctx.fill();
    }
    const ly = (inside && inside2 ? y - r - 22 * sc : py + 28 * sc);
    ctx.font = `600 ${13 * sc}px system-ui, sans-serif`; ctx.fillText(text, px, ly);
    ctx.font = `${11 * sc}px system-ui, sans-serif`; ctx.fillStyle = '#ccc'; ctx.fillText(sub, px, ly + 14 * sc);
  };
  const st = v => v.margin > v.rad ? 'visible' : v.margin > -v.rad ? 'on the skyline' : v.el > 0 ? 'behind terrain' : 'below horizon';
  tag(sx, sy, sr, '#ffcc33', 'Sun', `${s.el.toFixed(1)}° · ${compass(s.az)} ${s.az.toFixed(0)}° · ${st(s)}`);
  tag(ex, ey, er, '#4da6ff', `Earth · ${(100 * ep.k).toFixed(0)}% lit`, `${e.el.toFixed(1)}° · ${compass(e.az)} ${e.az.toFixed(0)}° · ${st(e)}`);
  renderHud(sun, e, ep);
}
function renderHud(sun, e, ep) {
  $('sky-hud').style.top = `${$('topbar').getBoundingClientRect().bottom + 8}px`;
  const s = sun.s, pct = clamp(sun.flat / SOLAR_CONST, 0, 1);
  const word = sun.f === 0 ? (s.el > 0 ? 'In shadow — the Sun is behind terrain' : 'Night — the Sun is below the horizon')
    : sun.f < 1 ? `Partial sunlight — ${(100 * sun.f).toFixed(0)}% of the Sun's disc above the skyline`
    : s.el < 1 ? 'Grazing sunlight — the Sun is right on the horizon' : s.el < 10 ? 'Low sunlight — long shadows' : 'Full sunlight';
  $('hud-title').textContent = `View from ${selLabel()} · ${fmtLat(S.sel.lat)}, ${fmtLon(S.sel.lon)} · facing ${compass(SKY.az0)} (${((SKY.az0 % 360) + 360) % 360 | 0}°)`;
  $('hud-meter').style.width = `${(100 * pct).toFixed(1)}%`;
  $('hud-irr').textContent = `${Math.round(sun.flat).toLocaleString()} W/m²`;
  $('hud-word').textContent = word;
  $('hud-detail').innerHTML = `Sunlight on level ground (${(100 * pct).toFixed(0)}% of overhead Sun) · a Sun-facing panel gets `
    + `<b>${Math.round(sun.panel).toLocaleString()} W/m²</b>. Earth is ${e.margin > 0 ? '<b style="color:#7ee08a">in view</b>' : '<b style="color:#ff7b7b">not in view</b>'}`
    + `, ${(100 * ep.k).toFixed(0)}% of its disc lit${e.margin <= 0 && sun.f === 0 ? ' — no sunlight or earthshine reaches the site' : ''}.`;
}
G2.cv.addEventListener('pointerdown', ev => {
  G2.drag = {x: ev.clientX, y: ev.clientY, az0: SKY.az0, el0: SKY.el0}; G2.cv.setPointerCapture(ev.pointerId); G2.cv.classList.add('dragging');
});
G2.cv.addEventListener('pointermove', ev => {
  if (!G2.drag) return;
  const L = skyLayout(), ppd = L.free / L.dpr / SKY.fov;
  SKY.az0 = G2.drag.az0 - (ev.clientX - G2.drag.x) / ppd;
  SKY.el0 = clamp(G2.drag.el0 + (ev.clientY - G2.drag.y) / ppd, -20, 80);
  requestDraw();
});
G2.cv.addEventListener('pointerup', () => { G2.drag = null; G2.cv.classList.remove('dragging'); });
G2.cv.addEventListener('wheel', ev => {
  ev.preventDefault(); SKY.fov = clamp(SKY.fov * Math.exp(ev.deltaY * 0.0012), 10, 360); requestDraw();
}, {passive: false});
G2.cv.addEventListener('dblclick', () => { aimSky(); requestDraw(); });
document.querySelectorAll('[data-look]').forEach(b => b.onclick = () => {
  if (!S.calc) return;
  const k = b.dataset.look;
  if (k === 'all') { SKY.fov = 360; SKY.el0 = 0; }
  else { const v = look(S.calc.g, k, S.t); SKY.az0 = v.az; SKY.fov = 50; SKY.el0 = Math.max(0, v.el - 10); }
  requestDraw();
});

/* ------------------------------------------------------------------ 3D globe (Plotly) */
const G3 = {div: $('globe3d'), ready: false, busy: false, pending: false};
function meshColor(sh) {
  const M = S.mesh, rows = new Array(M.nr);
  for (let r = 0, i = 0; r < M.nr; r++) {
    const row = new Array(M.nc);
    for (let c = 0; c < M.nc; c++, i++) {
      const cosi = Math.max(0, (M.nx[i] * sh[0] + M.ny[i] * sh[1] + M.nz[i] * sh[2]) / 127);
      row[c] = Math.min(1, M.refl[i] / 255 * (0.03 + 0.97 * cosi) * 1.35);
    }
    rows[r] = row;
  }
  return rows;
}
const SUN_R = 3.4, EARTH_R = 2.3, SUN_C = '#ffcc33', EARTH_C = '#4da6ff', SITE_C = '#ff4d6d';
function bodyTraceXYZ(u, rd) { const R = S.stat.r_moon_km; return {x: [u[0] * R * 1.025, u[0] * R * rd], y: [u[1] * R * 1.025, u[1] * R * rd], z: [u[2] * R * 1.025, u[2] * R * rd]}; }
// Site markers are small 3D spheres (mesh3d), not flat screen-facing markers, so terrain never cuts
// through them as the globe turns. Each sphere rests on the highest nearby mesh vertex.
const SITE_BALL = {sel: 32, saved: 20};                   // sphere radius [km]
function meshTopKm(lat, lon) {                            // highest drawn terrain near (lat, lon) [km above R]
  const M = S.mesh, near = (a, v) => { let b = 0; for (let k = 1; k < a.length; k++) if (Math.abs(a[k] - v) < Math.abs(a[b] - v)) b = k; return b; };
  const r0 = near(M.lat, lat), c0 = near(M.lon, ((lon % 360) + 360) % 360);   // mesh lon runs 0..360
  let top = -Infinity;
  for (let r = Math.max(0, r0 - 2); r <= Math.min(M.nr - 1, r0 + 2); r++)
    for (let dc = -2; dc <= 2; dc++) top = Math.max(top, M.h[r * M.nc + (c0 + dc + M.nc) % M.nc]);
  return M.exag * Math.max(top, demHeight(lat, lon)) / 1000;
}
const BALL = (() => {                                     // unit UV sphere, 10 rings x 16 segments
  const NR = 10, NS = 16, v = [], f = [];
  for (let i = 0; i <= NR; i++) for (let j = 0; j < NS; j++) {
    const th = Math.PI * i / NR, ph = 2 * Math.PI * j / NS;
    v.push([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]);
  }
  for (let i = 0; i < NR; i++) for (let j = 0; j < NS; j++) {
    const a = i * NS + j, b = i * NS + (j + 1) % NS, c = a + NS, d = b + NS;
    f.push([a, c, b], [b, c, d]);
  }
  return {v, f};
})();
function siteBalls(list, rad) {                           // one mesh3d of spheres + label anchors
  const R = S.stat.r_moon_km, o = {x: [], y: [], z: [], i: [], j: [], k: [], tops: []};
  list.forEach((st, n) => {
    const g = dirOf(st.lat, st.lon), rc = R + meshTopKm(st.lat, st.lon) + rad * 1.3, base = n * BALL.v.length;
    for (const p of BALL.v) { o.x.push(g[0] * rc + p[0] * rad); o.y.push(g[1] * rc + p[1] * rad); o.z.push(g[2] * rc + p[2] * rad); }
    for (const t of BALL.f) { o.i.push(base + t[0]); o.j.push(base + t[1]); o.k.push(base + t[2]); }
    o.tops.push({dir: g, xyz: g.map(c => c * (rc + rad)), label: st.label, sel: rad === SITE_BALL.sel});
  });
  return o;
}
// Labels are scene annotations: drawn over the 3D canvas so the Moon never clips the words. Plotly
// can't hide them behind the Moon by itself, so a label is shown only while its site faces the camera.
function siteLabelVisible(dir) {
  const e = G3.eye || {x: 1, y: 0, z: 0}, n = Math.hypot(e.x, e.y, e.z) || 1;
  return (dir[0] * e.x + dir[1] * e.y + dir[2] * e.z) / n > 0.12;
}
function siteAnnotations() {
  return (G3.labels || []).map(t => ({
    x: t.xyz[0], y: t.xyz[1], z: t.xyz[2], text: t.label, showarrow: false, yshift: t.sel ? 16 : 12,
    font: {color: t.sel ? '#fff' : '#ddd', size: t.sel ? 12 : 11}, bgcolor: 'rgba(0,0,0,0.55)', borderpad: 2,
    visible: siteLabelVisible(t.dir)}));
}
function onCamera(ev) {                                   // keep label visibility in step with the view
  const cam = ev && (ev['scene.camera'] || (ev.scene && ev.scene.camera));
  const eye = cam ? cam.eye : ev && ev['scene.camera.eye'];
  if (!eye || !G3.ready) return;
  G3.eye = eye;
  const ann = (G3.div.layout.scene.annotations || []), upd = {};
  (G3.labels || []).forEach((t, n) => { const v = siteLabelVisible(t.dir); if (ann[n] && ann[n].visible !== v) upd[`scene.annotations[${n}].visible`] = v; });
  if (Object.keys(upd).length) Plotly.relayout(G3.div, upd);
}
function build3D() {
  const M = S.mesh, R = S.stat.r_moon_km, X = [], Y = [], Z = [];
  for (let r = 0, i = 0; r < M.nr; r++) {
    const la = M.lat[r] * D2R, xr = [], yr = [], zr = [];
    for (let c = 0; c < M.nc; c++, i++) {
      const lo = M.lon[c] * D2R, rr = R + M.exag * M.h[i] / 1000;
      xr.push(+(rr * Math.cos(la) * Math.cos(lo)).toFixed(2)); yr.push(+(rr * Math.cos(la) * Math.sin(lo)).toFixed(2)); zr.push(+(rr * Math.sin(la)).toFixed(2));
    }
    X.push(xr); Y.push(yr); Z.push(zr);
  }
  const P = phaseInfo(S.t), path = (body, rd) => {
    const o = {x: [], y: [], z: []};
    for (let t = S.track.t0_ms; t <= spanEnd(); t += 3 * HOUR) { const u = norm(bodyAt(body, t)); o.x.push(u[0] * R * rd); o.y.push(u[1] * R * rd); o.z.push(u[2] * R * rd); }
    return o;
  };
  const sp = path('sun', SUN_R), ep = path('earth', EARTH_R);
  const data = [
    {type: 'surface', x: X, y: Y, z: Z, surfacecolor: meshColor(P.sh), colorscale: [[0, '#000000'], [1, '#ffffff']], cmin: 0, cmax: 1,
     // 'none' (not 'skip'): no tooltip, but Plotly still fires click events, so the globe can select a site
     showscale: false, hoverinfo: 'none', lighting: {ambient: 1, diffuse: 0, specular: 0, fresnel: 0}, name: 'Moon'},
    {type: 'scatter3d', mode: 'lines', ...sp, line: {color: SUN_C, width: 2, dash: 'dot'}, opacity: 0.5, hoverinfo: 'skip', name: 'Sun path (Moon-fixed)'},
    {type: 'scatter3d', mode: 'lines', ...ep, line: {color: EARTH_C, width: 2, dash: 'dot'}, opacity: 0.5, hoverinfo: 'skip', name: 'Earth path (Moon-fixed)'},
    {type: 'scatter3d', mode: 'lines+markers+text', ...bodyTraceXYZ(P.sh, SUN_R), text: ['', 'Sun'], textposition: 'top center',
     line: {color: SUN_C, width: 5}, marker: {size: [0, 22], color: SUN_C}, textfont: {color: SUN_C, size: 13}, hoverinfo: 'skip', name: 'Sun'},
    {type: 'scatter3d', mode: 'lines+markers+text', ...bodyTraceXYZ(P.eh, EARTH_R), text: ['', 'Earth'], textposition: 'top center',
     line: {color: EARTH_C, width: 5}, marker: {size: [0, 14], color: EARTH_C}, textfont: {color: EARTH_C, size: 13}, hoverinfo: 'skip', name: 'Earth'},
    {type: 'mesh3d', x: [], y: [], z: [], i: [], j: [], k: [], color: '#f2f2f2', flatshading: false, showlegend: true, hoverinfo: 'skip',
     lighting: {ambient: 0.55, diffuse: 0.7, specular: 0.35, roughness: 0.4}, lightposition: {x: 1e5, y: 1e5, z: 1e5}, name: 'Saved sites'},
    {type: 'mesh3d', x: [], y: [], z: [], i: [], j: [], k: [], color: SITE_C, flatshading: false, showlegend: true, hoverinfo: 'skip',
     lighting: {ambient: 0.55, diffuse: 0.7, specular: 0.5, roughness: 0.3}, lightposition: {x: 1e5, y: 1e5, z: 1e5}, name: 'Selected site'},
  ];
  const u = P.eh;                                         // look from Earth, as in the original viewer
  if (!G3.eye) G3.eye = {x: 1.7 * u[0], y: 1.7 * u[1], z: 1.7 * u[2]};
  Plotly.newPlot(G3.div, data, {
    template: 'plotly_dark', paper_bgcolor: 'black', font: {color: '#ddd'}, margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: true,
    legend: {x: 0.5, xanchor: 'center', y: 0.9, orientation: 'h', bgcolor: 'rgba(0,0,0,0)', font: {size: 11, color: '#ddd'}},
    scene: {xaxis: {visible: false}, yaxis: {visible: false}, zaxis: {visible: false}, aspectmode: 'data', bgcolor: 'black',
            dragmode: 'orbit', camera: {eye: G3.eye}, annotations: []},
    uirevision: 'keep',
  }, {scrollZoom: true, displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toImage']});
  G3.div.on('plotly_click', ev => {
    const p = ev.points && ev.points[0];
    if (!p || p.data.type !== 'surface') return;
    const ll = latlonOf([p.x, p.y, p.z]); placePoint(ll[0], ll[1]);
  });
  G3.div.on('plotly_relayouting', onCamera);
  G3.div.on('plotly_relayout', onCamera);
  G3.ready = true;
  update3DSites();
}
function update3DSites() {
  if (!G3.ready) return;
  const others = S.sites.filter(s => !(S.sel && sameSite(S.sel, s.lat, s.lon)));   // the selection gets its own ball
  const saved = siteBalls(others.map(s => ({...s, label: s.name})), SITE_BALL.saved);
  const sel = siteBalls(S.sel ? [{...S.sel, label: selLabel()}] : [], SITE_BALL.sel);
  G3.labels = [...saved.tops, ...sel.tops];
  Plotly.update(G3.div, {x: [saved.x, sel.x], y: [saved.y, sel.y], z: [saved.z, sel.z], i: [saved.i, sel.i], j: [saved.j, sel.j], k: [saved.k, sel.k]},
                {'scene.annotations': siteAnnotations()}, [5, 6]);
}
async function draw3D() {
  if (!G3.ready || S.view !== '3d') return;
  if (G3.busy) { G3.pending = true; return; }
  G3.busy = true;
  const P = phaseInfo(S.t), s = bodyTraceXYZ(P.sh, SUN_R), e = bodyTraceXYZ(P.eh, EARTH_R);
  await Plotly.restyle(G3.div, {x: [s.x, e.x], y: [s.y, e.y], z: [s.z, e.z]}, [3, 4]);
  await Plotly.restyle(G3.div, {surfacecolor: [meshColor(P.sh)]}, [0]);
  G3.busy = false;
  if (G3.pending) { G3.pending = false; draw3D(); }
}

/* ------------------------------------------------------------------ mini maps */
const MAPS = {
  eq: {cv: $('map-eq'), w: 480, h: 240,
       inv: (x, y, m) => [90 - (y + 0.5) / m.h * 180, (x + 0.5) / m.w * 360 - 180],
       fwd: (lat, lon, m) => ({x: (wrap180(lon) + 180) / 360 * m.w, y: (90 - lat) / 180 * m.h, ok: true})},
  np: {cv: $('map-np'), w: 220, h: 220, pole: 1}, sp: {cv: $('map-sp'), w: 220, h: 220, pole: -1},
};
// polar maps: orthographic from the pole out to 60°; north has 0° at the bottom, south 0° at the top, 90°E on the right
for (const m of [MAPS.np, MAPS.sp]) {
  const s = m.pole, k = 0.5 / Math.cos(60 * D2R);
  m.inv = (x, y, mm) => {
    const px = ((x + 0.5) / mm.w - 0.5) / k * 2 * 0.5, py = ((y + 0.5) / mm.h - 0.5) / k * 2 * 0.5, r = Math.hypot(px, py);
    if (r > Math.cos(60 * D2R) || r > 1) return null;
    return [s * Math.acos(Math.min(1, r)) * R2D, Math.atan2(px, s * py) * R2D];
  };
  m.fwd = (lat, lon, mm) => {
    const r = Math.cos(lat * D2R) * k;
    return {x: (0.5 + r * Math.sin(lon * D2R)) * mm.w, y: (0.5 + s * r * Math.cos(lon * D2R)) * mm.h, ok: s * lat >= 60};
  };
}
function buildMap(m) {
  m.cv.width = m.w; m.cv.height = m.h;
  const {tw, th} = S.tex, pix = [], tex = [], nv = [];
  for (let y = 0; y < m.h; y++) for (let x = 0; x < m.w; x++) {
    const ll = m.inv(x, y, m); if (!ll) continue;
    const tx = Math.min(tw - 1, Math.floor((ll[1] + 180) / 360 * tw)), ty = Math.min(th - 1, Math.floor((90 - ll[0]) / 180 * th));
    pix.push(y * m.w + x); tex.push(ty * tw + tx); nv.push(...dirOf(ll[0], ll[1]));
  }
  m.pix = Int32Array.from(pix); m.tex = Int32Array.from(tex); m.nv = Float32Array.from(nv);
  m.cv.addEventListener('click', ev => {
    const r = m.cv.getBoundingClientRect(), ll = m.inv((ev.clientX - r.left) / r.width * m.w, (ev.clientY - r.top) / r.height * m.h, m);
    if (ll) placePoint(ll[0], ll[1]);
  });
}
function drawMaps() {
  if (!S.track) return;
  const P = phaseInfo(S.t), sh = P.sh;
  for (const m of Object.values(MAPS)) {
    const ctx = m.cv.getContext('2d'), img = ctx.createImageData(m.w, m.h), d = img.data, g = S.tex.g;
    for (let k = 0; k < m.pix.length; k++) {
      const c = m.nv[3 * k] * sh[0] + m.nv[3 * k + 1] * sh[1] + m.nv[3 * k + 2] * sh[2];
      const v = g[m.tex[k]] * (c > 0 ? 1 : 0.38), o = m.pix[k] * 4;
      d[o] = d[o + 1] = d[o + 2] = v; d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const mark = (lat, lon, draw) => { const p = m.fwd(lat, lon, m); if (p.ok) draw(p.x, p.y); };
    ctx.font = '11px system-ui, sans-serif';
    for (const s of S.sites) mark(s.lat, s.lon, (x, y) => {
      ctx.fillStyle = '#fff'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    });
    mark(P.sLat, P.sLon, (x, y) => { ctx.fillStyle = '#ffcc33'; ctx.strokeStyle = '#000'; ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); });
    mark(P.eLat, P.eLon, (x, y) => { ctx.fillStyle = '#4da6ff'; ctx.strokeStyle = '#000'; ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); });
    if (S.sel) mark(S.sel.lat, S.sel.lon, (x, y) => {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = '#ff4d6d'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = '#ff4d6d'; ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2 * Math.PI); ctx.fill();
    });
  }
}

/* ------------------------------------------------------------------ site handling */
const selLabel = () => { const s = S.sel && S.sites.find(x => x.id === S.sel.id); return s ? s.name : (S.sel && S.sel.name) || 'Selected point'; };
function nextSiteId() {
  for (let i = 0; i < 26 * 27; i++) {
    const id = i < 26 ? String.fromCharCode(65 + i) : String.fromCharCode(64 + Math.floor(i / 26)) + String.fromCharCode(65 + i % 26);
    if (!S.sites.some(s => s.id === id)) return id;
  }
  return String(Date.now());
}
function addSite(lat, lon, name = '', note = '') {
  let n = 1; while (S.sites.some(s => s.name === `Point ${n}`)) n++;
  const id = nextSiteId(), s = {id, name: name || `Point ${n}`, note, lat, lon: wrap180(lon)};
  S.sites.push(s); saveSites(); return s;
}
const sameSite = (s, lat, lon) => Math.abs(s.lat - lat) < 1e-4 && Math.abs(wrap180(s.lon - lon)) < 1e-4;
function placePoint(lat, lon) {
  if (S.drop) {
    const s = addSite(lat, lon); setDrop(false); select(s.lat, s.lon, s.id);
    toast(`${s.name} saved at ${fmtSite(s)} — rename it from the ≡ menu`);
  } else select(lat, lon, null);
}
function select(lat, lon, id, name = null) {
  const moved = !S.sel || !sameSite(S.sel, clamp(lat, -89.99, 89.99), lon);
  S.sel = {lat: clamp(lat, -89.99, 89.99), lon: wrap180(lon), id, name};
  if (moved) SKY.az0 = null;                              // re-aim the site view at the Sun / Earth
  $('in-lat').value = S.sel.lat.toFixed(3); $('in-lon').value = S.sel.lon.toFixed(3);
  $('loading').hidden = false; $('loading').textContent = 'Tracing the terrain horizon…';
  setTimeout(() => {
    S.calc = analyse(S.sel.lat, S.sel.lon);
    $('loading').hidden = true;
    renderSites(); update3DSites(); renderStatic(); renderTime(); renderCompare();
    if (!S.laidOut) { S.laidOut = true; initWindows(); }   // first fill: place panels by their real size
  }, 15);
}
function setDrop(on) {
  S.drop = on;
  $('drop-btn').classList.toggle('armed', on);
  $('drop-btn').textContent = on ? 'Click the map or the globe to place the point… (Esc to cancel)' : '+ Drop new point';
  for (const el of [G2.cv, MAPS.eq.cv, MAPS.np.cv, MAPS.sp.cv]) el.classList.toggle('armed', on);
}
function renderSites() {
  $('site-list').innerHTML = S.sites.map(s =>
    `<button class="site-btn${S.sel && S.sel.id === s.id ? ' active' : ''}" data-id="${s.id}" title="${escapeHTML(s.name)}: ${fmtSite(s)}">
       ${escapeHTML(s.name)}<small>${fmtSite(s)}</small></button>`).join('') || '<p class="note">No saved sites. Tick sites in the Library or drop a point.</p>';
  for (const b of $('site-list').querySelectorAll('button'))
    b.onclick = () => { const s = S.sites.find(x => x.id === b.dataset.id); select(s.lat, s.lon, s.id); };
  drawMaps(); requestDraw(); renderLibraryChecks();
}
const escapeHTML = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

function renderLibrary() {
  const q = ($('lib-search').value || '').trim().toLowerCase();
  let html = '', shown = 0;
  LIBRARY.forEach(([group, items], gi) => {
    const rows = items.map((it, i) => ({it, i})).filter(({it}) => !q || it[0].toLowerCase().includes(q) || group.toLowerCase().includes(q));
    if (!rows.length) return;
    shown += rows.length;
    html += `<div class="lib-group"><span>${group} <small>(${rows.length})</small></span><button class="btn small" data-addgroup="${gi}">Compare all</button></div>`
      + rows.map(({it: [n, la, lo, approx], i}) => `<div class="lib-row" data-g="${gi}" data-i="${i}">
          <label title="Add to the saved sites (compared in Data → Compare data)"><input type="checkbox" data-check><span>${escapeHTML(n)}</span></label>
          <small>${approx ? '≈ ' : ''}${la.toFixed(2)}°, ${lo.toFixed(2)}°</small>
          <button class="btn small" data-view>View</button></div>`).join('');
  });
  $('lib-list').innerHTML = html || '<p class="note">No sites match.</p>';
  $('lib-count').textContent = `${shown} sites`;
  const entry = el => LIBRARY[+el.dataset.g][1][+el.dataset.i];
  for (const row of $('lib-list').querySelectorAll('.lib-row')) {
    const [n, la, lo] = entry(row), grp = LIBRARY[+row.dataset.g][0];
    row.querySelector('[data-view]').onclick = () => {
      const s = S.sites.find(x => sameSite(x, la, lo)); select(la, lo, s ? s.id : null, n);
    };
    row.querySelector('[data-check]').onchange = ev => {
      setInCompare(n, grp, la, lo, ev.target.checked);
      toast(ev.target.checked ? `${n} added to saved sites (Data → Compare data)` : `${n} removed`);
    };
  }
  for (const b of $('lib-list').querySelectorAll('[data-addgroup]')) b.onclick = () => {
    const [grp, items] = LIBRARY[+b.dataset.addgroup];
    items.forEach(([n, la, lo]) => setInCompare(n, grp, la, lo, true, false));
    afterSitesChanged(); toast(`${items.length} ${grp} sites added — see Data → Compare data`);
  };
  renderLibraryChecks();
}
function renderLibraryChecks() {
  for (const row of document.querySelectorAll('#lib-list .lib-row')) {
    const [, la, lo] = LIBRARY[+row.dataset.g][1][+row.dataset.i];
    row.querySelector('[data-check]').checked = S.sites.some(s => sameSite(s, la, lo));
    row.classList.toggle('active', !!(S.sel && sameSite(S.sel, la, lo)));
  }
}
function setInCompare(name, group, lat, lon, on, refresh = true) {
  const i = S.sites.findIndex(s => sameSite(s, lat, lon));
  if (on && i < 0) addSite(lat, lon, name, group);
  if (!on && i >= 0) { const [s] = S.sites.splice(i, 1); if (S.sel && S.sel.id === s.id) S.sel.id = null; }
  saveSites();
  if (refresh) afterSitesChanged();
}
function afterSitesChanged() {
  if (S.sel && !S.sel.id) { const s = S.sites.find(x => sameSite(x, S.sel.lat, S.sel.lon)); if (s) S.sel.id = s.id; }
  renderSites(); update3DSites(); renderCompare();
}

/* ------------------------------------------------------------------ panels */
function renderStatic() {                               // things that change with site or track, not with time
  const C = S.calc; if (!C) return;
  $('m-comm').textContent = `${fmtHM(C.dur.earth)} (${(100 * C.dur.earth / C.span).toFixed(0)}%)`;
  $('m-sun').textContent = `${fmtHM(C.dur.sun)} (${(100 * C.dur.sun / C.span).toFixed(0)}%)`;
  const w = C.win.earth;
  $('m-ewin-list').innerHTML = !w.length ? 'Earth is never above the terrain horizon in this span.' :
    '<ol>' + w.map(x => `<li>${x.openA ? '… ' : ''}${fmtShort(x.a)} – ${fmtShort(x.b)}${x.openB ? ' …' : ''} <span class="muted">(${fmtHM(x.b - x.a)})</span></li>`).join('') + '</ol>';
  $('a-site').textContent = `${S.sel.lat.toFixed(3)}°, ${S.sel.lon.toFixed(3)}°`;
  $('m-note').textContent = `${selLabel()} · ${fmtLat(S.sel.lat)}, ${fmtLon(S.sel.lon)} · ground ${Math.round(C.g.h_m).toLocaleString()} m (LOLA). `
    + `Totals cover ${fmtUTC(S.track.t0_ms)} + ${S.track.days} d; peak solar elevation ${C.maxSun.toFixed(1)}° on ${fmtShort(C.maxSunT)}.`;
  drawPanoramaPaths();
}
function stateText(s) {
  if (s.margin > s.rad) return ['visible', 'ok'];
  if (s.margin > -s.rad) return ['on the horizon line', 'warn'];
  return [s.el > 0 ? 'blocked by terrain' : 'below the horizon', 'bad'];
}
function renderTime() {
  if (!S.track) return;
  $('sim-time').textContent = `${fmtUTC(S.t)} UTC`;
  const P = phaseInfo(S.t);
  $('pos-info').innerHTML = `<b>${P.name}</b>, ${(100 * P.frac).toFixed(1)}% lit as seen from Earth · `
    + `<span style="color:#4da6ff">Earth</span> ${Math.round(P.eKm).toLocaleString()} km, sub-Earth ${P.eLat.toFixed(2)}°, ${wrap180(P.eLon).toFixed(2)}° · `
    + `<span style="color:#ffcc33">Sun</span> ${(P.sKm / 1e6).toFixed(2)} M km, sub-solar ${P.sLat.toFixed(2)}°, ${wrap180(P.sLon).toFixed(2)}°`;
  const R = $('sim-range'); R.value = Math.round((S.t - S.track.t0_ms) / (10 * MIN));
  drawMaps(); requestDraw();
  const C = S.calc; if (!C) return;
  const s = look(C.g, 'sun', S.t), e = look(C.g, 'earth', S.t), [st, sc] = stateText(s), [et, ec] = stateText(e);
  const chip = (id, txt, cls) => { const el = $(id); el.textContent = txt; el.className = 'chip' + (cls ? ' ' + cls : ''); };
  chip('m-solel', fmtDeg(s.el, 1), sc);
  chip('a-sel', fmtDeg(s.el), sc); chip('a-saz', fmtDeg(s.az, 1));
  chip('a-eel', fmtDeg(e.el), ec); chip('a-eaz', fmtDeg(e.az, 1));
  const until = (w, upLbl, downLbl, lblId, valId) => {
    const n = nextChange(w, S.t);
    $(lblId).textContent = n.vis ? downLbl : upLbl;
    chip(valId, n.at === null ? 'not in span' : (n.open ? '> ' : '') + fmtHM(n.at - S.t), n.vis ? 'ok' : 'bad');
  };
  until(C.win.sun, 'Time to sunrise', 'Time to sunset', 'a-sset-l', 'a-sset');
  until(C.win.earth, 'Time to Earthrise', 'Time to Earthset', 'a-eset-l', 'a-eset');
  const ew = C.win.earth, cur = ew.find(x => S.t >= x.a && S.t < x.b) || ew.find(x => x.a > S.t);
  chip('m-ewin', !ew.length ? 'none' : (ew.length === 1 && ew[0].openA && ew[0].openB) ? 'whole span'
       : cur ? `${fmtShort(cur.a)} – ${fmtShort(cur.b)}` : 'none left', cur && S.t >= cur.a ? 'ok' : '');
  $('a-state').innerHTML = `Sun is <b>${st}</b> (terrain horizon ${s.hz.toFixed(2)}° toward ${s.az.toFixed(0)}°). `
    + `Earth is <b>${et}</b> (terrain horizon ${e.hz.toFixed(2)}° toward ${e.az.toFixed(0)}°)`
    + `${e.margin > 0 ? ': direct communication possible.' : ': no direct line of sight to Earth.'}`;
  drawBars(); drawPanoramaNow(s, e);
}
function drawBars() {
  const C = S.calc, W = 320, L = 42, H = 13, t0 = S.track.t0_ms, span = C.span, x = t => L + (t - t0) / span * (W - L);
  const col = {sun: '#ffcc33', earth: '#4da6ff', both: '#7ee08a'};
  let g = '';
  [['sun', 'Sun'], ['earth', 'Earth'], ['both', 'Both']].forEach(([b, n], r) => {
    const y = 2 + r * (H + 5);
    g += `<text x="0" y="${y + 10}">${n}</text><rect x="${L}" y="${y}" width="${W - L}" height="${H}" fill="#2a2a33"/>`;
    for (const w of C.win[b]) g += `<rect x="${x(w.a).toFixed(1)}" y="${y}" width="${Math.max(0.8, x(w.b) - x(w.a)).toFixed(1)}" height="${H}" fill="${col[b]}"/>`;
  });
  const cx = x(S.t).toFixed(1);
  g += `<line x1="${cx}" x2="${cx}" y1="0" y2="56" stroke="#fff" stroke-width="1.5"/>`
     + `<text x="${L}" y="68">${fmtShort(t0)}</text><text x="${W}" y="68" text-anchor="end">${fmtShort(t0 + span)}</text>`;
  $('m-bars').innerHTML = `<svg viewBox="0 0 ${W} 72" width="100%" style="cursor:pointer;display:block">${g}</svg>`;
  $('m-bars').firstChild.onclick = ev => {
    const r = ev.currentTarget.getBoundingClientRect(), f = ((ev.clientX - r.left) / r.width * W - L) / (W - L);
    setTime(t0 + clamp(f, 0, 1) * span);
  };
}
const PAN = {W: 320, H: 150, pl: 22, pb: 14};
function panScale() {
  const hz = S.calc.g.hz; let hmax = -90, hmin = 90; for (const v of hz) { hmax = Math.max(hmax, v); hmin = Math.min(hmin, v); }
  const y0 = Math.floor(Math.min(-2, hmin - 1)), y1 = Math.ceil(Math.min(90, Math.max(10, hmax + 6)));
  return {y0, y1, X: az => PAN.pl + az / 360 * (PAN.W - PAN.pl), Y: el => (PAN.H - PAN.pb) * (1 - (el - y0) / (y1 - y0))};
}
function drawPanoramaPaths() {
  const C = S.calc, hz = C.g.hz, {y0, y1, X, Y} = panScale(), {W, H, pl, pb} = PAN;
  let g = `<rect x="${pl}" y="0" width="${W - pl}" height="${H - pb}" fill="#0b0d18"/>`, pts = `${X(0)},${Y(y0)} `;
  for (let i = 0; i <= N_AZ; i += 2) pts += `${X(i * 360 / N_AZ).toFixed(1)},${Y(Math.max(y0, hz[i % N_AZ])).toFixed(1)} `;
  g += `<polygon points="${pts}${X(360)},${Y(y0)}" fill="#4a4a52" stroke="#8a8a90" stroke-width="1"/>`;
  if (y0 < 0) g += `<line x1="${pl}" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="#666" stroke-dasharray="3 3" stroke-width=".7"/>`;
  for (const [b, c] of [['sun', '#ffcc33'], ['earth', '#4da6ff']])
    for (let t = S.track.t0_ms; t <= spanEnd(); t += HOUR) {
      const s = look(C.g, b, t); if (s.el < y0 || s.el > y1) continue;
      g += `<circle cx="${X(s.az).toFixed(1)}" cy="${Y(s.el).toFixed(1)}" r=".9" fill="${c}" opacity=".55"/>`;
    }
  [[0, 'N'], [90, 'E'], [180, 'S'], [270, 'W'], [360, 'N']].forEach(([a, n]) => g += `<text x="${X(a)}" y="${H - 3}" text-anchor="middle">${n}</text>`);
  [...new Set([y0, 0, y1])].forEach(v => g += `<text x="${pl - 3}" y="${clamp(Y(v) + 3, 8, H - pb)}" text-anchor="end">${v}°</text>`);
  S.panBase = g;
}
function drawPanoramaNow(s, e) {
  const {y0, y1, X, Y} = panScale();
  let g = S.panBase;
  for (const [b, c, r] of [[s, '#ffcc33', 5], [e, '#4da6ff', 4.5]]) {
    const cx = X(b.az);
    if (b.el > y1) g += `<path d="M${cx - 5},11 L${cx + 5},11 L${cx},3 Z" fill="${c}"/><text x="${cx + 7}" y="11" style="fill:${c}">${b.el.toFixed(0)}°</text>`;
    else if (b.el >= y0) g += `<circle cx="${cx}" cy="${Y(b.el)}" r="${r}" fill="${c}" stroke="#000"/>`;
  }
  $('panorama').innerHTML = g;
}

function metrics(lat, lon) {
  const C = analyse(lat, lon), pct = v => `${(100 * v / C.span).toFixed(0)}%`;
  return {C, rows: [
    ['Comm.', pct(C.dur.earth), `Communication (Earth line of sight) ${fmtHM(C.dur.earth)}`],
    ['Sun', pct(C.dur.sun), `Sunlight ${fmtHM(C.dur.sun)}`],
    ['Both', pct(C.dur.both), `Sun and Earth together ${fmtHM(C.dur.both)}`],
    ['Max ☉', fmtDeg(C.maxSun, 1), 'Maximum solar elevation'],
    ['E. win.', String(C.win.earth.length), 'Number of Earth visibility windows'],
  ]};
}
function renderCompare() {
  if (!S.track || $('tab-compare').hidden) return;
  let cols;
  if (S.cmpMode === 'sites') {
    const list = S.sites.map(s => ({head: s.name, title: `${s.name} · ${fmtSite(s)}, ground ${Math.round(demHeight(s.lat, s.lon))} m`, lat: s.lat, lon: s.lon}));
    if (S.sel && !S.sel.id) list.push({head: `${selLabel()} (not saved)`, title: `${selLabel()} ${fmtSite(S.sel)}`, lat: S.sel.lat, lon: S.sel.lon});
    cols = list.map(c => ({...c, rows: metrics(c.lat, c.lon).rows}));
  } else cols = S.snapshots.map((s, i) => ({head: s.head, title: s.title, rows: s.rows, del: i}));
  $('cmp-add-date').hidden = S.cmpMode !== 'dates';
  if (S.cmpMode === 'sites' && cols.length > 12) toast(`Comparing ${cols.length} sites…`);
  if (!cols.length) { $('cmp-table').innerHTML = `<p class="note">${S.cmpMode === 'sites' ? 'No saved sites yet.' : 'Add the current date span, change the start date, then add it again to compare.'}</p>`; return; }
  // one row per site or date span, one column per metric (hover a value for durations in hh:mm)
  const heads = cols[0].rows;
  $('cmp-table').innerHTML = `<table><tr><th></th>${heads.map(h => `<th title="${h[2].replace(/ \d.*$/, '')}">${h[0]}</th>`).join('')}</tr>`
    + cols.map(c => `<tr><td title="${escapeHTML(c.title)}"><b>${escapeHTML(c.head)}</b>${c.del !== undefined ? ` <button class="x" data-del="${c.del}" aria-label="Remove">×</button>` : ''}</td>`
      + c.rows.map(r => `<td title="${escapeHTML(r[2])}">${r[1]}</td>`).join('') + '</tr>').join('') + '</table>'
    + '<p class="note">Percentages are of the simulation span. Hover a value for the duration in hh:mm.</p>';
  for (const b of $('cmp-table').querySelectorAll('[data-del]')) b.onclick = () => { S.snapshots.splice(+b.dataset.del, 1); renderCompare(); };
  if (S.cmpMode === 'sites') $('cmp-table').querySelectorAll('tr').forEach((tr, i) => {
    const s = S.sites[i - 1]; if (!s) return;
    tr.style.cursor = 'pointer'; tr.title = `View ${s.name}`; tr.onclick = () => select(s.lat, s.lon, s.id);
  });
}

/* ------------------------------------------------------------------ time */
let drawQueued = false;
function requestDraw() {
  if (drawQueued) return; drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; S.view === '2d' ? drawSky() : draw3D(); });
}
function setTime(t) { S.t = clamp(t, S.track.t0_ms, spanEnd()); renderTime(); }
const stepMs = () => Math.max(10 * MIN, Math.round((spanEnd() - S.track.t0_ms) / 600 / (10 * MIN)) * 10 * MIN);
$('sim-range').addEventListener('input', e => setTime(S.track.t0_ms + e.target.value * 10 * MIN));
$('t-back').onclick = () => setTime(S.t - stepMs());
$('t-back-day').onclick = () => setTime(S.t - DAY);
$('t-fwd-day').onclick = () => setTime(S.t + DAY);
$('t-play').onclick = () => setPlaying(!S.playing);
function setPlaying(on) {
  S.playing = on; $('t-play').innerHTML = on ? '&#10074;&#10074;' : '&#9655;';
  clearInterval(S.timer);
  if (on) S.timer = setInterval(() => {
    if (G3.busy) return;
    if (S.t >= spanEnd()) { setPlaying(false); return; }
    setTime(S.t + stepMs());
  }, 90);
}
document.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(tag) || (tag === 'BUTTON' && e.key === ' ')) return;
  if (e.key === 'ArrowRight') setTime(S.t + stepMs());
  else if (e.key === 'ArrowLeft') setTime(S.t - stepMs());
  else if (e.key === ' ') { e.preventDefault(); setPlaying(!S.playing); }
  else if (e.key === 'Escape') { setDrop(false); closeMenus(); }
});

/* ------------------------------------------------------------------ data loading */
// Two ways to run: with the local server (python moon_app.py, any date, live JPL Horizons), or as a
// static website (python build_site.py) where Sun/Earth positions are pre-computed per month.
// Static data is loaded with <script> tags (not fetch) so index.html also works opened straight from disk.
const STATIC = window.LUNAR_STATIC || null;
const DATA = 'data/site/';
window.LUNAR_TRACK = window.LUNAR_TRACK || {};
function loadScript(src) {
  return new Promise((ok, fail) => {
    const s = document.createElement('script'); s.src = src; s.async = true;
    s.onload = ok; s.onerror = () => fail(new Error(`could not load ${src}`));
    document.head.appendChild(s);
  });
}
const loadStat = () => STATIC ? loadScript(DATA + 'static.js').then(() => window.LUNAR_MOON)
                              : fetch('/api/static').then(r => r.json());
const monthCache = new Map();
async function staticTrack(start, days) {
  let t0 = Date.parse(`${start}:00Z`); t0 -= t0 % HOUR;
  const t1 = t0 + days * DAY, first = Date.parse(`${STATIC.first}-01T00:00:00Z`);
  const [ly, lm] = STATIC.last.split('-').map(Number), end = Date.UTC(ly, lm, 1);
  if (!(t0 >= first && t1 <= end))
    throw new Error(`this website has Sun and Earth positions from ${STATIC.first} to ${STATIC.last}. `
      + 'Pick a start date and span inside that range (or run python moon_app.py locally for any date).');
  const sun = [], earth = [];
  let gT0 = null;
  for (let d = new Date(Date.UTC(new Date(t0).getUTCFullYear(), new Date(t0).getUTCMonth(), 1)); d.getTime() < t1; d.setUTCMonth(d.getUTCMonth() + 1)) {
    const key = d.toISOString().slice(0, 7);
    if (!monthCache.has(key)) monthCache.set(key, loadScript(`${DATA}track/${key}.js`).then(() => {
      const m = window.LUNAR_TRACK[key]; if (!m) throw new Error(`missing data for ${key}`); return m; }));
    const m = await monthCache.get(key);
    if (gT0 === null) gT0 = m.t0_ms;
    if (sun.length) { sun.length -= 3; earth.length -= 3; }   // months overlap by one hour: drop the duplicate
    for (let i = 0; i < m.sun.length; i++) { sun.push(m.sun[i]); earth.push(m.earth[i]); }
  }
  const i0 = Math.round((t0 - gT0) / HOUR), n = Math.round(days * 24) + 1;
  return {start, days, t0_ms: t0, dt_ms: HOUR, sun: sun.slice(3 * i0, 3 * (i0 + n)), earth: earth.slice(3 * i0, 3 * (i0 + n)),
          source: 'JPL Horizons (pre-computed)'};
}
async function loadTrack() {
  const d = $('in-date').value, t = $('in-time').value || '00:00', days = +$('in-span').value;
  if (!d) return;
  setPlaying(false);
  $('date-status').className = 'status'; $('date-status').textContent = 'Querying JPL Horizons…';
  $('loading').hidden = false; $('loading').textContent = 'Querying JPL Horizons for the Sun and Earth…';
  try {
    let j;
    if (STATIC) j = await staticTrack(`${d}T${t}`, days);
    else {
      const r = await fetch(`/api/track?start=${encodeURIComponent(`${d}T${t}`)}&days=${days}`); j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    }
    S.track = j; S.t = j.t0_ms;
    $('sim-range').max = Math.round((spanEnd() - j.t0_ms) / (10 * MIN));
    $('date-status').textContent = STATIC
      ? `Sun and Earth from JPL Horizons (pre-computed hourly, ${STATIC.first} to ${STATIC.last}): ${j.days} days.`
      : `Sun and Earth from JPL Horizons: ${j.days} days at ${j.dt_ms / MIN}-min steps.`;
    if (G3.ready) { Plotly.purge(G3.div); G3.ready = false; if (S.view === '3d') build3D(); }
    $('loading').hidden = true;
    if (S.sel) select(S.sel.lat, S.sel.lon, S.sel.id); else renderTime();
  } catch (err) {
    $('loading').hidden = true;
    $('date-status').className = 'status err'; $('date-status').textContent = `Could not load: ${err.message}`;
  }
}
async function loadTexture() {
  // the static texture is a data: URL, so reading its pixels is allowed even from file://
  const src = STATIC ? await loadScript(DATA + 'texture.js').then(() => window.LUNAR_TEXTURE) : '/api/texture.png';
  const img = new Image(); img.src = src; await img.decode();
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data, g = new Uint8Array(c.width * c.height);
  for (let i = 0; i < g.length; i++) g[i] = d[4 * i];
  S.tex = {g, tw: c.width, th: c.height};
}
async function init() {
  let now = new Date(Date.now() - Date.now() % HOUR);
  if (STATIC) {                                           // keep the default start inside the pre-computed range
    const lo = Date.parse(`${STATIC.first}-01T00:00:00Z`), [ly, lm] = STATIC.last.split('-').map(Number);
    const hi = Date.UTC(ly, lm, 1) - 31 * DAY;
    now = new Date(clamp(now.getTime(), lo, hi));
    $('in-date').min = `${STATIC.first}-01`; $('in-date').max = new Date(Date.UTC(ly, lm, 0)).toISOString().slice(0, 10);
    $('menu-date').querySelector('[data-act="date-reload"]').textContent = 'Reload Sun/Earth data';
  }
  $('in-date').value = now.toISOString().slice(0, 10); $('in-time').value = now.toISOString().slice(11, 16);
  try {
    const [stat] = await Promise.all([loadStat(), loadTexture()]);
    S.stat = stat;
    const m = stat.mesh;
    S.mesh = {lat: m.lat, lon: m.lon, nr: m.shape[0], nc: m.shape[1], exag: m.exag, h: new Int16Array(b64(m.h).buffer),
              nx: new Int8Array(b64(m.nx).buffer), ny: new Int8Array(b64(m.ny).buffer), nz: new Int8Array(b64(m.nz).buffer), refl: b64(m.refl)};
    S.dem = {h: new Int16Array(b64(stat.dem.h).buffer), NR: stat.dem.shape[0], NC: stat.dem.shape[1], ppd: stat.dem.ppd};
  } catch (err) {
    $('loading').textContent = STATIC ? `Could not load the Moon data (${err.message}). Build it with: python python/build_site.py`
      : `Could not load the Moon data (${err.message}). Start the app with: python moon_app.py`; return;
  }
  Object.values(MAPS).forEach(buildMap);
  renderLibrary();
  // optional link parameters: #site=lat,lon&view=3d&tab=compare&cmp=dates
  const q = new URLSearchParams(location.hash.slice(1)), ll = (q.get('site') || '').split(',').map(Number);
  S.sel = ll.length === 2 && ll.every(isFinite) ? {lat: ll[0], lon: ll[1], id: null}
                                                 : {lat: S.sites[0].lat, lon: S.sites[0].lon, id: S.sites[0].id};
  renderSites();
  await loadTrack();
  if (q.get('tab') === 'compare') document.querySelector('.tab[data-tab="compare"]').click();
  if (q.get('cmp') === 'dates') document.querySelector('.seg-btn[data-cmp="dates"]').click();
  setView(q.get('view') === '2d' ? '2d' : '3d');
}

/* ------------------------------------------------------------------ UI wiring */
function closeMenus() { document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open')); }
document.querySelectorAll('.menu-btn').forEach(b => b.onclick = e => {
  e.stopPropagation(); const m = $(b.dataset.menu), open = m.classList.contains('open'); closeMenus(); if (!open) m.classList.add('open');
});
document.addEventListener('click', e => { if (!e.target.closest('.menu')) closeMenus(); });
document.querySelectorAll('.menu button').forEach(b => b.onclick = () => { closeMenus(); ACTIONS[b.dataset.act](); });

const ACTIONS = {
  'export-csv'() {
    if (!S.calc) return;
    const rows = [['site', 'lat', 'lon', 'body', 'start_utc', 'end_utc', 'duration_h', 'starts_before_span', 'ends_after_span']];
    for (const b of ['sun', 'earth', 'both']) for (const w of S.calc.win[b])
      rows.push([selLabel(), S.sel.lat.toFixed(4), S.sel.lon.toFixed(4), b, fmtUTC(w.a), fmtUTC(w.b), ((w.b - w.a) / HOUR).toFixed(2), w.openA, w.openB]);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([rows.map(r => r.join(',')).join('\n')], {type: 'text/csv'}));
    a.download = `visibility_${S.sel.lat.toFixed(2)}_${S.sel.lon.toFixed(2)}.csv`; a.click();
  },
  'copy-summary'() {
    if (!S.calc) return;
    const C = S.calc, txt = `${selLabel()} (${fmtLat(S.sel.lat)}, ${fmtLon(S.sel.lon)}), ${fmtUTC(S.track.t0_ms)} UTC + ${S.track.days} d\n`
      + `Communication (Earth line of sight): ${fmtHM(C.dur.earth)}\nSunlight: ${fmtHM(C.dur.sun)}\nBoth: ${fmtHM(C.dur.both)}\n`
      + `Earth windows: ${C.win.earth.map(w => `${fmtShort(w.a)} – ${fmtShort(w.b)}`).join('; ') || 'none'}`;
    navigator.clipboard.writeText(txt).then(() => toast('Summary copied'), () => toast('Clipboard not available'));
  },
  'save-site'() {
    if (!S.sel) return;
    if (S.sel.id) { toast('This point is already a saved site'); return; }
    const s = addSite(S.sel.lat, S.sel.lon, S.sel.name || ''); S.sel.id = s.id; afterSitesChanged(); renderStatic(); toast(`${s.name} saved`);
  },
  'rename-site'() {
    const s = S.sel && S.sites.find(x => x.id === S.sel.id); if (!s) { toast('Select a saved site first'); return; }
    const n = prompt('Name for this site', s.name); if (n === null || !n.trim()) return;
    s.name = n.trim(); saveSites(); afterSitesChanged(); renderStatic();
  },
  'delete-site'() {
    const i = S.sel ? S.sites.findIndex(x => x.id === S.sel.id) : -1; if (i < 0) { toast('Select a saved site first'); return; }
    const [s] = S.sites.splice(i, 1); S.sel.id = null; S.sel.name = s.name; saveSites(); afterSitesChanged(); renderStatic(); toast(`${s.name} deleted`);
  },
  'reset-sites'() { S.sites = DEFAULT_SITES.map(s => ({...s})); saveSites(); if (S.sel) S.sel.id = null; afterSitesChanged(); },
  'clear-sites'() { S.sites = []; saveSites(); if (S.sel) S.sel.id = null; afterSitesChanged(); },
  'date-now'() { const n = new Date(); $('in-date').value = n.toISOString().slice(0, 10); $('in-time').value = n.toISOString().slice(11, 16); loadTrack(); },
  'date-reload'() { loadTrack(); },
};

document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === b));
  $('tab-summary').hidden = b.dataset.tab !== 'summary'; $('tab-compare').hidden = b.dataset.tab !== 'compare';
  renderCompare();
});
document.querySelectorAll('.seg-btn').forEach(b => b.onclick = () => {
  document.querySelectorAll('.seg-btn').forEach(x => x.classList.toggle('active', x === b));
  S.cmpMode = b.dataset.cmp; renderCompare();
});
$('cmp-add-date').onclick = () => {
  if (!S.sel || !S.track) return;
  const m = metrics(S.sel.lat, S.sel.lon);
  S.snapshots.push({head: fmtShort(S.track.t0_ms).slice(0, 5) + ` +${S.track.days}d`,
                    title: `${selLabel()} ${fmtSite(S.sel)}, ${fmtUTC(S.track.t0_ms)} + ${S.track.days} d`, rows: m.rows});
  renderCompare();
};
$('m-ewin-row').onclick = () => { $('m-ewin-list').hidden = !$('m-ewin-list').hidden; };
$('lib-btn').onclick = () => showWin($('lib-panel'), $('lib-panel').hidden);
$('lib-close').onclick = () => showWin($('lib-panel'), false);
$('lib-search').addEventListener('input', () => renderLibrary());
$('drop-btn').onclick = () => setDrop(!S.drop);
$('in-go').onclick = () => {
  const la = parseFloat($('in-lat').value), lo = parseFloat($('in-lon').value);
  if (isFinite(la) && isFinite(lo) && Math.abs(la) <= 90) placePoint(la, lo); else toast('Enter a latitude (−90…90) and longitude');
};
['in-lat', 'in-lon'].forEach(id => $(id).addEventListener('keydown', e => { if (e.key === 'Enter') $('in-go').click(); }));
let dateTimer;
['in-date', 'in-time', 'in-span'].forEach(id => $(id).addEventListener('change', () => { clearTimeout(dateTimer); dateTimer = setTimeout(loadTrack, 300); }));
$('v2d').onclick = () => setView('2d');
$('v3d').onclick = () => setView('3d');
function setView(v) {
  S.view = v;
  $('v2d').classList.toggle('active', v === '2d'); $('v3d').classList.toggle('active', v === '3d');
  G2.cv.hidden = v !== '2d'; G3.div.hidden = v !== '3d'; $('sky-hud').hidden = v !== '2d';
  $('hint').textContent = v === '2d' ? 'View from the selected site · drag to look around · scroll to zoom · double-click to re-aim'
                                     : 'Zoom in / zoom out with mouse wheel · drag to rotate · click to select a point';
  if (v === '3d') {
    if (!G3.ready && S.track) { $('loading').hidden = false; $('loading').textContent = 'Building the 3D Moon…'; setTimeout(() => { build3D(); $('loading').hidden = true; draw3D(); }, 20); }
    else { Plotly.Plots.resize(G3.div); draw3D(); }
  } else requestDraw();
}
window.addEventListener('resize', () => requestDraw());

/* ------------------------------------------------------------------ floating windows */
// Each panel can be dragged by its title bar, resized from the bottom-right corner, minimised to its
// title bar (– button or double-click), and the library can be closed. Layout is kept in localStorage.
const LAYOUT_KEY = 'lunar.layout.v2';
const wins = () => [...document.querySelectorAll('.win')];
function defaultLayout() {
  const W = innerWidth, H = innerHeight, top = 60, out = {};
  const left = wins().filter(w => w.dataset.dock === 'left'), right = wins().filter(w => w.dataset.dock === 'right');
  out['lib-panel'] = {x: Math.min(W - 460, 400), y: 70, w: 440, h: Math.min(640, H - 170), min: false};
  if (W < 900) {                                          // phones: all minimised, stacked down the left
    [...left, ...right].forEach((w, i) => out[w.id] = {x: 12, y: top + i * 50, min: true});
    out['lib-panel'] = {x: 12, y: 60, w: W - 24, h: H - 180, min: false};
    return out;
  }
  for (const [list, side] of [[left, 'left'], [right, 'right']]) {
    let y = top;
    for (const w of list) {
      const x = side === 'left' ? 12 : W - w.offsetWidth - 12, min = y > H - 220;
      out[w.id] = {x, y, min};
      y += (min ? 48 : Math.min(w.offsetHeight, H * 0.55)) + 10;
    }
  }
  return out;
}
function applyWin(w, st) {
  const min = !!st.min;
  w.classList.toggle('min', min);
  const b = w.querySelector('.win-min');
  b.innerHTML = min ? '&#9634;' : '&#8211;'; b.title = min ? 'Restore' : 'Minimize';
  if (st.w) { w.dataset.w = st.w; w.style.width = `${clamp(st.w, 240, innerWidth - 8)}px`; }
  if (st.h) w.dataset.h = st.h;
  const x = clamp(st.x, 80 - w.offsetWidth, innerWidth - 80), y = clamp(st.y, 0, innerHeight - 44);
  w.style.left = `${x}px`; w.style.top = `${y}px`;
  const room = Math.max(120, innerHeight - y - 8);
  if (min) { w.style.height = ''; w.style.maxHeight = ''; }
  else if (w.dataset.h) { w.style.height = `${clamp(+w.dataset.h, 120, room)}px`; w.style.maxHeight = ''; }
  else { w.style.height = ''; w.style.maxHeight = `${Math.max(160, innerHeight - y - 100)}px`; }
}
const winState = w => ({x: w.offsetLeft, y: w.offsetTop, min: w.classList.contains('min'),
                        w: w.dataset.w ? +w.dataset.w : undefined, h: w.dataset.h ? +w.dataset.h : undefined});
function saveLayout() {
  const o = {}; wins().forEach(w => o[w.id] = winState(w));
  try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(o)); } catch (e) {}
}
function initWindows(reset = false) {
  let saved = null;
  if (!reset) try { saved = JSON.parse(localStorage.getItem(LAYOUT_KEY)); } catch (e) {}
  wins().forEach(w => { w.classList.remove('min'); w.style.maxHeight = ''; w.style.height = ''; w.style.width = '';
                        delete w.dataset.w; delete w.dataset.h; });
  const def = defaultLayout();
  wins().forEach(w => applyWin(w, (saved && saved[w.id]) || def[w.id]));
  if (reset) saveLayout();
  updateMinAll();
}
function showWin(w, on) {
  w.hidden = !on;
  if (on) { w.style.zIndex = ++zTop; applyWin(w, {...winState(w), min: false}); }
}
function updateMinAll() {
  const anyOpen = wins().some(w => !w.hidden && !w.classList.contains('min'));
  $('min-all').textContent = anyOpen ? 'Minimize all' : 'Restore all';
}
let zTop = 20;
wins().forEach(w => {
  const head = w.querySelector('.win-head');
  const grip = document.createElement('div'); grip.className = 'win-resize'; grip.title = 'Drag to resize'; w.appendChild(grip);
  w.addEventListener('pointerdown', () => { w.style.zIndex = ++zTop; }, true);
  head.addEventListener('pointerdown', ev => {
    if (ev.target.closest('button, input')) return;
    const r = w.getBoundingClientRect(), dx = ev.clientX - r.left, dy = ev.clientY - r.top;
    head.setPointerCapture(ev.pointerId); w.classList.add('dragging');
    const move = e => applyWin(w, {...winState(w), x: e.clientX - dx, y: e.clientY - dy});
    const up = () => { head.removeEventListener('pointermove', move); head.removeEventListener('pointerup', up);
                       w.classList.remove('dragging'); saveLayout(); };
    head.addEventListener('pointermove', move); head.addEventListener('pointerup', up);
  });
  grip.addEventListener('pointerdown', ev => {
    ev.preventDefault(); ev.stopPropagation();
    const r = w.getBoundingClientRect(), x0 = ev.clientX, y0 = ev.clientY;
    grip.setPointerCapture(ev.pointerId); w.classList.add('resizing');
    const move = e => applyWin(w, {...winState(w), w: Math.round(r.width + e.clientX - x0), h: Math.round(r.height + e.clientY - y0)});
    const up = () => { grip.removeEventListener('pointermove', move); grip.removeEventListener('pointerup', up);
                       w.classList.remove('resizing'); saveLayout(); requestDraw(); };
    grip.addEventListener('pointermove', move); grip.addEventListener('pointerup', up);
  });
  grip.addEventListener('dblclick', () => { delete w.dataset.w; delete w.dataset.h; w.style.width = ''; applyWin(w, winState(w)); saveLayout(); });
  const toggle = () => { applyWin(w, {...winState(w), min: !w.classList.contains('min')}); saveLayout(); updateMinAll(); requestDraw(); };
  w.querySelector('.win-min').addEventListener('click', toggle);
  head.addEventListener('dblclick', ev => { if (!ev.target.closest('button, input')) toggle(); });
});
$('reset-layout').onclick = () => initWindows(true);
$('min-all').onclick = () => {
  const minimise = wins().some(w => !w.hidden && !w.classList.contains('min'));
  wins().filter(w => !w.hidden).forEach(w => applyWin(w, {...winState(w), min: minimise}));
  saveLayout(); updateMinAll(); if (S.calc) { aimSky(); requestDraw(); }
};
window.addEventListener('resize', () => wins().forEach(w => applyWin(w, winState(w))));
initWindows();

init();
