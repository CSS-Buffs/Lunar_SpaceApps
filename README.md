# Lunar_SpaceApps: a data-driven 3D Moon and orbit simulator

An interactive 3D model of the Moon and a high-fidelity lunar orbit simulator. Both are written in **Python** and **MATLAB**, and both are built from live NASA data services.

| NASA source | What we use it for |
|---|---|
| **LRO LOLA** laser altimeter (PDS) | Global shape of the Moon (`LDEM_4/16/64`) and terrain for impact detection |
| **NASA Moon Trek** (WMTS tiles and search API) | LRO WAC global mosaic as the true surface appearance, plus IAU feature names (maria, major craters, Apollo sites) |
| **LROC QuickMap** (Lunaserv WMS, numeric layers) | GLD100 stereo terrain for ~100 m close-ups, TiO₂ abundance and the GRAIL free-air gravity map |
| **JPL Horizons API** | Sun and Earth positions (third-body forces and real sunlight or phase for any date), LRO's real trajectory (initial state and ground truth), and sub-Earth/sub-solar points for checking orientation |
| **GRAIL** GRGM1200A gravity (PDS) | Spherical-harmonic lunar gravity field (only the degrees you request are downloaded) |
| NASA NAIF lunar frame kernel | Mean-Earth to Principal-Axis frame rotation (DE440) |

All downloads are cached in `data/`. Generated pages and plots go to `output/`.

## Python

```bash
pip install -r python/requirements.txt
cd python
python moon_viewer.py                                   # photo-textured Moon (Moon Trek WAC)
python moon_viewer.py --date 2026-11-24T12:00 --labels  # real sunlight/phase (Horizons) + names
python moon_viewer.py --date now                        # the Moon as it is lit right now
python moon_viewer.py --color gravity                   # GRAIL free-air anomaly (our SH synthesis)
python moon_viewer.py --color freeair | tio2 | slope | height
python moon_viewer.py --region -45 -41.5 -13.5 -9 --source gld100 --stride 1 --exaggeration 2 \
                      --date 2026-11-20T06:00           # Tycho close-up, GLD100 terrain at 128 px/deg
python simulate_orbit.py --lro --epoch 2024-08-20T00:00 --days 1 --compare-degrees 2,20,60,100
python simulate_orbit.py --alt 50 --inc 90 --days 2 --epoch 2026-11-20T00:00
python simulate_orbit.py --alt 8 --ecc 0.002 --inc 85 --days 3 --degree 60   # ends in an impact
python validate.py                                      # cross-checks against NASA data
```

The viewer writes an HTML page and opens it. Drag to rotate, scroll to zoom, right-drag to pan, double-click to reset.

## MATLAB (R2019b+)

```matlab
cd matlab
moon_viewer                                             % Moon Trek texture, rotate/zoom
moon_viewer('date','2026-11-24T12:00','labels',true)     % sunlight from Horizons + names
moon_viewer('color','tio2')                              % QuickMap layers: tio2, freeair
moon_viewer('region',[-45 -41.5 -13.5 -9],'source','gld100','exaggeration',2)
out = moon_orbit_sim('lro',true,'epoch','2024-08-20T00:00:00','days',1,'degree',100);
out = moon_orbit_sim('alt',50,'inc',90,'days',1);
```

The MATLAB code mirrors the Python implementation. It was written without access to MATLAB, so the Python side is the tested reference.

## Physics

* **Gravity:** GRAIL GRGM1200A spherical harmonics (default degree 100, up to 1200 available). It uses a fully-normalised Cunningham V/W recursion, JIT-compiled with numba.
* **Frames:** the simulation runs in the Moon-centred ICRF frame, which is what Horizons uses. Moon orientation comes from the IAU 2015 model with libration terms. GRAIL is evaluated in the DE440 Principal-Axis frame, and LOLA in the Mean-Earth frame.
* **Perturbations:** Earth and Sun positions come from the Horizons API (cubic-Hermite interpolated). Cannonball solar radiation pressure with lunar shadow is optional.
* **Integrator:** DOP853 in Python and ode113 in MATLAB, with tolerances of 1e-11. The run stops on impact with LOLA terrain.

## Validation (`python validate.py` → `output/validation_report.md`)

| Check | Result |
|---|---|
| Orientation vs Horizons sub-Earth/sub-solar points | ≤ 0.006° (≈190 m); NAIF documents ~0.005° for the IAU series |
| SH gravity vs finite-difference potential | 7e-7 relative (finite-difference noise level) |
| Our GRAIL free-air map vs QuickMap GRAIL map | correlation 0.88 |
| LOLA vs independent GLD100 stereo terrain (near side, 0.05°) | median 3 m, robust σ 17 m |
| 24 h prediction vs **real LRO** (3 epochs) | degree 2: 10–18 km · degree 60: 120–480 m · **degree 100–150: 33–133 m** |

Two corrections came out of these checks:
* **ME→PA rotation sign:** the sign of the NAIF ME→PA rotation was confirmed against LRO. The adopted convention is 2–5× more accurate than the opposite sign.
* **QuickMap pixel registration:** QuickMap returns corner-registered pixels. Correcting for this improved the terrain and gravity comparisons.

**Remaining limitations**
* LRO maneuvers and momentum dumps are not modeled.
* SRP uses a cannonball model and is off by default.
* Moon Trek DEM tiles are 8-bit images, so numeric terrain comes from LOLA and QuickMap GLD100.

## Layout

```
python/  nasa_http.py  horizons.py  quickmap.py  moontrek.py  lola_data.py  grail_gravity.py
         lunar_frames.py  lunar_physics.py  moon_viewer.py  simulate_orbit.py  validate.py
matlab/  nasa_get.m  horizons_time.m  horizons_vectors.m  quickmap_numeric.m  moontrek_mosaic.m
         moontrek_features.m  lola_read.m  grail_load.m  grail_accel.m  lunar_orientation.m
         lunar_constants.m  moon_viewer.m  moon_orbit_sim.m
data/    cached NASA downloads (git-ignored)      output/  generated HTML/PNG/report (git-ignored)
```
