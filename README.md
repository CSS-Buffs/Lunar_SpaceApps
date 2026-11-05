# Lunar_SpaceApps: 3D Moon from NASA LOLA data

An interactive 3D model of the Moon built from **NASA LRO Lunar Orbiter Laser Altimeter (LOLA)**
elevation data, with a lunar gravity / orbit simulator. It is written in both **Python** and **MATLAB**.

The data is downloaded on first run from the NASA PDS Geosciences Node
([LOLA GDR, cylindrical](https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/))
into `data/`.

| Grid       | Size          | Resolution  | File size |
|------------|---------------|-------------|-----------|
| `LDEM_4`   | 1440 × 720    | ~7.6 km/px  | 2 MB      |
| `LDEM_16`  | 5760 × 2880   | ~1.9 km/px  | 33 MB     |
| `LDEM_64`  | 23040 × 11520 | ~474 m/px   | 531 MB    |

Radius = DN × 0.5 m + 1737.4 km, in the Mean-Earth / Polar-Axis frame, with longitude positive east.

## Python

```bash
pip install -r python/requirements.txt
cd python
python moon_viewer.py                                  # global Moon in the browser
python moon_viewer.py --stride 1 --exaggeration 5      # full LDEM_4 detail
python moon_viewer.py --color gravity                  # or: slope, terrain
python moon_viewer.py --ppd 16 --stride 1 --region -50 -35 -20 -5    # Tycho close-up
python simulate_orbit.py --alt 50 --inc 90 --days 1    # LRO-like orbit
python simulate_orbit.py --alt 3 --inc 30 --days 1     # low orbit, hits terrain
```

The viewer writes `output/*.html` and opens it in your browser. Drag to rotate, scroll to zoom,
right-drag to pan and double-click to reset.

## MATLAB (R2019a+)

```matlab
cd matlab
moon_viewer                         % rotate by dragging, zoom with scroll wheel
moon_viewer(4, 1, 5)                % full LDEM_4 resolution, 5x relief
moon_viewer(16, 1, 2, [-50 -35 340 355])   % Tycho close-up
out = moon_orbit_sim('alt', 50, 'inc', 90, 'days', 1);
out = moon_orbit_sim('alt', 3,  'inc', 30, 'days', 1);   % terrain impact
```

Controls at the bottom of the figure change the relief exaggeration, the colour field
(elevation, gravity or slope) and the sun direction.

## Physics model

* **Gravity:** point mass + J2 + C22 from GRAIL GRGM1200A (GM = 4902.800 km³/s²), evaluated in the
  body-fixed frame.
* **Rotation:** synchronous rotation at the sidereal rate (27.3217 d). The Earth's tidal (third-body)
  perturbation is applied from the body-fixed +X direction.
* **Integration:** DOP853 (Python) / ode113 (MATLAB) with tight tolerances. Steps are capped at 10 s
  so terrain crossings aren't missed. The run stops at **impact with the LOLA terrain**.
* **Accuracy check:** the Jacobi constant (energy in the rotating frame) is exactly conserved by this
  model. Its drift (~1e-14 relative over 1 day in Python) measures numerical error.
* **Surface gravity map:** gravity plus centrifugal acceleration on the true LOLA surface.

**Limitations:** gravity is degree 2 only. Real low orbits are strongly perturbed by mascons, so
mission-grade work needs the full GRGM1200A spherical-harmonic field. Lunar obliquity (1.5°),
libration and solar perturbation are also ignored.

## Layout

```
python/  lola_data.py  lunar_physics.py  moon_viewer.py  simulate_orbit.py
matlab/  lola_read.m   lunar_constants.m lunar_gravity.m moon_viewer.m  moon_orbit_sim.m
data/    downloaded LOLA files (git-ignored)
output/  generated HTML / PNG (git-ignored)
```
