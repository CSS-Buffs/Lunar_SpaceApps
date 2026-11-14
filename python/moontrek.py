"""
NASA Moon Trek client  (https://trek.nasa.gov/moon/)

  * WMTS tiles of the LRO WAC global mosaic (true surface appearance), stitched
    into one equirectangular image
  * The Moon Trek search index for IAU nomenclature: maria, major craters and the
    Apollo landing sites, with coordinates, used as labels on the 3D model
  * Catalogue search of high-resolution DEM products available for a region
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import nasa_http

TILES = "https://trek.nasa.gov/tiles/Moon/EQ/{layer}/1.0.0/default/default028mm/{z}/{row}/{col}.{ext}"
SEARCH = "https://trek.nasa.gov/moon/TrekServices/ws/index/eq/searchItems?"
WAC_LAYER = "LRO_WAC_Mosaic_Global_303ppd_v02"

MAJOR_CRATERS = ["Tycho", "Copernicus", "Kepler", "Aristarchus", "Plato", "Clavius",
                 "Grimaldi", "Ptolemaeus", "Theophilus", "Langrenus", "Petavius",
                 "Tsiolkovskiy", "Korolev", "Hertzsprung", "Apollo", "Schrödinger",
                 "Shackleton", "Daedalus", "Von Kármán", "Mendeleev"]


def wac_mosaic(level: int = 3, bbox=None, layer: str = WAC_LAYER, ext: str = "jpg", workers: int = 8):
    """Stitch WAC mosaic tiles. Tile level z has 2^(z+1) x 2^z tiles of 256 px over
    lon -180..180, lat 90..-90 (z=3 -> 4096x2048 px, ~0.09 deg/px).
    bbox = (lon0, lat0, lon1, lat1) fetches only the tiles covering that region.
    Returns (gray float32 0..1 image, lat centres north->south, lon centres west->east)."""
    from PIL import Image
    nrow, ncol = 2 ** level, 2 ** (level + 1)
    tdeg = 180.0 / nrow                                   # degrees per tile
    if bbox is None:
        r0, r1, c0, c1 = 0, nrow - 1, 0, ncol - 1
    else:
        lon0, lat0, lon1, lat1 = bbox
        c0, c1 = int((lon0 + 180) // tdeg), int(min((lon1 + 180) // tdeg, ncol - 1))
        r0, r1 = int((90 - lat1) // tdeg), int(min((90 - lat0) // tdeg, nrow - 1))

    def fetch(rc):
        r, c = rc
        url = TILES.format(layer=layer, z=level, row=r, col=c, ext=ext)
        data = nasa_http.get_bytes(url, nasa_http.cache_path(f"moontrek/{layer}/{level}", url, "." + ext))
        return r, c, np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32) / 255.0

    out = np.zeros(((r1 - r0 + 1) * 256, (c1 - c0 + 1) * 256), np.float32)
    jobs = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    with ThreadPoolExecutor(workers) as ex:
        for r, c, tile in ex.map(fetch, jobs):
            out[(r - r0) * 256:(r - r0 + 1) * 256, (c - c0) * 256:(c - c0 + 1) * 256] = tile
    px = tdeg / 256
    lat = 90 - r0 * tdeg - (np.arange(out.shape[0]) + 0.5) * px
    lon = -180 + c0 * tdeg + (np.arange(out.shape[1]) + 0.5) * px
    return out, lat, lon


def _search(params: dict):
    url = nasa_http.build_url(SEARCH, params)
    return nasa_http.get_json(url, nasa_http.cache_path("moontrek/search", url, ".json"))["response"]["docs"]


def _point(doc):
    lon, lat = map(float, doc["shape"].replace("POINT (", "").rstrip(")").split())
    return lat, lon


def features():
    """List of dicts {name, type, lat, lon (east, -180..180)} for labels."""
    out = []
    for ptype in ["Mare, maria", "Oceanus, oceani", "Crewed Landing"]:
        for d in _search(dict(start=0, rows=100, facetKeys="itemType|productType",
                              facetValues=f"nomenclature|{ptype}")):
            if d.get("shape", "").startswith("POINT"):
                lat, lon = _point(d)
                out.append(dict(name=d["title"].replace(" LM descent stage", ""),
                                type=ptype.split(",")[0], lat=lat, lon=lon))
    for name in MAJOR_CRATERS:
        docs = _search(dict(start=0, rows=5, facetKeys="itemType", facetValues="nomenclature", key=name))
        exact = [d for d in docs if d["title"].lower() == name.lower() and d.get("shape", "").startswith("POINT")]
        if exact:
            lat, lon = _point(exact[0])
            out.append(dict(name=exact[0]["title"], type="Crater", lat=lat, lon=lon))
    return out


def dem_products_near(lat: float, lon: float):
    """Moon Trek DEM products whose bounding box contains (lat, lon) [lon -180..180]."""
    docs = _search(dict(start=0, rows=200, key="DEM", facetKeys="productCat2", facetValues="DEM"))
    hits = []
    for d in docs:
        try:
            x0, y0, x1, y1 = map(float, d["bbox"].split(","))
        except (KeyError, ValueError):
            continue
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            hits.append(dict(label=d["productLabel"], title=d["title"], bbox=(x0, y0, x1, y1)))
    return hits


if __name__ == "__main__":
    img, lat, lon = wac_mosaic(2)
    print("WAC mosaic", img.shape, img.min(), img.max())
    f = features()
    print(len(f), "features, e.g.", f[:3])
    print("DEMs at Tycho:", dem_products_near(-43.3, -11.4))
