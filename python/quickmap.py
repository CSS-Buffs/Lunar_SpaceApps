"""
LROC QuickMap data client.

QuickMap (https://quickmap.lroc.im-ldi.com) is served by the LROC "Lunaserv" WMS at
https://wms.im-ldi.com/ (formerly webmap.lroc.asu.edu). Layers with the suffix
"_numeric" can be requested as 32-bit float GeoTIFF, which returns real physical values
instead of a picture.

Numeric layers used here
    luna_wac_dtm_numeric_meters_absolute   GLD100 stereo DTM, planetary radius [m] (+-79 deg lat)
    luna_grail_freeair_numeric    GRAIL free-air gravity anomaly [mGal]
    luna_wac_tio2_numeric         TiO2 abundance [wt %] (+-70 deg lat)
Image layers
    luna_wac_global               WAC 100 m/px global mosaic
All requests are in simple cylindrical lat/lon (EPSG:4326, longitude -180..180).

Registration: comparing against LOLA shows this server returns corner-registered pixels
(pixel i is centred at lon0 + i*dlon, not lon0 + (i+0.5)*dlon); validate.py measures it.
Note: the non-"_absolute" GLD100 layer is offset by about -1742 m; use the absolute one.
"""
from __future__ import annotations

import io

import numpy as np

import nasa_http

WMS = "https://wms.im-ldi.com/?"
NODATA_LIMIT = 1e30

NUMERIC_LAYERS = {
    "gld100": ("luna_wac_dtm_numeric_meters_absolute", "m (radius)"),
    "freeair": ("luna_grail_freeair_numeric", "mGal"),
    "tio2": ("luna_wac_tio2_numeric", "wt% TiO2"),
}


def _getmap(layer: str, bbox, width: int, height: int, fmt: str) -> bytes:
    lon0, lat0, lon1, lat1 = bbox
    params = dict(SERVICE="WMS", VERSION="1.1.1", REQUEST="GetMap", SRS="EPSG:4326",
                  STYLES="", LAYERS=layer, FORMAT=fmt, WIDTH=int(width), HEIGHT=int(height),
                  BBOX=f"{lon0},{lat0},{lon1},{lat1}")
    url = nasa_http.build_url(WMS, params)
    ext = ".tif" if "tiff" in fmt else ".jpg" if "jpeg" in fmt else ".png"
    data = nasa_http.get_bytes(url, nasa_http.cache_path("quickmap", url, ext), timeout=600)
    if data[:5] == b"<?xml":
        raise RuntimeError(f"QuickMap WMS error for {layer}: {data.decode(errors='replace')}")
    return data


def numeric(name_or_layer: str, bbox=(-180, -90, 180, 90), width=1440, height=720):
    """Float grid for a numeric layer. Returns (values, lat, lon) with values (height,width),
    lat north->south and lon west->east at pixel centres; nodata -> NaN."""
    import tifffile
    layer = NUMERIC_LAYERS.get(name_or_layer, (name_or_layer,))[0]
    data = _getmap(layer, bbox, width, height, "image/tiff; mode=32bit")
    a = tifffile.imread(io.BytesIO(data)).astype(np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    a[(a < -NODATA_LIMIT) | (a > NODATA_LIMIT)] = np.nan
    lon0, lat0, lon1, lat1 = bbox
    lat = lat1 - np.arange(height) * (lat1 - lat0) / height       # corner-registered (see above)
    lon = lon0 + np.arange(width) * (lon1 - lon0) / width
    return a, lat, lon


def image(layer="luna_wac_global", bbox=(-180, -90, 180, 90), width=2048, height=1024):
    """RGB/grey image layer as uint8 array."""
    from PIL import Image
    data = _getmap(layer, bbox, width, height, "image/jpeg")
    return np.asarray(Image.open(io.BytesIO(data)))


def global_to_east360(a, lon):
    """Re-order a global -180..180 grid to 0..360 east longitude (matching LOLA LDEM)."""
    order = np.argsort(np.mod(lon, 360))
    return a[:, order], np.mod(lon, 360)[order]


if __name__ == "__main__":
    fa, lat, lon = numeric("freeair", width=720, height=360)
    print("GRAIL free-air [mGal]:", np.nanmin(fa), np.nanmax(fa))
    r, *_ = numeric("gld100", bbox=(-12.5, -44.5, -10.2, -42.1), width=300, height=300)
    print("GLD100 Tycho radius [km]:", np.nanmin(r) / 1e3, np.nanmax(r) / 1e3)
