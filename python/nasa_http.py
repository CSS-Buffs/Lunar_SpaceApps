"""Small HTTP helper with an on-disk cache shared by all NASA data clients."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USER_AGENT = "lunar-spaceapps-model/2.0 (NASA open data client)"


def cache_path(subdir: str, url: str, suffix: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return DATA_DIR / subdir / f"{h}{suffix}"


def get_bytes(url: str, cache: Path | None = None, timeout: float = 300, retries: int = 3) -> bytes:
    """GET url (with retries); if cache is given, read/write the result there."""
    if cache is not None and cache.exists() and cache.stat().st_size > 0:
        return cache.read_bytes()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Download failed after {retries} attempts: {url}\n{last}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(cache.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(cache)
    return data


def get_json(url: str, cache: Path | None = None, timeout: float = 120):
    return json.loads(get_bytes(url, cache, timeout).decode("utf-8", errors="replace"))


def build_url(base: str, params: dict) -> str:
    return base + urllib.parse.urlencode(params, safe=",:'@|")
