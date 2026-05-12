"""One-time GeoNames download + pickled-gazetteer build.

Downloads two small files from https://download.geonames.org/export/dump/
into `data/cache/geonames/` and produces `gazetteer.pkl` next to them so the
georef runner can load it in <1 s on subsequent runs.

    uv run python -m georeferencing.setup_gazetteer

Idempotent: skips downloads when the target file already exists with non-zero
size, and skips the pickle build when `gazetteer.pkl` exists (use --rebuild to
force).
"""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import httpx

from config import CACHE_DIR

from .gazetteer import Gazetteer

log = logging.getLogger(__name__)

GEONAMES_BASE = "https://download.geonames.org/export/dump"
USER_AGENT = "credleaf-research/0.1 (+https://github.com/Ala-ADN/credleaf)"

DOWNLOADS = [
    # (url_path, local_filename, member_inside_zip_or_None)
    ("cities1000.zip", "cities1000.zip", "cities1000.txt"),
    ("countryInfo.txt", "countryInfo.txt", None),
]


def geonames_dir() -> Path:
    return CACHE_DIR / "geonames"


def gazetteer_pickle_path() -> Path:
    return geonames_dir() / "gazetteer.pkl"


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip download (exists): %s", dest.name)
        return
    log.info("downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, headers={"User-Agent": USER_AGENT}, timeout=120.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                fh.write(chunk)


def _unzip_member(zip_path: Path, member: str, out_dir: Path) -> Path:
    out_path = out_dir / member
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("skip extract (exists): %s", out_path.name)
        return out_path
    log.info("extracting %s from %s", member, zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, path=out_dir)
    return out_path


def fetch_geonames(target_dir: Path | None = None) -> Path:
    """Ensure all required GeoNames files are present locally. Returns the dir."""
    target = target_dir or geonames_dir()
    target.mkdir(parents=True, exist_ok=True)
    for url_path, local_name, member in DOWNLOADS:
        url = f"{GEONAMES_BASE}/{url_path}"
        dest = target / local_name
        _download(url, dest)
        if member is not None:
            _unzip_member(dest, member, target)
    return target


def build(rebuild: bool = False) -> Path:
    target = fetch_geonames()
    pkl = gazetteer_pickle_path()
    if pkl.exists() and not rebuild:
        log.info("gazetteer pickle already present: %s (use --rebuild to force)", pkl)
        return pkl
    gz = Gazetteer.from_geonames(target)
    gz.save_pickle(pkl)
    return pkl


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild", action="store_true", help="rebuild pickle even if present")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pkl = build(rebuild=args.rebuild)
    print(f"\ngazetteer ready: {pkl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
