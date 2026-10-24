"""Download and unpack registered public datasets into data/raw/<key>/.

Usage:
    python -m aip_data.download --list
    python -m aip_data.download cmapss
    python -m aip_data.download all
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from aip_core import DATASETS, paths
from aip_core.logging import get_logger

log = get_logger("aip_data.download")
CHUNK = 1 << 20  # 1 MiB


def download_dataset(key: str, force: bool = False) -> Path:
    spec = DATASETS[key]
    target_dir = paths.raw / key
    archive = paths.raw / spec.archive_name

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        log.info("%s already present at %s (use --force to redownload)", key, target_dir)
        return target_dir

    paths.ensure()
    log.info("Downloading %s from %s", spec.name, spec.url)
    with requests.get(spec.url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
    log.info("Saved archive: %s (%.1f MB)", archive, archive.stat().st_size / 1e6)

    target_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target_dir)
        log.info("Extracted to %s", target_dir)
    else:
        log.warning("%s is not a zip; leaving archive in place for manual handling", archive)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", nargs="?", help="dataset key, or 'all'")
    parser.add_argument("--list", action="store_true", help="list registered datasets")
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    args = parser.parse_args()

    if args.list or not args.key:
        for k, spec in DATASETS.items():
            print(f"{k:14s} phase {spec.phase}  {spec.name}")
        return

    keys = list(DATASETS) if args.key == "all" else [args.key]
    for k in keys:
        if k not in DATASETS:
            raise SystemExit(f"Unknown dataset '{k}'. Use --list.")
        download_dataset(k, force=args.force)


if __name__ == "__main__":
    main()
