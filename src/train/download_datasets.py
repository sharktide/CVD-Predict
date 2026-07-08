#!/usr/bin/env python3
"""Download datasets from PhysioNet for stroke-prediction model training.

Usage
-----
    python -m src.train.download_datasets            # download all datasets
    python -m src.train.download_datasets mimic3_waveform cves  # specific ones
    python -m src.train.download_datasets --list      # show available datasets

PhysioNet requires credentialed access for some datasets.  Set the environment
variables ``PHYSIONET_USER`` and ``PHYSIONET_PASS`` (or place them in a ``.env``
file at the project root) before running.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth

from .config import DATASETS, PHYSIONET_PASS, PHYSIONET_USER, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION = requests.Session()
if PHYSIONET_USER and PHYSIONET_PASS:
    SESSION.auth = HTTPBasicAuth(PHYSIONET_USER, PHYSIONET_PASS)


def _progress_bar(resp: requests.Response, total: int | None = None) -> Iterator[bytes]:
    """Yield chunks while printing a simple progress bar."""
    chunk_size = 1024 * 256  # 256 KB
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if chunk:
            downloaded += len(chunk)
            yield chunk
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    if total:
        print()


def _get(url: str, dest: Path, desc: str = "") -> None:
    """Download *url* to *dest*, skipping if it already exists."""
    if dest.exists():
        log.info("Already exists, skipping: %s", dest.name)
        return

    log.info("Downloading %s → %s", desc or url, dest.name)
    resp = SESSION.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0)) or None

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in _progress_bar(resp, total):
            f.write(chunk)
    tmp.rename(dest)


def _list_remote_dir(url: str) -> list[str]:
    """Return file/directory names listed at a PhysioNet directory page."""
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    # PhysioNet directory listings use simple anchor tags
    names: list[str] = []
    for line in resp.text.splitlines():
        if 'href="' in line:
            href = line.split('href="')[1].split('"')[0]
            name = href.rstrip("/").split("/")[-1]
            if name and name not in (".", "..", "about", "contact", "sitemap"):
                names.append(name)
    return names


def _download_tree(base_url: str, local_dir: Path, skip_extensions: tuple[str, ...] = ()) -> None:
    """Recursively download all files under *base_url* into *local_dir*."""
    stack: list[str] = [base_url]
    while stack:
        current = stack.pop()
        rel = current[len(base_url):]
        target = local_dir / rel
        target.mkdir(parents=True, exist_ok=True)

        try:
            entries = _list_remote_dir(current)
        except requests.HTTPError:
            # It's a file, not a directory — download it
            fname = current.rstrip("/").split("/")[-1]
            if not any(fname.endswith(ext) for ext in skip_extensions):
                _get(current, target / fname, desc=fname)
            continue

        for entry in entries:
            child_url = urljoin(current, entry)
            if entry.endswith("/"):
                stack.append(child_url)
            else:
                if not any(entry.endswith(ext) for ext in skip_extensions):
                    _get(child_url, target / entry, desc=entry)


def _extract_if_compressed(path: Path) -> None:
    """Extract .tar.gz / .zip archives sitting next to raw data."""
    for archive in path.rglob("*.tar.gz"):
        log.info("Extracting %s", archive.name)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=archive.parent)
    for archive in path.rglob("*.zip"):
        log.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(path=archive.parent)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dataset(name: str, *, force: bool = False) -> Path:
    """Download a single dataset by its registry key.

    Returns the local directory where data was stored.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")

    info = DATASETS[name]
    local_dir: Path = info["local_dir"]

    if force and local_dir.exists():
        import shutil
        shutil.rmtree(local_dir)

    log.info("=== Dataset: %s ===", name)
    log.info("  %s", info["description"])

    if not PHYSIONET_USER or not PHYSIONET_PASS:
        log.warning(
            "PHYSIONET_USER / PHYSIONET_PASS not set. "
            "Some datasets require credentialed access."
        )

    base_url = info["url"]
    _download_tree(base_url, local_dir, skip_extensions=(".html", ".txt", ".md"))
    _extract_if_compressed(local_dir)

    log.info("Finished %s → %s", name, local_dir)
    return local_dir


def download_all(*, force: bool = False) -> dict[str, Path]:
    """Download every registered dataset. Returns {name: path}."""
    results: dict[str, Path] = {}
    for name in DATASETS:
        try:
            results[name] = download_dataset(name, force=force)
        except Exception:
            log.exception("Failed to download dataset '%s'", name)
    return results


def list_datasets() -> None:
    """Print a table of available datasets."""
    print(f"\n{'Key':<22} {'PhysioNet name':<20} {'Description'}")
    print("-" * 90)
    for key, info in DATASETS.items():
        print(f"{key:<22} {info['physionet_name']:<20} {info['description'][:55]}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PhysioNet datasets for stroke-prediction training."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset keys to download (default: all).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available datasets and exit."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if data exists."
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    targets = args.datasets if args.datasets else list(DATASETS.keys())
    for name in targets:
        download_dataset(name, force=args.force)


if __name__ == "__main__":
    main()
