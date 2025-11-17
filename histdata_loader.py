"""
Utilities for downloading and converting HistData.com SPX/USD minute bars.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from tempfile import TemporaryDirectory
import zipfile

import pandas as pd

from histdata import download_hist_data

PAIR = "spxusd"
PAIR_LABEL = "SPX"


def download_spx_histdata(
    target_dir: Path,
    start_year: int = 2000,
    end_year: Optional[int] = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Download zipped HistData files for SPX/USD 1-minute data covering the
    requested year range.

    Note: Downloads complete years. For current year, downloads up to the current month.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    end_year = end_year or now.year
    downloaded = []

    years = list(range(start_year, end_year + 1))
    total = len(years)

    for idx, year in enumerate(years, start=1):
        # For past years, download the entire year (month=None)
        # For current year, we'll handle it differently if needed
        is_current_year = (year == now.year)

        # Use the expected filename format from histdata library
        zip_name = f"DAT_ASCII_{PAIR.upper()}_M1_{year}.zip"
        target_path = target_dir / zip_name

        if target_path.exists():
            if verbose:
                print(f"[{idx:04d}/{total:04d}] Skipping {year} (already downloaded)")
            continue

        if verbose:
            pct = (idx / total) * 100
            print(
                f"[{idx:04d}/{total:04d}] {pct:5.1f}% Downloading {PAIR.upper()} {year}",
                flush=True,
            )

        try:
            # For past years: month=None downloads the entire year
            # For current year: month=None should work too (gets partial year)
            filename = download_hist_data(
                year=str(year),
                month=None,
                pair=PAIR,
                time_frame="M1",
                platform="ASCII",
                output_directory=str(target_dir),
                verbose=False,  # Suppress per-file verbose output
            )
            if filename:
                # The library returns a path string, convert to Path and ensure it exists
                file_path = Path(filename)
                if file_path.exists():
                    downloaded.append(file_path)
                elif target_path.exists():
                    # Fallback: check if file exists at expected location
                    downloaded.append(target_path)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to download {year}: {e}")
            continue

    return downloaded


def build_prices_from_histdata(
    raw_dir: Path,
    output_path: Path,
    limit_files: Optional[Iterable[Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convert the downloaded HistData zip files into a single parquet file with
    timezone-aware timestamps.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    zips = sorted(limit_files or raw_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No HistData zip files found under {raw_dir}. Run download_spx_histdata first."
        )

    if verbose:
        print(f"Building price data from {len(zips)} zip files...")

    frames = []
    total = len(zips)
    for idx, zip_path in enumerate(zips, start=1):
        if verbose and idx % 50 == 0:  # Progress every 50 files
            pct = (idx / total) * 100
            print(f"  [{idx:04d}/{total:04d}] {pct:5.1f}% processed", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_files:
                continue
            with zf.open(csv_files[0]) as fh:
                df = pd.read_csv(
                    fh,
                    sep=";",
                    header=None,
                    names=["timestamp", "open", "high", "low", "close", "volume"],
                    dtype={"timestamp": str},
                )
            df["ts"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S")
            df = df[["ts", "close"]]
            frames.append(df)

    if not frames:
        raise RuntimeError("No CSV data extracted from HistData archives.")

    if verbose:
        print(f"Consolidating {len(frames)} data frames...")
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop_duplicates(subset="ts").sort_values("ts")
    prices["ts"] = prices["ts"].dt.tz_localize("Etc/GMT+5").dt.tz_convert("America/New_York")
    prices = prices.set_index("ts").rename(columns={"close": PAIR_LABEL})
    prices.to_csv(output_path, index_label="timestamp")
    return prices


def ensure_histdata_prices(
    output_path: Path,
    start_year: int = 2000,
    end_year: Optional[int] = None,
    rebuild: bool = False,
    verbose: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Ensure the consolidated SPX price parquet exists, downloading + rebuilding when necessary.

    Args:
        output_path: Path to save the final consolidated CSV
        start_year: First year to download
        end_year: Last year to download (defaults to current year)
        rebuild: If True, rebuild from zip files even if output exists
        verbose: Show progress messages
        cache_dir: Directory to cache downloaded zip files. If None, uses data/histdata_cache.
                  Set to False to use temporary directory (slower, no caching).
    """
    output_path = Path(output_path)

    if rebuild or not output_path.exists():
        if verbose:
            print(f"{'Rebuilding' if rebuild else 'Building'} price data from {start_year}...")

        # Use persistent cache directory by default for faster subsequent runs
        if cache_dir is None:
            cache_dir = output_path.parent / "histdata_cache"

        if cache_dir is False:
            # Use temporary directory (old behavior)
            with TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                download_spx_histdata(tmp_path, start_year=start_year, end_year=end_year, verbose=verbose)
                prices = build_prices_from_histdata(tmp_path, output_path, verbose=verbose)
        else:
            # Use persistent cache directory (faster!)
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Using cache directory: {cache_path}")
            download_spx_histdata(cache_path, start_year=start_year, end_year=end_year, verbose=verbose)
            prices = build_prices_from_histdata(cache_path, output_path, verbose=verbose)

        if verbose:
            print(f"[OK] Price data saved to {output_path}")
    else:
        if verbose:
            print(f"Loading cached price data from {output_path}")
        prices = pd.read_csv(output_path, parse_dates=["timestamp"])
        prices = prices.set_index("timestamp")
        if prices.index.tz is None:
            prices.index = (
                prices.index.tz_localize("UTC").tz_convert("America/New_York")
            )
    return prices


__all__ = [
    "download_spx_histdata",
    "build_prices_from_histdata",
    "ensure_histdata_prices",
]


