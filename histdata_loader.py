"""
Utilities for downloading and converting HistData.com SPX/USD minute bars.

Data is downloaded as ZIP files temporarily, then extracted and saved as CSV files (prices_YYYY.csv).
ZIP files are kept in histdata_cache/ for faster re-downloads, but final data is always CSV.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile

# Prevent __pycache__ creation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import pandas as pd

from histdata import download_hist_data

PAIR = "spxusd"
PAIR_LABEL = "SPX"


def _download_single_year(
    year: int,
    target_dir: Path,
    verbose: bool = True,
    current_year: Optional[int] = None,
) -> Optional[Path]:
    """Download a single year of data. Helper for parallel downloads."""
    zip_name = f"DAT_ASCII_{PAIR.upper()}_M1_{year}.zip"
    target_path = target_dir / zip_name

    if target_path.exists():
        return target_path

    try:
        # For current year, HistData requires month parameter - skip it
        if current_year and year == current_year:
            if verbose:
                print(f"Warning: Skipping {year} (current year requires month parameter)")
            return None
            
        filename = download_hist_data(
            year=str(year),
            month=None,
            pair=PAIR,
            time_frame="M1",
            platform="ASCII",
            output_directory=str(target_dir),
            verbose=False,
        )
        if filename:
            file_path = Path(filename)
            if file_path.exists():
                return file_path
            elif target_path.exists():
                return target_path
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to download {year}: {e}")
    return None


def download_spx_histdata(
    target_dir: Path,
    start_year: int = 2000,
    end_year: Optional[int] = None,
    verbose: bool = True,
    max_workers: int = 4,
) -> list[Path]:
    """
    Download zipped HistData files for SPX/USD 1-minute data covering the
    requested year range.

    Note: Downloads complete years. For current year, downloads up to the current month.
    Uses parallel downloads for faster performance.
    
    Args:
        target_dir: Directory to save downloaded zip files
        start_year: First year to download
        end_year: Last year to download (defaults to current year)
        verbose: Show progress messages
        max_workers: Number of parallel download threads (default: 4)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    end_year = end_year or now.year
    downloaded = []

    years = list(range(start_year, end_year + 1))
    total = len(years)
    
    # Filter out years that already exist
    years_to_download = []
    for year in years:
        zip_name = f"DAT_ASCII_{PAIR.upper()}_M1_{year}.zip"
        target_path = target_dir / zip_name
        if not target_path.exists():
            years_to_download.append(year)
        else:
            downloaded.append(target_path)
            if verbose:
                print(f"Skipping {year} (already downloaded)")

    if not years_to_download:
        if verbose:
            print(f"All {total} years already downloaded")
        return downloaded

    if verbose:
        print(f"Downloading {len(years_to_download)} years in parallel (max {max_workers} workers)...")

    # Download in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {
            executor.submit(_download_single_year, year, target_dir, verbose, now.year): year
            for year in years_to_download
        }
        
        completed = 0
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    downloaded.append(result)
                    if verbose:
                        pct = (completed / len(years_to_download)) * 100
                        print(f"[{completed}/{len(years_to_download)}] {pct:5.1f}% Downloaded {year}")
            except Exception as e:
                if verbose:
                    print(f"Error downloading {year}: {e}")

    return downloaded


def build_prices_from_histdata(
    raw_dir: Path,
    output_dir: Path,
    limit_files: Optional[Iterable[Path]] = None,
    verbose: bool = True,
) -> dict[int, pd.DataFrame]:
    """
    Convert the downloaded HistData zip files into year-based CSV files.
    
    Returns a dictionary mapping year to DataFrame.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zips = sorted(limit_files or raw_dir.glob("*.zip"))
    if not zips:
        if verbose:
            print(f"Warning: No HistData zip files found under {raw_dir}. Skipping build.")
        return {}

    if verbose:
        print(f"Building price data from {len(zips)} zip files...")

    # Group by year
    year_data = {}
    
    for zip_path in zips:
        # Extract year from filename: DAT_ASCII_SPXUSD_M1_2019.zip -> 2019
        year = None
        try:
            year_str = zip_path.stem.split("_")[-1]
            year = int(year_str)
        except (ValueError, IndexError):
            if verbose:
                print(f"Warning: Could not extract year from {zip_path.name}, skipping")
            continue
        
        if verbose:
            print(f"  Processing {year}...", end=" ", flush=True)
        
        frames = []
        with zipfile.ZipFile(zip_path) as zf:
            csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            for csv_file in csv_files:
                with zf.open(csv_file) as fh:
                    df = pd.read_csv(
                        fh,
                        sep=";",
                        header=None,
                        names=["timestamp", "open", "high", "low", "close", "volume"],
                        dtype={"timestamp": str},
                        engine="c",  # Use C engine for faster parsing
                    )
                    df["ts"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S")
                    df = df[["ts", "close"]]
                    frames.append(df)
        
        if frames:
            prices = pd.concat(frames, ignore_index=True)
            prices = prices.drop_duplicates(subset="ts").sort_values("ts")
            prices["ts"] = prices["ts"].dt.tz_localize("Etc/GMT+5").dt.tz_convert("America/New_York")
            prices = prices.set_index("ts").rename(columns={"close": PAIR_LABEL})
            
            # Save year file
            year_file = output_dir / f"prices_{year}.csv"
            prices.to_csv(year_file, index_label="timestamp")
            year_data[year] = prices
            
            if verbose:
                print(f"✓ Saved {len(prices):,} rows to {year_file.name}")
        else:
            if verbose:
                print("✗ No data extracted")

    if not year_data:
        raise RuntimeError("No CSV data extracted from HistData archives.")
    
    if verbose:
        print(f"\n[OK] Created {len(year_data)} year files in {output_dir}")
    
    return year_data


def load_prices_by_year(data_dir: Path, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Load price data from year-based CSV files and combine them seamlessly.
    
    Args:
        data_dir: Directory containing price files (prices_YYYY.csv)
        start_year: First year to load (inclusive)
        end_year: Last year to load (inclusive)
    
    Returns:
        Combined DataFrame with all years, sorted by timestamp
    """
    data_dir = Path(data_dir)
    now = datetime.now()
    end_year = end_year or now.year
    
    # Find all year files
    year_files = sorted(data_dir.glob("prices_*.csv"))
    
    if not year_files:
        raise FileNotFoundError(f"No price files found in {data_dir}")
    
    frames = []
    for year_file in year_files:
        # Extract year from filename: prices_2019.csv -> 2019
        try:
            year = int(year_file.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        
        # Filter by year range
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue
        
        df = pd.read_csv(
            year_file,
            parse_dates=["timestamp"],
            index_col="timestamp",
            engine="c",  # Use C engine for faster parsing
        )
        # Ensure index is DatetimeIndex (sometimes parse_dates doesn't work correctly)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Ensure timezone is set to America/New_York
        if df.index.tz is None:
            # If no timezone, assume UTC and convert
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            # If timezone exists, convert to America/New_York
            df.index = df.index.tz_convert("America/New_York")
        frames.append(df)
    
    if not frames:
        raise FileNotFoundError(f"No price files found for years {start_year}-{end_year}")
    
    # Combine and sort
    prices = pd.concat(frames, axis=0)
    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    
    return prices


def ensure_histdata_prices(
    output_dir: Path,
    start_year: int = 2019,
    end_year: Optional[int] = None,
    rebuild: bool = False,
    verbose: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Ensure year-based SPX price CSV files exist, downloading + rebuilding when necessary.
    
    Files are saved as prices_YYYY.csv (one per year) to keep files manageable.

    Args:
        output_dir: Directory to save year-based CSV files
        start_year: First year to download
        end_year: Last year to download (defaults to current year)
        rebuild: If True, rebuild from zip files even if files exist
        verbose: Show progress messages
        cache_dir: Directory to cache downloaded zip files. If None, uses data/histdata_cache.
                  Set to False to use temporary directory (slower, no caching).
    
    Returns:
        Combined DataFrame with all years loaded
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now()
    requested_end_year = end_year or now.year
    # Don't try to download current year (requires month parameter) - use previous year as max
    effective_end_year = min(requested_end_year, now.year - 1) if requested_end_year >= now.year else requested_end_year
    
    # Check which year files are missing
    missing_years = []
    for year in range(start_year, effective_end_year + 1):
        year_file = output_dir / f"prices_{year}.csv"
        if rebuild or not year_file.exists():
            missing_years.append(year)
    
    if missing_years:
        if verbose:
            print(f"{'Rebuilding' if rebuild else 'Downloading'} price data for years: {missing_years}")

        # Use persistent cache directory by default for faster subsequent runs
        if cache_dir is None:
            cache_dir = output_dir / "histdata_cache"

        if cache_dir is False:
            # Use temporary directory (old behavior)
            with TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                download_spx_histdata(tmp_path, start_year=min(missing_years), end_year=max(missing_years), verbose=verbose)
                build_prices_from_histdata(tmp_path, output_dir, verbose=verbose)
        else:
            # Use persistent cache directory (faster!)
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Using cache directory: {cache_path}")
            download_spx_histdata(cache_path, start_year=min(missing_years), end_year=max(missing_years), verbose=verbose)
            # Only build if we have zip files
            built_data = build_prices_from_histdata(cache_path, output_dir, verbose=verbose)
            if not built_data and verbose:
                print(f"Note: No new data files built. Using existing files if available.")

        if verbose:
            print(f"[OK] Price data files saved to {output_dir}")
    else:
        if verbose:
            print(f"All price files exist in {output_dir}")
    
    # Load and return combined data (use original end_year for loading, not effective_end_year)
    return load_prices_by_year(output_dir, start_year=start_year, end_year=end_year)


__all__ = [
    "download_spx_histdata",
    "build_prices_from_histdata",
    "ensure_histdata_prices",
    "load_prices_by_year",
]


