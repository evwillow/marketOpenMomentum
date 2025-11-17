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
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow()
    end_year = end_year or now.year
    downloaded = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            zip_name = f"{PAIR.upper()}_{year}_{month:02d}.zip"
            target_path = target_dir / zip_name
            if target_path.exists():
                continue
            if verbose:
                print(f"Downloading {PAIR.upper()} {year}-{month:02d}")
            filename = download_hist_data(
                year=str(year),
                month=str(month),
                pair=PAIR,
                time_frame="M1",
                platform="ASCII",
                output_directory=str(target_dir),
                verbose=verbose,
            )
            if filename:
                downloaded.append(target_dir / filename)
    return downloaded


def build_prices_from_histdata(
    raw_dir: Path,
    output_path: Path,
    limit_files: Optional[Iterable[Path]] = None,
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

    frames = []
    for zip_path in zips:
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
) -> pd.DataFrame:
    """
    Ensure the consolidated SPX price parquet exists, downloading + rebuilding when necessary.
    """
    output_path = Path(output_path)

    if rebuild or not output_path.exists():
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            download_spx_histdata(tmp_path, start_year=start_year, end_year=end_year)
            prices = build_prices_from_histdata(tmp_path, output_path)
    else:
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


