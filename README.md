# Market Open Momentum

A machine learning strategy for trading SPX/USD momentum at market open using minute-level data.

## Overview

This project implements a gradient-boosted classifier that predicts SPX/USD price movements based on engineered momentum, volatility, and oscillator features. The strategy is backtested and validated before deployment.

## Project Structure

```
marketOpenMomentum/
├── notebooks/              # Jupyter notebooks (workflow)
│   ├── 00_environment.ipynb      # Setup and Alpaca connectivity
│   ├── 01_feature_engineering.ipynb  # Data download and feature creation
│   ├── 02_model_training.ipynb       # XGBoost model training
│   ├── 03_backtest.ipynb             # Strategy backtesting
│   └── 04_diagnostics.ipynb          # Data validation and tests
├── data/                   # Generated data files (not in Git - too large)
├── tests/                  # Unit tests
├── momentum_lib.py         # Core library functions
├── histdata_loader.py      # Historical data downloader
├── setup_data.py           # Script to generate all data files
└── export_notebooks_to_pdf.py  # Script to export notebooks as PDFs
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file with your Alpaca API credentials:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Generate Data Files

The large data files (prices.csv, features.csv) are excluded from Git due to size limits. Generate them locally:

```bash
python setup_data.py
```

This will:
- Download historical SPX/USD data from HistData.com (2019-present)
- Compute engineered features
- Train the XGBoost model

**Note:** First run takes 10-20 minutes. Subsequent runs are faster due to caching.

### 4. Run Notebooks

Execute the notebooks in order:
1. `00_environment.ipynb` - Verify setup
2. `01_feature_engineering.ipynb` - Download/process data
3. `02_model_training.ipynb` - Train model
4. `03_backtest.ipynb` - Backtest strategy
5. `04_diagnostics.ipynb` - Validate results

## Exporting Notebooks to PDF

To generate PDF versions of all notebooks:

```bash
python export_notebooks_to_pdf.py
```

PDFs will be saved to `notebook_pdfs/` directory.

## Data Files

Large data files are excluded from Git:
- `data/prices.csv` (~66 MB) - Historical price data
- `data/features.csv` (~268 MB) - Engineered features
- `data/histdata_cache/*.zip` - Cached downloads

These can be regenerated using `setup_data.py`. See `data/README.md` for details.

## Testing

Run the test suite:

```bash
pytest
```

Or use the diagnostics notebook which includes automated tests.

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## License

[Add your license here]

