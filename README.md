# UPRO/SPXU Morning-Momentum Strategy

A deterministic, rule-based trading strategy that exploits early-morning momentum divergence between leveraged ETFs. The strategy enters simultaneous positions in UPRO (3x bull) and SPXU (3x bear) at market open, then closes the losing side after a 5-minute evaluation window.

## Strategy Overview

### Core Mechanism

This is a **duel strategy** where two opposing leveraged ETFs compete for dominance in the first 5 minutes of trading. The strategy captures directional momentum while hedging initial market uncertainty.

**Daily Workflow:**

1. **Morning Cleanup (09:30)** - Close any overnight positions from previous day
2. **PDT Gate Check** - Verify Pattern Day Trading limits allow new positions
3. **Duel Entry (09:30)** - Buy equal dollar amounts of UPRO and SPXU
4. **Winner Detection (09:31-09:35)** - Monitor 5-minute window for 0.5% spread
5. **Close Loser** - Sell underperforming ETF when winner emerges
6. **Close Winner** - Exit remaining position at next day open OR same day close (configurable)

### Key Features

- **Deterministic execution** - No machine learning, fully rule-based
- **PDT-aware** - Tracks rolling 5-day pattern day trade limits
- **Configurable exit modes** - Switch between overnight hold or same-day close
- **Backtested 2010-present** - Historical simulation with realistic slippage
- **SPX proxy simulation** - Uses SPX data to simulate UPRO/SPXU behavior (3x/-3x leverage)

## Project Structure

```
marketOpenMomentum/
├── config.py                       # Strategy parameters (PDT, slippage, exit modes)
├── momentum_lib.py                 # Core backtest engine and duel logic
├── histdata_loader.py              # SPX data downloader (HistData.com)
├── export_notebooks_to_pdf.py      # Notebook PDF export utility
├── notebooks/                      # Analysis workflow
│   ├── 00_environment.ipynb        # Environment setup and configuration
│   ├── 01_feature_engineering.ipynb # SPX data loading and UPRO/SPXU simulation
│   ├── 02_model_training.ipynb     # Placeholder (no ML required)
│   ├── 03_backtest.ipynb           # Complete backtest with performance stats
│   └── 04_diagnostics.ipynb        # Data validation and determinism tests
└── data/                           # Generated data (excluded from Git)
    ├── prices_YYYY.csv             # Year-based SPX price files
    ├── upro_prices.csv             # Simulated UPRO prices
    ├── spxu_prices.csv             # Simulated SPXU prices
    ├── equity_curve.csv            # Backtest equity curve
    └── trade_log.csv               # Complete trade history
```

## Configuration

All strategy parameters are centralized in `config.py`:

**Position Sizing:**
- `POSITION_SIZE_PCT` - Fraction of cash to allocate (default: 0.95)

**Winner Detection:**
- `SPREAD_THRESHOLD` - Minimum spread to declare winner (default: 0.005 = 0.5%)
- Winner window is fixed at 5 minutes (09:31-09:35)

**Exit Modes:**
- `EXIT_MODE = "next_open"` - Hold winner overnight, exit at next day 09:30 (1 day trade)
- `EXIT_MODE = "same_close"` - Exit winner at 16:00 same day (2 day trades)

**PDT Rules:**
- `USE_PDT_LIMITS` - Enable/disable PDT tracking (default: True)
- `PDT_THRESHOLD_EQUITY` - Equity level where PDT rules apply (default: $25,000)
- `PDT_MAX_TRADES` - Max day trades in rolling 5 business days (default: 3)

**Trading Costs:**
- `SLIPPAGE_BPS` - Slippage per trade in basis points (default: 5 bps)

**Backtest Parameters:**
- `START_CAPITAL` - Starting equity for simulation (default: $10,000)
- `START_YEAR` - Beginning year for backtest (default: 2010)
- `UPRO_LEVERAGE` / `SPXU_LEVERAGE` - Leverage multipliers (3.0 / -3.0)

## Installation

### Requirements

- Python 3.10+
- Jupyter Notebook/Lab
- Internet connection for initial SPX data download

### Setup

```bash
# Clone repository
git clone <repository_url>
cd marketOpenMomentum

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run Notebooks in Sequence

Execute notebooks in order to generate data and run backtest:

```bash
jupyter notebook notebooks/
```

**Notebook Workflow:**

1. **00_environment.ipynb** - Load strategy configuration and verify setup
2. **01_feature_engineering.ipynb** - Download SPX data, simulate UPRO/SPXU, filter to regular hours
3. **02_model_training.ipynb** - Placeholder (strategy is rule-based, no training needed)
4. **03_backtest.ipynb** - Run complete backtest with equity curves and performance metrics
5. **04_diagnostics.ipynb** - Validate data quality, PDT logic, and determinism

### 2. Initial Data Download

On first run, `01_feature_engineering.ipynb` will:
- Download SPX/USD 1-minute data from HistData.com (2010-present, parallel downloads)
- Save as year-based CSV files (`prices_2010.csv`, `prices_2011.csv`, etc.)
- Cache ZIP files in `data/histdata_cache/` for faster re-downloads
- Simulate UPRO/SPXU from SPX data using 3x/-3x leverage
- Filter to regular market hours (09:30-16:00 ET)

**Note:** Initial download uses parallel workers for speed. Subsequent runs use cached files.

### 3. Analyze Results

After running `03_backtest.ipynb`, review:
- **Equity curve** - Visual performance over time
- **Trade log** - Complete record of all entries/exits (saved to `data/trade_log.csv`)
- **Performance stats** - CAGR, Sharpe ratio, max drawdown, MAR ratio
- **Trade breakdown** - P&L by role (entry, cleanup, loser, winner)

### 4. Validate Determinism

Run `04_diagnostics.ipynb` to verify:
- No NaN values or duplicate timestamps
- UPRO/SPXU data properly aligned
- Winner window timestamps exist for all trading days
- PDT tracker correctly enforces limits
- Backtest produces identical results on re-run (regression test)

## Data Management

### SPX Data Source

Historical SPX/USD data comes from HistData.com:
- Free 1-minute OHLCV bars
- Available from 2010 onwards (some years may have limited data)
- Saved as year-based CSV files to keep file sizes manageable
- Parallel downloads for faster initial setup

### Data Files (Excluded from Git)

Large data files are excluded via `.gitignore`:
- `data/prices_*.csv` - Year-based SPX price files (~10-20 MB each)
- `data/upro_prices.csv` - Simulated UPRO prices
- `data/spxu_prices.csv` - Simulated SPXU prices
- `data/equity_curve.csv` - Backtest equity curve
- `data/trade_log.csv` - Trade history
- `data/histdata_cache/*.zip` - Cached downloads

**To regenerate:** Simply delete files and re-run notebooks.

## Performance Metrics

The backtest calculates:

- **Total Return** - Cumulative percentage gain/loss
- **CAGR** - Compound Annual Growth Rate
- **Sharpe Ratio** - Risk-adjusted returns (annualized)
- **Max Drawdown** - Largest peak-to-trough decline
- **Win Rate** - Percentage of profitable bars
- **MAR Ratio** - CAGR divided by absolute max drawdown

## Testing

### Automated Validation

Run `04_diagnostics.ipynb` to execute:
- Data quality checks (no NaNs, duplicates, alignment)
- Timestamp validation (09:30-09:35 window exists)
- PDT tracker unit tests
- Determinism regression tests (identical results on re-run)

### Manual Testing

Test configuration changes by modifying `config.py` and re-running backtest:

```python
# Example: Test higher PDT limit
from config import default_config
default_config.pdt_max_trades = 5
```

## Export Notebooks to PDF

Generate PDF documentation of all notebooks:

```bash
python export_notebooks_to_pdf.py
```

PDFs are saved to `notebook_pdfs/` directory with execution results included.

## Architecture Notes

### Data Flow

1. **SPX Download** - `histdata_loader.py` fetches and caches raw ZIP files
2. **CSV Conversion** - Extracts to year-based CSV files for faster loading
3. **ETF Simulation** - `momentum_lib.simulate_leveraged_etfs()` creates UPRO/SPXU from SPX
4. **Regular Hours Filter** - Removes pre-market and after-hours data
5. **Backtest Engine** - `momentum_lib.run_backtest()` executes strategy logic
6. **Results Export** - Equity curve and trade log saved as CSV

### Backtest Engine

The `run_backtest()` function in `momentum_lib.py` implements:
- **Morning cleanup** - Closes overnight positions
- **PDT tracking** - Rolling 5-day counter with equity threshold
- **Duel entry** - Calculates whole shares for equal dollar allocation
- **Winner window** - Monitors 5-minute spread with deterministic timestamps
- **Execution** - Applies configurable slippage (buy at close × 1.0005, sell at close × 0.9995)
- **Exit logic** - Handles both next-day and same-day exit modes

### Determinism

All backtest operations are fully deterministic:
- Fixed timestamps (09:30:00, 09:31:00, ..., 09:35:00)
- Consistent slippage application
- Whole share integer rounding
- Reproducible random state (where applicable)

Running the same backtest twice produces identical equity curves and trade logs.

## Limitations

### SPX as Proxy

Real UPRO/SPXU prices differ from simulated values due to:
- Compounding effects (leveraged ETFs reset daily)
- Tracking error and fees
- Bid-ask spreads not captured in simulation

For production use, replace simulation with actual UPRO/SPXU historical data.

### Data Availability

HistData.com provides SPX/USD data from 2010 onwards. Some earlier years may have limited or missing data. The loader will skip unavailable years gracefully.

### Slippage Model

Current slippage is a simple percentage offset. Real slippage varies by:
- Market conditions (volatility)
- Order size relative to average volume
- Bid-ask spread dynamics

Adjust `SLIPPAGE_BPS` in `config.py` based on live trading experience.

## Future Enhancements

Potential improvements:
- Live trading integration via Alpaca API
- Real UPRO/SPXU price data (replace simulation)
- Dynamic slippage based on volatility
- Multiple timeframe analysis
- Stop-loss and take-profit rules
- Position sizing based on equity
- Risk parity allocation adjustments

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading leveraged ETFs involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own due diligence before trading real capital.
