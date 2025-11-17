"""
Configuration for UPRO/SPXU Morning-Momentum Strategy.

This strategy simulates leveraged ETF behavior using SPX as a proxy:
- UPRO simulated as 3x SPX daily returns
- SPXU simulated as -3x SPX daily returns

All parameters are deterministic and consistent across backtest and live trading.
"""

from dataclasses import dataclass
from typing import Literal

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

# Position sizing (fraction of available cash)
POSITION_SIZE_PCT = 0.95  # Use 95% of cash for the duel

# Winner detection threshold
SPREAD_THRESHOLD = 0.005  # 0.5% spread between UPRO/SPXU to declare winner

# Winner exit mode
ExitMode = Literal["next_open", "same_close"]
EXIT_MODE: ExitMode = "next_open"  # Default for <$25k accounts

# PDT (Pattern Day Trader) enforcement
USE_PDT_LIMITS = True  # Set False to disable PDT checks
PDT_THRESHOLD_EQUITY = 25000  # Equity threshold for PDT rule
PDT_MAX_TRADES = 3  # Max day trades in rolling 5 business days

# Trading costs
SLIPPAGE_BPS = 5  # 5 basis points (0.05%) slippage per trade

# Starting capital for backtest
START_CAPITAL = 10000  # $10k starting equity

# Leverage multipliers for simulated ETFs
UPRO_LEVERAGE = 3.0   # 3x long SPX
SPXU_LEVERAGE = -3.0  # 3x short SPX

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Historical data range
START_YEAR = 2010  # Begin backtest from 2010
END_YEAR = None    # None = current year

# Market hours (Eastern Time)
MARKET_OPEN = "09:30:00"
MARKET_CLOSE = "16:00:00"

# Winner detection window (5 minutes from entry)
WINNER_CHECK_MINUTES = [1, 2, 3, 4, 5]  # Check at 09:31, 09:32, 09:33, 09:34, 09:35

# File paths
DATA_DIR = "data"
SPX_PRICES_FILE = "spx_prices.csv"
UPRO_PRICES_FILE = "upro_prices.csv"
SPXU_PRICES_FILE = "spxu_prices.csv"
TRADE_LOG_FILE = "trade_log.csv"
EQUITY_CURVE_FILE = "equity_curve.csv"


@dataclass
class StrategyConfig:
    """Dataclass for strategy configuration."""

    # Position sizing
    position_size_pct: float = POSITION_SIZE_PCT

    # Entry/exit rules
    spread_threshold: float = SPREAD_THRESHOLD
    exit_mode: ExitMode = EXIT_MODE

    # PDT settings
    use_pdt_limits: bool = USE_PDT_LIMITS
    pdt_threshold_equity: float = PDT_THRESHOLD_EQUITY
    pdt_max_trades: int = PDT_MAX_TRADES

    # Costs
    slippage_bps: float = SLIPPAGE_BPS

    # Capital
    start_capital: float = START_CAPITAL

    # Leverage
    upro_leverage: float = UPRO_LEVERAGE
    spxu_leverage: float = SPXU_LEVERAGE

    # Data range
    start_year: int = START_YEAR
    end_year: int | None = END_YEAR

    def get_slippage_multiplier(self, side: Literal["buy", "sell"]) -> float:
        """Get the slippage multiplier for a trade."""
        if side == "buy":
            return 1 + (self.slippage_bps / 10000)
        else:  # sell
            return 1 - (self.slippage_bps / 10000)


# Default configuration instance
default_config = StrategyConfig()


def get_config() -> StrategyConfig:
    """Get the default strategy configuration."""
    return default_config


__all__ = [
    "StrategyConfig",
    "default_config",
    "get_config",
    "POSITION_SIZE_PCT",
    "SPREAD_THRESHOLD",
    "EXIT_MODE",
    "USE_PDT_LIMITS",
    "PDT_THRESHOLD_EQUITY",
    "PDT_MAX_TRADES",
    "SLIPPAGE_BPS",
    "START_CAPITAL",
    "UPRO_LEVERAGE",
    "SPXU_LEVERAGE",
    "START_YEAR",
    "END_YEAR",
    "MARKET_OPEN",
    "MARKET_CLOSE",
    "WINNER_CHECK_MINUTES",
]