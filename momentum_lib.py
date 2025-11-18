"""
Core logic for UPRO/SPXU Morning-Momentum Strategy.

This module implements the complete duel strategy including:
- SPX to UPRO/SPXU simulation (3x/-3x leverage)
- Morning cleanup logic
- PDT tracking and gating
- Duel entry with equal position sizing
- 5-minute winner detection window
- Loser/winner exit logic
- Deterministic execution with slippage
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal, Optional

# Prevent __pycache__ creation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from config import StrategyConfig, default_config


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================


def load_spx_data(csv_path: Path) -> pd.DataFrame:
    """
    Load SPX 1-minute price data from CSV (legacy function for single file).

    Returns DataFrame with DatetimeIndex (ET timezone) and 'close' column.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")

    # Ensure timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")

    # Keep only 'SPX' or 'close' column
    if "SPX" in df.columns:
        df = df.rename(columns={"SPX": "close"})

    return df[["close"]].sort_index()


def load_prices_from_data_dir(data_dir: Path, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Load price data from year-based CSV files (prices_YYYY.csv).
    
    This is a convenience wrapper that uses histdata_loader.load_prices_by_year.
    
    Args:
        data_dir: Directory containing price files
        start_year: First year to load (inclusive)
        end_year: Last year to load (inclusive)
    
    Returns:
        Combined DataFrame with all years, sorted by timestamp
    """
    from histdata_loader import load_prices_by_year
    return load_prices_by_year(data_dir, start_year=start_year, end_year=end_year)


def simulate_leveraged_etfs(spx_df: pd.DataFrame, config: StrategyConfig = default_config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate UPRO (3x long) and SPXU (-3x short) from SPX minute data.

    Uses simple daily return scaling to approximate leveraged ETF behavior.
    Returns (upro_df, spxu_df) with matching indexes and 'close' columns.
    """
    spx = spx_df["close"].copy()

    # Get daily returns for SPX
    daily_spx = spx.resample("D").last()
    daily_ret = daily_spx.pct_change(fill_method=None).fillna(0)

    # Scale returns by leverage
    upro_daily_ret = daily_ret * config.upro_leverage
    spxu_daily_ret = daily_ret * config.spxu_leverage

    # Build cumulative price series starting at SPX initial price
    start_price = spx.iloc[0]
    upro_daily = start_price * (1 + upro_daily_ret).cumprod()
    spxu_daily = start_price * (1 + spxu_daily_ret).cumprod()

    # Broadcast daily prices to minute bars (forward fill)
    upro_min = upro_daily.reindex(spx.index, method="ffill")
    spxu_min = spxu_daily.reindex(spx.index, method="ffill")

    # Add intraday noise based on SPX intraday moves (scaled by leverage)
    spx_daily = daily_spx.reindex(spx.index, method="ffill")
    intraday_ret = spx / spx_daily - 1

    upro_intraday = upro_min * (1 + intraday_ret * config.upro_leverage)
    spxu_intraday = spxu_min * (1 + intraday_ret * config.spxu_leverage)

    upro_df = pd.DataFrame({"close": upro_intraday})
    spxu_df = pd.DataFrame({"close": spxu_intraday})

    return upro_df, spxu_df


def filter_regular_hours(df: pd.DataFrame, market_open: str = "09:30:00", market_close: str = "16:00:00") -> pd.DataFrame:
    """Filter DataFrame to regular trading hours only (09:30-16:00 ET)."""
    open_time = pd.to_datetime(market_open).time()
    close_time = pd.to_datetime(market_close).time()

    mask = (df.index.time >= open_time) & (df.index.time <= close_time)
    return df[mask]


# ============================================================================
# PDT TRACKING
# ============================================================================


class PDTTracker:
    """
    Track day trades over rolling 5 business days.

    A day trade is defined as opening and closing a position on the same day.
    """

    def __init__(self, max_trades: int = 3):
        self.max_trades = max_trades
        self.trades: deque = deque(maxlen=5)  # Store (date, count) for last 5 business days

    def add_day_trade(self, date: pd.Timestamp, count: int = 1) -> None:
        """Record day trade(s) for a given date."""
        # Check if date already exists
        for i, (d, c) in enumerate(self.trades):
            if d == date.date():
                self.trades[i] = (d, c + count)
                return

        # Add new date
        self.trades.append((date.date(), count))

    def get_count(self, current_date: pd.Timestamp) -> int:
        """Get total day trades in rolling 5 business days ending at current_date."""
        lookback_start = (current_date - BDay(4)).date()

        total = 0
        for date, count in self.trades:
            if date >= lookback_start:
                total += count

        return total

    def can_trade(self, current_date: pd.Timestamp, equity: float, threshold: float = 25000) -> bool:
        """Check if trading is allowed based on PDT rules."""
        if equity >= threshold:
            return True  # PDT rules don't apply

        return self.get_count(current_date) < self.max_trades


# ============================================================================
# POSITION & TRADE TRACKING
# ============================================================================


@dataclass
class Position:
    """Represents a position in UPRO or SPXU."""
    ticker: Literal["UPRO", "SPXU"]
    shares: int
    entry_price: float
    entry_time: pd.Timestamp

    def value(self, current_price: float) -> float:
        """Current market value of position."""
        return self.shares * current_price

    def pnl(self, current_price: float) -> float:
        """Unrealized P&L."""
        return self.shares * (current_price - self.entry_price)


@dataclass
class Trade:
    """Record of a completed trade."""
    ticker: Literal["UPRO", "SPXU"]
    side: Literal["buy", "sell"]
    shares: int
    price: float
    timestamp: pd.Timestamp
    pnl: float = 0.0
    role: Literal["entry", "cleanup", "loser", "winner"] = "entry"


# ============================================================================
# BACKTEST ENGINE
# ============================================================================


@dataclass
class BacktestState:
    """State variables for backtest simulation."""
    equity: float
    cash: float
    positions: dict[Literal["UPRO", "SPXU"], Position | None] = field(default_factory=lambda: {"UPRO": None, "SPXU": None})
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    pdt_tracker: PDTTracker = field(default_factory=PDTTracker)

    # Duel state
    duel_active: bool = False
    duel_entry_time: pd.Timestamp | None = None
    winner: Literal["UPRO", "SPXU"] | None = None
    winner_exit_time: pd.Timestamp | None = None


def execute_trade(
    state: BacktestState,
    ticker: Literal["UPRO", "SPXU"],
    side: Literal["buy", "sell"],
    shares: int,
    price: float,
    timestamp: pd.Timestamp,
    config: StrategyConfig,
    role: Literal["entry", "cleanup", "loser", "winner"] = "entry",
) -> None:
    """Execute a trade and update state."""
    if shares == 0:
        return

    # Apply slippage
    fill_price = price * config.get_slippage_multiplier(side)

    if side == "buy":
        cost = shares * fill_price
        state.cash -= cost
        state.positions[ticker] = Position(ticker, shares, fill_price, timestamp)
        pnl = 0.0
    else:  # sell
        proceeds = shares * fill_price
        state.cash += proceeds

        # Calculate P&L
        if state.positions[ticker]:
            pnl = shares * (fill_price - state.positions[ticker].entry_price)
        else:
            pnl = 0.0

        state.positions[ticker] = None

    # Record trade
    trade = Trade(ticker, side, shares, fill_price, timestamp, pnl, role)
    state.trades.append(trade)

    # Update equity
    state.equity = state.cash
    for pos in state.positions.values():
        if pos:
            state.equity += pos.value(price)  # Use market price for equity calc


def morning_cleanup(
    state: BacktestState,
    upro_price: float,
    spxu_price: float,
    timestamp: pd.Timestamp,
    config: StrategyConfig,
) -> None:
    """
    Close any existing positions at market open (09:30).
    This ensures we start each day clean.
    """
    for ticker in ["UPRO", "SPXU"]:
        pos = state.positions[ticker]
        if pos:
            price = upro_price if ticker == "UPRO" else spxu_price
            execute_trade(state, ticker, "sell", pos.shares, price, timestamp, config, role="cleanup")


def check_pdt_and_enter_duel(
    state: BacktestState,
    upro_price: float,
    spxu_price: float,
    timestamp: pd.Timestamp,
    config: StrategyConfig,
) -> bool:
    """
    Check PDT limits and enter duel if allowed.

    Returns True if duel was entered, False if skipped.
    """
    # PDT check
    if config.use_pdt_limits:
        can_trade = state.pdt_tracker.can_trade(timestamp, state.equity, config.pdt_threshold_equity)
        if not can_trade:
            return False

    # Calculate position sizes
    position_value = state.cash * config.position_size_pct
    value_per_leg = position_value / 2

    upro_shares = int(value_per_leg / upro_price)
    spxu_shares = int(value_per_leg / spxu_price)

    # Skip if either side is 0 shares
    if upro_shares == 0 or spxu_shares == 0:
        return False

    # Enter duel
    execute_trade(state, "UPRO", "buy", upro_shares, upro_price, timestamp, config, role="entry")
    execute_trade(state, "SPXU", "buy", spxu_shares, spxu_price, timestamp, config, role="entry")

    state.duel_active = True
    state.duel_entry_time = timestamp
    state.winner = None

    return True


def check_winner_window(
    state: BacktestState,
    upro_price: float,
    spxu_price: float,
    timestamp: pd.Timestamp,
    config: StrategyConfig,
) -> bool:
    """
    Check if winner can be declared based on 5-minute window.

    Returns True if loser was closed, False otherwise.
    """
    if not state.duel_active or state.winner is not None:
        return False

    upro_pos = state.positions["UPRO"]
    spxu_pos = state.positions["SPXU"]

    if not upro_pos or not spxu_pos:
        return False

    # Calculate returns
    ret_upro = (upro_price - upro_pos.entry_price) / upro_pos.entry_price
    ret_spxu = (spxu_price - spxu_pos.entry_price) / spxu_pos.entry_price
    spread = abs(ret_upro - ret_spxu)

    # Check time window (within 5 minutes of entry)
    minutes_elapsed = (timestamp - state.duel_entry_time).total_seconds() / 60

    if minutes_elapsed <= 5:
        # Check if spread threshold met
        if spread >= config.spread_threshold:
            # Declare winner
            state.winner = "UPRO" if ret_upro > ret_spxu else "SPXU"
            loser = "SPXU" if state.winner == "UPRO" else "UPRO"

            # Close loser
            loser_pos = state.positions[loser]
            loser_price = spxu_price if loser == "SPXU" else upro_price
            execute_trade(state, loser, "sell", loser_pos.shares, loser_price, timestamp, config, role="loser")

            # Set winner exit time
            if config.exit_mode == "next_open":
                state.winner_exit_time = None  # Will exit next day at open
            else:  # same_close
                # Exit at 16:00 today
                exit_time = timestamp.replace(hour=16, minute=0, second=0)
                state.winner_exit_time = exit_time

            return True

    elif minutes_elapsed > 5 and state.winner is None:
        # End of window - declare leader as winner
        state.winner = "UPRO" if ret_upro > ret_spxu else "SPXU"
        loser = "SPXU" if state.winner == "UPRO" else "UPRO"

        # Close loser at current bar
        loser_pos = state.positions[loser]
        loser_price = spxu_price if loser == "SPXU" else upro_price
        execute_trade(state, loser, "sell", loser_pos.shares, loser_price, timestamp, config, role="loser")

        # Set winner exit time
        if config.exit_mode == "next_open":
            state.winner_exit_time = None
        else:
            exit_time = timestamp.replace(hour=16, minute=0, second=0)
            state.winner_exit_time = exit_time

        return True

    return False


def check_winner_exit(
    state: BacktestState,
    upro_price: float,
    spxu_price: float,
    timestamp: pd.Timestamp,
    config: StrategyConfig,
) -> bool:
    """
    Check if winner should be exited.

    Returns True if winner was closed, False otherwise.
    """
    if state.winner is None:
        return False

    winner_pos = state.positions[state.winner]
    if not winner_pos:
        return False  # Already closed

    should_exit = False

    if config.exit_mode == "same_close":
        # Exit at 16:00 today
        if timestamp.time() >= time(16, 0):
            should_exit = True
    else:  # next_open
        # Exit at next day 09:30
        if timestamp.date() > state.duel_entry_time.date() and timestamp.time() >= time(9, 30):
            should_exit = True

    if should_exit:
        winner_price = upro_price if state.winner == "UPRO" else spxu_price
        execute_trade(state, state.winner, "sell", winner_pos.shares, winner_price, timestamp, config, role="winner")

        # Update PDT tracker
        day_trades = 2 if config.exit_mode == "same_close" else 1
        state.pdt_tracker.add_day_trade(state.duel_entry_time, day_trades)

        # Reset duel state
        state.duel_active = False
        state.winner = None
        state.duel_entry_time = None

        return True

    return False


def run_backtest(
    upro_df: pd.DataFrame,
    spxu_df: pd.DataFrame,
    config: StrategyConfig = default_config,
) -> tuple[BacktestState, pd.DataFrame, pd.DataFrame]:
    """
    Run complete backtest of UPRO/SPXU morning momentum strategy.

    Returns:
        - BacktestState with final state
        - equity_curve DataFrame
        - trades_df DataFrame
    """
    # Initialize state
    state = BacktestState(
        equity=config.start_capital,
        cash=config.start_capital,
        pdt_tracker=PDTTracker(max_trades=config.pdt_max_trades),
    )

    # Align data
    combined = pd.DataFrame({
        "upro": upro_df["close"],
        "spxu": spxu_df["close"],
    }).dropna()

    # Ensure index is DatetimeIndex
    if not isinstance(combined.index, pd.DatetimeIndex):
        combined.index = pd.to_datetime(combined.index, utc=True)
        if combined.index.tz is not None:
            combined.index = combined.index.tz_convert("America/New_York")
        else:
            combined.index = combined.index.tz_localize("America/New_York")

    # Group by date for daily processing
    for date, day_data in combined.groupby(combined.index.date):
        # Get 09:30 bar
        open_bars = day_data[day_data.index.time == time(9, 30)]
        if open_bars.empty:
            continue

        open_bar = open_bars.iloc[0]
        open_time = open_bar.name
        upro_open = open_bar["upro"]
        spxu_open = open_bar["spxu"]

        # 1. Morning cleanup
        morning_cleanup(state, upro_open, spxu_open, open_time, config)

        # 2. Enter duel (if PDT allows)
        entered = check_pdt_and_enter_duel(state, upro_open, spxu_open, open_time, config)

        # 3. Process rest of day minute by minute
        for timestamp, row in day_data.iterrows():
            upro_price = row["upro"]
            spxu_price = row["spxu"]

            # Check winner window
            check_winner_window(state, upro_price, spxu_price, timestamp, config)

            # Check winner exit
            check_winner_exit(state, upro_price, spxu_price, timestamp, config)

            # Update equity curve
            current_equity = state.cash
            for pos in state.positions.values():
                if pos:
                    price = upro_price if pos.ticker == "UPRO" else spxu_price
                    current_equity += pos.value(price)

            state.equity_curve.append((timestamp, current_equity))

    # Convert to DataFrames
    equity_df = pd.DataFrame(state.equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")

    trades_df = pd.DataFrame([
        {
            "timestamp": t.timestamp,
            "ticker": t.ticker,
            "side": t.side,
            "shares": t.shares,
            "price": t.price,
            "pnl": t.pnl,
            "role": t.role,
        }
        for t in state.trades
    ])

    return state, equity_df, trades_df


def calculate_performance_stats(equity_curve: pd.DataFrame) -> dict:
    """Calculate performance statistics from equity curve."""
    returns = equity_curve["equity"].pct_change().dropna()

    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    # Total return
    total_return = (equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0]) - 1

    # CAGR
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Sharpe (annualized, assuming 252 trading days)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Max drawdown
    cummax = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - cummax) / cummax
    max_drawdown = drawdown.min()

    # Win rate (from returns)
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "mar": cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0,
    }


__all__ = [
    "load_spx_data",
    "simulate_leveraged_etfs",
    "filter_regular_hours",
    "PDTTracker",
    "Position",
    "Trade",
    "BacktestState",
    "run_backtest",
    "calculate_performance_stats",
]