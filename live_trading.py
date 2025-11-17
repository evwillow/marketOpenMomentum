"""
Production-oriented helper for running the dual-leg UPRO/SPXU session.
The notebook `04_live_trading.ipynb` imports this module to keep itself lightweight.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import alpaca_trade_api as tradeapi
import pytz

from momentum_lib import bootstrap_env


@dataclass
class LiveMomentumTrader:
    api: tradeapi.REST
    timezone: str = "America/New_York"
    position_size_pct: float = 0.95
    wait_minutes: int = 5
    spread_threshold: float = 0.005

    def __post_init__(self):
        self.tz = pytz.timezone(self.timezone)

    def ny_now(self) -> datetime:
        return datetime.now(self.tz)

    def wait_until(self, target_time: datetime):
        while True:
            now = self.ny_now()
            if now >= target_time:
                break
            remaining = (target_time - now).total_seconds()
            sleep_window = max(1.0, min(30.0, remaining / 2))
            print(f"Waiting for market open ({remaining/60:.1f} min remaining)")
            time.sleep(sleep_window)

    def market_is_open(self) -> bool:
        try:
            clk = self.api.get_clock()
            return bool(getattr(clk, "is_open", False))
        except Exception as exc:
            print(f"get_clock error: {exc}")
            return False

    def latest_price(self, symbol: str, tries: int = 5) -> float:
        last_err = None
        for i in range(tries):
            try:
                trade = self.api.get_latest_trade(symbol)
                price = float(getattr(trade, "price", getattr(trade, "p")))
                if price > 0:
                    return price
            except Exception as exc:  # pragma: no cover network
                last_err = exc
            time.sleep(0.25 + 0.15 * i)
        raise RuntimeError(f"Unable to fetch quote for {symbol}: {last_err}")

    def wait_filled(self, order_id: str, timeout: int = 60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                order = self.api.get_order(order_id)
                if order.status == "filled":
                    return order
            except Exception:
                pass
            time.sleep(0.35)
        raise RuntimeError(f"Order {order_id} did not fill within {timeout}s")

    def pdt_can_trade(self) -> bool:
        try:
            acct = self.api.get_account()
            equity = float(acct.equity)
            daytrades = int(getattr(acct, "daytrade_count", 0))
            if equity < 25_000 and daytrades >= 3:
                print(f"PDT guard triggered: equity={equity:.2f}, daytrades={daytrades}")
                return False
            return True
        except Exception as exc:
            print(f"PDT check failed: {exc}")
            return False

    def cash_only_allocation(self) -> float:
        acct = self.api.get_account()
        cash = float(acct.cash)
        duel_notional = cash * self.position_size_pct
        return max(0.0, duel_notional / 2.0)

    def can_place_opg_now(self) -> bool:
        now = self.ny_now()
        hhmm = now.hour * 100 + now.minute
        return hhmm <= 928 or hhmm >= 1900

    def existing_opg_sell(self, symbol: str) -> bool:
        try:
            opens = self.api.list_orders(status="open", direction="desc")
            for order in opens:
                if (
                    order.symbol == symbol
                    and getattr(order, "time_in_force", "") == "opg"
                    and order.side == "sell"
                ):
                    return True
        except Exception:
            pass
        return False

    def buy_whole_shares(self, symbol: str, alloc_cash: float):
        price = self.latest_price(symbol)
        qty = int(alloc_cash // price)
        if qty < 1:
            print(f"Insufficient cash to buy {symbol} (need at least ${price:.2f}).")
            return None
        for _ in range(3):
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day",
                )
                print(f"Submitted market buy: {symbol} x{qty} (id={order.id})")
                return order
            except Exception as exc:
                qty -= 1
                print(f"Buy failure for {symbol}: {exc}. Retrying with qty={qty}.")
                if qty < 1:
                    break
                time.sleep(0.4)
        print(
            f"Could not purchase whole shares of {symbol} with allocation ${alloc_cash:,.2f}."
        )
        return None

    def morning_cleanup_if_any(self) -> bool:
        try:
            positions = {
                p.symbol: p
                for p in self.api.list_positions()
                if p.symbol in ("UPRO", "SPXU")
            }
            if not positions:
                return False
            print(
                f"Positions detected from prior session: "
                f"{ {s: p.qty for s, p in positions.items()} }"
            )
            if self.market_is_open():
                for symbol, pos in positions.items():
                    qty_str = pos.qty
                    entry_px = float(pos.avg_entry_price)
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=qty_str,
                        side="sell",
                        type="market",
                        time_in_force="day",
                    )
                    print(
                        f"Cleanup sell submitted for {symbol} x{qty_str} (id={order.id})"
                    )
                    try:
                        fill = self.wait_filled(order.id)
                        sell_px = float(fill.filled_avg_price)
                        qty = float(qty_str)
                        pnl = (sell_px - entry_px) * qty
                        pnl_pct = (sell_px / entry_px - 1.0) * 100.0
                        print(f"Cleanup P/L for {symbol}: ${pnl:,.2f} ({pnl_pct:.2f}%)")
                    except Exception as exc:
                        print(f"Cleanup confirmation failed for {symbol}: {exc}")
            else:
                if self.can_place_opg_now():
                    for symbol, pos in positions.items():
                        if self.existing_opg_sell(symbol):
                            print(
                                f"OPG cleanup already pending for {symbol}; skipping duplicate request."
                            )
                            continue
                        qty_str = pos.qty
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty_str,
                            side="sell",
                            type="market",
                            time_in_force="opg",
                        )
                        print(
                            f"Scheduled OPG cleanup for {symbol} x{qty_str} (id={order.id})"
                        )
                else:
                    open_time = self.ny_now().replace(
                        hour=9, minute=30, second=0, microsecond=0
                    )
                    print("Waiting for open to submit cleanup market orders...")
                    self.wait_until(open_time)
                    return self.morning_cleanup_if_any()
            print("Cleanup finished; exiting to avoid overlapping trades.")
            return True
        except Exception as exc:
            print(f"Cleanup check failed: {exc}")
            return False

    def run_session(self):
        if self.morning_cleanup_if_any():
            return

        open_time = self.ny_now().replace(hour=9, minute=30, second=0, microsecond=0)
        self.wait_until(open_time)
        if not self.market_is_open():
            print("Market not open (holiday/half-day/clock issue). Exiting.")
            return

        if not self.pdt_can_trade():
            return

        allocation = self.cash_only_allocation()
        if allocation <= 0:
            print("No cash available for entries.")
            return

        self.buy_whole_shares("SPXU", allocation)
        time.sleep(1.5)
        self.buy_whole_shares("UPRO", allocation)

        time.sleep(5)

        upro_entry = self.latest_price("UPRO")
        spxu_entry = self.latest_price("SPXU")
        print(f"Entry quotes: UPRO={upro_entry:.4f}, SPXU={spxu_entry:.4f}")

        cutoff = self.ny_now() + timedelta(minutes=self.wait_minutes)
        winner = None
        while self.ny_now() < cutoff and winner is None:
            upro_now = self.latest_price("UPRO")
            spxu_now = self.latest_price("SPXU")
            upro_ret = (upro_now - upro_entry) / upro_entry
            spxu_ret = (spxu_now - spxu_entry) / spxu_entry
            spread = abs(upro_ret - spxu_ret)
            print(
                f"Tick | UPRO={upro_ret:.4%} SPXU={spxu_ret:.4%} spread={spread:.4%}"
            )
            if spread >= self.spread_threshold:
                winner = "UPRO" if upro_ret > spxu_ret else "SPXU"
                break
            time.sleep(5)

        if winner is None:
            upro_now = self.latest_price("UPRO")
            spxu_now = self.latest_price("SPXU")
            winner = (
                "UPRO"
                if (upro_now - upro_entry) / upro_entry
                > (spxu_now - spxu_entry) / spxu_entry
                else "SPXU"
            )

        loser = "SPXU" if winner == "UPRO" else "UPRO"
        print(f"Winner: {winner}. Closing loser: {loser}.")
        self._close_loser_and_schedule_winner(loser, winner)

    def _close_loser_and_schedule_winner(self, loser: str, winner: str):
        try:
            loser_pos = self.api.get_position(loser)
            loser_qty_int = int(float(loser_pos.qty))
            loser_entry_px = float(loser_pos.avg_entry_price)
            if loser_qty_int <= 0:
                print(f"No quantity to close for {loser}.")
            else:
                sell_order = self.api.submit_order(
                    symbol=loser,
                    qty=loser_qty_int,
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
                print(f"Submitted market sell for {loser} x{loser_qty_int} (id={sell_order.id})")
                fill = self.wait_filled(sell_order.id)
                sell_px = float(fill.filled_avg_price)
                pnl = (sell_px - loser_entry_px) * loser_qty_int
                pnl_pct = (sell_px / loser_entry_px - 1.0) * 100.0
                print(f"Realized P/L on {loser}: ${pnl:,.2f} ({pnl_pct:.2f}%)")
        except Exception as exc:
            print(f"Unable to close loser {loser}: {exc}")
            return

        try:
            winner_pos = self.api.get_position(winner)
            winner_qty = int(float(winner_pos.qty))
            if winner_qty > 0:
                if self.can_place_opg_now():
                    if self.existing_opg_sell(winner):
                        print(f"OPG sell already scheduled for {winner}.")
                    else:
                        opg_order = self.api.submit_order(
                            symbol=winner,
                            qty=winner_qty,
                            side="sell",
                            type="market",
                            time_in_force="opg",
                        )
                        print(
                            f"Scheduled OPG sell for {winner} x{winner_qty} (id={opg_order.id})"
                        )
                else:
                    print("OPG window closed; winner will be handled next session via cleanup.")
            else:
                print(f"No quantity remaining to schedule for {winner}.")
        except Exception as exc:
            print(f"OPG scheduling failed for {winner}: {exc}")
            return

        try:
            hold_qty = int(float(self.api.get_position(winner).qty))
            print(f"Holding overnight: {winner} qty={hold_qty}")
        except Exception:
            print("Winner position not visible yet; check Alpaca portal for holdings.")


def build_trader(env_path: Path = Path(".env"), **kwargs) -> LiveMomentumTrader:
    bootstrap_env(env_path)
    api = tradeapi.REST()
    return LiveMomentumTrader(api=api, **kwargs)


__all__ = ["LiveMomentumTrader", "build_trader"]


