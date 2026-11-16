"""
Turns a backtester's trade list into what the simulator needs, and blocks scalping.

Trade CSV columns (one row per trade, ONE micro contract, costs already deducted):
    entry_time, exit_time, pnl_per_micro, mae_per_micro
mae_per_micro = worst running P&L of that trade (<= min(pnl, 0)).
Times are Eastern Time unless they carry a timezone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["entry_time", "exit_time", "pnl_per_micro", "mae_per_micro"]
ET = "America/New_York"


def load_trades(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    d = df[REQUIRED].copy()
    for col in ("entry_time", "exit_time"):
        t = pd.to_datetime(d[col])
        t = t.dt.tz_localize(ET) if t.dt.tz is None else t.dt.tz_convert(ET)
        d[col] = t.dt.tz_localize(None)                      # ET wall-clock time
    d = d.sort_values("entry_time").reset_index(drop=True)
    if (d["exit_time"] < d["entry_time"]).any():
        raise ValueError("Some trades exit before they enter.")
    if (d["mae_per_micro"] > np.minimum(d["pnl_per_micro"], 0) + 1e-9).any():
        raise ValueError("mae_per_micro must be <= min(pnl_per_micro, 0) on every trade.")
    d["hold_s"] = (d["exit_time"] - d["entry_time"]).dt.total_seconds()
    # Tradeify trading day runs 18:00 ET -> 17:00 ET next day, so +6h puts a session on one date.
    d["session_day"] = (d["entry_time"] + pd.Timedelta(hours=6)).dt.date
    d["exit_session_day"] = (d["exit_time"] + pd.Timedelta(hours=6)).dt.date
    return d


def compliance(d: pd.DataFrame, prof: dict) -> dict:
    """Hard gate. Any issue means the strategy is rejected, not warned about."""
    c, s = prof["conduct"], prof["session"]
    flat_h, flat_m = map(int, s["flat_by"].split(":"))
    own_h, own_m = map(int, s["own_flatten_at"].split(":"))
    exit_clock = d["exit_time"].dt.hour * 60 + d["exit_time"].dt.minute
    # exits in the 16:45-17:59 window are late; exits from 18:00 onward belong to the next session
    late = int(((exit_clock >= flat_h * 60 + flat_m) & (d["exit_time"].dt.hour < 18)).sum())
    cross = int((d["session_day"] != d["exit_session_day"]).sum())
    per_day = d.groupby("session_day").size()
    prev_exit = d.groupby("session_day")["exit_time"].shift(1)
    overlaps = int((d["entry_time"] < prev_exit).sum())

    rep = {
        "n_trades": int(len(d)),
        "pct_over_10s": float((d["hold_s"] > 10).mean()),
        "pct_over_min_hold": float((d["hold_s"] >= c["own_min_hold_seconds"]).mean()),
        "median_hold_min": float(d["hold_s"].median() / 60),
        "max_trades_per_day": int(per_day.max()),
        "late_exits": late,
        "cross_session_trades": cross,
        "overlapping_trades": overlaps,
    }
    issues = []
    if rep["pct_over_min_hold"] < 1.0:
        issues.append(f"Scalping: {int((d['hold_s'] < c['own_min_hold_seconds']).sum())} trades "
                      f"lasted under {c['own_min_hold_seconds']}s. Every trade must last at least that long.")
    if rep["median_hold_min"] < c["own_median_hold_minutes"]:
        issues.append(f"Median hold is {rep['median_hold_min']:.1f} min; "
                      f"minimum is {c['own_median_hold_minutes']} min.")
    if rep["max_trades_per_day"] > c["own_max_trades_per_day"]:
        issues.append(f"Up to {rep['max_trades_per_day']} trades in one day; "
                      f"limit is {c['own_max_trades_per_day']}.")
    if late:
        issues.append(f"{late} trades exit at or after {s['flat_by']} ET. Flatten by {s['own_flatten_at']}.")
    if cross:
        issues.append(f"{cross} trades are held across the daily close. Overnight is forbidden.")
    if overlaps:
        issues.append(f"{overlaps} overlapping trades. One position at a time only.")
    rep["issues"] = issues
    rep["ok"] = not issues
    return rep


def daily_records(d: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day: end-of-day P&L and worst intraday running P&L (per micro)."""
    rows = []
    for day, g in d.groupby("session_day"):
        run = worst = 0.0
        for p, m in zip(g["pnl_per_micro"], g["mae_per_micro"]):
            worst = min(worst, run + m)
            run += p
        worst = min(worst, run)
        rows.append({"day": day, "pnl": run, "mae": worst})
    return pd.DataFrame(rows)
