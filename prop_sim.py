"""
Prop-firm rules simulator driven by a firm profile (see profiles/*.yaml).

A strategy is summarised as a list of TRADING DAYS, each with two numbers per ONE micro contract:
    pnl : end-of-day profit/loss in USD
    mae : worst running profit/loss reached during that day in USD (always <= min(pnl, 0))
The simulator scales both by the number of micros, checks the trailing floor against the
intraday worst point (the firm enforces it in real time), and applies the firm's rules.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import yaml

EPS = 1e-9


def load_profile(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _active(pnl: float, mae: float) -> bool:
    """A day with no trades (all zeros) is not a trading day."""
    return not (pnl == 0 and mae == 0)


def simulate_eval(days, prof: dict, micros: int, max_days: int | None = None) -> dict:
    e = prof["evaluation"]
    start = prof["account"]["start_balance"]
    dd = e["max_drawdown"]
    lock_buf = prof["drawdown"]["lock_buffer"]
    micros = min(micros, e["max_micros"])

    bal = peak = float(start)
    floor, locked = start - dd, False
    best, n = 0.0, 0

    for pnl1, mae1 in days:
        if not _active(pnl1, mae1):
            continue
        pnl, mae = pnl1 * micros, mae1 * micros
        if bal + mae <= floor + EPS:                       # touching the floor fails instantly
            return {"result": "breach", "days": n + 1}
        bal += pnl
        n += 1
        best = max(best, pnl)
        if not locked:                                     # EOD ratchet, then lock
            peak = max(peak, bal)
            floor = peak - dd
            if bal >= start + dd + lock_buf:
                floor, locked = start + lock_buf, True
        profit = bal - start
        if (profit >= e["profit_target"] and n >= e["min_trading_days"]
                and best <= e["consistency"] * profit + EPS):
            return {"result": "pass", "days": n}
        if max_days and n >= max_days:
            break
    return {"result": "timeout", "days": n}


def simulate_funded(days, prof: dict, micros: int, cushion: float = 1000.0,
                    max_payouts: int = 3, max_days: int = 250) -> dict:
    """Funded Flex account. `cushion` = minimum room kept above the locked floor AFTER a payout.
    Requesting a payout locks the floor at start+lock_buffer immediately, so paying out too
    early can leave almost no room. The simulator only requests when the cushion holds."""
    f = prof["funded_flex"]
    start = prof["account"]["start_balance"]
    dd = f["max_drawdown"]
    lock_buf = prof["drawdown"]["lock_buffer"]
    locked_floor = start + lock_buf

    bal = peak = float(start)
    floor, locked = start - dd, False
    tier = f["start_micros"]
    wins, cycle_start, first = 0, float(start), True
    received, payouts, n = 0.0, 0, 0

    for pnl1, mae1 in days:
        if not _active(pnl1, mae1):
            continue
        for thr, m in f["scaling"]:                        # limits step up on prior EOD equity
            if bal - start >= thr:
                tier = max(tier, m)
        c = min(micros, tier)
        pnl, mae = pnl1 * c, mae1 * c
        if bal + mae <= floor + EPS:
            return {"result": "breach", "payouts": payouts, "received": received, "days": n + 1}
        bal += pnl
        n += 1
        if pnl >= f["winning_day_threshold"]:
            wins += 1
        if not locked:
            peak = max(peak, bal)
            floor = peak - dd
            if bal >= start + dd + lock_buf:
                floor, locked = locked_floor, True

        if wins >= f["winning_days_per_payout"]:
            amt = min(f["payout_fraction"] * (bal - start), f["payout_cap"])
            eligible = first or (bal - cycle_start) > 0
            if eligible and amt >= f["min_payout"] and (bal - amt) - locked_floor >= cushion:
                floor, locked = locked_floor, True         # lock happens at the request
                bal -= amt
                received += amt * f["profit_split_trader"]
                payouts += 1
                wins, cycle_start, first = 0, bal, False
                if payouts >= max_payouts:
                    return {"result": "done", "payouts": payouts, "received": received, "days": n}
        if n >= max_days:
            break
    return {"result": "timeout", "payouts": payouts, "received": received, "days": n}


def _block_bootstrap(pnl, mae, rng, block):
    n = len(pnl)
    block = min(block, n)
    while True:
        s = int(rng.integers(0, n - block + 1))
        for k in range(block):
            yield float(pnl[s + k]), float(mae[s + k])


def _validate(pnl, mae):
    pnl, mae = np.asarray(pnl, float), np.asarray(mae, float)
    if len(pnl) != len(mae) or len(pnl) < 30:
        raise ValueError("Need >= 30 trading days with matching pnl/mae arrays.")
    if np.any(mae > np.minimum(pnl, 0) + EPS):
        raise ValueError("mae must be <= min(pnl, 0) for every day (worst point includes the close).")
    return pnl, mae


def monte_carlo(pnl, mae, prof: dict, micros: int, n_paths: int = 5000, seed: int = 0,
                block: int = 5, eval_max_days: int = 120, cushion: float = 1000.0,
                funded_max_payouts: int = 3) -> dict:
    pnl, mae = _validate(pnl, mae)
    rng = np.random.default_rng(seed)
    ev = {"pass": 0, "breach": 0, "timeout": 0}
    pass_days = []
    for _ in range(n_paths):
        gen = itertools.islice(_block_bootstrap(pnl, mae, rng, block), eval_max_days * 3)
        r = simulate_eval(gen, prof, micros, eval_max_days)
        ev[r["result"]] += 1
        if r["result"] == "pass":
            pass_days.append(r["days"])

    fu_paths = max(1000, n_paths // 2)
    got1, breach_first, recv, npay = 0, 0, [], []
    for _ in range(fu_paths):
        gen = itertools.islice(_block_bootstrap(pnl, mae, rng, block), 750)
        r = simulate_funded(gen, prof, micros, cushion, funded_max_payouts)
        recv.append(r["received"])
        npay.append(r["payouts"])
        if r["payouts"] >= 1:
            got1 += 1
        elif r["result"] == "breach":
            breach_first += 1

    p_pass = ev["pass"] / n_paths
    out = {
        "eval_p_pass": p_pass,
        "eval_p_breach": ev["breach"] / n_paths,
        "eval_p_timeout": ev["timeout"] / n_paths,
        "eval_median_days_to_pass": float(np.median(pass_days)) if pass_days else None,
        "funded_p_first_payout": got1 / fu_paths,
        "funded_p_breach_before_first_payout": breach_first / fu_paths,
        "funded_mean_trader_receipts": float(np.mean(recv)),
        "funded_mean_payouts": float(np.mean(npay)),
    }
    out.update(economics(p_pass, out["funded_mean_trader_receipts"], prof))
    return out


def economics(p_pass: float, expected_receipts: float, prof: dict) -> dict:
    """Expected spend to obtain one funded account vs expected trader receipts from it."""
    fee, reset = prof["costs"]["evaluation_fee"], prof["costs"]["reset_fee"]
    if p_pass <= 0:
        return {"expected_spend_to_pass": float("inf"), "expected_net_per_funded": float("-inf"),
                "breakeven_pass_prob": None}
    spend = fee + (1.0 / p_pass - 1.0) * reset              # geometric attempts: buy once, reset on fails
    denom = expected_receipts - fee + reset
    p_be = (reset / denom) if denom > 0 else None           # pass prob where spend == receipts
    return {"expected_spend_to_pass": spend,
            "expected_net_per_funded": expected_receipts - spend,
            "breakeven_pass_prob": p_be}
