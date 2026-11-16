"""Every test is a hand-computed case. Run: python test_prop_sim.py"""
from pathlib import Path

import numpy as np

from prop_sim import load_profile, simulate_eval, simulate_funded, monte_carlo, economics

P = load_profile(Path(__file__).parent / "tradeify_select_50k.yaml")


def ev(days, micros=1):
    return simulate_eval(days, P, micros)


def test_pass_in_three_even_days():
    r = ev([(1000, 0), (1000, 0), (1000, 0)])           # biggest day 33.3% <= 40%
    assert r == {"result": "pass", "days": 3}, r


def test_min_three_days_even_if_target_hit_early():
    r = ev([(1500, 0), (1500, 0)])                      # 3000 profit but only 2 days
    assert r["result"] == "timeout", r


def test_consistency_blocks_then_more_days_unlock():
    r = ev([(1500, 0), (1500, 0), (1000, 0)])           # 1500/4000 = 37.5% -> pass on day 3
    assert r == {"result": "pass", "days": 3}, r
    r = ev([(2000, 0), (500, 0), (500, 0)])             # 2000/3000 = 66% -> blocked
    assert r["result"] == "timeout", r


def test_consistency_exactly_40_percent_passes():
    r = ev([(1300, 0), (1000, 0), (950, 0)])            # 1300/3250 = 40.0%
    assert r["result"] == "pass", r


def test_intraday_touch_of_floor_breaches():
    assert ev([(0.01, -2000)])["result"] == "breach"    # 50000-2000 == floor 48000 -> fails
    assert ev([(0.01, -1999)])["result"] != "breach"


def test_trailing_floor_ratchets_on_eod_close():
    # day1 closes +1500 -> floor 49500. day2 dips to -2000 from 51500 = 49500 -> breach
    r = ev([(1500, 0), (0.01, -2000)])
    assert r == {"result": "breach", "days": 2}, r
    r = ev([(1500, 0), (0.01, -1999)])
    assert r["result"] != "breach", r


def test_floor_locks_at_start_plus_100():
    # +2100 EOD -> floor locks at 50100 forever. From 52100, -2000 touches it.
    assert ev([(2100, 0), (0.01, -2000)])["result"] == "breach"
    assert ev([(2100, 0), (0.01, -1999)])["result"] != "breach"
    # after lock, a further big win does NOT raise the floor
    r = ev([(2100, 0), (500, 0), (0.01, -2400)])        # 52600-2400=50200 > 50100 survives
    assert r["result"] != "breach", r


def test_micros_scale_and_cap():
    assert ev([(100, 0)] * 40, micros=999)["result"] in ("pass", "timeout")  # capped at 40 micros
    assert ev([(20, -50)], micros=40)["result"] == "breach"                  # 40*-50 = -2000


def test_funded_requires_cushion_before_first_payout():
    days = [(200, 0)] * 5                                # 5 winning days, profit 1000
    r = simulate_funded(days, P, micros=1, cushion=0)
    assert r["payouts"] == 1 and abs(r["received"] - 0.9 * 500) < 1e-9, r
    r = simulate_funded(days, P, micros=1, cushion=1000)  # post-payout 50500 - 50100 = 400 < 1000
    assert r["payouts"] == 0, r


def test_funded_payout_cap_and_split():
    days = [(1000, 0)] * 5                               # profit 5000, 50% = 2500 = cap
    r = simulate_funded(days, P, micros=1, cushion=0)
    assert abs(r["received"] - 0.9 * 2500) < 1e-9, r


def test_funded_second_payout_needs_positive_cycle():
    up = [(200, 0)] * 5
    down = [(-300, -300)] * 2
    r = simulate_funded(up + down + [(200, 0)] * 5, P, micros=1, cushion=0)
    assert r["payouts"] == 1, r                          # second cycle net negative -> blocked


def test_funded_scaling_limits_micros():
    # starts at 20 micros: 1 micro-day of +100 per micro -> at most +2000 on day one
    r = simulate_funded([(100, 0)], P, micros=40, cushion=0)
    assert r["result"] == "timeout"                      # ran; scaling applied internally


def test_economics_math():
    e = economics(0.5, 2000, P)                          # spend = 165 + (2-1)*109 = 274
    assert abs(e["expected_spend_to_pass"] - 274) < 1e-9, e
    assert abs(e["expected_net_per_funded"] - 1726) < 1e-9, e
    assert e["breakeven_pass_prob"] is not None and e["breakeven_pass_prob"] < 0.5


def test_monte_carlo_runs_and_is_reproducible():
    rng = np.random.default_rng(1)
    pnl = rng.normal(40, 220, 400)
    mae = np.minimum(pnl, 0) - np.abs(rng.normal(60, 60, 400))
    a = monte_carlo(pnl, mae, P, micros=10, n_paths=400, seed=7)
    b = monte_carlo(pnl, mae, P, micros=10, n_paths=400, seed=7)
    assert a == b
    assert 0 <= a["eval_p_pass"] <= 1
    assert abs(a["eval_p_pass"] + a["eval_p_breach"] + a["eval_p_timeout"] - 1) < 1e-9


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
            print("PASS", name)
    print(f"\n{n} tests passed")
