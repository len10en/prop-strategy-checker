import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from prop_sim import load_profile, monte_carlo
from trades import load_trades, compliance, daily_records

PROFILE = Path(__file__).parent / "tradeify_select_50k.yaml"
prof = load_profile(PROFILE)

st.set_page_config(page_title="Prop Strategy Checker", layout="wide")
st.title("Prop Strategy Checker")
st.caption(f"Rules: {prof['firm']} {prof['plan']}. Edit tradeify_select_50k.yaml if the firm changes them.")


def demo_trades(seed=3, days=160):
    """SYNTHETIC data only for trying the screen. It is not a real strategy."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(days):
        day = pd.Timestamp("2024-01-02") + pd.tseries.offsets.BDay(k)
        t = day + pd.Timedelta(hours=9, minutes=40)
        for _ in range(int(rng.integers(1, 5))):
            hold = int(rng.integers(8, 70))
            pnl = float(rng.normal(3, 25))
            mae = min(pnl, 0) - abs(float(rng.normal(8, 8)))
            rows.append((t, t + pd.Timedelta(minutes=hold), round(pnl, 2), round(mae, 2)))
            t += pd.Timedelta(minutes=hold + int(rng.integers(5, 40)))
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "pnl_per_micro", "mae_per_micro"])


with st.sidebar:
    st.header("1. Strategy trades")
    source = st.radio("Source", ["Upload trades CSV", "Demo data (synthetic)"])
    up = st.file_uploader("CSV", type="csv") if source == "Upload trades CSV" else None
    st.header("2. Settings")
    micros = st.slider("Micro contracts per trade", 1, prof["evaluation"]["max_micros"], 10)
    cushion = st.slider("Cushion kept above locked floor before a payout ($)", 0, 1800, 1000, 100)
    paths = st.select_slider("Simulation paths", [1000, 2000, 5000, 10000], 2000)
    seed = st.number_input("Random seed", value=0, step=1)

if source == "Demo data (synthetic)":
    raw = demo_trades()
    st.warning("DEMO DATA: synthetic trades. Results below mean nothing about any real strategy.")
elif up is not None:
    raw = pd.read_csv(up)
else:
    st.info("Upload a trades CSV with columns: entry_time, exit_time, pnl_per_micro, mae_per_micro.")
    st.stop()

try:
    trades = load_trades(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.subheader("Rule check")
rep = compliance(trades, prof)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", rep["n_trades"])
c2.metric("Median hold (min)", f"{rep['median_hold_min']:.1f}")
c3.metric("Trades held 60s or more", f"{rep['pct_over_min_hold']:.0%}")
c4.metric("Max trades in a day", rep["max_trades_per_day"])
if not rep["ok"]:
    for msg in rep["issues"]:
        st.error(msg)
    st.stop()
st.success("No scalping, no overnight, no overlaps. Strategy is eligible to be simulated.")

days = daily_records(trades)
st.subheader("Daily results (per 1 micro)")
st.line_chart(days.set_index("day")["pnl"].cumsum())
if len(days) < 250:
    st.warning(f"Only {len(days)} trading days. Treat every probability below as unreliable "
               "until you have at least 250 days spanning different market conditions.")

if st.button("Run simulation", type="primary"):
    with st.spinner("Simulating evaluations and funded accounts..."):
        r = monte_carlo(days["pnl"].values, days["mae"].values, prof, micros,
                        n_paths=int(paths), seed=int(seed), cushion=float(cushion))
    st.subheader(f"Results at {micros} micros")
    a, b, c, d = st.columns(4)
    a.metric("Chance of passing", f"{r['eval_p_pass']:.1%}")
    b.metric("Chance of breaching", f"{r['eval_p_breach']:.1%}")
    c.metric("Median days to pass", r["eval_median_days_to_pass"] or "n/a")
    d.metric("Chance of a first payout once funded", f"{r['funded_p_first_payout']:.1%}")
    e, f, g = st.columns(3)
    e.metric("Expected spend to get funded", f"${r['expected_spend_to_pass']:,.0f}")
    f.metric("Expected payouts per funded account", f"${r['funded_mean_trader_receipts']:,.0f}")
    be = r["breakeven_pass_prob"]
    g.metric("Break-even pass chance", f"{be:.1%}" if be else "n/a")
    if r["expected_net_per_funded"] < 0:
        st.error("Expected net is negative. Do not buy the evaluation on this strategy.")
    elif be and r["eval_p_pass"] < be * 1.5:
        st.warning("Pass chance is close to break-even. A small error in costs would erase the edge.")
    else:
        st.success("Positive expected value in simulation. Forward test before spending money.")
    st.session_state["last"] = r

if st.button("Compare position sizes"):
    rows = []
    for m in [2, 5, 10, 15, 20, 30, 40]:
        if m > prof["evaluation"]["max_micros"]:
            continue
        r = monte_carlo(days["pnl"].values, days["mae"].values, prof, m,
                        n_paths=1500, seed=int(seed), cushion=float(cushion))
        rows.append({"micros": m, "pass %": round(100 * r["eval_p_pass"], 1),
                     "breach %": round(100 * r["eval_p_breach"], 1),
                     "first payout %": round(100 * r["funded_p_first_payout"], 1),
                     "expected net $": round(r["expected_net_per_funded"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
