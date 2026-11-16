from pathlib import Path
import pandas as pd
from prop_sim import load_profile
from trades import load_trades, compliance, daily_records
P = load_profile(Path(__file__).parent / "tradeify_select_50k.yaml")

def mk(rows):
    return load_trades(pd.DataFrame(rows, columns=["entry_time","exit_time","pnl_per_micro","mae_per_micro"]))

def test_scalp_is_rejected():
    d = mk([("2025-03-03 09:40","2025-03-03 09:40:20",5,-2)])   # 20 s trade
    r = compliance(d, P); assert not r["ok"] and any("Scalping" in i for i in r["issues"]), r

def test_clean_strategy_passes():
    d = mk([("2025-03-03 09:40","2025-03-03 10:10",20,-5),("2025-03-03 10:30","2025-03-03 11:15",-10,-12)])
    assert compliance(d, P)["ok"]

def test_late_exit_and_overnight_rejected():
    d = mk([("2025-03-03 16:00","2025-03-03 16:50",5,-1)]); assert compliance(d, P)["late_exits"] == 1
    d = mk([("2025-03-03 16:00","2025-03-04 09:40",5,-1)]); assert compliance(d, P)["cross_session_trades"] == 1

def test_overlap_rejected():
    d = mk([("2025-03-03 09:40","2025-03-03 10:40",5,-1),("2025-03-03 10:00","2025-03-03 10:50",5,-1)])
    assert compliance(d, P)["overlapping_trades"] == 1

def test_session_day_and_intraday_worst_point():
    d = mk([("2025-03-03 09:40","2025-03-03 10:10",30,-5),("2025-03-03 10:30","2025-03-03 11:15",-20,-40)])
    day = daily_records(d).iloc[0]
    assert day["pnl"] == 10 and day["mae"] == -10        # 30 then worst -40 => -10 running, close +10
    # an 18:30 ET entry belongs to the NEXT calendar day's session
    d = mk([("2025-03-03 18:30","2025-03-03 19:30",5,-1)]); assert str(d["session_day"][0]) == "2025-03-04"

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
