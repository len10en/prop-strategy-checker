# Prop Strategy Checker

Simulates a trading strategy against Tradeify Select 50K (Flex) rules: evaluation, funded
payouts, and expected cost. Rules live in `tradeify_select_50k.yaml`.

## Run locally (Windows)
Double-click `Run.bat`, or:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

## Tests
    python test_prop_sim.py
    python test_trades.py

## Trades CSV format
entry_time, exit_time, pnl_per_micro, mae_per_micro (one micro contract, costs deducted).
