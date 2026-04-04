"""
Roll Yield Ingest — All Commodities
Fetches settlement prices from LSEG/Refinitiv and saves roll_yield_data.parquet
"""

import datetime
import logging
import pandas as pd
import refinitiv.data as rd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

_BASE    = Path(__file__).parent
END_DATE = datetime.date.today().isoformat()

def _get_start_date():
    """If parquet exists, start from latest date (to catch any updates). Else full history."""
    pq = _BASE / "roll_yield_data.parquet"
    if pq.exists():
        existing = pd.read_parquet(pq, columns=["Date"])
        latest   = pd.to_datetime(existing["Date"]).max().date()
        # Go back 5 days to catch any late settlements
        start = (latest - datetime.timedelta(days=10)).isoformat()
        log.info(f"Incremental mode: fetching from {start} (parquet latest = {latest})")
        return start, True
    log.info("Full history mode: fetching from 2015-01-01")
    return "2015-01-01", False

COMM_CONFIG = {
    "KC":  {"name": "Arabica",      "spot": "KCc2",  "yr1": "KCc7",  "curve": [f"KCc{i}"  for i in range(1,9)]},
    "LRC": {"name": "Robusta",      "spot": "LRCc2", "yr1": "LRCc8", "curve": [f"LRCc{i}" for i in range(1,9)]},
    "CC":  {"name": "NYC Cocoa",    "spot": "CCc2",  "yr1": "CCc7",  "curve": [f"CCc{i}"  for i in range(1,9)]},
    "LCC": {"name": "LDN Cocoa",    "spot": "LCCc2", "yr1": "LCCc7", "curve": [f"LCCc{i}" for i in range(1,9)]},
    "SB":  {"name": "Sugar",        "spot": "SBc1",  "yr1": "SBc5",  "curve": [f"SBc{i}"  for i in range(1,9)]},
    "CT":  {"name": "Cotton",       "spot": "CTc2",  "yr1": "CTc7",  "curve": [f"CTc{i}"  for i in range(1,9)]},
    "LSU": {"name": "White Sugar",  "spot": "LSUc1", "yr1": "LSUc6", "curve": [f"LSUc{i}" for i in range(1,9)]},
    "C":   {"name": "Corn",         "spot": "Cc1",   "yr1": "Cc6",   "curve": [f"Cc{i}"   for i in range(1,9)]},
    "W":   {"name": "Wheat",        "spot": "Wc1",   "yr1": "Wc6",   "curve": [f"Wc{i}"   for i in range(1,9)]},
    "KW":  {"name": "Wheat (KCB)",  "spot": "KWc1",  "yr1": "KWc6",  "curve": [f"KWc{i}"  for i in range(1,9)]},
    "OJ":  {"name": "Orange Juice", "spot": "OJc2",  "yr1": "OJc7",  "curve": [f"OJc{i}"  for i in range(1,9)]},
}

def fetch_all_rics(start_date):
    """Fetch all RICs in one batch call."""
    all_rics = []
    for cfg in COMM_CONFIG.values():
        all_rics.extend(cfg["curve"])
    all_rics = list(dict.fromkeys(all_rics))

    log.info(f"Fetching {len(all_rics)} RICs from {start_date} to {END_DATE}...")
    df = rd.get_history(
        universe=all_rics,
        fields=["TR.SETTLEMENTPRICE"],
        start=start_date,
        end=END_DATE,
        interval="daily",
    )
    df.index = pd.to_datetime(df.index)
    df.columns = [str(c) for c in df.columns]
    log.info(f"  -> {len(df)} rows, {len(df.columns)} RICs returned")
    return df

def build_parquet(raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw wide data into long format with one row per (Date, Commodity)."""
    rows = []
    for comm, cfg in COMM_CONFIG.items():
        curve_cols = cfg["curve"]
        spot_col   = cfg["spot"]
        yr1_col    = cfg["yr1"]

        # Only use dates where both spot and 1yr are present
        available = [c for c in curve_cols if c in raw.columns]
        if spot_col not in raw.columns or yr1_col not in raw.columns:
            log.warning(f"  {comm}: spot or 1yr RIC not in data — skipping")
            continue

        df_c = raw[available].dropna(subset=[spot_col, yr1_col]).copy()
        if df_c.empty:
            log.warning(f"  {comm}: no data after dropna — skipping")
            continue
        df_c.index.name = "Date"
        df_c = df_c.reset_index()

        df_c["Commodity"]      = comm
        df_c["Spot"]           = df_c[spot_col]
        df_c["OneYr"]          = df_c[yr1_col]
        df_c["Roll_Yield_1yr"] = df_c[spot_col] / df_c[yr1_col] - 1

        # Rename curve cols to c1-c8
        rename = {col: f"c{i+1}" for i, col in enumerate(curve_cols) if col in df_c.columns}
        df_c = df_c.rename(columns=rename)

        keep = ["Date", "Commodity", "Spot", "OneYr", "Roll_Yield_1yr"] + [f"c{i}" for i in range(1,9)]
        keep = [k for k in keep if k in df_c.columns]
        rows.append(df_c[keep])

        log.info(f"  {comm}: {len(df_c)} rows  |  latest roll yield = {df_c['Roll_Yield_1yr'].iloc[-1]:.2%}")

    return pd.concat(rows, ignore_index=True)

def main():
    log.info("=" * 60)
    log.info(f"Roll Yield Ingest  |  {END_DATE}")
    log.info("=" * 60)

    start_date, incremental = _get_start_date()
    rd.open_session()
    try:
        raw    = fetch_all_rics(start_date)
        new_df = build_parquet(raw)
        out    = _BASE / "roll_yield_data.parquet"

        if incremental and out.exists():
            existing = pd.read_parquet(out)
            existing["Date"] = pd.to_datetime(existing["Date"])
            # Remove overlapping dates then append
            cutoff  = new_df["Date"].min()
            existing = existing[existing["Date"] < cutoff]
            df = pd.concat([existing, new_df], ignore_index=True).sort_values(["Commodity","Date"])
            log.info(f"Appended {len(new_df)} new rows to {len(existing)} existing rows")
        else:
            df = new_df

        df.to_parquet(out, index=False)
        log.info(f"Saved -> {out}  |  {len(df)} rows  |  {df['Commodity'].nunique()} commodities")
        log.info(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    finally:
        rd.close_session()
    log.info("=" * 60)

if __name__ == "__main__":
    main()
