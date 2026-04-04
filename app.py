"""
Roll Yield Monitor — All Commodities
1-Year Roll Yield = Spot / 1yr − 1
Demo mode: generates synthetic data if parquet not found
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Roll Yield Monitor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
  [data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:#fafafa!important;color:#1d1d1f!important}
  [data-testid="stHeader"]{background:transparent!important}
  .block-container{padding-top:2rem!important;padding-bottom:1.5rem;max-width:1500px}
  hr{border:none!important;border-top:1px solid #e8e8ed!important;margin:.4rem 0!important}
  [data-testid="stRadio"] label,[data-testid="stRadio"] label p{color:#1d1d1f!important}
  [data-testid="stExpander"]{border:1px solid #e8e8ed!important;border-radius:8px!important;background:#fff!important}
  h1,h2,h3{color:#1d1d1f!important}
  html,body,[class*="css"]{color:#1d1d1f!important}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NAVY  = "#0a2463"
BLACK = "#1d1d1f"
_BASE = Path(__file__).parent

COMM_CONFIG = {
    "KC":   {"name": "Arabica",      "spot": "KCc2",  "yr1": "KCc7",  "curve": [f"KCc{i}"  for i in range(1,9)], "color": "#0a2463"},
    "LRC":  {"name": "Robusta",      "spot": "LRCc2", "yr1": "LRCc8", "curve": [f"LRCc{i}" for i in range(1,9)], "color": "#8b1a00"},
    "CC":   {"name": "NYC Cocoa",    "spot": "CCc2",  "yr1": "CCc7",  "curve": [f"CCc{i}"  for i in range(1,9)], "color": "#e8a020"},
    "LCC":  {"name": "LDN Cocoa",    "spot": "LCCc2", "yr1": "LCCc7", "curve": [f"LCCc{i}" for i in range(1,9)], "color": "#4a7fb5"},
    "SB":   {"name": "Sugar",        "spot": "SBc1",  "yr1": "SBc5",  "curve": [f"SBc{i}"  for i in range(1,9)], "color": "#1a6b1a"},
    "CT":   {"name": "Cotton",       "spot": "CTc2",  "yr1": "CTc7",  "curve": [f"CTc{i}"  for i in range(1,9)], "color": "#7b2d8b"},
    "LSU":  {"name": "White Sugar",  "spot": "LSUc1", "yr1": "LSUc6", "curve": [f"LSUc{i}" for i in range(1,9)], "color": "#c0392b"},
    "C":    {"name": "Corn",         "spot": "Cc1",   "yr1": "Cc6",   "curve": [f"Cc{i}"   for i in range(1,9)], "color": "#f39c12"},
    "W":    {"name": "Wheat",        "spot": "Wc1",   "yr1": "Wc6",   "curve": [f"Wc{i}"   for i in range(1,9)], "color": "#d35400"},
    "KW":   {"name": "Wheat (KCB)",  "spot": "KWc1",  "yr1": "KWc6",  "curve": [f"KWc{i}"  for i in range(1,9)], "color": "#795548"},
    "OJ":   {"name": "Orange Juice", "spot": "OJc2",  "yr1": "OJc7",  "curve": [f"OJc{i}"  for i in range(1,9)], "color": "#e67e22"},
}

COMMS     = list(COMM_CONFIG.keys())
NAMES     = {k: v["name"] for k, v in COMM_CONFIG.items()}
COLORS    = {k: v["color"] for k, v in COMM_CONFIG.items()}
MONTHS    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,Helvetica Neue,sans-serif", color=BLACK, size=10),
)

def lbl(text):
    return (f"<div style='background:{NAVY};padding:5px 13px;border-radius:5px;"
            f"margin-bottom:8px'><span style='font-size:.78rem;font-weight:500;"
            f"letter-spacing:.07em;text-transform:uppercase;color:#dde4f0'>{text}</span></div>")

# ── Demo data generator ───────────────────────────────────────────────────────
def _generate_demo():
    dates = pd.bdate_range("2020-01-01", pd.Timestamp.today())
    np.random.seed(42)
    rows = []
    for comm, cfg in COMM_CONFIG.items():
        spot0   = {"KC": 150, "LRC": 2000, "CC": 2500, "LCC": 1800,
                   "SB": 15,  "CT": 80,    "LSU": 400, "C": 450,
                   "W": 550,  "KW": 570,   "OJ": 130}[comm]
        spot    = spot0 * np.exp(np.cumsum(np.random.normal(0, 0.008, len(dates))))
        # backwardation baseline varies by commodity
        base_ry = np.random.uniform(0.03, 0.15)
        ry      = base_ry + np.random.normal(0, 0.02, len(dates))
        ry      = pd.Series(ry).rolling(20).mean().fillna(base_ry).values
        yr1     = spot / (1 + ry)
        for i, d in enumerate(dates):
            # build curve: monotonic from spot to yr1
            spread = yr1[i] - spot[i]
            curve  = [spot[i] + spread * (j / 7) for j in range(8)]
            rows.append({
                "Date": d, "Commodity": comm,
                "Spot": round(spot[i], 2), "OneYr": round(yr1[i], 2),
                "Roll_Yield_1yr": round(ry[i], 6),
                **{f"c{j+1}": round(curve[j], 2) for j in range(8)},
            })
    return pd.DataFrame(rows)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    pq = _BASE / "roll_yield_data.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
        df["Date"] = pd.to_datetime(df["Date"])
        return df, False
    return _generate_demo(), True

df, is_demo = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='font-family:\"Playfair Display\",Georgia,serif;color:#0a2463;"
    "font-weight:400;letter-spacing:-.01em;margin-bottom:2px'>Roll Yield Monitor</h2>",
    unsafe_allow_html=True,
)
if is_demo:
    st.info("Demo mode — synthetic data. Run roll_yield_ingest.py to load live data.")
st.markdown("<hr>", unsafe_allow_html=True)

# ── Date bounds ───────────────────────────────────────────────────────────────
min_d         = df["Date"].min().date()
max_d         = df["Date"].max().date()
default_start = (df["Date"].max() - pd.DateOffset(years=3)).date()

# ── Collapsible filters ───────────────────────────────────────────────────────
with st.expander("Controls", expanded=True):
    c1, c2 = st.columns([3, 5])
    with c1:
        sel_comms = st.multiselect(
            "Commodities",
            options=COMMS,
            default=["KC", "LRC", "CC"],
            format_func=lambda x: f"{x} — {NAMES[x]}",
            key="ms_comms",
        )
    with c2:
        date_range = st.slider(
            "Date range", min_value=min_d, max_value=max_d,
            value=(default_start, max_d), key="sl_dates",
        )

start_d, end_d = date_range
df_fil = df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Roll Yield Line Chart (multi-commodity)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(lbl("1-Year Roll Yield (%) — Spot / 1yr − 1"), unsafe_allow_html=True)

fig_line = go.Figure()
for comm in sel_comms:
    s = df_fil[df_fil["Commodity"] == comm].sort_values("Date")
    fig_line.add_trace(go.Scatter(
        x=s["Date"], y=(s["Roll_Yield_1yr"] * 100).round(2),
        name=NAMES[comm], mode="lines",
        line=dict(color=COLORS[comm], width=1.8),
        hovertemplate=f"<b>{NAMES[comm]}</b>  %{{x|%d %b %Y}}  %{{y:.1f}}%<extra></extra>",
    ))
fig_line.add_hline(y=0, line_dash="dot", line_color="#aaaaaa", line_width=1)
fig_line.update_layout(
    height=370,
    xaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9, color=BLACK),
               ticksuffix="%", title="Roll Yield (%)"),
    legend=dict(orientation="h", y=1.02, x=0, font=dict(size=8, color=BLACK), bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(t=10, b=10, l=4, r=4), **_D,
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Roll Yield Ranking + Percentile (side by side)
# ═══════════════════════════════════════════════════════════════════════════════
col_rank, col_pct = st.columns(2)

# Latest date in filtered range
latest_date = df_fil["Date"].max()
df_latest   = df_fil[df_fil["Date"] == latest_date].set_index("Commodity")

with col_rank:
    st.markdown(lbl(f"Roll Yield Ranking · {latest_date.strftime('%d/%m/%Y')}"), unsafe_allow_html=True)
    rank_rows = []
    for comm in COMMS:
        if comm in df_latest.index:
            ry = df_latest.loc[comm, "Roll_Yield_1yr"] * 100
            rank_rows.append({"Rank": 0, "Commodity": NAMES[comm], "Roll Yield (1yr)": f"{ry:+.1f}%", "_ry": ry})
    rank_df = pd.DataFrame(rank_rows).sort_values("_ry", ascending=False).reset_index(drop=True)
    rank_df["Rank"] = rank_df.index + 1

    fig_rank = go.Figure(go.Table(
        columnwidth=[30, 100, 80],
        header=dict(
            values=["Rank", "Commodity", "Roll Yield (1yr)"],
            fill_color=NAVY, font=dict(color="white", size=10),
            align="center", height=28,
        ),
        cells=dict(
            values=[rank_df["Rank"], rank_df["Commodity"], rank_df["Roll Yield (1yr)"]],
            fill_color=[["white" if i % 2 == 0 else "#f5f5f7" for i in range(len(rank_df))]],
            font=dict(color=[
                ["white"]*len(rank_df),
                [BLACK]*len(rank_df),
                [("#1a6b1a" if r > 0 else "#8b0000") for r in rank_df["_ry"]],
            ], size=10),
            align="center", height=24,
        ),
    ))
    fig_rank.update_layout(height=340, margin=dict(t=0, b=0, l=0, r=0), **_D)
    st.plotly_chart(fig_rank, use_container_width=True)

with col_pct:
    st.markdown(lbl("Roll Yield Percentile vs Full History"), unsafe_allow_html=True)
    pct_rows = []
    for comm in COMMS:
        hist = df[df["Commodity"] == comm]["Roll_Yield_1yr"].dropna()
        if hist.empty:
            continue
        cur = df_latest.loc[comm, "Roll_Yield_1yr"] if comm in df_latest.index else np.nan
        if np.isnan(cur):
            continue
        pct = float((hist < cur).mean() * 100)
        pct_rows.append({"Commodity": NAMES[comm], "Percentile": round(pct, 1), "color": COLORS[comm]})

    pct_df = pd.DataFrame(pct_rows).sort_values("Percentile", ascending=True)
    fig_pct = go.Figure(go.Bar(
        x=pct_df["Percentile"], y=pct_df["Commodity"],
        orientation="h", marker_color=pct_df["color"],
        text=pct_df["Percentile"].map(lambda x: f"{x:.0f}th"),
        textposition="outside", textfont=dict(size=9, color=BLACK),
    ))
    fig_pct.add_vline(x=50, line_dash="dot", line_color="#aaaaaa", line_width=1)
    fig_pct.add_vline(x=80, line_dash="dot", line_color="#e07b39", line_width=1)
    fig_pct.update_layout(
        height=340,
        xaxis=dict(range=[0, 115], showgrid=False, tickfont=dict(size=9, color=BLACK), ticksuffix="%"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
        margin=dict(t=0, b=0, l=4, r=60), **_D,
    )
    st.plotly_chart(fig_pct, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Forward Curves (latest + last 4 days + last 4 weeks)
# ═══════════════════════════════════════════════════════════════════════════════
curve_comm = st.selectbox(
    "Commodity for curves", COMMS,
    format_func=lambda x: f"{x} — {NAMES[x]}", key="curve_comm",
)
curve_color = COLORS[curve_comm]
curve_cols  = [f"c{i}" for i in range(1, 9)]
curve_labels= [f"{curve_comm}c{i}" for i in range(1, 9)]

df_comm = df_fil[df_fil["Commodity"] == curve_comm].sort_values("Date")
all_dates_sorted = df_comm["Date"].drop_duplicates().sort_values()

# Latest, last 4 trading days, last 4 weekly snapshots
latest_4d = all_dates_sorted.iloc[-4:].tolist() if len(all_dates_sorted) >= 4 else all_dates_sorted.tolist()
weekly_idx = list(range(-1, -len(all_dates_sorted), -5))[:4]
latest_4w  = [all_dates_sorted.iloc[i] for i in sorted(weekly_idx)]

day_colors  = ["#1d1d1f", "#c0392b", "#82c982", "#aaaaaa"]
week_colors = ["#1d1d1f", "#c0392b", "#82c982", "#aaaaaa"]

st.markdown(lbl(f"Forward Curves · {NAMES[curve_comm]}"), unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3)

def _curve_fig(dates, colors, title):
    fig = go.Figure()
    for d, col in zip(dates, colors):
        row = df_comm[df_comm["Date"] == d]
        if row.empty:
            continue
        y = [row.iloc[0][c] for c in curve_cols]
        fig.add_trace(go.Scatter(
            x=curve_labels, y=y, mode="lines+markers",
            name=d.strftime("%d/%m/%Y"),
            line=dict(color=col, width=2), marker=dict(size=5),
            hovertemplate="%{x}  %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color=BLACK), x=0.5, xanchor="center"),
        height=320,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9, color=BLACK)),
        legend=dict(font=dict(size=8, color=BLACK), bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(t=35, b=10, l=4, r=4), **_D,
    )
    return fig

with fc1:
    latest_row = df_comm[df_comm["Date"] == all_dates_sorted.iloc[-1]]
    y_latest   = [latest_row.iloc[0][c] for c in curve_cols]
    fig_latest = go.Figure(go.Scatter(
        x=curve_labels, y=y_latest, mode="lines+markers",
        line=dict(color=curve_color, width=2.5), marker=dict(size=6),
        hovertemplate="%{x}  %{y:.2f}<extra></extra>",
    ))
    fig_latest.update_layout(
        title=dict(text=f"Latest · {all_dates_sorted.iloc[-1].strftime('%d/%m/%Y')}",
                   font=dict(size=11, color=BLACK), x=0.5, xanchor="center"),
        height=320,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9, color=BLACK)),
        margin=dict(t=35, b=10, l=4, r=4), **_D,
    )
    st.plotly_chart(fig_latest, use_container_width=True)

with fc2:
    st.plotly_chart(_curve_fig(latest_4d, day_colors, "Last 4 Days"), use_container_width=True)

with fc3:
    st.plotly_chart(_curve_fig(latest_4w, week_colors, "Last 4 Weeks"), use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Roll Yield Heatmap (Years × Months, per commodity)
# ═══════════════════════════════════════════════════════════════════════════════
hm_comm = st.selectbox(
    "Commodity for Heatmap", COMMS,
    format_func=lambda x: f"{x} — {NAMES[x]}", key="hm_comm",
)
st.markdown(lbl(f"Roll Yield Heatmap · {NAMES[hm_comm]} · Monthly Avg"), unsafe_allow_html=True)

hm_s = df_fil[df_fil["Commodity"] == hm_comm].copy()
hm_s["Year"]  = hm_s["Date"].dt.year
hm_s["Month"] = hm_s["Date"].dt.month

pivot = (
    hm_s.groupby(["Year", "Month"])["Roll_Yield_1yr"]
    .mean()
    .reset_index()
    .pivot(index="Year", columns="Month", values="Roll_Yield_1yr")
)
pivot.columns = [MONTHS[m - 1] for m in pivot.columns]
pivot = pivot.sort_index(ascending=False)

z      = (pivot.values.astype(float) * 100).round(2)
years  = [str(y) for y in pivot.index]
months = list(pivot.columns)
text_mat = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

fig_hm = go.Figure(go.Heatmap(
    z=z, x=months, y=years,
    text=text_mat,
    texttemplate="%{text}",
    textfont=dict(size=8, color=BLACK),
    colorscale=[
        [0.0, "#8b0000"],
        [0.4, "#f5c6cb"],
        [0.5, "#ffffff"],
        [0.6, "#d4edda"],
        [1.0, "#1a6b1a"],
    ],
    zmid=0,
    colorbar=dict(
        title=dict(text="Roll Yield %", font=dict(size=9, color=BLACK)),
        tickfont=dict(size=8, color=BLACK), ticksuffix="%",
        thickness=12, len=0.8,
    ),
    hoverongaps=False,
    hovertemplate="<b>%{y} · %{x}</b><br>Avg Roll Yield: %{z:.1f}%<extra></extra>",
))
fig_hm.update_layout(
    height=max(300, len(years) * 28),
    xaxis=dict(side="top", tickfont=dict(size=9, color=BLACK), showgrid=False),
    yaxis=dict(tickfont=dict(size=9, color=BLACK), showgrid=False),
    margin=dict(t=40, b=10, l=60, r=10), **_D,
)
st.plotly_chart(fig_hm, use_container_width=True)
