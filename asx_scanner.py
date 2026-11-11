"""
ASX Large-Cap Daily Scanner
- จัดกลุ่ม: Breakout / Breakdown / ขาขึ้น / ขาลง / Sideways
- Timeframe: Day, กราฟย้อนหลัง 1 ปี
- RSI 9-3-3: RSI(9), เส้น K = SMA3 ของ RSI, เส้น D = SMA3 ของ K
- สร้างรายงาน HTML (เปิดในมือถือได้) + CSV ทุกวัน

ติดตั้ง:  pip install yfinance pandas numpy plotly
รัน:      python asx_scanner.py
"""

import os
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------ ตั้งค่า ------------------
# รายชื่อหุ้นใหญ่ใน ASX (แก้ไข/เพิ่มได้) ตัวที่ดึงข้อมูลไม่ได้จะถูกข้ามอัตโนมัติ
UNIVERSE = [
    "BHP", "CBA", "CSL", "NAB", "WBC", "ANZ", "MQG", "WES", "GMG", "FMG",
    "RIO", "TLS", "WDS", "WOW", "TCL", "ALL", "QBE", "STO", "XRO", "REA",
    "COL", "SUN", "NST", "RMD", "PME", "WTC", "EVN", "SIG", "BXB", "IAG",
    "ORG", "S32", "MIN", "PLS", "LYC", "QAN", "CPU", "ASX", "SCG", "JHX",
    "COH", "SHL", "TLC", "NXT", "CAR", "SGP", "APA", "AGL", "ALQ", "AZJ",
]
TOP_N = 30                 # เลือก 30 ตัวที่มีมูลค่าซื้อขายเฉลี่ยสูงสุด (most active)
ACTIVE_LOOKBACK = 20       # ใช้มูลค่าซื้อขายเฉลี่ย 20 วันล่าสุดวัดความ active

RSI_LEN, K_LEN, D_LEN = 9, 3, 3   # RSI 9-3-3
OVERSOLD, OVERBOUGHT = 30, 70
EMA_FAST, EMA_SLOW = 50, 200
BREAKOUT_LOOKBACK = 20     # ทะลุจุดสูงสุด/ต่ำสุด 20 วัน
BREAKOUT_VOL_MULT = 1.5    # วอลุ่มต้องมากกว่าค่าเฉลี่ย 1.5 เท่า
RECENT_DAYS = 3            # breakout ที่เกิดภายใน 3 วันล่าสุดยังนับ

OUT_DIR = "docs"           # โฟลเดอร์รายงาน (docs ใช้กับ GitHub Pages ได้เลย)


# ------------------ อินดิเคเตอร์ ------------------
def rsi(close, length):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def add_indicators(df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"], RSI_LEN)
    df["K"] = df["RSI"].rolling(K_LEN).mean()
    df["D"] = df["K"].rolling(D_LEN).mean()
    df["EMA_F"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_S"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["HH"] = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    df["LL"] = df["Low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    df["VOL_AVG"] = df["Volume"].rolling(BREAKOUT_LOOKBACK).mean().shift(1)

    vol_ok = df["Volume"] > BREAKOUT_VOL_MULT * df["VOL_AVG"]
    df["BREAKOUT"] = (df["Close"] > df["HH"]) & vol_ok
    df["BREAKDOWN"] = (df["Close"] < df["LL"]) & vol_ok

    k_up = (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))
    k_dn = (df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1))
    # Long: K ตัด D ขึ้นในโซนล่าง (ฟื้นจาก oversold) หรือวัน breakout
    df["LONG"] = (k_up & (df["K"] < 40)) | df["BREAKOUT"]
    # Sell: K ตัด D ลงในโซนบน (อ่อนแรงจาก overbought) หรือวัน breakdown
    df["SELL"] = (k_dn & (df["K"] > 60)) | df["BREAKDOWN"]
    return df


def classify(df):
    last = df.iloc[-1]
    recent = df.iloc[-RECENT_DAYS:]
    if recent["BREAKOUT"].any():
        return "Breakout"
    if recent["BREAKDOWN"].any():
        return "Breakdown"
    if last["Close"] > last["EMA_F"] > last["EMA_S"]:
        return "ขาขึ้น"
    if last["Close"] < last["EMA_F"] < last["EMA_S"]:
        return "ขาลง"
    return "Sideways"


def last_signal(df):
    sig = df[df["LONG"] | df["SELL"]]
    if sig.empty:
        return "-", None
    row = sig.iloc[-1]
    days_ago = len(df) - 1 - df.index.get_loc(sig.index[-1])
    return ("LONG" if row["LONG"] else "SELL"), int(days_ago)


# ------------------ ดึงข้อมูล ------------------
def fetch(tickers):
    syms = [t + ".AX" for t in tickers]
    raw = yf.download(syms, period="2y", interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    data = {}
    for t, s in zip(tickers, syms):
        try:
            df = raw[s].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(df) >= EMA_SLOW:
            data[t] = df
        else:
            print(f"ข้าม {t}: ข้อมูลไม่พอ")
    return data


def pick_most_active(data):
    value = {t: (df["Close"] * df["Volume"]).iloc[-ACTIVE_LOOKBACK:].mean()
             for t, df in data.items()}
    top = sorted(value, key=value.get, reverse=True)[:TOP_N]
    return top, value


# ------------------ กราฟ ------------------
def make_chart(t, df, group):
    d = df.iloc[-252:]  # 1 ปีล่าสุด
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                 low=d["Low"], close=d["Close"], name="ราคา"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA_F"], name=f"EMA{EMA_FAST}",
                             line=dict(width=1.2, color="#f2a900")), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA_S"], name=f"EMA{EMA_SLOW}",
                             line=dict(width=1.2, color="#7b61ff")), 1, 1)
    lg, sl = d[d["LONG"]], d[d["SELL"]]
    fig.add_trace(go.Scatter(x=lg.index, y=lg["Low"] * 0.985, mode="markers",
                             name="LONG", marker=dict(symbol="triangle-up",
                             size=11, color="#16a34a")), 1, 1)
    fig.add_trace(go.Scatter(x=sl.index, y=sl["High"] * 1.015, mode="markers",
                             name="SELL", marker=dict(symbol="triangle-down",
                             size=11, color="#dc2626")), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI9",
                             line=dict(width=1, color="#888")), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["K"], name="K3",
                             line=dict(width=1.4, color="#2563eb")), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["D"], name="D3",
                             line=dict(width=1.4, color="#ea580c")), 2, 1)
    for lvl in (OVERSOLD, OVERBOUGHT):
        fig.add_hline(y=lvl, line_dash="dot", line_color="#aaa", row=2, col=1)
    fig.update_layout(title=f"{t} — {group}", height=520,
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# ------------------ รายงาน ------------------
def build_report(rows, charts, today):
    order = ["Breakout", "ขาขึ้น", "Sideways", "ขาลง", "Breakdown"]
    tbl = pd.DataFrame(rows)
    tbl["_o"] = tbl["กลุ่ม"].map({g: i for i, g in enumerate(order)})
    tbl = tbl.sort_values(["_o", "RSI9"], ascending=[True, False]).drop(columns="_o")

    parts = [f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASX Scan {today}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0 auto;max-width:1000px;padding:12px;}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}
.long{{color:#16a34a;font-weight:600}} .sell{{color:#dc2626;font-weight:600}}
h2{{margin-top:28px}} .note{{font-size:12px;color:#666}}
</style></head><body>
<h1>ASX Large-Cap Scan — {today}</h1>
<p class="note">Timeframe Day · RSI {RSI_LEN}-{K_LEN}-{D_LEN} · EMA{EMA_FAST}/{EMA_SLOW} ·
Breakout = ปิดเหนือจุดสูงสุด {BREAKOUT_LOOKBACK} วันด้วยวอลุ่ม &gt; {BREAKOUT_VOL_MULT}x ·
ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน</p>"""]

    for g in order:
        sub = tbl[tbl["กลุ่ม"] == g]
        if sub.empty:
            continue
        parts.append(f"<h2>{g} ({len(sub)})</h2><table><tr><th>หุ้น</th><th>ราคา</th>"
                     "<th>RSI9</th><th>K/D</th><th>สัญญาณล่าสุด</th></tr>")
        for _, r in sub.iterrows():
            cls = "long" if r["สัญญาณ"] == "LONG" else "sell" if r["สัญญาณ"] == "SELL" else ""
            ago = "" if r["วันที่แล้ว"] is None else f" ({r['วันที่แล้ว']} วันก่อน)"
            parts.append(f"<tr><td><a href='#{r['หุ้น']}'>{r['หุ้น']}</a></td>"
                         f"<td>{r['ราคา']:.2f}</td><td>{r['RSI9']:.1f}</td>"
                         f"<td>{r['K']:.1f}/{r['D']:.1f}</td>"
                         f"<td class='{cls}'>{r['สัญญาณ']}{ago}</td></tr>")
        parts.append("</table>")

    parts.append("<h2>กราฟรายตัว</h2>")
    first = True
    for t in tbl["หุ้น"]:
        parts.append(f"<div id='{t}'>")
        parts.append(charts[t].to_html(full_html=False,
                                       include_plotlyjs="cdn" if first else False))
        parts.append("</div>")
        first = False
    parts.append("</body></html>")
    return "\n".join(parts), tbl


def main():
    today = dt.date.today().isoformat()
    print("กำลังดึงข้อมูล...")
    data = fetch(UNIVERSE)
    top, _ = pick_most_active(data)

    rows, charts = [], {}
    for t in top:
        df = add_indicators(data[t])
        group = classify(df)
        sig, ago = last_signal(df)
        last = df.iloc[-1]
        rows.append({"หุ้น": t, "กลุ่ม": group, "ราคา": last["Close"],
                     "RSI9": last["RSI"], "K": last["K"], "D": last["D"],
                     "สัญญาณ": sig, "วันที่แล้ว": ago})
        charts[t] = make_chart(t, df, group)

    html, tbl = build_report(rows, charts, today)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in (f"asx_scan_{today}.html", "index.html"):
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(html)
    tbl.to_csv(os.path.join(OUT_DIR, f"asx_scan_{today}.csv"),
               index=False, encoding="utf-8-sig")

    print(tbl.to_string(index=False))
    print(f"\nบันทึกรายงานที่ {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
