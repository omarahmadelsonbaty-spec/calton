import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
SYMBOL_RE = re.compile(r"^[A-Z0-9^][A-Z0-9^.=\-]{0,20}$")

TIMEFRAMES = {
    "1m": ("5d", "1m"),
    "5m": ("1mo", "5m"),
    "15m": ("1mo", "15m"),
    "30m": ("60d", "30m"),
    "1h": ("3mo", "60m"),
    "4h": ("60d", "60m"),
    "1d": ("1y", "1d"),
    "1w": ("2y", "1wk"),
}

MULTIPLIERS = {
    "ES=F": 50.0, "MES=F": 5.0,
    "NQ=F": 20.0, "MNQ=F": 2.0,
    "RTY=F": 50.0, "M2K=F": 5.0,
    "YM=F": 5.0, "MYM=F": 0.5,
    "CL=F": 1000.0, "MCL=F": 100.0,
    "GC=F": 100.0, "MGC=F": 10.0,
    "SI=F": 5000.0,
}


class CalcRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=21)
    direction: Literal["LONG", "SHORT"]
    equity: float = Field(gt=0, le=1e12)
    risk_pct: float = Field(gt=0, le=100)
    rr: float = Field(gt=0, le=100)
    timeframe: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"] = "1d"
    entry_price: Optional[float] = Field(default=None, gt=0, le=1e12)


def atr(bars, period: int = 14) -> Optional[float]:
    if len(bars) < 2:
        return None
    h, l, c = bars["High"].tolist(), bars["Low"].tolist(), bars["Close"].tolist()
    trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    if not trs:
        return None
    n = min(period, len(trs))
    val = sum(trs[:n]) / n
    for tr in trs[n:]:
        val = (val * (n - 1) + tr) / n
    return val if math.isfinite(val) and val > 0 else None


def kind_and_unit(symbol: str) -> tuple[str, str]:
    if symbol.endswith("=F"):
        return "future", "contracts"
    if symbol.endswith("-USD"):
        return "crypto", "units"
    if symbol.endswith("=X"):
        return "fx", "units"
    return "equity", "shares"


def decimals(price: float) -> int:
    if price >= 1000:
        return 2
    if price >= 1:
        return 4
    if price >= 0.01:
        return 6
    return 8


app = FastAPI(title="Position")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/calc")
def calc(req: CalcRequest):
    symbol = req.symbol.strip().upper()
    if not SYMBOL_RE.match(symbol):
        raise HTTPException(400, "Invalid symbol")

    t = yf.Ticker(symbol)
    last = None
    try:
        p = float(t.fast_info.last_price)
        if math.isfinite(p) and p > 0:
            last = p
    except Exception:
        pass

    period, interval = TIMEFRAMES[req.timeframe]
    hist = t.history(period=period, interval=interval, auto_adjust=False)
    if hist.empty:
        raise HTTPException(404, "No market data")

    bars = hist[["High", "Low", "Close"]].dropna()
    if req.timeframe == "4h":
        bars = bars.resample("4h", origin="start_day", offset="9h30min").agg(
            {"High": "max", "Low": "min", "Close": "last"}
        ).dropna()

    if len(bars) < 16:
        raise HTTPException(422, "Insufficient history")

    if last is None:
        last = float(bars["Close"].iloc[-1])

    completed = bars.iloc[:-1]
    vol = atr(completed)
    if vol is None:
        raise HTTPException(422, "Volatility unavailable")

    entry = req.entry_price if req.entry_price is not None else last
    dp = decimals(entry)
    entry = round(entry, dp)

    sign = 1 if req.direction == "LONG" else -1
    window = completed.tail(20)

    if sign == 1:
        swing = float(window["Low"].min())
        stop = round(min(swing - 0.2 * vol, entry - vol), dp)
        if stop >= entry:
            raise HTTPException(422, "Stop above entry")
    else:
        swing = float(window["High"].max())
        stop = round(max(swing + 0.2 * vol, entry + vol), dp)
        if stop <= entry:
            raise HTTPException(422, "Stop below entry")

    dist = abs(entry - stop)
    kind, unit = kind_and_unit(symbol)
    mult = MULTIPLIERS.get(symbol, 1.0)
    if kind == "fx" and symbol.startswith("USD") and symbol.endswith("=X") and not symbol.endswith("USD=X"):
        mult = 1.0 / entry

    risk_budget = req.equity * (req.risk_pct / 100.0)
    risk_per_unit = dist * mult
    raw_qty = risk_budget / risk_per_unit
    qty = math.floor(raw_qty) if kind in ("equity", "future") else math.floor(raw_qty * 1e6) / 1e6

    if qty <= 0:
        raise HTTPException(422, "Account too small for 1 unit")

    target = round(entry + sign * dist * req.rr, dp)

    return {
        "symbol": symbol,
        "unit": unit,
        "quantity": qty,
        "entry": entry,
        "stop": stop,
        "stop_pct": round(dist / entry * 100, 2),
        "target": target,
        "risk_at_size": round(qty * risk_per_unit, 2),
        "reward_at_target": round(qty * dist * req.rr * mult, 2),
        "notional": round(qty * entry * mult, 2),
        "priced_at": datetime.now(timezone.utc).isoformat(),
    }