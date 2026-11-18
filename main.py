"""Position Engine API.

Automatic stop = recent swing extreme +/- ATR buffer (with universal percentage fallback).
Position size is then derived from account equity and risk.

Data source: Yahoo Finance via yfinance. Quotes can be delayed. This is a
calculator, not an execution venue. Instrument sizing is deliberately limited
to instruments whose contract multiplier is known below.
"""
import asyncio
import logging
import math
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("position-engine")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

TICKER_REFRESH_SECONDS = 24 * 60 * 60
TICKER_RETRY_SECONDS = 600
QUOTE_TTL_SECONDS = 8.0
ATR_PERIOD = 14
SWING_LOOKBACK = 20
ATR_BUFFER = 0.20

TIMEFRAMES = {
    "1m": ("5d", "1m", "1-minute", 60),
    "5m": ("1mo", "5m", "5-minute", 300),
    "15m": ("1mo", "15m", "15-minute", 300),
    "30m": ("60d", "30m", "30-minute", 600),
    "1h": ("3mo", "60m", "1-hour", 900),
    "4h": ("60d", "60m", "4-hour", 1800),
    "1d": ("1y", "1d", "daily", 3600),
    "1w": ("2y", "1wk", "weekly", 14400),
}

SYMBOL_RE = re.compile(r"[A-Z0-9^][A-Z0-9^.=\-]{0,19}")
PLAIN_TICKER_RE = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+)?")
NASDAQ_DIRECTORIES = (
    ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
    ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
)
CORE_TICKERS = {
    "AAPL": "Apple Inc.", "AMZN": "Amazon.com Inc.", "AMD": "Advanced Micro Devices",
    "BTC-USD": "Bitcoin / US Dollar", "ETH-USD": "Ethereum / US Dollar",
    "SOL-USD": "Solana / US Dollar", "EURUSD=X": "Euro / US Dollar",
    "GBPUSD=X": "British Pound / US Dollar",
    "ES=F": "E-mini S&P 500 Futures", "MES=F": "Micro E-mini S&P 500 Futures",
    "NQ=F": "E-mini Nasdaq-100 Futures", "MNQ=F": "Micro E-mini Nasdaq-100 Futures",
    "RTY=F": "E-mini Russell 2000 Futures", "M2K=F": "Micro E-mini Russell 2000 Futures",
    "YM=F": "E-mini Dow Futures", "MYM=F": "Micro E-mini Dow Futures",
    "CL=F": "Crude Oil WTI Futures", "MCL=F": "Micro WTI Crude Oil Futures",
    "GC=F": "Gold Futures", "MGC=F": "Micro Gold Futures", "SI=F": "Silver Futures",
}
# Dollar P/L multiplier per 1.0 price move. Unsupported futures are rejected.
FUTURES_MULTIPLIERS = {
    "ES=F": 50.0, "MES=F": 5.0, "NQ=F": 20.0, "MNQ=F": 2.0,
    "RTY=F": 50.0, "M2K=F": 5.0, "YM=F": 5.0, "MYM=F": 0.5,
    "CL=F": 1000.0, "MCL=F": 100.0, "GC=F": 100.0,
    "MGC=F": 10.0, "SI=F": 5000.0,
}
CRYPTO_SUFFIXES = ("-USD",)
UNIT_NAMES = {"equity": "shares", "future": "contracts", "crypto": "units"}

def instrument_kind(symbol: str) -> str:
    if symbol.endswith("=F"):
        return "future"
    if symbol.endswith(CRYPTO_SUFFIXES):
        return "crypto"
    if symbol.endswith("=X"):
        return "fx"
    return "equity"

_tickers = [{"symbol": s, "name": n} for s, n in CORE_TICKERS.items()]
_tickers_ready = False

def parse_symbol_directory(raw: str, symbol_column: str) -> dict[str, str]:
    lines = raw.splitlines()
    if not lines:
        return {}
    header = [x.strip() for x in lines[0].split("|")]
    try:
        si, ni, ti = header.index(symbol_column), header.index("Security Name"), header.index("Test Issue")
    except ValueError:
        return {}
    found = {}
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) <= max(si, ni, ti) or parts[ti].strip() == "Y":
            continue
        symbol = parts[si].strip().replace(".", "-")
        if PLAIN_TICKER_RE.fullmatch(symbol):
            found[symbol] = parts[ni].strip()
    return found

async def load_tickers() -> bool:
    global _tickers, _tickers_ready
    merged = dict(CORE_TICKERS)
    complete = True
    async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 PositionEngine"}) as client:
        for url, col in NASDAQ_DIRECTORIES:
            try:
                response = await client.get(url)
                response.raise_for_status()
                merged.update(parse_symbol_directory(response.text, col))
            except Exception as exc:
                complete = False
                log.warning("Ticker directory refresh failed: %s", exc)
    merged.update(CORE_TICKERS)
    _tickers = [{"symbol": s, "name": n} for s, n in sorted(merged.items())]
    _tickers_ready = True
    return complete

async def ticker_refresh_loop():
    while True:
        try:
            ok = await load_tickers()
        except Exception:
            ok = False
        await asyncio.sleep(TICKER_REFRESH_SECONDS if ok else TICKER_RETRY_SECONDS)

@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(ticker_refresh_loop())
    try:
        yield
    finally:
        task.cancel()

class TTLCache:
    def __init__(self, max_items=5000):
        self.data = {}
        self.lock = threading.Lock()
        self.max_items = max_items
    def get(self, key):
        with self.lock:
            item = self.data.get(key)
        if item is None:
            return None
        expires, value = item
        return value if time.monotonic() < expires else None
    def set(self, key, value, ttl):
        with self.lock:
            if len(self.data) >= self.max_items:
                self.data.clear()
            self.data[key] = (time.monotonic() + ttl, value)

_quote_cache = TTLCache()
_bars_cache = TTLCache()

class NoQuoteError(Exception):
    pass

@dataclass(frozen=True)
class Quote:
    price: float
    fetched_at: datetime

def positive(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None

def fetch_quote(symbol: str) -> Quote:
    ticker = yf.Ticker(symbol)
    price = None
    try:
        price = positive(ticker.fast_info.last_price)
    except Exception:
        pass
    if price is None:
        hist = ticker.history(period="5d", auto_adjust=False)
        if not hist.empty:
            closes = hist["Close"].dropna()
            if not closes.empty:
                price = positive(closes.iloc[-1])
    if price is None:
        raise NoQuoteError(symbol)
    return Quote(price, datetime.now(timezone.utc))

def get_quote(symbol: str) -> Quote:
    cached = _quote_cache.get(symbol)
    if cached is not None:
        return cached
    quote = fetch_quote(symbol)
    _quote_cache.set(symbol, quote, QUOTE_TTL_SECONDS)
    return quote

def fetch_bars(symbol: str, timeframe: str):
    period, interval, _label, ttl = TIMEFRAMES[timeframe]
    hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if hist.empty:
        return None
    bars = hist[["High", "Low", "Close"]].dropna()
    if timeframe == "4h":
        try:
            bars = bars.resample("4h", origin="start_day", offset="9h30min").agg(
                {"High": "max", "Low": "min", "Close": "last"}).dropna()
        except Exception:
            return None
    if len(bars) < ATR_PERIOD + 2:
        return None
    return bars

def get_bars(symbol: str, timeframe: str):
    key = (symbol, timeframe)
    cached = _bars_cache.get(key)
    if cached is not None:
        return cached
    bars = fetch_bars(symbol, timeframe)
    if bars is not None:
        _bars_cache.set(key, bars, TIMEFRAMES[timeframe][3])
    return bars

def wilder_atr(bars, period=ATR_PERIOD) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    highs, lows, closes = bars["High"].tolist(), bars["Low"].tolist(), bars["Close"].tolist()
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]),
               abs(lows[i] - closes[i-1])) for i in range(1, len(closes))]
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return positive(atr)

def price_decimals(price: float) -> int:
    if price >= 1000: return 2
    if price >= 1: return 4
    if price >= 0.01: return 6
    return 8

class CalcRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    direction: Literal["LONG", "SHORT"]
    equity: float = Field(gt=0, le=1e12)
    risk_pct: float = Field(gt=0, le=100)
    rr: float = Field(gt=0, le=100)
    timeframe: str = "1d"
    entry_price: Optional[float] = Field(default=None, gt=0)

def build_plan(symbol: str, quote: Quote, req: CalcRequest, bars) -> dict:
    kind = instrument_kind(symbol)
    if kind == "future" and symbol not in FUTURES_MULTIPLIERS:
        raise HTTPException(400, f"Contract specification unavailable for {symbol}.")
    if kind == "fx":
        raise HTTPException(400, "FX sizing is not enabled until account-currency conversion and lot specifications are available.")

    multiplier = FUTURES_MULTIPLIERS.get(symbol, 1.0)
    sign = 1 if req.direction == "LONG" else -1
    base_price = req.entry_price if req.entry_price is not None else quote.price
    dp = price_decimals(base_price)
    entry = round(base_price, dp)

    atr = wilder_atr(bars)
    if atr is None:
        raise HTTPException(422, f"Not enough {req.timeframe} history to calculate a reliable stop.")

    # Exclude the newest bar from swing detection: it may still be forming.
    completed = bars.iloc[:-1] if len(bars) > 1 else bars
    window = completed.tail(SWING_LOOKBACK)
    if len(window) < min(10, SWING_LOOKBACK):
        window = bars

    warnings = []
    if sign == 1:
        swing = float(window["Low"].min()) if not window.empty else entry * 0.985
        raw_stop = swing - ATR_BUFFER * atr
        if raw_stop >= entry:
            fallback_distance = entry * 0.015
            raw_stop = entry - fallback_distance
            swing = raw_stop
            warnings.append("No valid support stop found below price; applied default 1.5% safety stop-loss.")
    else:
        swing = float(window["High"].max()) if not window.empty else entry * 1.015
        raw_stop = swing + ATR_BUFFER * atr
        if raw_stop <= entry:
            fallback_distance = entry * 0.015
            raw_stop = entry + fallback_distance
            swing = raw_stop
            warnings.append("No valid resistance stop found above price; applied default 1.5% safety stop-loss.")

    stop = round(raw_stop, dp)
    distance = abs(entry - stop)
    if distance <= 0:
        distance = entry * 0.015
        stop = round(entry - sign * distance, dp)
        warnings.append("Stop distance too small; applied 1.5% minimum safety distance.")

    risk_budget = req.equity * req.risk_pct / 100.0
    risk_per_unit = distance * multiplier
    decimals = 0 if kind in ("equity", "future") else 6
    scale = 10 ** decimals
    quantity = math.floor((risk_budget / risk_per_unit) * scale + 1e-10) / scale
    quantity = int(quantity) if decimals == 0 else round(quantity, decimals)

    risk_at_size = quantity * risk_per_unit
    target = round(entry + sign * distance * req.rr, dp)
    reward = quantity * distance * req.rr * multiplier
    notional = quantity * entry * multiplier

    if quantity <= 0:
        warnings.append("Risk budget is too small for one unit at this stop.")
    warnings.append("Stop is a market-structure estimate. Gaps, slippage and fees can increase actual loss.")
    return {
        "symbol": symbol, "direction": req.direction,
        "unit": UNIT_NAMES[kind], "quantity": quantity,
        "entry": entry, "stop": stop, "target": target,
        "timeframe": TIMEFRAMES[req.timeframe][2],
        "swing_level": round(swing, dp), "atr": round(atr, dp + 2),
        "atr_buffer": ATR_BUFFER, "lookback_bars": len(window),
        "stop_pct": round(distance / entry * 100, 3),
        "rr": req.rr, "risk_budget": round(risk_budget, 2),
        "risk_at_size": round(risk_at_size, 2),
        "reward_at_target": round(reward, 2),
        "notional": round(notional, 2),
        "priced_at": quote.fetched_at.isoformat(),
        "warnings": warnings,
    }

app = FastAPI(title="Position Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-cache"})

@app.get("/healthz")
def healthz():
    return {"ok": True, "tickers": len(_tickers), "tickers_loaded": _tickers_ready}

@app.get("/api/search")
def search(q: str = Query("", max_length=20)):
    query = q.strip().upper()
    if not query:
        return []
    return [item for item in _tickers if item["symbol"].startswith(query)]

@app.post("/api/calc")
def calculate(req: CalcRequest):
    symbol = req.symbol.strip().upper()
    if not SYMBOL_RE.fullmatch(symbol):
        raise HTTPException(400, "Invalid symbol.")
    if req.timeframe not in TIMEFRAMES:
        raise HTTPException(400, "Unsupported timeframe.")
    if instrument_kind(symbol) == "future" and symbol not in FUTURES_MULTIPLIERS:
        raise HTTPException(400, f"Contract specification unavailable for {symbol}.")
    try:
        quote = get_quote(symbol)
        bars = get_bars(symbol, req.timeframe)
    except NoQuoteError:
        raise HTTPException(404, f"No price available for {symbol}.")
    except Exception as exc:
        log.exception("Market data request failed for %s", symbol)
        raise HTTPException(503, "Market data unavailable. Try again shortly.")
    if bars is None:
        raise HTTPException(422, f"No usable {req.timeframe} price history for {symbol}.")
    return build_plan(symbol, quote, req, bars)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))