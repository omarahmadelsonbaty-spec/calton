import asyncio
import json
import logging
import math
import re
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("calton")

BASE_DIR = Path(__file__).resolve().parent
TICKER_CACHE_FILE = BASE_DIR / "tickers_cache.json"

SYMBOL_RE = re.compile(r"^[A-Z0-9^][A-Z0-9^.=\-]{0,20}$")
PLAIN_TICKER_RE = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)?$")

REFRESH_OK_SECONDS = 24 * 60 * 60
REFRESH_ERR_SECONDS = 120
MARKET_CACHE_TTL_SECONDS = 15.0

NASDAQ_URLS = (
    ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
    ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
)
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

CORE_TICKERS = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "DIA": "SPDR Dow Jones ETF",
    "IWM": "iShares Russell 2000 ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "ABT": "Abbott Laboratories",
    "ACN": "Accenture",
    "ADBE": "Adobe",
    "AMD": "AMD",
    "AMZN": "Amazon",
    "AVGO": "Broadcom",
    "BA": "Boeing",
    "BAC": "Bank of America",
    "BRK-B": "Berkshire Hathaway Class B",
    "CRM": "Salesforce",
    "COST": "Costco",
    "CVX": "Chevron",
    "DIS": "Walt Disney",
    "GOOGL": "Alphabet Class A",
    "GOOG": "Alphabet Class C",
    "HD": "Home Depot",
    "INTC": "Intel",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "LLY": "Eli Lilly",
    "MA": "Mastercard",
    "MCD": "McDonald's",
    "META": "Meta Platforms",
    "MRK": "Merck",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
    "ORCL": "Oracle",
    "PEP": "PepsiCo",
    "PG": "Procter & Gamble",
    "PLTR": "Palantir Technologies",
    "SMCI": "Super Micro Computer",
    "TMO": "Thermo Fisher Scientific",
    "TSLA": "Tesla",
    "UNH": "UnitedHealth",
    "V": "Visa",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "XRP-USD": "XRP",
    "DOGE-USD": "Dogecoin",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "ES=F": "E-mini S&P 500",
    "MES=F": "Micro E-mini S&P 500",
    "NQ=F": "E-mini Nasdaq 100",
    "MNQ=F": "Micro E-mini Nasdaq 100",
    "RTY=F": "E-mini Russell 2000",
    "M2K=F": "Micro E-mini Russell 2000",
    "YM=F": "E-mini Dow",
    "MYM=F": "Micro E-mini Dow",
    "CL=F": "Crude Oil WTI",
    "MCL=F": "Micro Crude Oil WTI",
    "GC=F": "Gold",
    "MGC=F": "Micro Gold",
    "SI=F": "Silver",
    "^GSPC": "S&P 500 Index",
    "^NDX": "Nasdaq 100 Index",
    "^DJI": "Dow Jones Index",
    "^RUT": "Russell 2000 Index",
    "^VIX": "VIX",
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


def _load_disk_cache() -> dict[str, str]:
    merged = dict(CORE_TICKERS)
    if TICKER_CACHE_FILE.exists():
        try:
            data = json.loads(TICKER_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str):
                        merged[k.strip().upper()] = v.strip()
        except Exception as exc:
            log.warning("Failed reading %s: %s", TICKER_CACHE_FILE, exc)
    merged.update(CORE_TICKERS)
    return merged


def _save_disk_cache(mapping: dict[str, str]) -> None:
    try:
        TICKER_CACHE_FILE.write_text(json.dumps(mapping), encoding="utf-8")
    except Exception as exc:
        log.warning("Failed writing %s: %s", TICKER_CACHE_FILE, exc)


_ticker_map: dict[str, str] = _load_disk_cache()
_tickers: list[tuple[str, str]] = sorted(_ticker_map.items())
_market_cache: dict[tuple[str, str], tuple[float, Optional[float], object]] = {}


def parse_directory(raw: str, sym_col: str) -> dict[str, str]:
    lines = raw.splitlines()
    if not lines:
        return {}
    cols = [c.strip() for c in lines[0].split("|")]
    try:
        si = cols.index(sym_col)
        ni = cols.index("Security Name")
        ti = cols.index("Test Issue")
    except ValueError:
        return {}

    out: dict[str, str] = {}
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) <= max(si, ni, ti):
            continue
        if parts[ti].strip() == "Y":
            continue
        sym = parts[si].strip().replace(".", "-")
        name = parts[ni].strip()
        if sym and PLAIN_TICKER_RE.match(sym):
            out[sym] = name
    return out


def parse_sec_tickers(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for row in payload.values():
        if not isinstance(row, dict):
            continue
        raw_sym = str(row.get("ticker", "")).strip().upper().replace(".", "-")
        title = str(row.get("title", "")).strip()
        if raw_sym and PLAIN_TICKER_RE.match(raw_sym):
            out[raw_sym] = title or raw_sym
    return out


async def refresh_tickers() -> bool:
    global _ticker_map, _tickers
    merged = dict(_ticker_map)
    any_success = False

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": "calton-app/1.0 (contact@local.dev)"},
    ) as client:
        for url, col in NASDAQ_URLS:
            try:
                res = await client.get(url)
                res.raise_for_status()
                parsed = parse_directory(res.text, col)
                if parsed:
                    merged.update(parsed)
                    any_success = True
            except Exception as exc:
                log.warning("Ticker fetch failed (%s): %s", url, exc)

        try:
            res = await client.get(SEC_COMPANY_TICKERS_URL)
            res.raise_for_status()
            sec_parsed = parse_sec_tickers(res.json())
            if sec_parsed:
                merged.update(sec_parsed)
                any_success = True
        except Exception as exc:
            log.warning("SEC ticker fetch failed (%s): %s", SEC_COMPANY_TICKERS_URL, exc)

    merged.update(CORE_TICKERS)
    _ticker_map = merged
    _tickers = sorted(merged.items())
    if any_success:
        _save_disk_cache(merged)
    log.info("Loaded %d symbols (synced=%s)", len(_tickers), any_success)
    return any_success


async def ticker_loop():
    while True:
        try:
            ok = await refresh_tickers()
        except Exception:
            ok = False
        await asyncio.sleep(REFRESH_OK_SECONDS if ok else REFRESH_ERR_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(ticker_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


class CalcRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=21)
    direction: Literal["LONG", "SHORT"]
    equity: float = Field(gt=0, le=1e12)
    risk_pct: float = Field(gt=0, le=100)
    rr: float = Field(gt=0, le=100)
    timeframe: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"] = "5m"
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


def _fetch_market_sync(symbol: str, timeframe: str):
    key = (symbol, timeframe)
    now = time.monotonic()
    cached = _market_cache.get(key)
    if cached and (now - cached[0]) < MARKET_CACHE_TTL_SECONDS:
        return cached[1], cached[2]

    t = yf.Ticker(symbol)
    last = None
    try:
        p = float(t.fast_info.last_price)
        if math.isfinite(p) and p > 0:
            last = p
    except Exception:
        pass

    period, interval = TIMEFRAMES[timeframe]
    hist = t.history(period=period, interval=interval, auto_adjust=False)
    if hist.empty:
        return None, None

    bars = hist[["High", "Low", "Close"]].dropna()
    if timeframe == "4h":
        kind, _ = kind_and_unit(symbol)
        resample_kwargs = (
            {"origin": "start_day", "offset": "9h30min"}
            if kind == "equity"
            else {"origin": "epoch"}
        )
        bars = bars.resample("4h", **resample_kwargs).agg(
            {"High": "max", "Low": "min", "Close": "last"}
        ).dropna()

    _market_cache[key] = (now, last, bars)
    return last, bars


app = FastAPI(title="calton", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/tickers")
async def tickers(q: str = Query("", max_length=32), limit: int = Query(50, ge=1, le=200)):
    query = q.strip().upper()
    if not query:
        return [{"symbol": s, "name": n} for s, n in _tickers[:limit]]

    # Ticker-symbol-only matching: starts-with first, then substring in ticker symbol.
    prefix = []
    contains = []
    for s, n in _tickers:
        if s.startswith(query):
            prefix.append({"symbol": s, "name": n})
        elif query in s:
            contains.append({"symbol": s, "name": n})
        if len(prefix) + len(contains) >= limit * 2:
            break

    return (prefix + contains)[:limit]


@app.post("/api/calc")
async def calc(req: CalcRequest):
    symbol = req.symbol.strip().upper()
    if not SYMBOL_RE.match(symbol):
        raise HTTPException(400, "Invalid symbol")

    last, bars = await asyncio.to_thread(_fetch_market_sync, symbol, req.timeframe)
    if bars is None or bars.empty:
        raise HTTPException(404, "No market data")

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
    if dist <= 0:
        raise HTTPException(422, "Invalid stop distance")

    kind, unit = kind_and_unit(symbol)
    mult = MULTIPLIERS.get(symbol, 1.0)
    if kind == "fx" and symbol.startswith("USD") and symbol.endswith("=X") and not symbol.endswith("USD=X"):
        mult = 1.0 / entry

    risk_budget = req.equity * (req.risk_pct / 100.0)
    risk_per_unit = dist * mult
    if risk_per_unit <= 0:
        raise HTTPException(422, "Invalid risk per unit")

    raw_qty = risk_budget / risk_per_unit
    if kind == "future":
        qty = math.floor(raw_qty)
    elif kind == "equity":
        qty = math.floor(raw_qty * 1e4) / 1e4
    else:
        qty = math.floor(raw_qty * 1e6) / 1e6

    if qty <= 0:
        raise HTTPException(422, "Account too small")

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