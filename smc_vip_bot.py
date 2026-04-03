# smc_vip_bot.py
import asyncio
import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

# ===================== CONFIG =====================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "-1001234567890"))
ADMIN_ID = int(os.getenv("ADMIN_ID", "123456789"))

MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", 80))
EXTREME_CONFIDENCE = int(os.getenv("EXTREME_CONFIDENCE", 92))

TP1_RR = float(os.getenv("TP1_RR", 2))
TP2_RR = float(os.getenv("TP2_RR", 3))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 5))  # minutes
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", 60))  # seconds

SYMBOLS = {
    "GOLD": "GC=F",
    "BTCUSD": "BTC-USD",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "NAS100": "^NDX",
    "US30": "^DJI",
}

# ===================== LOGGER =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SMC_BOT")

# ===================== TELEGRAM BOT =====================
bot = Bot(token=TELEGRAM_TOKEN, parse_mode="Markdown")
dp = Dispatcher()

# ===================== HELPER FUNCTIONS =====================
def get_data(symbol, interval, lookback="60d"):
    df = yf.download(symbol, interval=interval, period=lookback, progress=False)
    df.dropna(inplace=True)
    return df

def detect_swings(df, order=5):
    from scipy.signal import argrelextrema
    highs = df['High'].values
    lows = df['Low'].values
    swing_highs = argrelextrema(highs, np.greater, order=order)[0]
    swing_lows = argrelextrema(lows, np.less, order=order)[0]
    return swing_highs, swing_lows

def detect_bos(df):
    swing_highs, swing_lows = detect_swings(df)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    last_high, prev_high = df['High'].iloc[swing_highs[-1]], df['High'].iloc[swing_highs[-2]]
    last_low, prev_low = df['Low'].iloc[swing_lows[-1]], df['Low'].iloc[swing_lows[-2]]
    if last_high > prev_high:
        return "bullish"
    if last_low < prev_low:
        return "bearish"
    return None

def detect_liquidity_sweep(df):
    highs, lows = df['High'], df['Low']
    if highs.iloc[-1] > highs.iloc[-5:-1].max():
        return "buy_side_liquidity"
    if lows.iloc[-1] < lows.iloc[-5:-1].min():
        return "sell_side_liquidity"
    return None

def detect_order_block(df):
    last = df.iloc[-3]
    if last['Close'] < last['Open']:
        return ("bearish", last['High'], last['Low'])
    if last['Close'] > last['Open']:
        return ("bullish", last['High'], last['Low'])
    return None

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr.iloc[-1]

def score_setup(daily_bias, bos, liquidity, ob):
    score = 0
    reasons = []
    if daily_bias:
        score += 30; reasons.append("Daily bias alignment")
    if bos:
        score += 25; reasons.append("30m BOS/MSS")
    if liquidity:
        score += 20; reasons.append("Liquidity sweep")
    if ob:
        score += 25; reasons.append("Order Block POI")
    return score, reasons

def generate_chart(df, entry, sl, tp1, tp2, file="chart.png"):
    apds = [
        mpf.make_addplot([entry]*len(df)),
        mpf.make_addplot([sl]*len(df)),
        mpf.make_addplot([tp1]*len(df)),
        mpf.make_addplot([tp2]*len(df)),
    ]
    mpf.plot(df.tail(100), type='candle', style='charles', addplot=apds, savefig=file)
    return file

def format_analysis(symbol, bias, reasons):
    text = f"📊 *SMC MARKET ANALYSIS — {symbol}*\n\nDaily Bias: *{bias}*\nConfluences:\n"
    for r in reasons: text += f"• {r}\n"
    return text

def format_signal(symbol, signal):
    text = f"""
🚨 *VIP TRADE SIGNAL*

Instrument: *{symbol}*
Direction: *{signal['direction'].upper()}*

Entry: `{signal['entry']:.2f}`
Stop Loss: `{signal['sl']:.2f}`

🎯 TP1: `{signal['tp1']:.2f}`
🎯 TP2: `{signal['tp2']:.2f}`

📈 Confidence: *{signal['score']}%*
"""
    return text

# ===================== SIGNAL GENERATOR =====================
async def generate_signal(symbol):
    try:
        daily = get_data(symbol, "1d")
        m30 = get_data(symbol, "30m")
        daily_bias = detect_bos(daily)
        bos = detect_bos(m30)
        liquidity = detect_liquidity_sweep(m30)
        ob = detect_order_block(m30)
        if not ob: return None
        score, reasons = score_setup(daily_bias, bos, liquidity, ob)
        if score < MIN_CONFIDENCE: return None
        direction, high, low = ob
        entry = (high + low) / 2
        atr = calculate_atr(m30)
        sl = low - atr if direction=="bullish" else high + atr
        risk = abs(entry - sl)
        tp1 = entry + risk*TP1_RR if direction=="bullish" else entry - risk*TP1_RR
        tp2 = entry + risk*TP2_RR if direction=="bullish" else entry - risk*TP2_RR
        chart_file = generate_chart(m30, entry, sl, tp1, tp2)
        analysis_text = format_analysis(symbol, daily_bias, reasons)
        signal_text = format_signal(symbol, {"direction":direction, "entry":entry, "sl":sl, "tp1":tp1, "tp2":tp2, "score":score})
        await bot.send_message(CHANNEL_ID, analysis_text)
        await bot.send_message(CHANNEL_ID, signal_text)
        with open(chart_file,"rb") as f:
            await bot.send_photo(CHANNEL_ID, f)
        logger.info(f"Signal sent for {symbol} | Confidence: {score}%")
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")

# ===================== SCHEDULER =====================
scheduler = AsyncIOScheduler()

async def scan_markets():
    for name, ticker in SYMBOLS.items():
        await generate_signal(ticker)

def start_scheduler():
    scheduler.add_job(lambda: asyncio.create_task(scan_markets()), "interval", minutes=SCAN_INTERVAL)
    scheduler.start()

# ===================== TELEGRAM COMMANDS =====================
@dp.message(commands=["help"])
async def cmd_help(message: Message):
    await message.answer("/help - Show commands\n/status - Show last signals\n/pause /resume - Admin only")

# ===================== MAIN =====================
async def main():
    start_scheduler()
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
