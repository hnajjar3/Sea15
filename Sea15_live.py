import requests
import pandas as pd
import numpy as np
import os
import csv
import time
import schedule
import random
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus


# ==============================================================================
# 🌊 PROJECT SEA 15: THE CAPTAIN (FINAL GOLDEN COPY)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
FMP_API_KEY = 'iuRa8nmjQdLOx66QtJiSWFitzOMqu6QF'
ALPACA_KEY = "PKG4U3SE6Z7EE66TZX4GLAMPUX"
ALPACA_SECRET = "AS4aE8iuwj4nB1YUoMCHamSfBcodV6sX3i5cCdYLuS3M"            

# SYSTEM SETTINGS
LOG_FILE = "sea15_live_log.csv"
REPORT_DIR = "weekly_reports"

# SCHEDULE (PACIFIC TIME as requested: 06:35 = 9:35 ET)
ENTRY_TIME = "06:35"   
EXIT_TIME  = "12:55"   
REPORT_TIME = "09:00"  # Saturday Morning Report

# RISK MANAGEMENT
RISK_PER_TRADE = 1000.00       
STOP_BUFFER = 0.05             
MAX_SHORTS = 5         # Daily Limit (Matches Backtest)
MAX_LONGS = 5          # Daily Limit (Matches Backtest)

# OPTIMIZED STRATEGY PARAMETERS (MATCHING ROBUST BACKTEST)
GAP_UP_MIN = 0.05              
GAP_UP_MAX = 0.13      # Cap at 13%
GAP_DOWN_THRESHOLD = -0.05     
MIN_PRICE = 1.00       # Matches Backtest
MAX_PRICE = 50.00      # Matches Backtest
MIN_VOLUME = 10000     # Liquidity Filter

# ------------------------------------------------------------------------------
# 2. SETUP CLIENTS
# ------------------------------------------------------------------------------
BASE_URL = "https://financialmodelingprep.com/api/v3"
alpaca = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def fetch_url(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200: return response.json()
        return []
    except: return []

# ------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def check_market_status():
    try:
        clock = alpaca.get_clock()
        if not clock.is_open:
            print(f"💤 Market is CLOSED today. Next open: {clock.next_open}")
            return False
        return True
    except: return True 

def is_tradable(ticker):
    try:
        asset = alpaca.get_asset(ticker)
        return asset.tradable and asset.status == 'active'
    except: return False

def execute_alpaca_order(ticker, action, shares):
    print(f"   🚀 SENDING: {action} {shares} {ticker}...")
    if not is_tradable(ticker):
        print(f"   ❌ SKIPPED: {ticker} not tradable.")
        return False

    side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
    try:
        order_data = MarketOrderRequest(
            symbol=ticker, qty=shares, side=side, time_in_force=TimeInForce.DAY
        )
        alpaca.submit_order(order_data)
        print(f"   ✅ FILLED: {ticker}")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def log_entry(ticker, type_, price, gap, shares):
    # Logs the entry signal to a simple CSV for debugging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Ticker', 'Type', 'Entry', 'Gap', 'Shares'])
        writer.writerow([timestamp, ticker, type_, price, gap, shares])

# ------------------------------------------------------------------------------
# 4. TRADING LOGIC
# ------------------------------------------------------------------------------
def get_nasdaq_tickers():
    print("📡 Fetching Real-Time Ticker List from FMP...")
    # 'volumeMoreThan' ensures we only look at liquid stocks
    url = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=5000&volumeMoreThan={MIN_VOLUME}&priceMoreThan={MIN_PRICE}&apikey={FMP_API_KEY}"
    data = fetch_url(url)
    if isinstance(data, list):
        return [x['symbol'] for x in data]
    return []

def get_realtime_quotes(tickers):
    chunk_size = 500
    all_quotes = []
    print(f"⚡ Fetching Quotes for {len(tickers)} tickers...")
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        url = f"{BASE_URL}/quote/{','.join(chunk)}?apikey={FMP_API_KEY}"
        data = fetch_url(url)
        if isinstance(data, list): all_quotes.extend(data)
    return pd.DataFrame(all_quotes)

def job_entry_scan():
    print("\n" + "="*60)
    print(f"🌊 SEA 15 ENTRY SCAN | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if not check_market_status(): return

    tickers = get_nasdaq_tickers()
    if not tickers: return
    df = get_realtime_quotes(tickers)
    if df.empty: return

    # Process Gaps
    df = df.dropna(subset=['open', 'previousClose', 'price'])
    df = df[ (df['open'] > 0) & (df['previousClose'] > 0) ].copy()
    df['gap_pct'] = (df['open'] - df['previousClose']) / df['previousClose']

    # 1. OPTIMIZED FILTERS ($1-$50, 5-13%)
    shorts = df[ 
        (df['gap_pct'] >= GAP_UP_MIN) & 
        (df['gap_pct'] <= GAP_UP_MAX) & 
        (df['price'] >= MIN_PRICE) & 
        (df['price'] <= MAX_PRICE)
    ].copy()

    longs = df[ 
        (df['gap_pct'] <= GAP_DOWN_THRESHOLD) & 
        (df['price'] >= MIN_PRICE) & 
        (df['price'] <= MAX_PRICE)
    ].copy()

    # 2. RANDOM SAMPLING (The "Robust" Selection)
    # Shuffle and pick top N to simulate random selection from valid pool
    
    # --- Execute Shorts ---
    if not shorts.empty:
        candidates = shorts.to_dict('records')
        random.shuffle(candidates) # <--- Randomize
        selected = candidates[:MAX_SHORTS]
        
        print(f"\n🐻 FOUND {len(shorts)} VALID SHORTS. RANDOMLY SELECTED {len(selected)}:")
        for row in selected:
            entry = row['price']
            shares = int(RISK_PER_TRADE / (entry * STOP_BUFFER))
            if shares > 0:
                print(f"   🔥 SHORT: {row['symbol']} @ ${entry} (Gap {row['gap_pct']:.1%})")
                if execute_alpaca_order(row['symbol'], "SELL", shares):
                    log_entry(row['symbol'], "Short", entry, row['gap_pct'], shares)
    else:
        print("\n🐻 No Short Signals.")

    # --- Execute Longs ---
    if not longs.empty:
        candidates = longs.to_dict('records')
        random.shuffle(candidates)
        selected = candidates[:MAX_LONGS]
        
        print(f"\n🐂 FOUND {len(longs)} VALID LONGS. RANDOMLY SELECTED {len(selected)}:")
        for row in selected:
            entry = row['price']
            shares = int(RISK_PER_TRADE / (entry * STOP_BUFFER))
            if shares > 0:
                print(f"   🔥 BUY: {row['symbol']} @ ${entry} (Gap {row['gap_pct']:.1%})")
                if execute_alpaca_order(row['symbol'], "BUY", shares):
                    log_entry(row['symbol'], "Long", entry, row['gap_pct'], shares)
    else:
        print("\n🐂 No Long Signals.")

def job_exit_flatten():
    print(f"\n🛑 MARKET CLOSE - FLATTENING ACCOUNT...")
    try:
        alpaca.close_all_positions(cancel_orders=True)
        print("✅ ALL POSITIONS CLOSED. SLEEPING.")
    except Exception as e:
        print(f"❌ FLAT FAILED: {e}")

# ------------------------------------------------------------------------------
# 5. SATURDAY KPI REPORTING
# ------------------------------------------------------------------------------
def job_weekly_report():
    print("\n" + "="*60)
    print(f"📊 GENERATING WEEKLY KPI REPORT | {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    
    try:
        # Fetch last 7 days of closed orders
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        req = GetOrdersRequest(status=OrderStatus.CLOSED, limit=500, after=start_date)
        orders = alpaca.get_orders(req)
        
        if not orders:
            print("   ⚠️ No orders found this week.")
            return

        trade_list = []
        for o in orders:
            if o.filled_qty is None or float(o.filled_qty) == 0: continue
            trade_list.append({
                'Date': o.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'Symbol': o.symbol,
                'Side': o.side,
                'Qty': o.filled_qty,
                'FillPrice': o.filled_avg_price,
                'Notional': float(o.filled_qty) * float(o.filled_avg_price) if o.filled_avg_price else 0
            })
            
        if trade_list:
            df = pd.DataFrame(trade_list)
            filename = f"{REPORT_DIR}/weekly_ledger_{end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(filename, index=False)
            print(f"   ✅ Report Saved: {filename}")
            print(f"   ✅ Total Trades: {len(df)}")
            print(f"   ✅ Total Volume: ${df['Notional'].sum():,.2f}")
            
    except Exception as e:
        print(f"   ❌ REPORT FAILED: {e}")

# ------------------------------------------------------------------------------
# 6. MASTER SCHEDULER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🌊 SEA 15 CAPTAIN INITIALIZED")
    print(f"⏰ Entry: {ENTRY_TIME} | Exit: {EXIT_TIME}")
    print(f"🎲 Mode:  Random Selection (Limit {MAX_SHORTS} Short / {MAX_LONGS} Long)")
    print(f"📊 Report: Saturdays @ {REPORT_TIME}")
    
    # Schedule
    schedule.every().day.at(ENTRY_TIME).do(job_entry_scan)
    schedule.every().day.at(EXIT_TIME).do(job_exit_flatten)
    schedule.every().saturday.at(REPORT_TIME).do(job_weekly_report)

    while True:
        schedule.run_pending()
        time.sleep(60)