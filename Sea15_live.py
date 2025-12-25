import requests
import pandas as pd
import numpy as np
import os
import csv
import time
import schedule
import random
import concurrent.futures
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderClass
# ==============================================================================
# 🌊 PROJECT SEA 15: THE TURBO CAPTAIN (MULTI-THREADED)
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
MAX_WORKERS = 20  # Number of parallel threads for fetching/trading

# SCHEDULE (PACIFIC TIME)
PREP_TIME  = "06:00"   # Pre-load tickers 30 mins before open
ENTRY_TIME = "06:30"   # Trigger exactly at Open
EXIT_TIME  = "12:55"   
REPORT_TIME = "09:00"  # Saturday

# EXECUTION TIMING
SECONDS_DELAY = 15     # Wait 15s after open for volatility to stabilize

# RISK MANAGEMENT
RISK_PER_TRADE = 1000.00        
STOP_BUFFER = 0.05             
MAX_SHORTS = 5         
MAX_LONGS = 5          

# STRATEGY PARAMETERS
GAP_UP_MIN = 0.05              
GAP_UP_MAX = 0.13      
GAP_DOWN_THRESHOLD = -0.05     
MIN_PRICE = 1.00       
MAX_PRICE = 50.00      
MIN_VOLUME = 10000     

# ------------------------------------------------------------------------------
# 2. SETUP CLIENTS & GLOBAL STATE
# ------------------------------------------------------------------------------
BASE_URL = "https://financialmodelingprep.com/api/v3"
alpaca = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

# Global list to hold tickers so we don't fetch them at the bell
TICKER_UNIVERSE = []

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

# ------------------------------------------------------------------------------
# 3. HIGH-SPEED NETWORK FUNCTIONS
# ------------------------------------------------------------------------------
def fetch_batch_quotes(tickers):
    """Worker function to fetch a single batch of quotes."""
    if not tickers: return []
    try:
        url = f"{BASE_URL}/quote/{','.join(tickers)}?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def get_realtime_quotes_turbo(tickers):
    """
    ⚡ PARALLEL FETCH: Downloads 5000+ quotes in ~1 second.
    """
    chunk_size = 500
    # Split tickers into chunks
    batches = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    all_quotes = []
    
    print(f"⚡ TURBO FETCH: Launching {len(batches)} parallel requests...")
    start_ts = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_batch_quotes, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            all_quotes.extend(future.result())
            
    print(f"⚡ FETCH COMPLETE: {len(all_quotes)} quotes in {time.time() - start_ts:.2f}s")
    return pd.DataFrame(all_quotes)

def execute_order_async(ticker, action, shares, price, gap):
    """Worker function to execute a single order WITH ATTACHED STOP LOSS."""
    print(f"   🚀 FIRE: {action} {shares} {ticker} @ ~${price}...")
    
    side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
    
    # 1. Calculate the Hard Stop Price based on the LIVE quote
    # We use the 'price' passed from the scanner (which is the live quote)
    if action == "BUY":
        stop_price = round(price * (1 - STOP_BUFFER), 2)
    else: # SHORT
        stop_price = round(price * (1 + STOP_BUFFER), 2)

    try:
        # 2. Construct the Bracket Order
        # This sends: Market Entry + Stop Loss Exit simultaneously
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,  # <--- Tells Alpaca this is a complex order
            stop_loss=StopLossRequest(stop_price=stop_price) # <--- The Protection
        )
        
        alpaca.submit_order(order_data)
        
        # Log success
        log_entry(ticker, action, price, gap, shares)
        print(f"   ✅ SENT: {ticker} (SL: ${stop_price})")
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR {ticker}: {e}")
        return False

# ------------------------------------------------------------------------------
# 4. STANDARD HELPERS
# ------------------------------------------------------------------------------
def check_market_status():
    try:
        clock = alpaca.get_clock()
        if not clock.is_open:
            print(f"💤 Market is CLOSED. Next open: {clock.next_open}")
            return False
        return True
    except: return True 

def log_entry(ticker, type_, price, gap, shares):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Ticker', 'Type', 'Entry', 'Gap', 'Shares'])
        writer.writerow([timestamp, ticker, type_, price, gap, shares])

# ------------------------------------------------------------------------------
# 5. CORE LOGIC JOBS
# ------------------------------------------------------------------------------
def job_update_universe():
    """Runs Pre-Market to load the gun."""
    global TICKER_UNIVERSE
    print("\n" + "="*60)
    print(f"📡 PRE-MARKET PREP | {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    url = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=6000&volumeMoreThan={MIN_VOLUME}&priceMoreThan={MIN_PRICE}&apikey={FMP_API_KEY}"
    try:
        data = requests.get(url).json()
        if isinstance(data, list):
            TICKER_UNIVERSE = [x['symbol'] for x in data]
            print(f"✅ UNIVERSE UPDATED: {len(TICKER_UNIVERSE)} tickers ready in memory.")
        else:
            print("❌ FAILED to download ticker list.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def job_entry_scan_turbo():
    global TICKER_UNIVERSE
    
    print("\n" + "="*60)
    print(f"🌊 SEA 15 TURBO SCAN | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if not check_market_status(): return
    
    # 0. WAIT FOR DUST TO SETTLE
    print(f"⏳ Market Open. Waiting {SECONDS_DELAY}s for liquidity...")
    time.sleep(SECONDS_DELAY)

    # 1. CHECK UNIVERSE
    if not TICKER_UNIVERSE:
        print("⚠️ Universe empty. Emergency fetch...")
        job_update_universe()
        if not TICKER_UNIVERSE: return

    # 2. FAST FETCH
    df = get_realtime_quotes_turbo(TICKER_UNIVERSE)
    if df.empty: return

    # 3. PROCESS GAPS
    # We use 'price' (Realtime) vs 'previousClose'
    df = df.dropna(subset=['price', 'previousClose'])
    df = df[ (df['price'] > 0) & (df['previousClose'] > 0) ].copy()
    df['gap_pct'] = (df['price'] - df['previousClose']) / df['previousClose']

    # 4. FILTER
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

    # 5. RANDOM SELECTION
    orders_to_send = []

    # --- Prepare Shorts ---
    if not shorts.empty:
        candidates = shorts.to_dict('records')
        random.shuffle(candidates)
        selected = candidates[:MAX_SHORTS]
        print(f"\n🐻 SHORTS ({len(shorts)} found, picking {len(selected)}):")
        for row in selected:
            shares = int(RISK_PER_TRADE / (row['price'] * STOP_BUFFER))
            if shares > 0:
                orders_to_send.append((row['symbol'], "SELL", shares, row['price'], row['gap_pct']))
    else:
        print("\n🐻 No Shorts.")

    # --- Prepare Longs ---
    if not longs.empty:
        candidates = longs.to_dict('records')
        random.shuffle(candidates)
        selected = candidates[:MAX_LONGS]
        print(f"\n🐂 LONGS ({len(longs)} found, picking {len(selected)}):")
        for row in selected:
            shares = int(RISK_PER_TRADE / (row['price'] * STOP_BUFFER))
            if shares > 0:
                orders_to_send.append((row['symbol'], "BUY", shares, row['price'], row['gap_pct']))
    else:
        print("\n🐂 No Longs.")

    # 6. TURBO EXECUTION (PARALLEL)
    if orders_to_send:
        print(f"\n⚡ TURBO EXEC: Firing {len(orders_to_send)} Orders Simultaneously...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(execute_order_async, t, a, s, p, g) 
                for t, a, s, p, g in orders_to_send
            ]
            concurrent.futures.wait(futures)
        print("✅ EXECUTION COMPLETE.")
    else:
        print("💤 No valid trades to execute.")

def job_exit_flatten():
    print(f"\n🛑 MARKET CLOSE - FLATTENING ACCOUNT...")
    try:
        alpaca.close_all_positions(cancel_orders=True)
        print("✅ ALL POSITIONS CLOSED. SLEEPING.")
    except Exception as e:
        print(f"❌ FLAT FAILED: {e}")

def job_weekly_report():
    print("\n" + "="*60)
    print(f"📊 WEEKLY REPORT | {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    try:
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
            print(f"   ✅ Saved: {filename}")
            print(f"   ✅ Volume: ${df['Notional'].sum():,.2f}")
            
    except Exception as e:
        print(f"   ❌ REPORT FAILED: {e}")

# ------------------------------------------------------------------------------
# 6. MASTER SCHEDULER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🌊 SEA 15 TURBO CAPTAIN INITIALIZED")
    print(f"⏰ Prep:  {PREP_TIME} (PT)")
    print(f"⏰ Entry: {ENTRY_TIME} (PT) + {SECONDS_DELAY}s delay")
    print(f"⏰ Exit:  {EXIT_TIME} (PT)")
    print(f"🚀 Speed: Multi-Threaded ({MAX_WORKERS} Workers)")
    
    # 1. Pre-Market Prep
    schedule.every().day.at(PREP_TIME).do(job_update_universe)
    
    # 2. The Trade Trigger (Starts at 06:30, waits 15s internal)
    schedule.every().day.at(ENTRY_TIME).do(job_entry_scan_turbo)
    
    # 3. Exit & Report
    schedule.every().day.at(EXIT_TIME).do(job_exit_flatten)
    schedule.every().saturday.at(REPORT_TIME).do(job_weekly_report)

    # Run Prep immediately if starting during the day for testing
    if not TICKER_UNIVERSE:
        job_update_universe()

    while True:
        schedule.run_pending()
        time.sleep(1) # Tighter loop for responsiveness