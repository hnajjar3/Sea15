import requests
import pandas as pd
import numpy as np
import os
import csv
import time
import schedule
import random
import threading
import concurrent.futures
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderClass

# ==============================================================================
# 🌊 PROJECT SEA 15: THE TURBO CAPTAIN (FINAL GOLDEN COPY)
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
CACHE_DIR = "live_cache"
MAX_WORKERS = 20  # Number of parallel threads for fetching/trading

# SCHEDULE (PACIFIC TIME)
PREP_TIME  = "03:00"   # Pre-load tickers well before open
ENTRY_TIME = "06:30"   # Trigger logic starts at Open
EXIT_TIME  = "12:55"   
REPORT_TIME = "09:00"  # Saturday

# EXECUTION TIMING
SECONDS_DELAY = 15     # Wait 15s after open for volatility to stabilize

# RISK MANAGEMENT
POSITION_SIZE = 2000.00        # Fixed Position Size (mimics AVG_TRADE_SIZE in BT)
STOP_LOSS_PCT = 0.015          # 1.5% Stop Loss
MAX_SHORTS = 5         
MAX_LONGS = 5          

# STRATEGY PARAMETERS
GAP_UP_MIN = 0.03
GAP_UP_MAX = 0.13
GAP_DOWN_THRESHOLD = -0.03
GAP_DOWN_FLOOR = -0.15          # <--- NEW: Floor to avoid falling knives
MIN_PRICE = 1.00       
MAX_PRICE = 50.00      
MIN_VOLUME = 60_000            # Increased to 50k
MAX_VOLUME = 600_000           # Soft Cap Ceiling
MAX_DOLLAR_VOL = 500_000_000  # <--- NEW: Liquidity Cap
HIGH_VOL_SAMPLE_RATE = 0.010   # Keep 1% of stocks > MAX_VOLUME

# ------------------------------------------------------------------------------
# 2. SETUP CLIENTS & GLOBAL STATE
# ------------------------------------------------------------------------------
BASE_URL = "https://financialmodelingprep.com/api/v3"
alpaca = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

# Global list to hold tickers
TICKER_UNIVERSE = []
# Lock to prevent file corruption during parallel logging
FILE_LOCK = threading.Lock()

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# ------------------------------------------------------------------------------
# 3. HIGH-SPEED NETWORK FUNCTIONS & FILTERS
# ------------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16)
)
def fetch_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"⚠️ Network Error: {e}")
        return []

def get_etf_blacklist():
    """
    Fetches a list of all ETF tickers from FMP to use as a blacklist.
    """
    cache_path = os.path.join(CACHE_DIR, 'etf_blacklist.pkl')
    # Check if cached today
    if os.path.exists(cache_path):
        # Optional: Check file age to force refresh daily
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if file_time.date() == datetime.now().date():
            return pd.read_pickle(cache_path)

    print("🛡️ Fetching ETF Blacklist from FMP...")
    endpoint = f"{BASE_URL}/etf/list?apikey={FMP_API_KEY}"
    try:
        data = fetch_url(endpoint)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            etf_tickers = df['symbol'].tolist()
            pd.to_pickle(set(etf_tickers), cache_path)
            return set(etf_tickers)
        return set()
    except Exception as e:
        print(f"⚠️ Error fetching ETF list: {e}")
        return set()

def get_clean_universe():
    """
    Runs the full filtering logic (Blacklist, Sector, Dollar Volume).
    """
    # 1. Get the Auto-Blacklist (All known ETFs from FMP)
    etf_blacklist = get_etf_blacklist()

    # 2. Define the "Zombie" Blacklist
    manual_blacklist = [
        # --- Leveraged ETFs & Derivatives ---
        'UVXY', 'NVDS', 'TSDD', 'CONI', 'BTF', 'DGXX',
        # --- Micro-Cap / Illiquid ---
        'WKSP', 'FTCI', 'DPRO', 'PXLW', 'JNVR', 'EUDA', 'RADX', 'CD'
    ]

    print("📡 Fetching new Ticker List from FMP...")
    # Request sector/industry/exchange to filter
    # Note: We filter strictly by NASDAQ exchange in the query
    endpoint = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=10000&apikey={FMP_API_KEY}"

    try:
        data = fetch_url(endpoint)
        if isinstance(data, list):
            df = pd.DataFrame(data)

            # --- FILTERING STAGE 1: IDENTIFIED TOXIC ASSETS ---
            # Remove FMP ETFs and our Manual "Zombie" list
            df = df[~df['symbol'].isin(etf_blacklist)]
            df = df[~df['symbol'].isin(manual_blacklist)]

            # --- FILTERING STAGE 2: SECTOR LOCKDOWN ---
            if 'sector' in df.columns:
                toxic_sectors = ['Bogus']
                df = df[~df['sector'].isin(toxic_sectors)]

            # --- FILTERING STAGE 3: KEYWORD SAFETY NET ---
            toxic_keywords = [
                'ETF', '2X', '3X', 'BULL', 'BEAR', 'INVERSE', 'SHORT',
                'TRUST', 'FUND', 'LP', 'SPAC', 'ETN', 'ACQUISITION',
                'CRYPTO', 'BITCOIN', 'BLOCKCHAIN'
            ]
            pattern = '|'.join(toxic_keywords)
            # Filter Company Name
            df = df[~df['companyName'].str.upper().str.contains(pattern, na=False)]

            # --- FILTERING STAGE 4: LIQUIDITY & PRICE ---
            # Also calculate Dollar Volume (Price * Volume)
            # FMP Screener usually returns last close data
            if 'price' in df.columns and 'volume' in df.columns:
                df['dollar_vol'] = df['price'] * df['volume']

                df = df[
                    (df['price'] > 2.00) &
                    (df['volume'] > 10000) &
                    (df['dollar_vol'] < MAX_DOLLAR_VOL) # <--- NEW CAP
                ]

            # Country Check (Default FMP screener might include others, but we usually want US)
            # BT has US_STOCKS_ONLY = False, so we SKIP country check to match BT.

            tickers = df['symbol'].tolist()
            print(f"✅ Filtered Universe Size: {len(tickers)} tickers")
            return tickers
        return []
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []

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
    """Worker function to execute a Bracket Order (Entry + SL)."""
    print(f"   🚀 FIRE: {action} {shares} {ticker} @ ~${price}...")
    
    side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
    
    # 1. Calculate Hard Stop Price
    if action == "BUY":
        stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
    else: # SHORT
        stop_price = round(price * (1 + STOP_LOSS_PCT), 2)

    try:
        # 2. Construct Bracket Order
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,  # <--- SAFETY ON
            stop_loss=StopLossRequest(stop_price=stop_price)
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
    """Thread-safe logging to CSV."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(LOG_FILE)
    
    with FILE_LOCK:  # <--- Critical for multi-threading
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Date', 'Ticker', 'Type', 'Entry', 'Gap', 'Shares'])
            writer.writerow([timestamp, ticker, type_, price, gap, shares])

# ------------------------------------------------------------------------------
# 5. CORE LOGIC JOBS
# ------------------------------------------------------------------------------
def job_update_universe():
    """Runs Pre-Market to load the gun using Robust Filters."""
    global TICKER_UNIVERSE
    print("\n" + "="*60)
    print(f"📡 PRE-MARKET PREP | {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    cache_path = os.path.join(CACHE_DIR, 'universe_cache.pkl')

    # Try to load cache if we are just restarting the script mid-day
    # But usually we want to run this fresh at 3AM

    clean_tickers = get_clean_universe()
    if clean_tickers:
        TICKER_UNIVERSE = clean_tickers
        pd.to_pickle(TICKER_UNIVERSE, cache_path)
        print(f"✅ UNIVERSE UPDATED: {len(TICKER_UNIVERSE)} tickers ready in memory.")
    else:
        print("❌ FAILED to update universe. Trying to load old cache...")
        if os.path.exists(cache_path):
            TICKER_UNIVERSE = pd.read_pickle(cache_path)
            print(f"⚠️ LOADED CACHED UNIVERSE: {len(TICKER_UNIVERSE)} tickers.")

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
    df = df.dropna(subset=['price', 'previousClose', 'volume'])
    df = df[ (df['price'] > 0) & (df['previousClose'] > 0) ].copy()
    df['gap_pct'] = (df['price'] - df['previousClose']) / df['previousClose']

    # 4. FILTER (With Volume Soft Cap)
    # Basic Filters first
    candidates = df[
        (df['price'] >= MIN_PRICE) & 
        (df['price'] <= MAX_PRICE) &
        (df['volume'] >= MIN_VOLUME)
    ].copy()

    # Apply Volume Soft Cap (Imitate Backtest Logic)
    # Split into High Vol and Low Vol
    high_vol = candidates[candidates['volume'] > MAX_VOLUME].copy()
    low_vol = candidates[candidates['volume'] <= MAX_VOLUME].copy()

    # Sample High Vol
    if not high_vol.empty:
        # We use a random mask to keep ~10%
        # This matches the "Soft Ceiling" logic
        high_vol = high_vol.sample(frac=HIGH_VOL_SAMPLE_RATE)

    # Recombine
    candidates = pd.concat([low_vol, high_vol])

    if candidates.empty:
        print("💤 No candidates after filtering.")
        return

    shorts = candidates[
        (candidates['gap_pct'] >= GAP_UP_MIN) &
        (candidates['gap_pct'] <= GAP_UP_MAX)
    ].copy()

    longs = candidates[
        (candidates['gap_pct'] <= GAP_DOWN_THRESHOLD) &
        (candidates['gap_pct'] >= GAP_DOWN_FLOOR)  # <--- NEW: Floor logic applied
    ].copy()

    # 5. RANDOM SELECTION
    orders_to_send = []

    # --- Prepare Shorts ---
    if not shorts.empty:
        short_list = shorts.to_dict('records')
        random.shuffle(short_list)
        selected = short_list[:MAX_SHORTS]
        print(f"\n🐻 SHORTS ({len(shorts)} found, picking {len(selected)}):")
        for row in selected:
            shares = int(POSITION_SIZE / row['price'])
            if shares > 0:
                orders_to_send.append((row['symbol'], "SELL", shares, row['price'], row['gap_pct']))
    else:
        print("\n🐻 No Shorts.")

    # --- Prepare Longs ---
    if not longs.empty:
        long_list = longs.to_dict('records')
        random.shuffle(long_list)
        selected = long_list[:MAX_LONGS]
        print(f"\n🐂 LONGS ({len(longs)} found, picking {len(selected)}):")
        for row in selected:
            shares = int(POSITION_SIZE / row['price'])
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
    print(f"🛡️ Safety: Bracket Orders (Auto-SL attached)")
    
    # 1. Pre-Market Prep
    schedule.every().day.at(PREP_TIME).do(job_update_universe)
    
    # 2. The Trade Trigger
    schedule.every().day.at(ENTRY_TIME).do(job_entry_scan_turbo)
    
    # 3. Exit & Report
    schedule.every().day.at(EXIT_TIME).do(job_exit_flatten)
    schedule.every().saturday.at(REPORT_TIME).do(job_weekly_report)

    # Run Prep immediately if starting mid-day
    if not TICKER_UNIVERSE:
        job_update_universe()

    while True:
        schedule.run_pending()
        time.sleep(1)