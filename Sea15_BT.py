
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = 'iuRa8nmjQdLOx66QtJiSWFitzOMqu6QF'   # <--- PASTE KEY HERE
BASE_URL = "https://financialmodelingprep.com/api/v3"

# Cache Settings
CACHE_DIR = 'market_data_cache'
FORCE_UPDATE = False

# Strategy Parameters
# 1. SHORT STRATEGY (The "Sweet Spot")
GAP_UP_MIN = 0.05       # Short > 5%
GAP_UP_MAX = 0.15       # UPDATED: Cap at 15% to avoid "Widowmaker" squeezes

# 2. LONG STRATEGY (The "Dip Buy")
GAP_DOWN_THRESHOLD = -0.05   # Long if Gap < -5%

MIN_PRICE = 2.00           # Raised to $2.00 to filter out total garbage
LOOKBACK_DAYS = 1095       # UPDATED: 3 Years (3 * 365)

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class APIError(Exception):
    pass

# ==========================================
# ROBUST API HANDLER
# ==========================================
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(APIError)
)
def fetch_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 429 or response.status_code >= 500:
            print(f"⚠️ Rate Limit/Server Error ({response.status_code}). Backing off...")
            raise APIError(f"Status {response.status_code}")
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network Error: {e}")
        raise APIError(str(e))

# ==========================================
# DATA FETCHING
# ==========================================
def get_nasdaq_tickers():
    cache_path = os.path.join(CACHE_DIR, 'nasdaq_tickers_list.pkl')
    if not FORCE_UPDATE and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
    print("📡 Fetching new Ticker List from FMP...")
    endpoint = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=10000&apikey={API_KEY}"
    try:
        data = fetch_url(endpoint)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            # Pre-filter for liquidity to save time
            if 'price' in df.columns and 'volume' in df.columns:
                df = df[ (df['price'] > 1.00) & (df['volume'] > 10000) ]
            tickers = df['symbol'].tolist()
            pd.to_pickle(tickers, cache_path)
            return tickers
        return []
    except Exception:
        return []

def get_historical_data(ticker):
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    
    # Check if cache exists
    if not FORCE_UPDATE and os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        # We use existing cache. If you want to force redownload for 3 years data, 
        # delete the cache folder or set FORCE_UPDATE = True
        return df
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    endpoint = f"{BASE_URL}/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={API_KEY}"
    
    try:
        data = fetch_url(endpoint)
        if isinstance(data, dict) and 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            df.to_pickle(cache_path)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ==========================================
# BACKTEST LOGIC
# ==========================================
def run_backtest(tickers):
    print(f"⚡ Starting Backtest on {len(tickers)} tickers...")
    print(f"   Lookback: {LOOKBACK_DAYS} days")
    print(f"   Settings: Short {int(GAP_UP_MIN*100)}-{int(GAP_UP_MAX*100)}%, Long Gap < {int(GAP_DOWN_THRESHOLD*100)}%")
    
    trade_log = []
    
    for i, ticker in enumerate(tickers):
        if i % 100 == 0: print(f"   Processed {i}/{len(tickers)}...")
            
        df = get_historical_data(ticker)
        
        if df.empty or len(df) < 5:
            continue
            
        # --- DATA CLEANING ---
        df = df[(df['open'] > 0.01) & (df['close'] > 0.01)].copy()
        
        # Calculate Gaps
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        # --- STRATEGY LOGIC ---
        
        # SHORT (Sweet Spot: 5% < Gap < 15%)
        short_mask = (df['gap_pct'] > GAP_UP_MIN) & (df['gap_pct'] <= GAP_UP_MAX) & (df['open'] > MIN_PRICE)
        if short_mask.any():
            shorts = df[short_mask].copy()
            shorts['pnl_pct'] = (shorts['open'] - shorts['close']) / shorts['open']
            
            for date, row in shorts.iterrows():
                trade_log.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Short',
                    'Price': row['open'], 'Gap': row['gap_pct'], 'PnL': row['pnl_pct']
                })

        # LONG (Dip Buy: Gap < -5%)
        long_mask = (df['gap_pct'] < GAP_DOWN_THRESHOLD) & (df['open'] > MIN_PRICE)
        if long_mask.any():
            longs = df[long_mask].copy()
            longs['pnl_pct'] = (longs['close'] - longs['open']) / longs['open']
            
            for date, row in longs.iterrows():
                trade_log.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Long',
                    'Price': row['open'], 'Gap': row['gap_pct'], 'PnL': row['pnl_pct']
                })
                
    return pd.DataFrame(trade_log)

# ==========================================
# REPORTING & VISUALIZATION
# ==========================================
def generate_report(results):
    if results.empty:
        print("No trades found.")
        return

    print("\n" + "="*60)
    print("📊 3-YEAR STRATEGY REPORT")
    print("="*60)

    results['Date'] = pd.to_datetime(results['Date'])
    
    plt.figure(figsize=(12, 6))

    for t_type in ['Short', 'Long']:
        subset = results[results['Type'] == t_type].copy()
        if subset.empty: continue
        
        subset = subset.sort_values('Date')
        
        # --- METRICS ---
        win_rate = (len(subset[subset['PnL'] > 0]) / len(subset)) * 100
        avg_ret = subset['PnL'].mean() * 100
        total_pnl = subset['PnL'].sum()
        
        # Equity Curve
        subset['Equity'] = subset['PnL'].cumsum()
        
        # Drawdown
        subset['HWM'] = subset['Equity'].cummax()
        subset['DD'] = subset['Equity'] - subset['HWM']
        max_dd = subset['DD'].min()

        # Weekly Stats (Added back in)
        weekly_pnl = subset.set_index('Date')['PnL'].resample('W').sum()
        best_week = weekly_pnl.max()
        worst_week = weekly_pnl.min()
        
        print("\n" + "="*60)
        print("🌊 PROJECT SEA 15: ALGORITHMIC REPORT 🌊")
        print("="*60)
        print(f"   Trades:         {len(subset)}")
        print(f"   Win Rate:       {win_rate:.2f}%")
        print(f"   Avg Return:     {avg_ret:.2f}%")
        print(f"   Total Units:    {total_pnl:.2f} R")
        print(f"   Max Drawdown:   {max_dd:.2f} R")
        print(f"   Best Week:      {best_week:.2f} R")
        print(f"   Worst Week:     {worst_week:.2f} R")
        
        plt.plot(subset['Date'], subset['Equity'], label=f"{t_type} ({total_pnl:.1f}R)")

    plt.title(f'Equity Curve (3 Years): Short 5-15% | Long < -5%')
    plt.xlabel('Date')
    plt.ylabel('Accumulated Risk Units (R)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('nasdaq_3yr_curve.png')
    print("\n✅ Chart saved to 'nasdaq_3yr_curve.png'")
    
    results.to_csv('nasdaq_3yr_results.csv', index=False)
    print("✅ Data saved to 'nasdaq_3yr_results.csv'")

if __name__ == "__main__":
    if API_KEY == 'YOUR_FMP_KEY':
        print("❌ ERROR: Please insert your FMP API Key")
    else:
        valid_tickers = get_nasdaq_tickers()
        if valid_tickers:
            # Note: Set FORCE_UPDATE = True if you need to re-download 3 years of data
            # FORCE_UPDATE = True 
            results = run_backtest(valid_tickers)
            generate_report(results)