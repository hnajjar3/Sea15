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
API_KEY = 'iuRa8nmjQdLOx66QtJiSWFitzOMqu6QF' 
BASE_URL = "https://financialmodelingprep.com/api/v3"
CACHE_DIR = 'market_data_cache'
GAP_UP_MIN, GAP_UP_MAX = 0.05, 0.13
GAP_DOWN_THRESHOLD = -0.05
MIN_PRICE, MAX_PRICE = 1.00, 50.00
MIN_VOLUME = 10000 
LOOKBACK_DAYS = 1095
MAX_TRADES_PER_DAY = 10  # 5 Short + 5 Long
FORCE_UPDATE = False

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
    # We ask for a broader list, but filter strictly later
    endpoint = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=10000&apikey={API_KEY}"
    try:
        data = fetch_url(endpoint)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            # Basic pre-filter to reduce download size
            if 'price' in df.columns and 'volume' in df.columns:
                df = df[ (df['price'] > 0.50) & (df['volume'] > 1000) ]
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
# BACKTEST LOGIC (OPTIMIZED)
# ==========================================
def run_backtest_robust(tickers):
    print(f"⚡ Starting Robust Backtest (No Look-Ahead + Daily Limits)...")
    
    all_signals = []
    
    for i, ticker in enumerate(tickers):
        if i % 100 == 0: print(f"   Scanning {i}/{len(tickers)}...")
            
        df = get_historical_data(ticker)
        if df.empty or len(df) < 5: continue
            
        # 1. FIX: Calculate Yesterday's Volume
        # We shift volume by 1 so 'vol_prev' is available at 9:30 AM today
        df['vol_prev'] = df['volume'].shift(1)
        
        # 2. Filter Validity
        df = df[(df['open'] > 0.01) & (df['close'] > 0.01) & (df['vol_prev'] > MIN_VOLUME)].copy()
        
        # Calculate Gaps
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        # 3. Identify Shorts
        short_mask = (
            (df['gap_pct'] >= GAP_UP_MIN) & 
            (df['gap_pct'] <= GAP_UP_MAX) & 
            (df['open'] >= MIN_PRICE) & 
            (df['open'] <= MAX_PRICE)
        )
        if short_mask.any():
            shorts = df[short_mask].copy()
            for date, row in shorts.iterrows():
                pnl = (row['open'] - row['close']) / row['open']
                all_signals.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Short',
                    'Price': row['open'], 'Gap': row['gap_pct'], 'PnL': pnl
                })

        # 4. Identify Longs
        long_mask = (
            (df['gap_pct'] <= GAP_DOWN_THRESHOLD) & 
            (df['open'] >= MIN_PRICE) & 
            (df['open'] <= MAX_PRICE)
        )
        if long_mask.any():
            longs = df[long_mask].copy()
            for date, row in longs.iterrows():
                pnl = (row['close'] - row['open']) / row['open']
                all_signals.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Long',
                    'Price': row['open'], 'Gap': row['gap_pct'], 'PnL': pnl
                })

    # --- THE REALITY CHECK (Daily Limits) ---
    print("🎲 Applying Daily Limit Logic (Random Selection)...")
    full_df = pd.DataFrame(all_signals)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    
    final_trades = []
    
    # Group by Date to simulate a single trading day
    for date, group in full_df.groupby('Date'):
        # Split into Short/Long
        shorts = group[group['Type'] == 'Short']
        longs = group[group['Type'] == 'Long']
        
        # Randomly pick up to 5 Shorts
        if len(shorts) > (MAX_TRADES_PER_DAY // 2):
            shorts = shorts.sample(n=(MAX_TRADES_PER_DAY // 2), random_state=42)
        
        # Randomly pick up to 5 Longs
        if len(longs) > (MAX_TRADES_PER_DAY // 2):
            longs = longs.sample(n=(MAX_TRADES_PER_DAY // 2), random_state=42)
            
        final_trades.extend(shorts.to_dict('records'))
        final_trades.extend(longs.to_dict('records'))
        
    return pd.DataFrame(final_trades)

# ==========================================
# REPORTING & VISUALIZATION
# ==========================================
def generate_report(results):
    if results.empty:
        print("No trades found.")
        return

    print("\n" + "="*60)
    print("📊 3-YEAR OPTIMIZED STRATEGY REPORT")
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

        # Weekly Stats
        weekly_pnl = subset.set_index('Date')['PnL'].resample('W').sum()
        best_week = weekly_pnl.max()
        worst_week = weekly_pnl.min()
        
        print("\n" + "="*60)
        print(f"🌊 SEA 15 OPTIMIZED: {t_type.upper()}")
        print("="*60)
        print(f"   Trades:         {len(subset)}")
        print(f"   Win Rate:       {win_rate:.2f}%")
        print(f"   Avg Return:     {avg_ret:.2f}%")
        print(f"   Total Units:    {total_pnl:.2f} R")
        print(f"   Max Drawdown:   {max_dd:.2f} R")
        print(f"   Best Week:      {best_week:.2f} R")
        print(f"   Worst Week:     {worst_week:.2f} R")
        
        plt.plot(subset['Date'], subset['Equity'], label=f"{t_type} ({total_pnl:.1f}R)")

    plt.title(f'Equity Curve: Gap {int(GAP_UP_MIN*100)}-{int(GAP_UP_MAX*100)}% | Price ${int(MIN_PRICE)}-${int(MAX_PRICE)}')
    plt.xlabel('Date')
    plt.ylabel('Accumulated Risk Units (R)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('nasdaq_optimized_curve.png')
    print("\n✅ Chart saved to 'nasdaq_optimized_curve.png'")
    
    results.to_csv('nasdaq_optimized_results.csv', index=False)
    print("✅ Data saved to 'nasdaq_optimized_results.csv'")

if __name__ == "__main__":
    if API_KEY == 'YOUR_FMP_KEY':
        print("❌ ERROR: Please insert your FMP API Key")
    else:
        valid_tickers = get_nasdaq_tickers()
        if valid_tickers:
            # Note: Set FORCE_UPDATE = True if you need to re-download 3 years of data
            results = run_backtest_robust(valid_tickers)
            generate_report(results)