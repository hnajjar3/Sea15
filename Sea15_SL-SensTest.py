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

# Strategy Params
GAP_UP_MIN, GAP_UP_MAX = 0.05, 0.13
GAP_DOWN_THRESHOLD = -0.05
MIN_PRICE, MAX_PRICE = 1.00, 50.00
MIN_VOLUME = 10000 

# We will test these Stop Loss levels simultaneously
SL_LEVELS = [0.05, 0.10, 0.15, 0.20, None] 

# Backtest Params
LOOKBACK_DAYS = 1095
MAX_TRADES_PER_DAY = 10 
FORCE_UPDATE = False

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class APIError(Exception):
    pass

# ==========================================
# API HANDLER
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

def get_nasdaq_tickers():
    cache_path = os.path.join(CACHE_DIR, 'nasdaq_tickers_list.pkl')
    if not FORCE_UPDATE and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
    print("📡 Fetching new Ticker List...")
    endpoint = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=10000&apikey={API_KEY}"
    try:
        data = fetch_url(endpoint)
        if isinstance(data, list):
            df = pd.DataFrame(data)
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
    if not FORCE_UPDATE and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
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
# MULTI-SL BACKTEST LOGIC
# ==========================================
def calculate_pnl(row, trade_type, sl_pct):
    """Helper to calculate PnL for a specific SL level"""
    open_price = row['open']
    close_price = row['close']
    high_price = row['high']
    low_price = row['low']
    
    if trade_type == 'Short':
        if sl_pct is None:
            return (open_price - close_price) / open_price
        
        stop_price = open_price * (1 + sl_pct)
        if high_price >= stop_price:
            return -sl_pct # Stopped out
        else:
            return (open_price - close_price) / open_price

    elif trade_type == 'Long':
        if sl_pct is None:
            return (close_price - open_price) / open_price
        
        stop_price = open_price * (1 - sl_pct)
        if low_price <= stop_price:
            return -sl_pct # Stopped out
        else:
            return (close_price - open_price) / open_price
            
    return 0.0

def run_sensitivity_test(tickers):
    print(f"⚡ Starting Sensitivity Analysis on SL Levels: {SL_LEVELS}")
    
    all_signals = []
    
    for i, ticker in enumerate(tickers):
        if i % 100 == 0: print(f"   Scanning {i}/{len(tickers)}...")
        df = get_historical_data(ticker)
        if df.empty or len(df) < 5: continue
            
        df['vol_prev'] = df['volume'].shift(1)
        df = df[(df['open'] > 0.01) & (df['close'] > 0.01) & (df['vol_prev'] > MIN_VOLUME)].copy()
        
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        # --- SHORTS ---
        short_mask = (
            (df['gap_pct'] >= GAP_UP_MIN) & 
            (df['gap_pct'] <= GAP_UP_MAX) & 
            (df['open'] >= MIN_PRICE) & 
            (df['open'] <= MAX_PRICE)
        )
        if short_mask.any():
            shorts = df[short_mask].copy()
            for date, row in shorts.iterrows():
                trade_data = {
                    'Date': date, 'Ticker': ticker, 'Type': 'Short',
                    'Price': row['open'], 'Gap': row['gap_pct']
                }
                # Calculate PnL for EVERY SL level simultaneously
                for sl in SL_LEVELS:
                    col_name = f"PnL_{str(sl)}" if sl else "PnL_NoSL"
                    trade_data[col_name] = calculate_pnl(row, 'Short', sl)
                
                all_signals.append(trade_data)

        # --- LONGS ---
        long_mask = (
            (df['gap_pct'] <= GAP_DOWN_THRESHOLD) & 
            (df['open'] >= MIN_PRICE) & 
            (df['open'] <= MAX_PRICE)
        )
        if long_mask.any():
            longs = df[long_mask].copy()
            for date, row in longs.iterrows():
                trade_data = {
                    'Date': date, 'Ticker': ticker, 'Type': 'Long',
                    'Price': row['open'], 'Gap': row['gap_pct']
                }
                # Calculate PnL for EVERY SL level simultaneously
                for sl in SL_LEVELS:
                    col_name = f"PnL_{str(sl)}" if sl else "PnL_NoSL"
                    trade_data[col_name] = calculate_pnl(row, 'Long', sl)

                all_signals.append(trade_data)

    # --- DAILY LIMIT SELECTION ---
    print("🎲 Applying Daily Limit Logic (Ensuring identical trades for all SLs)...")
    full_df = pd.DataFrame(all_signals)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    
    final_trades = []
    
    for date, group in full_df.groupby('Date'):
        shorts = group[group['Type'] == 'Short']
        longs = group[group['Type'] == 'Long']
        
        # Use random_state=42 to ensure we pick the SAME trades 
        # that we would have picked in previous single-runs
        if len(shorts) > (MAX_TRADES_PER_DAY // 2):
            shorts = shorts.sample(n=(MAX_TRADES_PER_DAY // 2), random_state=42)
        if len(longs) > (MAX_TRADES_PER_DAY // 2):
            longs = longs.sample(n=(MAX_TRADES_PER_DAY // 2), random_state=42)
            
        final_trades.extend(shorts.to_dict('records'))
        final_trades.extend(longs.to_dict('records'))
        
    return pd.DataFrame(final_trades)

# ==========================================
# COMPARISON REPORT
# ==========================================
def generate_comparison_report(results):
    if results.empty:
        print("No trades found.")
        return

    results['Date'] = pd.to_datetime(results['Date'])
    
    print("\n" + "="*80)
    print("🔬 STOP LOSS SENSITIVITY ANALYSIS")
    print("="*80)
    
    summary_stats = []

    plt.figure(figsize=(14, 8))
    
    # Iterate through our columns (PnL_0.05, PnL_0.1, etc)
    sl_cols = [c for c in results.columns if c.startswith('PnL_')]
    sl_cols.sort() # Ensure consistent order

    for col in sl_cols:
        # Calculate stats for this SL level
        total_r = results[col].sum()
        avg_r = results[col].mean() * 100
        win_rate = (len(results[results[col] > 0]) / len(results)) * 100
        
        # Drawdown calculation
        equity_curve = results.sort_values('Date')[col].cumsum()
        hwm = equity_curve.cummax()
        dd = equity_curve - hwm
        max_dd = dd.min()
        
        # --- FIXED LABEL FORMATTING ---
        raw_sl = col.replace("PnL_", "")
        
        if raw_sl == "NoSL":
            sl_label = "No SL"
        else:
            try:
                sl_label = f"{float(raw_sl)*100:.0f}%"
            except ValueError:
                sl_label = raw_sl # Fallback just in case
        
        summary_stats.append({
            "SL Level": sl_label,
            "Total R": total_r,
            "Win Rate": win_rate,
            "Avg Return": avg_r,
            "Max DD": max_dd
        })
        
        plt.plot(results.sort_values('Date')['Date'], equity_curve, label=f"SL {sl_label} (Total: {total_r:.1f}R)")

    # Print Text Table
    stats_df = pd.DataFrame(summary_stats)
    print(stats_df.to_string(index=False))
    
    print("\n" + "="*80)
    # Find best based on Total R
    best_perf = stats_df.loc[stats_df['Total R'].idxmax()]
    print(f"🏆 OPTIMAL CONFIGURATION: {best_perf['SL Level']} (Total: {best_perf['Total R']:.2f}R)")
    print("="*80)

    plt.title('Equity Curve Comparison: Finding the Optimal Stop Loss')
    plt.xlabel('Date')
    plt.ylabel('Accumulated Risk Units (R)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sl_sensitivity_analysis.png')
    print("✅ Chart saved to 'sl_sensitivity_analysis.png'")
    
    results.to_csv('sl_sensitivity_results.csv', index=False)
    print("✅ Data saved to 'sl_sensitivity_results.csv'")

if __name__ == "__main__":
    if API_KEY == 'YOUR_FMP_KEY':
        print("❌ ERROR: Please insert your FMP API Key")
    else:
        valid_tickers = get_nasdaq_tickers()
        if valid_tickers:
            results = run_sensitivity_test(valid_tickers)
            generate_comparison_report(results)