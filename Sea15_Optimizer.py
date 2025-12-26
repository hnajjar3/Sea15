import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION: OPTIMIZATION RANGES
# ==========================================
NUM_EXPERIMENTS = 2000       # How many different parameter sets to test
FORCE_REGEN_WIDE_POOL = True # Set TRUE for the first run to build the 'Wide' dataset

# PARAMETER RANGES (The Algorithm will pick random values within these)
RANGES = {
    'STOP_LOSS': (0.01, 0.10),      # Test stops from 1% to 10%
    'GAP_MIN':   (0.02, 0.10),      # Test gap minimums from 2% to 10%
    'GAP_MAX':   (0.10, 0.40),      # Test gap maximums from 10% to 40%
}

# HARD CONSTRAINTS (Fixed Filters)
SMA_PERIOD = 10
MIN_VOLUME = 50_000
MIN_PRICE = 2.00
LOOKBACK_DAYS = 1095  # 3 Years

# API & FILES
API_KEY = 'iuRa8nmjQdLOx66QtJiSWFitzOMqu6QF' 
BASE_URL = "https://financialmodelingprep.com/api/v3"
CACHE_DIR = 'market_data_cache_unified'       # Shared Cache (Efficient)
WIDE_POOL_FILE = 'sea_master_pool_WIDE.csv'   # Separate Pool File (Safe)

# FRICTION (Crucial for realistic optimization)
SLIPPAGE_PCT = 0.001
COMMISSION = 1.00
TRADE_SIZE = 2000

# ==========================================
# 2. DATA HARVESTING (LOOSE FILTERS)
# ==========================================
def fetch_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def get_nasdaq_tickers():
    # Try to load from existing cache first to save time
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, 'nasdaq_tickers_list.pkl')
    if os.path.exists(cache_path): return pd.read_pickle(cache_path)
    
    # Fallback to API
    print("📡 Fetching new Ticker List from FMP...")
    endpoint = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=10000&apikey={API_KEY}"
    data = fetch_url(endpoint)
    if isinstance(data, list):
        df = pd.DataFrame(data)
        tickers = df['symbol'].tolist()
        pd.to_pickle(tickers, cache_path)
        return tickers
    return []

def get_historical_data(ticker):
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if os.path.exists(cache_path): return pd.read_pickle(cache_path)
    # If not in cache, we skip it to keep this script fast & independent
    # (It relies on the cache built by your main script)
    return pd.DataFrame()

def generate_wide_pool(tickers):
    """
    Generates a pool with VERY LOOSE filters so we can filter it later dynamically.
    """
    if not FORCE_REGEN_WIDE_POOL and os.path.exists(WIDE_POOL_FILE):
        print(f"📂 Loading Wide Pool from {WIDE_POOL_FILE}...")
        return pd.read_csv(WIDE_POOL_FILE)

    print(f"⚡ Generating WIDE Signal Pool (Gap > 2%, No Max)...")
    all_signals = []
    
    # We only scan tickers that already exist in your cache
    cached_files = [f.replace('.pkl','') for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    valid_tickers = list(set(tickers) & set(cached_files))
    
    for i, ticker in enumerate(valid_tickers):
        if i % 500 == 0: print(f"   Scanning {i}/{len(valid_tickers)}...")
        
        try:
            df = pd.read_pickle(os.path.join(CACHE_DIR, f"{ticker}.pkl"))
        except: continue
        
        if df.empty or len(df) < SMA_PERIOD + 5: continue
            
        # Techs
        df['vol_prev'] = df['volume'].shift(1)
        df['sma'] = df['close'].rolling(window=SMA_PERIOD).mean().shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        # LOOSE Filter: Gap > 2% (0.02) OR Gap < -2% (-0.02)
        mask = (df['open'] > MIN_PRICE) & (df['vol_prev'] > MIN_VOLUME) & \
               ((df['gap_pct'] > 0.02) | (df['gap_pct'] < -0.02))
        
        df = df[mask].copy()
        
        for date, row in df.iterrows():
            # Trend Check 
            trend_short = row['open'] < row['sma']
            trend_long = row['open'] > row['sma']
            
            # Store RAW data needed to re-calculate PnL dynamically
            all_signals.append({
                'Date': date,
                'Ticker': ticker,
                'Entry': row['open'],
                'Close': row['close'],
                'High': row['high'],
                'Low': row['low'],
                'Gap': row['gap_pct'],
                'Volume': row['vol_prev'],
                'Trend_Short': trend_short,
                'Trend_Long': trend_long
            })

    pool = pd.DataFrame(all_signals)
    pool.to_csv(WIDE_POOL_FILE, index=False)
    return pool

# ==========================================
# 3. OPTIMIZATION WORKER
# ==========================================
def run_experiment(pool_df, params, experiment_id):
    """
    Runs ONE full historical backtest with a specific set of parameters.
    Returns the performance metrics for that set.
    """
    # Unpack Parameters
    sl_pct = params['sl']
    gap_min = params['gap_min']
    gap_max = params['gap_max']
    
    # 1. Dynamic Filtering (Vectorized = Fast)
    # Short Logic: Gap is Positive, Price < SMA (Trend), Gap within range
    # NEW (Trend Filter Disabled - matches your best manual run)
    short_mask = (pool_df['Gap'] >= gap_min) & (pool_df['Gap'] <= gap_max)
    
    # Long Logic: Gap is Negative (Mirror range), Price > SMA
    # We mirror the gap range for longs: e.g. -13% to -5%
    long_mask = (pool_df['Gap'] <= -gap_min) & (pool_df['Gap'] >= -gap_max) & (pool_df['Trend_Long'] == True)
    
    # Slice Data
    shorts = pool_df[short_mask].copy()
    longs = pool_df[long_mask].copy()
    
    if shorts.empty and longs.empty: return None

    # 2. Dynamic PnL Calculation
    
    # SHORT PnL
    if not shorts.empty:
        stop_prices_s = shorts['Entry'] * (1 + sl_pct)
        stopped_out_s = shorts['High'] >= stop_prices_s
        shorts['PnL'] = np.where(stopped_out_s, -sl_pct, (shorts['Entry'] - shorts['Close']) / shorts['Entry'])
    
    # LONG PnL
    if not longs.empty:
        stop_prices_l = longs['Entry'] * (1 - sl_pct)
        stopped_out_l = longs['Low'] <= stop_prices_l
        longs['PnL'] = np.where(stopped_out_l, -sl_pct, (longs['Close'] - longs['Entry']) / longs['Entry'])
    
    # Merge
    combined = pd.concat([shorts, longs])
    
    # 3. Apply Friction
    # Net PnL = Gross - Slippage - Comm%
    friction = (SLIPPAGE_PCT * 2) + (COMMISSION / TRADE_SIZE)
    combined['Net_PnL'] = combined['PnL'] - friction
    
    # 4. Calculate Metrics
    if len(combined) < 100: return None # Ignore non-statistically significant sets
    
    avg_pnl = combined['Net_PnL'].mean()
    total_r = combined['Net_PnL'].sum()
    win_rate = (combined['Net_PnL'] > 0).mean()
    
    # Sharpe Approximation (Daily)
    daily = combined.groupby('Date')['Net_PnL'].sum()
    if daily.std() == 0: return None
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
    
    return {
        'id': experiment_id,
        'sl': sl_pct,
        'gap_min': gap_min,
        'gap_max': gap_max,
        'Sharpe': sharpe,
        'Total_R': total_r,
        'Win_Rate': win_rate,
        'Trades': len(combined)
    }

# ==========================================
# 4. MAIN OPTIMIZER LOOP
# ==========================================
def optimize():
    # Load Tickers & Generate Wide Pool
    tickers = get_nasdaq_tickers()
    pool = generate_wide_pool(tickers)
    
    if pool.empty: 
        print("❌ Error: Wide Pool is empty. Ensure you have cached data from the main script.")
        return

    print(f"\n🚀 Starting {NUM_EXPERIMENTS} random experiments...")
    print(f"Searching space: SL={RANGES['STOP_LOSS']}, GapMin={RANGES['GAP_MIN']}")

    # Generate Random Parameter Sets
    param_sets = []
    for i in range(NUM_EXPERIMENTS):
        # Random sampling from uniform distributions
        p = {
            'sl': np.round(np.random.uniform(*RANGES['STOP_LOSS']), 3),
            'gap_min': np.round(np.random.uniform(*RANGES['GAP_MIN']), 3),
        }
        # Gap Max must be > Gap Min
        p['gap_max'] = np.round(np.random.uniform(p['gap_min'] + 0.05, RANGES['GAP_MAX'][1]), 3)
        param_sets.append(p)

    # Run Parallel
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(run_experiment)(pool, params, i) for i, params in enumerate(param_sets)
    )
    
    # Clean Results
    results = [r for r in results if r is not None]
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("❌ No valid results found.")
        return

    # Save Raw Data
    res_df.to_csv("optimization_results.csv", index=False)
    
    # ==========================================
    # 5. ANALYSIS & VISUALIZATION
    # ==========================================
    print("\n🏆 OPTIMIZATION RESULTS 🏆")
    print("Top 10 Configurations by Sharpe Ratio:")
    print(res_df.sort_values('Sharpe', ascending=False).head(10)[['sl','gap_min','gap_max','Sharpe','Total_R','Trades']])
    
    best = res_df.sort_values('Sharpe', ascending=False).iloc[0]
    print(f"\n✅ BEST CONFIGURATION FOUND:")
    print(f"   Stop Loss: {best['sl']*100:.1f}%")
    print(f"   Gap Range: {best['gap_min']*100:.1f}% to {best['gap_max']*100:.1f}%")
    print(f"   Sharpe:    {best['Sharpe']:.2f}")
    
    # PLOT: Heatmap of Sharpe (Stop Loss vs Gap Min)
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(res_df['sl']*100, res_df['gap_min']*100, c=res_df['Sharpe'], cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(sc, label='Sharpe Ratio')
    plt.xlabel('Stop Loss %')
    plt.ylabel('Gap Min %')
    plt.title('Optimization Heatmap: Stop Loss vs Gap Size')
    plt.grid(True, alpha=0.3)
    plt.savefig("optimization_heatmap.png")
    print("✅ Heatmap saved to optimization_heatmap.png")

if __name__ == "__main__":
    optimize()
