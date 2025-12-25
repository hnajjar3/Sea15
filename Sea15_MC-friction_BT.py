
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

# STRATEGY PARAMS
GAP_UP_MIN, GAP_UP_MAX = 0.05, 0.13
GAP_DOWN_THRESHOLD = -0.05
MIN_PRICE, MAX_PRICE = 1.00, 50.00
MIN_VOLUME = 10000 
LOOKBACK_DAYS = 1095
STOP_LOSS_PCT = 0.05        # 5% Hard Stop

# ⚠️ FRICTION PARAMETERS (THE REALITY CHECK)
SLIPPAGE_PCT = 0.001        # 0.1% Slippage per side (Entry AND Exit)
COMMISSION_PER_TRADE = 1.00 # $1.00 per trade
AVG_TRADE_SIZE = 2000       # Size of 1 Unit (R) in Dollars (used to calc commission drag)

# SIMULATION PARAMS
NUM_SIMULATIONS = 50        
MAX_TRADES_PER_DAY = 10     
FORCE_UPDATE = False        

if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
class APIError(Exception): pass

# ==========================================
# 1. DATA HARVESTING
# ==========================================
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=16))
def fetch_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200: return response.json()
        return []
    except: return []

def get_nasdaq_tickers():
    cache_path = os.path.join(CACHE_DIR, 'nasdaq_tickers_list.pkl')
    if not FORCE_UPDATE and os.path.exists(cache_path): return pd.read_pickle(cache_path)
    print("📡 Fetching Ticker List...")
    url = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=10000&apikey={API_KEY}"
    data = fetch_url(url)
    if isinstance(data, list):
        df = pd.DataFrame(data)
        if 'price' in df.columns: df = df[df['price']>0.5]
        tickers = df['symbol'].tolist()
        pd.to_pickle(tickers, cache_path)
        return tickers
    return []

def get_historical_data(ticker):
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if not FORCE_UPDATE and os.path.exists(cache_path): return pd.read_pickle(cache_path)
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    url = f"{BASE_URL}/historical-price-full/{ticker}?from={start}&to={end}&apikey={API_KEY}"
    data = fetch_url(url)
    if isinstance(data, dict) and 'historical' in data:
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        df.to_pickle(cache_path)
        return df
    return pd.DataFrame()

# ==========================================
# 2. MASTER POOL WITH FRICTION
# ==========================================
def generate_master_pool(tickers):
    print(f"⚡ Generating Master Pool (SL: {STOP_LOSS_PCT*100}%, Slip: {SLIPPAGE_PCT*100}%, Comm: ${COMMISSION_PER_TRADE})...")
    all_signals = []
    
    # Calculate Commission Impact in % terms relative to Trade Size
    # If Trade Size = $2000 and Comm = $1, impact is 0.05%
    comm_pct = COMMISSION_PER_TRADE / AVG_TRADE_SIZE
    
    for i, ticker in enumerate(tickers):
        if i % 100 == 0: print(f"   Scanning {i}/{len(tickers)}...")
        df = get_historical_data(ticker)
        if df.empty or len(df) < 5: continue
            
        df['vol_prev'] = df['volume'].shift(1)
        df = df[(df['open']>0.01) & (df['close']>0.01) & (df['vol_prev']>MIN_VOLUME)].copy()
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        # --- SHORTS ---
        s_mask = (df['gap_pct']>=GAP_UP_MIN) & (df['gap_pct']<=GAP_UP_MAX) & \
                 (df['open']>=MIN_PRICE) & (df['open']<=MAX_PRICE)
        if s_mask.any():
            shorts = df[s_mask].copy()
            for date, row in shorts.iterrows():
                stop_price = row['open'] * (1 + STOP_LOSS_PCT)
                
                # Check Stop Loss
                if row['high'] >= stop_price:
                    # Stopped Out
                    raw_pnl = -STOP_LOSS_PCT
                    # Slip on Entry + Slip on Stop Exit
                    total_slip = SLIPPAGE_PCT + SLIPPAGE_PCT 
                else:
                    # Normal Close
                    raw_pnl = (row['open'] - row['close']) / row['open']
                    # Slip on Entry + Slip on Close Exit
                    total_slip = SLIPPAGE_PCT + SLIPPAGE_PCT
                
                # Net PnL = Raw - Slippage - Commission
                net_pnl = raw_pnl - total_slip - comm_pct
                
                all_signals.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Short', 
                    'PnL': net_pnl
                })

        # --- LONGS ---
        l_mask = (df['gap_pct']<=GAP_DOWN_THRESHOLD) & \
                 (df['open']>=MIN_PRICE) & (df['open']<=MAX_PRICE)
        if l_mask.any():
            longs = df[l_mask].copy()
            for date, row in longs.iterrows():
                stop_price = row['open'] * (1 - STOP_LOSS_PCT)
                
                # Check Stop Loss
                if row['low'] <= stop_price:
                    raw_pnl = -STOP_LOSS_PCT
                    total_slip = SLIPPAGE_PCT + SLIPPAGE_PCT
                else:
                    raw_pnl = (row['close'] - row['open']) / row['open']
                    total_slip = SLIPPAGE_PCT + SLIPPAGE_PCT

                # Net PnL
                net_pnl = raw_pnl - total_slip - comm_pct

                all_signals.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Long', 
                    'PnL': net_pnl
                })
                
    return pd.DataFrame(all_signals)

# ==========================================
# 3. MONTE CARLO ENGINE
# ==========================================
def run_monte_carlo(master_pool):
    print(f"\n🎲 Running {NUM_SIMULATIONS} Simulations (Real-World Friction)...")
    master_pool['Date'] = pd.to_datetime(master_pool['Date'])
    limit_per_side = MAX_TRADES_PER_DAY // 2
    equity_curves = []
    
    for i in range(NUM_SIMULATIONS):
        if (i+1) % 10 == 0: print(f"   Running Simulation {i+1}/{NUM_SIMULATIONS}...")
        sim_trades = []
        seed = 42 + i
        
        for date, group in master_pool.groupby('Date'):
            shorts = group[group['Type'] == 'Short']
            longs = group[group['Type'] == 'Long']
            
            if len(shorts) > limit_per_side:
                shorts = shorts.sample(n=limit_per_side, random_state=seed)
            if len(longs) > limit_per_side:
                longs = longs.sample(n=limit_per_side, random_state=seed)
                
            sim_trades.extend(shorts.to_dict('records'))
            sim_trades.extend(longs.to_dict('records'))
            
        sim_df = pd.DataFrame(sim_trades).sort_values('Date')
        sim_df['Equity'] = sim_df['PnL'].cumsum()
        
        curve = sim_df[['Date', 'Equity']].set_index('Date')
        curve.columns = [f'Run_{i+1}']
        equity_curves.append(curve)

    return pd.concat(equity_curves, axis=1).ffill().fillna(0)

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_multiverse(all_curves):
    plt.figure(figsize=(12, 8))
    for col in all_curves.columns:
        plt.plot(all_curves.index, all_curves[col], alpha=0.15, color='gray', linewidth=1)
        
    mean_curve = all_curves.mean(axis=1)
    plt.plot(mean_curve.index, mean_curve, color='#007BFF', linewidth=2.5, label='Average (Net of Fees)')
    
    final_vals = all_curves.iloc[-1]
    best_run = final_vals.idxmax()
    worst_run = final_vals.idxmin()
    
    plt.plot(all_curves.index, all_curves[best_run], 'g--', alpha=0.8, label=f'Best Net (+{final_vals.max():.0f}R)')
    plt.plot(all_curves.index, all_curves[worst_run], 'r--', alpha=0.8, label=f'Worst Net (+{final_vals.min():.0f}R)')
    
    plt.title(f'Monte Carlo Stress Test: 5% SL + Slippage & Commissions', fontsize=14)
    plt.ylabel('Net Profit (Risk Units)', fontsize=12)
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    filename = 'sea15_friction_test.png'
    plt.savefig(filename)
    print(f"\n✅ Chart saved to {filename}")
    print(f"   Avg Net Profit: {final_vals.mean():.2f} R")
    print(f"   Worst Case Net: {final_vals.min():.2f} R")
    print(f"   Best Case Net:  {final_vals.max():.2f} R")

if __name__ == "__main__":
    valid_tickers = get_nasdaq_tickers()
    if valid_tickers:
        # NOTE: New filename for the friction-adjusted pool
        master_file = 'sea15_master_pool_friction.csv'
        
        if os.path.exists(master_file) and not FORCE_UPDATE:
            print("📂 Loading Friction Pool from disk...")
            master_pool = pd.read_csv(master_file)
        else:
            master_pool = generate_master_pool(valid_tickers)
            master_pool.to_csv(master_file, index=False)
            
        if not master_pool.empty:
            curves = run_monte_carlo(master_pool)
            plot_multiverse(curves)