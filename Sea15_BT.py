import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==========================================
# CONFIGURATION
# ==========================================
# NEW GLOBALS
BENCHMARK_TICKER = 'QQQ'
RISK_FREE_RATE = 0.03
VAR_CONFIDENCE = 0.95
STARTING_CAPITAL = 100000

# TOGGLES
ENABLE_MC = True             # Set to True for Monte Carlo, False for Single Run
ENABLE_FRICTION = True       # Set to True to include Slippage & Commissions

# *** TREND FILTER ***
ENABLE_TREND_FILTER = False   # <--- Keep True for safety in liquid markets
SMA_PERIOD = 10              # Trend definition

# *** VOLUME SOFT CAP ***
MIN_VOLUME = 50_000          # The Floor (Hard Filter)
MAX_VOLUME = 500_000         # <--- NEW: The Soft Ceiling (500k)
HIGH_VOL_SAMPLE_RATE = 0.10  # <--- NEW: Only keep 10% of stocks above MAX_VOLUME

# API & DATA
API_KEY = 'iuRa8nmjQdLOx66QtJiSWFitzOMqu6QF' 
BASE_URL = "https://financialmodelingprep.com/api/v3"
CACHE_DIR = 'market_data_cache_unified'
MASTER_POOL_FILE = 'sea_master_pool_unified.csv'

# STRATEGY PARAMS
GAP_UP_MIN, GAP_UP_MAX = 0.05, 0.13
GAP_DOWN_THRESHOLD = -0.05
MIN_PRICE, MAX_PRICE = 1.00, 50.00
LOOKBACK_DAYS = 1095        # 3 Years
STOP_LOSS_PCT = 0.05        # 5% Stop Loss

# SIMULATION PARAMS
MAX_TRADES_PER_DAY = 10     
NUM_SIMULATIONS = 50        
FORCE_UPDATE = False        
FORCE_REGEN_POOL = False     # Set True once to ensure Volume column exists

# FRICTION PARAMS
SLIPPAGE_PCT = 0.001        # 0.1% per side
COMMISSION_PER_TRADE = 1.00 
AVG_TRADE_SIZE = 2000       

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class APIError(Exception):
    pass

# ==========================================
# 1. DATA HARVESTING
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
    
    print("📡 Fetching new Ticker List from FMP...")
    endpoint = f"{BASE_URL}/stock-screener?exchange=nasdaq&limit=10000&apikey={API_KEY}"
    try:
        data = fetch_url(endpoint)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            if 'price' in df.columns and 'volume' in df.columns:
                df = df[ (df['price'] > 2.00) & (df['volume'] > 10000) ]
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
# 2. MASTER POOL GENERATION
# ==========================================

def generate_master_pool(tickers):
    if not FORCE_REGEN_POOL and os.path.exists(MASTER_POOL_FILE):
        print(f"📂 Loading Master Signal Pool from {MASTER_POOL_FILE}...")
        pool = pd.read_csv(MASTER_POOL_FILE)
        # Check if new MCMC columns exist, otherwise regenerate
        required_cols = ['Trend_Aligned', 'Volume', 'Raw_PnL', 'High', 'Gap_Pct']
        if all(col in pool.columns for col in required_cols):
            return pool
        print("⚠️ Pool missing columns (MCMC data). Regenerating...")

    print(f"⚡ Generating Signal Pool (Filter: {ENABLE_TREND_FILTER}, SMA: {SMA_PERIOD})...")
    all_signals = []
    
    for i, ticker in enumerate(tickers):
        if i % 100 == 0: print(f"   Scanning {i}/{len(tickers)}...")
            
        df = get_historical_data(ticker)
        if df.empty or len(df) < SMA_PERIOD + 5: continue
            
        # Techs
        df['vol_prev'] = df['volume'].shift(1)
        df['sma'] = df['close'].rolling(window=SMA_PERIOD).mean().shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        # Filter Validity (Min Volume Only)
        df = df[(df['open'] > MIN_PRICE) & (df['vol_prev'] > MIN_VOLUME)].copy()
        
        # --- SHORTS ---
        short_mask = (df['gap_pct'] >= GAP_UP_MIN) & (df['gap_pct'] <= GAP_UP_MAX)
        if short_mask.any():
            shorts = df[short_mask].copy()
            for date, row in shorts.iterrows():
                is_aligned = True
                if ENABLE_TREND_FILTER:
                    is_aligned = row['open'] < row['sma']
                
                entry = row['open']
                stop = entry * (1 + STOP_LOSS_PCT)
                
                if row['high'] >= stop:
                    raw_pnl = -STOP_LOSS_PCT
                    outcome = 'Stopped Out'
                else:
                    raw_pnl = (entry - row['close']) / entry
                    outcome = 'Win' if raw_pnl > 0 else 'Loss'

                # MERGED DICTIONARY (Original + New MCMC Fields)
                all_signals.append({
                    'Date': date, 
                    'Ticker': ticker, 
                    'Type': 'Short',
                    'Raw_PnL': raw_pnl,          # <--- Critical for Simulation
                    'Outcome': outcome,
                    'Trend_Aligned': is_aligned,
                    'Entry_Price': entry,
                    'Volume': row['vol_prev'],
                    # NEW FIELDS FOR MCMC:
                    'High': row['high'],  
                    'Low': row['low'],    
                    'Close': row['close'],
                    'Gap_Pct': row['gap_pct']
                })

        # --- LONGS ---
        long_mask = (df['gap_pct'] <= GAP_DOWN_THRESHOLD)
        if long_mask.any():
            longs = df[long_mask].copy()
            for date, row in longs.iterrows():
                is_aligned = True
                if ENABLE_TREND_FILTER:
                    is_aligned = row['open'] > row['sma']

                entry = row['open']
                stop = entry * (1 - STOP_LOSS_PCT)
                
                if row['low'] <= stop:
                    raw_pnl = -STOP_LOSS_PCT
                    outcome = 'Stopped Out'
                else:
                    raw_pnl = (row['close'] - entry) / entry
                    outcome = 'Win' if raw_pnl > 0 else 'Loss'

                # MERGED DICTIONARY (Original + New MCMC Fields)
                all_signals.append({
                    'Date': date, 
                    'Ticker': ticker, 
                    'Type': 'Long',
                    'Raw_PnL': raw_pnl,          # <--- Critical for Simulation
                    'Outcome': outcome,
                    'Trend_Aligned': is_aligned,
                    'Entry_Price': entry,
                    'Volume': row['vol_prev'],
                    # NEW FIELDS FOR MCMC:
                    'High': row['high'], 
                    'Low': row['low'],   
                    'Close': row['close'],
                    'Gap_Pct': row['gap_pct']
                })

    pool_df = pd.DataFrame(all_signals)
    pool_df.to_csv(MASTER_POOL_FILE, index=False)
    return pool_df

# ==========================================
# 3. SIMULATION ENGINE (SOFT CAP APPLIED)
# ==========================================
def apply_friction(row):
    if not ENABLE_FRICTION: return row['Raw_PnL']
    comm_pct = COMMISSION_PER_TRADE / AVG_TRADE_SIZE
    total_slip = SLIPPAGE_PCT * 2
    return row['Raw_PnL'] - total_slip - comm_pct

# ==========================================
# OPTIMIZED SIMULATION ENGINE (NUMPY)
# ==========================================
def simulation_worker(dates, ticker_map, pnl_map, vol_map, type_map, seed):
    """
    Worker function that runs ONE simulation using pure Numpy for speed.
    """
    np.random.seed(seed)
    daily_results = []
    
    unique_dates = np.unique(dates)
    
    for d in unique_dates:
        # 1. Get indices for this day
        day_mask = (dates == d)
        
        # 2. Split by Type (Long/Short)
        # We pre-calculated these masks, but slicing arrays is fast
        day_indices = np.where(day_mask)[0]
        
        if len(day_indices) == 0: continue
            
        # Extract data for this day
        curr_vols = vol_map[day_indices]
        curr_types = type_map[day_indices]
        curr_pnls = pnl_map[day_indices]
        
        # 3. Apply Soft Cap (Vectorized)
        # Logic: Keep all Low Vol, Keep 10% of High Vol
        high_vol_mask = curr_vols > MAX_VOLUME
        low_vol_mask = ~high_vol_mask
        
        # Indices relative to the current day slice
        high_vol_local_idx = np.where(high_vol_mask)[0]
        low_vol_local_idx = np.where(low_vol_mask)[0]
        
        # Sample High Vol
        if len(high_vol_local_idx) > 0:
            keep_n = int(len(high_vol_local_idx) * HIGH_VOL_SAMPLE_RATE)
            # Ensure at least 1 if exists, or strictly 10%? 
            # Using strict rate usually, but let's ensure we don't drop everything
            keep_n = max(1, keep_n) 
            selected_high = np.random.choice(high_vol_local_idx, size=keep_n, replace=False)
        else:
            selected_high = np.array([], dtype=int)
            
        # Combine indices
        kept_local_indices = np.concatenate([low_vol_local_idx, selected_high])
        
        if len(kept_local_indices) == 0: continue

        # 4. Split Longs/Shorts from the survivor pool
        survivor_types = curr_types[kept_local_indices]
        survivor_indices = day_indices[kept_local_indices] # Map back to global global_idx
        
        short_local = np.where(survivor_types == 0)[0] # 0 for Short
        long_local = np.where(survivor_types == 1)[0]  # 1 for Long
        
        # 5. Limit Selection (Max Trades Per Day)
        limit = MAX_TRADES_PER_DAY // 2
        
        final_picks = []
        
        # Sample Shorts
        if len(short_local) > limit:
            picked_s = np.random.choice(short_local, size=limit, replace=False)
            final_picks.append(survivor_indices[picked_s])
        else:
            final_picks.append(survivor_indices[short_local])
            
        # Sample Longs
        if len(long_local) > limit:
            picked_l = np.random.choice(long_local, size=limit, replace=False)
            final_picks.append(survivor_indices[picked_l])
        else:
            final_picks.append(survivor_indices[long_local])
            
        # Flatten
        final_indices = np.concatenate(final_picks).astype(int)
        
        # Store PnL
        # We only need the PnL sum for the curve usually, but to match your report we need records
        # For speed, we just sum them here, or return indices to rebuild DF later.
        # Let's return indices to rebuild the DF accurately.
        daily_results.append(final_indices)

    return np.concatenate(daily_results)

def run_simulation_optimized(master_pool):
    print(f"\n🚀 Starting Parallel Monte Carlo ({NUM_SIMULATIONS} Runs)...")
    
    # 1. Pre-process Data for Numpy
    # Filter Trend Aligned globally first if that switch is ON
    if ENABLE_TREND_FILTER:
        df = master_pool[master_pool['Trend_Aligned'] == True].copy()
    else:
        df = master_pool.copy()
        
    df['Net_PnL'] = df.apply(apply_friction, axis=1)
    
    # Convert dates to integers for fast indexing
    df['Date_Int'] = pd.to_datetime(df['Date']).astype('int64') // 10**9 
    
    # Map 'Type' to int (Short=0, Long=1)
    df['Type_Int'] = np.where(df['Type'] == 'Short', 0, 1)
    
    # Create Arrays
    dates_arr = df['Date_Int'].values
    pnl_arr = df['Net_PnL'].values
    vol_arr = df['Volume'].values
    type_arr = df['Type_Int'].values
    idx_arr = df.index.values # Keep original indices to reconstruct
    
    # 2. Run Parallel Simulations
    # Using joblib to utilize all CPU cores
    results_indices = Parallel(n_jobs=-1, verbose=5)(
        delayed(simulation_worker)(
            dates_arr, idx_arr, pnl_arr, vol_arr, type_arr, 42 + i
        ) for i in range(NUM_SIMULATIONS)
    )
    
    # 3. Reconstruct DataFrames
    simulations = []
    print("📊 Reconstructing Trade Logs...")
    for i, indices in enumerate(results_indices):
        sim_df = df.loc[indices].copy()
        sim_df['Sim_ID'] = i + 1
        sim_df = sim_df.sort_values('Date')
        sim_df['Equity'] = sim_df['Net_PnL'].cumsum()
        simulations.append(sim_df)
        
    return simulations

# ==========================================
# 4. ANALYSIS: VOLUME CURVE
# ==========================================
def analyze_by_volume(master_pool):
    print("\n🔬 Analyzing Performance by Volume...")
    if 'Volume' not in master_pool.columns: return

    if 'Net_PnL' not in master_pool.columns:
        master_pool['Net_PnL'] = master_pool.apply(apply_friction, axis=1)

    if ENABLE_TREND_FILTER:
        df = master_pool[master_pool['Trend_Aligned'] == True].copy()
    else:
        df = master_pool.copy()

    try:
        df['Vol_Bucket'] = pd.qcut(df['Volume'], q=10, labels=False)
    except: return

    stats = df.groupby('Vol_Bucket').agg({
        'Volume': ['min', 'max', 'mean'],
        'Net_PnL': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    
    stats.columns = ['Min_Vol', 'Max_Vol', 'Avg_Vol', 'Count', 'Avg_Net_PnL', 'Win_Rate']
    stats['Avg_Net_PnL'] *= 100
    stats['Win_Rate'] *= 100
    
    print("\n" + "="*80)
    print(f"{'Bucket':<6} | {'Volume Range':<25} | {'Count':<6} | {'Win Rate':<8} | {'Avg Net PnL':<8}")
    print("-" * 80)
    for i, row in stats.iterrows():
        vol_range = f"{row['Min_Vol']:.0f} - {row['Max_Vol']:.0f}"
        print(f"{i:<6} | {vol_range:<25} | {int(row['Count']):<6} | {row['Win_Rate']:6.2f}% | {row['Avg_Net_PnL']:6.2f}%")
    print("="*80)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(stats.index, stats['Avg_Net_PnL'], color='skyblue', label='Net PnL %')
    ax2 = plt.gca().twinx()
    ax2.plot(stats.index, stats['Win_Rate'], color='green', marker='o', linewidth=2, label='Win Rate %')
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Net Profit by Volume (Trend Filter: {ENABLE_TREND_FILTER})")
    plt.xlabel("Volume Decile (0=Low, 9=High)")
    plt.ylabel("Avg Net PnL (%)")
    ax2.set_ylabel("Win Rate (%)")
    plt.xticks(stats.index, [f"{x/1000:.0f}k" for x in stats['Avg_Vol']])
    plt.savefig("volume_curve.png")
    print("✅ Volume Curve saved to volume_curve.png")

# ==========================================
# 5. REPORTING
# ==========================================
def calculate_financial_metrics(sim_df, bench_ret):
    """
    Calculates the full 13-metric financial suite for a single simulation run.
    """
    # 1. Prepare Daily Equity Curve
    sim_df = sim_df.copy()
    
    # --- CRITICAL FIX: Ensure dates are datetime objects ---
    sim_df['Date'] = pd.to_datetime(sim_df['Date'])
    
    # Dollar PnL per day
    daily_pnl = sim_df.groupby('Date')['Net_PnL'].sum() * AVG_TRADE_SIZE
    
    # Construct Equity Series
    equity_curve = STARTING_CAPITAL + daily_pnl.cumsum()
    equity_df = pd.DataFrame({'Equity': equity_curve})
    
    # Align with Benchmark
    start_date = equity_df.index.min()
    end_date = equity_df.index.max()
    
    if bench_ret.empty:
        bench_subset = pd.Series(0, index=equity_df.index)
    else:
        bench_subset = bench_ret.loc[start_date:end_date]
    
    # Reindex to Union of dates (to handle days with no trades)
    full_idx = equity_df.index.union(bench_subset.index).sort_values()
    equity_df = equity_df.reindex(full_idx).ffill().fillna(STARTING_CAPITAL)
    bench_subset = bench_subset.reindex(full_idx).fillna(0)
    
    # Calculate Returns
    equity_df['Prev'] = equity_df['Equity'].shift(1).fillna(STARTING_CAPITAL)
    rp = (equity_df['Equity'] - equity_df['Prev']) / equity_df['Prev']
    rm = bench_subset
    
    # --- METRICS CALCULATION ---
    days = (equity_df.index.max() - equity_df.index.min()).days
    years = days / 365.25
    if years == 0: years = 0.001
    
    # 1. CAGR
    total_ret = (equity_df['Equity'].iloc[-1] / STARTING_CAPITAL)
    cagr = (total_ret ** (1/years)) - 1
    
    # 2. Volatility (Annualized)
    vol = rp.std() * np.sqrt(252)
    
    # 3. Sharpe Ratio
    rf_daily = RISK_FREE_RATE / 252
    excess_ret = rp - rf_daily
    sharpe = (excess_ret.mean() / rp.std()) * np.sqrt(252) if rp.std() > 0 else 0
    
    # 4. Sortino Ratio (Downside Vol)
    downside_std = rp[rp < 0].std() * np.sqrt(252)
    sortino = (excess_ret.mean() * 252) / downside_std if downside_std > 0 else 0
    
    # 5. Max Drawdown
    roll_max = equity_df['Equity'].cummax()
    dd = (equity_df['Equity'] - roll_max) / roll_max
    max_dd = dd.min()
    
    # 6. Beta & Alpha
    covariance = np.cov(rp, rm)[0][1]
    variance = np.var(rm)
    beta = covariance / variance if variance > 0 else 0
    
    # Alpha (Annualized) = Rp - (Rf + Beta * (Rm - Rf))
    # We use annualized means for this formula
    rp_ann = rp.mean() * 252
    rm_ann = rm.mean() * 252
    alpha = rp_ann - (RISK_FREE_RATE + beta * (rm_ann - RISK_FREE_RATE))
    
    # 7. Treynor Ratio
    treynor = (rp_ann - RISK_FREE_RATE) / beta if beta != 0 else 0
    
    # 8. Information Ratio
    tracking_error = (rp - rm).std() * np.sqrt(252)
    info_ratio = (rp_ann - rm_ann) / tracking_error if tracking_error > 0 else 0
    
    # 9. Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # 10. Omega Ratio (Threshold 0)
    # Simple probability weighted ratio of gains vs losses
    gains = rp[rp > 0].sum()
    losses = abs(rp[rp < 0].sum())
    omega = gains / losses if losses > 0 else 0
    
    # 11. VaR (Value at Risk) 95% Daily
    # The 5th percentile of daily returns
    var_95 = np.percentile(rp, (1 - VAR_CONFIDENCE) * 100)
    
    # 12. CVaR (Conditional Value at Risk) 95% Daily
    # Average of returns worse than VaR
    cvar_95 = rp[rp <= var_95].mean()

    # Basic Trade Stats
    wins = sim_df[sim_df['Net_PnL'] > 0]
    win_rate = (len(wins) / len(sim_df))
    
    return {
        'CAGR': cagr,
        'Vol (Ann)': vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max DD %': max_dd,
        'Beta': beta,
        'Alpha (Ann)': alpha,
        'Treynor': treynor,
        'Info Ratio': info_ratio,
        'Calmar': calmar,
        'Omega': omega,
        'VaR 95%': var_95,
        'CVaR 95%': cvar_95,
        'Win Rate %': win_rate,
        'Trades': len(sim_df),
        'Equity Curve': equity_df['Equity'] # For Charting
    }

def generate_report(simulations, benchmark_ret):
    if not simulations: return
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    
    if ENABLE_MC:
        print(f"\n📊 MONTE CARLO REPORT ({NUM_SIMULATIONS} Runs)")
        all_stats = []
        equity_curves = []
        
        for sim in simulations:
            metrics = calculate_financial_metrics(sim, benchmark_ret)
            
            # Separate the curve from the stats dict
            curve = metrics.pop('Equity Curve')
            curve.name = f"Run_{sim['Sim_ID'].iloc[0]}"
            equity_curves.append(curve)
            
            all_stats.append(metrics)
            
        kpi_df = pd.DataFrame(all_stats)
        
        # Format and Print
        print("="*100)
        print(f"{'Metric':<20} | {'Mean':<12} | {'Min (Worst)':<12} | {'Max (Best)':<12}")
        print("-" * 100)
        
        pct_metrics = ['CAGR', 'Vol (Ann)', 'Max DD %', 'Win Rate %', 'VaR 95%', 'CVaR 95%', 'Alpha (Ann)']
        
        for col in kpi_df.columns:
            mean_val = kpi_df[col].mean()
            min_val = kpi_df[col].min()
            max_val = kpi_df[col].max()
            
            if col in pct_metrics:
                print(f"{col:<20} | {mean_val*100:11.2f}% | {min_val*100:11.2f}% | {max_val*100:11.2f}%")
            else:
                print(f"{col:<20} | {mean_val:12.2f} | {min_val:12.2f} | {max_val:12.2f}")
                
        print("="*100)
        kpi_df.to_csv(f"summary_mc_{timestamp}.csv", index=False)
        
        # Charting
        full_curves = pd.concat(equity_curves, axis=1).ffill().fillna(STARTING_CAPITAL)
        plt.figure(figsize=(12, 8))
        for col in full_curves.columns:
            plt.plot(full_curves.index, full_curves[col], alpha=0.1, color='gray', linewidth=1)
        avg_curve = full_curves.mean(axis=1)
        plt.plot(avg_curve.index, avg_curve, color='#007BFF', linewidth=2.5, label='Average Outcome')
        plt.title(f"Monte Carlo ({NUM_SIMULATIONS} Runs) | Sharpe: {kpi_df['Sharpe'].mean():.2f} | CAGR: {kpi_df['CAGR'].mean()*100:.1f}%")
        plt.xlabel("Date"); plt.ylabel("Equity ($)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(f"chart_mc_{timestamp}.png")
        print(f"✅ Chart saved.")

def get_benchmark_data():
    print(f"📡 Fetching Benchmark ({BENCHMARK_TICKER}) Data...")
    df = get_historical_data(BENCHMARK_TICKER)
    if df.empty: return pd.Series(dtype=float)
    df = df.sort_index()
    return df['close'].pct_change().fillna(0)

def process_portfolio_daily(sim_trades, benchmark_ret):
    if sim_trades.empty: return pd.DataFrame()
    sim_trades = sim_trades.copy()
    
    # Force conversion of Date column to Datetime objects to match Benchmark
    sim_trades['Date'] = pd.to_datetime(sim_trades['Date'])

    sim_trades['PnL_Dollar'] = sim_trades['Net_PnL'] * AVG_TRADE_SIZE
    daily_pnl = sim_trades.groupby('Date')['PnL_Dollar'].sum()
    
    if not daily_pnl.empty:
        start_date = daily_pnl.index.min()
        end_date = daily_pnl.index.max()
        
        # Slice benchmark to match simulation dates
        bench_subset = benchmark_ret.loc[start_date:end_date] if not benchmark_ret.empty else pd.Series(dtype=float)
        
        # Now both indexes are Datetime objects, so union/sort will work
        full_index = daily_pnl.index.union(bench_subset.index).sort_values()
        
        df = pd.DataFrame(index=full_index)
        df['Bench_Ret'] = bench_subset.reindex(full_index).fillna(0)
        df['Daily_PnL'] = daily_pnl.reindex(full_index).fillna(0)
        df['Equity'] = STARTING_CAPITAL + df['Daily_PnL'].cumsum()
        df['Prev_Equity'] = df['Equity'].shift(1).fillna(STARTING_CAPITAL)
        df['Port_Ret'] = df['Daily_PnL'] / df['Prev_Equity']
        return df
        
    return pd.DataFrame()

def generate_excel_report_full(simulations, benchmark_data):
    if not simulations: return
    # Find the median (representative) simulation
    final_equities = [sim['Equity'].iloc[-1] for sim in simulations]
    median_idx = np.argsort(final_equities)[len(final_equities)//2]
    best_sim = simulations[median_idx]
    
    print(f"\n📝 Generating Excel Report...")
    
    # 1. Generate Daily Log
    daily_df = process_portfolio_daily(best_sim, benchmark_data)
    
    # 2. Generate Full Financial Metrics (Re-using the robust function)
    metrics_dict = calculate_financial_metrics(best_sim, benchmark_data)
    
    # Remove the large Series object so it doesn't clutter the Summary sheet
    if 'Equity Curve' in metrics_dict:
        del metrics_dict['Equity Curve']
        
    filename = f"Sea15_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary Sheet with ALL 13 KPIs
            summary_series = pd.Series(metrics_dict, name="Value")
            summary_series.to_excel(writer, sheet_name="Summary")
            
            # Daily Log Sheet
            if not daily_df.empty: daily_df.to_excel(writer, sheet_name="Daily_Log")
            
            # Trade Ledger Sheet
            best_sim.to_excel(writer, sheet_name="Trade_Ledger", index=False)
            
        print(f"✅ Excel Report saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving Excel: {e}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if API_KEY == 'YOUR_FMP_KEY':
        print("❌ ERROR: Please insert your FMP API Key")
    else:
        valid_tickers = get_nasdaq_tickers()
        if valid_tickers:
            bench_ret = get_benchmark_data()
            pool = generate_master_pool(valid_tickers)
            if not pool.empty:
                sim_results = run_simulation_optimized(pool)
                analyze_by_volume(pool)
                generate_report(sim_results, bench_ret) 
                generate_excel_report_full(sim_results, bench_ret)