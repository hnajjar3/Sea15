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
# NEW GLOBALS
BENCHMARK_TICKER = 'QQQ'
RISK_FREE_RATE = 0.03
VAR_CONFIDENCE = 0.95
STARTING_CAPITAL = 100000

# TOGGLES
ENABLE_MC = True             # Set to True for Monte Carlo, False for Single Run
ENABLE_FRICTION = True       # Set to True to include Slippage & Commissions

# *** CRITICAL: TREND FILTER RESTORED ***
ENABLE_TREND_FILTER = True   # <--- The only way to win in liquid markets
SMA_PERIOD = 50              # Trend definition

# API & DATA
API_KEY = 'iuRa8nmjQdLOx66QtJiSWFitzOMqu6QF' 
BASE_URL = "https://financialmodelingprep.com/api/v3"
CACHE_DIR = 'market_data_cache_unified'
MASTER_POOL_FILE = 'sea_master_pool_unified.csv'

# STRATEGY PARAMS
GAP_UP_MIN, GAP_UP_MAX = 0.05, 0.13
GAP_DOWN_THRESHOLD = -0.05
MIN_PRICE, MAX_PRICE = 1.00, 50.00
MIN_VOLUME = 50_000         # 50k Floor
LOOKBACK_DAYS = 1095        # 3 Years
STOP_LOSS_PCT = 0.15        # 15% Stop Loss

# SIMULATION PARAMS
MAX_TRADES_PER_DAY = 10     
NUM_SIMULATIONS = 50        
FORCE_UPDATE = False        
FORCE_REGEN_POOL = True     # Set True once to rebuild with SMA & Volume columns

# FRICTION PARAMS (Realistic for >50k Volume)
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
# 2. MASTER POOL GENERATION (WITH SMA & VOL)
# ==========================================
def generate_master_pool(tickers):
    if not FORCE_REGEN_POOL and os.path.exists(MASTER_POOL_FILE):
        print(f"📂 Loading Master Signal Pool from {MASTER_POOL_FILE}...")
        pool = pd.read_csv(MASTER_POOL_FILE)
        # Validation
        required_cols = ['Trend_Aligned', 'Volume']
        if all(col in pool.columns for col in required_cols):
            return pool
        print("⚠️ Pool missing columns. Regenerating...")

    print(f"⚡ Generating Signal Pool (Filter: {ENABLE_TREND_FILTER}, SMA: {SMA_PERIOD})...")
    all_signals = []
    
    for i, ticker in enumerate(tickers):
        if i % 100 == 0: print(f"   Scanning {i}/{len(tickers)}...")
            
        df = get_historical_data(ticker)
        if df.empty or len(df) < SMA_PERIOD + 5: continue
            
        # 1. Techs
        df['vol_prev'] = df['volume'].shift(1)
        df['sma'] = df['close'].rolling(window=SMA_PERIOD).mean().shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        # 2. Filter Validity
        df = df[(df['open'] > MIN_PRICE) & (df['vol_prev'] > MIN_VOLUME)].copy()
        
        # --- SHORTS ---
        short_mask = (df['gap_pct'] >= GAP_UP_MIN) & (df['gap_pct'] <= GAP_UP_MAX)
        if short_mask.any():
            shorts = df[short_mask].copy()
            for date, row in shorts.iterrows():
                # Trend Logic: Aligned if Open < SMA
                is_aligned = True
                if ENABLE_TREND_FILTER:
                    is_aligned = row['open'] < row['sma']
                
                # PnL Logic
                entry = row['open']
                stop = entry * (1 + STOP_LOSS_PCT)
                
                if row['high'] >= stop:
                    raw_pnl = -STOP_LOSS_PCT
                    outcome = 'Stopped Out'
                else:
                    raw_pnl = (entry - row['close']) / entry
                    outcome = 'Win' if raw_pnl > 0 else 'Loss'

                all_signals.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Short',
                    'Raw_PnL': raw_pnl, 'Outcome': outcome,
                    'Entry_Price': entry,
                    'Volume': row['vol_prev'],
                    'Trend_Aligned': is_aligned
                })

        # --- LONGS ---
        long_mask = (df['gap_pct'] <= GAP_DOWN_THRESHOLD)
        if long_mask.any():
            longs = df[long_mask].copy()
            for date, row in longs.iterrows():
                # Trend Logic: Aligned if Open > SMA
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

                all_signals.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'Long',
                    'Raw_PnL': raw_pnl, 'Outcome': outcome,
                    'Entry_Price': entry,
                    'Volume': row['vol_prev'],
                    'Trend_Aligned': is_aligned
                })

    pool_df = pd.DataFrame(all_signals)
    pool_df.to_csv(MASTER_POOL_FILE, index=False)
    return pool_df

# ==========================================
# 3. SIMULATION ENGINE
# ==========================================
def apply_friction(row):
    if not ENABLE_FRICTION: return row['Raw_PnL']
    comm_pct = COMMISSION_PER_TRADE / AVG_TRADE_SIZE
    total_slip = SLIPPAGE_PCT * 2
    return row['Raw_PnL'] - total_slip - comm_pct

def run_simulation(master_pool):
    print(f"\n🚀 Simulation: Filter={ENABLE_TREND_FILTER}, VolFloor={MIN_VOLUME}, Stop={STOP_LOSS_PCT*100}%")
    
    master_pool['Net_PnL'] = master_pool.apply(apply_friction, axis=1)
    master_pool['Date'] = pd.to_datetime(master_pool['Date'])
    
    simulations = []
    
    for i in range(NUM_SIMULATIONS):
        if (i+1) % 10 == 0: print(f"   Simulating Run {i+1}...")
        seed = 42 + i
        sim_trades = []
        
        for date, group in master_pool.groupby('Date'):
            shorts = group[group['Type'] == 'Short']
            longs = group[group['Type'] == 'Long']
            
            # Apply Filter
            if ENABLE_TREND_FILTER:
                shorts = shorts[shorts['Trend_Aligned'] == True]
                longs = longs[longs['Trend_Aligned'] == True]
            
            # Sample
            limit = MAX_TRADES_PER_DAY // 2
            if len(shorts) > limit: shorts = shorts.sample(n=limit, random_state=seed)
            if len(longs) > limit: longs = longs.sample(n=limit, random_state=seed)
            
            sim_trades.extend(shorts.to_dict('records'))
            sim_trades.extend(longs.to_dict('records'))
            
        sim_df = pd.DataFrame(sim_trades)
        if sim_df.empty: continue
        
        sim_df = sim_df.sort_values('Date')
        sim_df['Equity'] = sim_df['Net_PnL'].cumsum()
        sim_df['Sim_ID'] = i + 1
        simulations.append(sim_df)
        
    return simulations

# ==========================================
# 4. ANALYSIS: VOLUME CURVE
# ==========================================
def analyze_by_volume(master_pool):
    print("\n🔬 Analyzing Performance by Volume...")
    if 'Volume' not in master_pool.columns: return

    # Filter only the trades that WOULD be taken (Trend Aligned)
    # Also ensure Net_PnL is present
    if 'Net_PnL' not in master_pool.columns:
        # Calculate it if simulation hasn't run yet
        master_pool['Net_PnL'] = master_pool.apply(apply_friction, axis=1)

    if ENABLE_TREND_FILTER:
        df = master_pool[master_pool['Trend_Aligned'] == True].copy()
    else:
        df = master_pool.copy()

    try:
        df['Vol_Bucket'] = pd.qcut(df['Volume'], q=10, labels=False)
    except: return

    # Use NET PnL to see real edge after friction
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
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(stats.index, stats['Avg_Net_PnL'], color='skyblue', label='Net PnL %')
    ax2 = plt.gca().twinx()
    ax2.plot(stats.index, stats['Win_Rate'], color='green', marker='o', linewidth=2, label='Win Rate %')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Net Profit by Volume (Trend Filter: {ENABLE_TREND_FILTER})")
    plt.xlabel("Volume Decile (0=Low, 9=High)")
    plt.ylabel("Avg Net PnL (%)")
    ax2.set_ylabel("Win Rate (%)")
    
    # Label x-axis
    plt.xticks(stats.index, [f"{x/1000:.0f}k" for x in stats['Avg_Vol']])
    
    plt.savefig("volume_curve.png")
    print("✅ Volume Curve saved to volume_curve.png")

# ==========================================
# 5. REPORTING & VISUALIZATION (Existing)
# ==========================================
def calculate_kpis(df):
    """Calculates KPIs for a single dataframe of trades."""
    if df.empty: return {}
    wins = df[df['Net_PnL'] > 0]
    win_rate = (len(wins) / len(df)) * 100
    stopped = df[df['Outcome'] == 'Stopped Out']
    stop_rate = (len(stopped) / len(df)) * 100
    avg_ret = df['Net_PnL'].mean() * 100
    total_r = df['Net_PnL'].sum()
    df = df.sort_values('Date').copy() 
    df['Equity'] = df['Net_PnL'].cumsum()
    df['HWM'] = df['Equity'].cummax()
    df['DD'] = df['Equity'] - df['HWM']
    max_dd = df['DD'].min()
    return {
        'Trades': len(df),
        'Win Rate %': win_rate,
        'Stop Rate %': stop_rate,
        'Avg Return %': avg_ret,
        'Total R': total_r,
        'Max DD R': max_dd
    }

def generate_report(simulations):
    if not simulations:
        print("❌ No trades generated.")
        return
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    if ENABLE_MC:
        print(f"\n📊 MONTE CARLO REPORT ({NUM_SIMULATIONS} Runs)")
        all_kpis = []
        equity_curves = []
        for sim in simulations:
            kpis = calculate_kpis(sim)
            all_kpis.append(kpis)
            curve = sim[['Date', 'Equity']].set_index('Date')
            curve.columns = [f"Run_{sim['Sim_ID'].iloc[0]}"]
            equity_curves.append(curve)
        kpi_df = pd.DataFrame(all_kpis)
        print("="*60)
        print(f"{'Metric':<20} | {'Mean':<10} | {'Min (Worst)':<10} | {'Max (Best)':<10}")
        print("-" * 60)
        for col in kpi_df.columns:
            mean_val = kpi_df[col].mean()
            min_val = kpi_df[col].min()
            max_val = kpi_df[col].max()
            print(f"{col:<20} | {mean_val:10.2f} | {min_val:10.2f} | {max_val:10.2f}")
        print("="*60)
        summary_file = f"summary_mc_{timestamp}.csv"
        kpi_df.to_csv(summary_file, index=False)
        full_curves = pd.concat(equity_curves, axis=1).ffill().fillna(0)
        plt.figure(figsize=(12, 8))
        for col in full_curves.columns:
            plt.plot(full_curves.index, full_curves[col], alpha=0.1, color='gray', linewidth=1)
        avg_curve = full_curves.mean(axis=1)
        plt.plot(avg_curve.index, avg_curve, color='#007BFF', linewidth=2.5, label='Average Outcome')
        plt.title(f"Monte Carlo ({NUM_SIMULATIONS} Runs) - SL: {STOP_LOSS_PCT*100}% - Friction: {ENABLE_FRICTION}")
        plt.ylabel("Accumulated R")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plot_file = f"chart_mc_{timestamp}.png"
        plt.savefig(plot_file)
        print(f"✅ Chart saved to {plot_file}")
    else:
        sim = simulations[0]
        kpis = calculate_kpis(sim)
        print("\n" + "="*60)
        print(f"📊 SINGLE RUN REPORT")
        print("="*60)
        for k, v in kpis.items():
            print(f"   {k:<15}: {v:.2f}")
        trade_file = f"trades_single_{timestamp}.csv"
        sim.to_csv(trade_file, index=False)
        plt.figure(figsize=(12, 6))
        plt.plot(sim['Date'], sim['Equity'], label=f"Equity (Total: {kpis['Total R']:.2f}R)")
        plt.title(f"Equity Curve - SL: {STOP_LOSS_PCT*100}% - Friction: {ENABLE_FRICTION}")
        plt.ylabel("Accumulated R")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_file = f"chart_single_{timestamp}.png"
        plt.savefig(plot_file)
        print(f"✅ Chart saved to {plot_file}")

def get_benchmark_data():
    print(f"📡 Fetching Benchmark ({BENCHMARK_TICKER}) Data...")
    df = get_historical_data(BENCHMARK_TICKER)
    if df.empty: return pd.Series(dtype=float)
    df = df.sort_index()
    return df['close'].pct_change().fillna(0)

def process_portfolio_daily(sim_trades, benchmark_ret):
    if sim_trades.empty: return pd.DataFrame()
    sim_trades = sim_trades.copy()
    sim_trades['PnL_Dollar'] = sim_trades['Net_PnL'] * AVG_TRADE_SIZE
    daily_pnl = sim_trades.groupby('Date')['PnL_Dollar'].sum()
    if not daily_pnl.empty:
        start_date = daily_pnl.index.min()
        end_date = daily_pnl.index.max()
        bench_subset = benchmark_ret.loc[start_date:end_date] if not benchmark_ret.empty else pd.Series(dtype=float)
        full_index = daily_pnl.index.union(bench_subset.index).sort_values()
        df = pd.DataFrame(index=full_index)
        df['Bench_Ret'] = bench_subset.reindex(full_index).fillna(0)
        df['Daily_PnL'] = daily_pnl.reindex(full_index).fillna(0)
        df['Equity'] = STARTING_CAPITAL + df['Daily_PnL'].cumsum()
        df['Prev_Equity'] = df['Equity'].shift(1).fillna(STARTING_CAPITAL)
        df['Port_Ret'] = df['Daily_PnL'] / df['Prev_Equity']
        return df
    return pd.DataFrame()

def calculate_advanced_metrics(daily_df):
    if daily_df.empty: return {}
    df = daily_df.dropna().copy()
    if df.empty: return {}
    rp = df['Port_Ret']
    rm = df['Bench_Ret']
    rf_daily = RISK_FREE_RATE / 252.0
    ann_factor = 252
    days = (df.index[-1] - df.index[0]).days
    if days > 0:
        cagr = (df['Equity'].iloc[-1] / STARTING_CAPITAL) ** (365/days) - 1
    else:
        cagr = 0
    vol = rp.std() * np.sqrt(ann_factor)
    if len(rp) > 1 and rp.std() > 0 and rm.std() > 0:
        cov_matrix = np.cov(rp, rm)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    else:
        beta = 0
    rp_mean = rp.mean()
    rm_mean = rm.mean()
    alpha_daily = rp_mean - (rf_daily + beta * (rm_mean - rf_daily))
    alpha_ann = alpha_daily * ann_factor
    rp_ann = rp_mean * ann_factor
    if abs(beta) > 0.0001:
        treynor = (rp_ann - RISK_FREE_RATE) / beta
    else:
        treynor = np.nan
    active_ret = rp - rm
    tracking_error = active_ret.std() * np.sqrt(ann_factor)
    if tracking_error > 0:
        info_ratio = (active_ret.mean() * ann_factor) / tracking_error
    else:
        info_ratio = np.nan
    df['HWM'] = df['Equity'].cummax()
    df['DD_Pct'] = (df['Equity'] - df['HWM']) / df['HWM']
    max_dd = df['DD_Pct'].min()
    if max_dd < 0:
        calmar = cagr / abs(max_dd)
    else:
        calmar = np.nan
    threshold = rf_daily
    excess = rp - threshold
    positive_sum = excess[excess > 0].sum()
    negative_sum = abs(excess[excess < 0].sum())
    if negative_sum > 0:
        omega = positive_sum / negative_sum
    else:
        omega = np.inf
    var_95 = np.percentile(rp, (1 - VAR_CONFIDENCE) * 100)
    cvar_95 = rp[rp <= var_95].mean()
    sharpe = (rp_ann - RISK_FREE_RATE) / vol if vol > 0 else 0
    downside_std = rp[rp < 0].std() * np.sqrt(ann_factor)
    sortino = (rp_ann - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0
    return {
        'CAGR': cagr,
        'Vol (Ann)': vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max DD %': max_dd,
        'Beta': beta,
        'Alpha (Ann)': alpha_ann,
        'Treynor': treynor,
        'Info Ratio': info_ratio,
        'Calmar': calmar,
        'Omega': omega,
        'VaR 95% (Daily)': var_95,
        'CVaR 95% (Daily)': cvar_95
    }

def generate_excel_report_full(simulations, benchmark_data):
    if not simulations: return
    final_equities = [sim['Equity'].iloc[-1] for sim in simulations]
    median_idx = np.argsort(final_equities)[len(final_equities)//2]
    best_sim = simulations[median_idx]
    print(f"\n📝 Generating Excel Report (Using Median Run #{best_sim['Sim_ID'].iloc[0]})...")
    daily_df = process_portfolio_daily(best_sim, benchmark_data)
    metrics = calculate_advanced_metrics(daily_df)
    filename = f"Sea15_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            summary_series = pd.Series(metrics, name="Value")
            summary_series.to_excel(writer, sheet_name="Summary")
            if not daily_df.empty:
                w_df = daily_df.resample('W').agg({
                    'Daily_PnL': 'sum',
                    'Equity': 'last',
                    'Port_Ret': lambda x: (1+x).prod() - 1
                })
                w_df.to_excel(writer, sheet_name="Weekly_Analysis")
                daily_df.to_excel(writer, sheet_name="Daily_Log")
            best_sim.to_excel(writer, sheet_name="Trade_Ledger", index=False)
        print(f"✅ Excel Report saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving Excel report: {e}")

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
                sim_results = run_simulation(pool)
                # CALL THE ANALYSIS FUNCTION
                analyze_by_volume(pool)
                
                generate_report(sim_results)
                generate_excel_report_full(sim_results, bench_ret)