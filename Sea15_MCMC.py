import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & PRIORS
# ==========================================
MCMC_STEPS = 5000
BURN_IN = 1000
TEMPERATURE = 0.5 # Lower T makes it pickier (exploits peaks)

# Fixed Constraints
MAX_TRADES_PER_DAY = 10
COMMISSION = 1.00
SLIPPAGE_PCT = 0.002
AVG_TRADE_SIZE = 2000

# Priors
PRIORS = {
    'stop_loss':    (0.005, 0.10, 'float'),
    'gap_min':      (0.02, 0.15, 'float'),
    'gap_max':      (0.05, 0.40, 'float'),
    'sma_period':   (0, 50, 'int'),
    'min_volume':   (50000, 500000, 'int'),
    'max_volume':   (500000, 5000000, 'int'),
    'high_vol_decay': (0.01, 1.0, 'float') # 1.0 = Keep All, 0.1 = Keep 10%
}

CACHE_DIR = 'market_data_cache_unified'
MASTER_POOL_FILE = 'sea_master_pool_unified.csv'

# ==========================================
# 2. DATA LOADER
# ==========================================
def load_data():
    if not os.path.exists(MASTER_POOL_FILE):
        print(f"❌ Error: {MASTER_POOL_FILE} not found.")
        return pd.DataFrame()

    print(f"📂 Loading Master Pool: {MASTER_POOL_FILE}...")
    df = pd.read_csv(MASTER_POOL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Int'] = df['Date'].astype('int64') // 10**9

    req_cols = ['Date_Int', 'Volume', 'Gap_Pct', 'High', 'Low', 'Close', 'Entry_Price', 'Trend_Aligned']
    # If missing new Trend columns, just warn (so it works with old pool files too)
    for c in req_cols:
        if c not in df.columns: return pd.DataFrame()

    # Fill missing Trend columns if loading an old CSV
    if 'Trend_10' not in df.columns:
        print("⚠️ Warning: Trend_10/20/50 missing. MCMC will fallback to Trend_Aligned for all periods.")
        df['Trend_10'] = df['Trend_Aligned']
        df['Trend_20'] = df['Trend_Aligned']
        df['Trend_50'] = df['Trend_Aligned']

    return df

# ==========================================
# 3. MCMC ENGINE
# ==========================================
class Sea15_MCMC:
    def __init__(self, pool_df):
        self.pool = pool_df
        # Pre-compute numpy arrays
        self.dates = pool_df['Date_Int'].values
        self.vols = pool_df['Volume'].values
        self.gaps = pool_df['Gap_Pct'].values
        self.trends = pool_df['Trend_Aligned'].values
        self.trend_10 = pool_df['Trend_10'].values
        self.trend_20 = pool_df['Trend_20'].values
        self.trend_50 = pool_df['Trend_50'].values
        self.types = np.where(pool_df['Type'] == 'Short', 0, 1)
        self.entries = pool_df['Entry_Price'].values
        self.highs = pool_df['High'].values
        self.lows = pool_df['Low'].values
        self.closes = pool_df['Close'].values

        self.current_params = self.random_params()
        self.current_score = -np.inf
        self.history = []

    def random_params(self):
        p = {}
        for k, (min_v, max_v, dtype) in PRIORS.items():
            val = np.random.uniform(min_v, max_v)
            if dtype == 'int': val = int(round(val))
            p[k] = val
        if p['gap_max'] <= p['gap_min']: p['gap_max'] = p['gap_min'] + 0.05
        return p

    def propose(self, current_p):
        new_p = current_p.copy()
        keys = list(PRIORS.keys())
        num_mutations = np.random.randint(1, 3)
        targets = np.random.choice(keys, num_mutations, replace=False)

        for k in targets:
            min_v, max_v, dtype = PRIORS[k]
            sigma = (max_v - min_v) * 0.15 # Step size
            val = new_p[k] + np.random.normal(0, sigma)
            val = max(min_v, min(val, max_v))
            if dtype == 'int': val = int(round(val))
            new_p[k] = val

        if new_p['gap_max'] <= new_p['gap_min']: new_p['gap_max'] = new_p['gap_min'] + 0.02
        return new_p

    def run_simulation(self, params):
        # 1. Filtering

        # Trend Selection (Nearest Neighbor Mapping)
        sma = params['sma_period']
        if sma <= 0:
            mask_trend = np.ones(len(self.pool), dtype=bool)
        elif sma < 15:
            mask_trend = (self.trend_10 == True)
        elif sma < 35:
            mask_trend = (self.trend_20 == True)
        else:
            mask_trend = (self.trend_50 == True)

        # Volume Filter (Min)
        mask_vol_min = (self.vols >= params['min_volume'])

        # Volume Soft Cap (Deterministic Sampling)
        # Logic: If Vol <= Max, keep. If Vol > Max, keep if hash(date+ticker) % 100 < rate*100
        # For MCMC speed and stability, we use a simpler deterministic mask relative to the array index
        # (Assuming the array is shuffled or somewhat random, taking every Nth element works as a sample)
        # BUT 'indices' changes every loop.
        # Vectorized implementation:

        max_v = params['max_volume']
        rate = params['high_vol_decay']

        # High Vol Mask
        is_high_vol = (self.vols > max_v)

        # "Random" Deterministic Noise for sampling
        # We can use the 'Volume' itself as a hash source: (Volume % 100) / 100.0
        # If (Vol % 100) < (rate * 100), we keep it.
        # This is deterministic and fast.

        hash_val = (self.vols % 10007) / 10007.0 # Prime modulus
        keep_high_vol = (hash_val <= rate)

        # Combined Vol Mask: (Low Vol) OR (High Vol AND Keep)
        mask_vol_final = mask_vol_min & ((~is_high_vol) | keep_high_vol)

        mask_short = (self.types == 0) & (self.gaps >= params['gap_min']) & (self.gaps <= params['gap_max'])
        mask_long = (self.types == 1) & (self.gaps <= -params['gap_min']) & (self.gaps >= -params['gap_max'])

        final_mask = mask_vol_final & mask_trend & (mask_short | mask_long)
        indices = np.where(final_mask)[0]

        if len(indices) < 20: return np.array([]) # Loosened constraint for MCMC exploration

        # 2. PnL Calc
        entries = self.entries[indices]
        highs = self.highs[indices]
        lows = self.lows[indices]
        closes = self.closes[indices]
        types = self.types[indices]

        sl = params['stop_loss']
        pnls = np.zeros(len(indices))

        s_idx = (types == 0)
        if np.any(s_idx):
            stops_s = entries[s_idx] * (1 + sl)
            stopped_s = highs[s_idx] >= stops_s
            pnls[s_idx] = np.where(stopped_s, -sl, (entries[s_idx] - closes[s_idx]) / entries[s_idx])

        l_idx = (types == 1)
        if np.any(l_idx):
            stops_l = entries[l_idx] * (1 - sl)
            stopped_l = lows[l_idx] <= stops_l
            pnls[l_idx] = np.where(stopped_l, -sl, (closes[l_idx] - entries[l_idx]) / entries[l_idx])

        pnls -= (SLIPPAGE_PCT * 2) + (COMMISSION / AVG_TRADE_SIZE)

        # 3. Daily Limit (Priority by Volume)
        sim_dates = self.dates[indices]
        sim_vols = self.vols[indices]

        # Create DataFrame for grouping
        df_run = pd.DataFrame({'Date': sim_dates, 'PnL': pnls, 'Vol': sim_vols})
        df_run = df_run.sort_values(['Date', 'Vol'], ascending=[True, False])

        # Take Top N
        df_lim = df_run.groupby('Date').head(MAX_TRADES_PER_DAY)

        # Daily Sums
        return df_lim.groupby('Date')['PnL'].sum().values

    def calculate_score(self, daily_returns):
        if len(daily_returns) < 20: return -10.0

        mean_ret = np.mean(daily_returns)
        neg_rets = daily_returns[daily_returns < 0]
        downside_std = np.std(neg_rets) if len(neg_rets) > 0 else 1e-6
        if downside_std == 0: downside_std = 1e-6

        sortino = (mean_ret / downside_std) * np.sqrt(252)

        # Robustness Penalty (Optional): Penalize very low trade counts heavily
        if len(daily_returns) < 100: sortino *= 0.5

        return sortino

    def run(self):
        print(f"🚀 Starting MCMC Optimization ({MCMC_STEPS} Steps, T={TEMPERATURE})...")

        # Initialize
        self.current_score = self.calculate_score(self.run_simulation(self.current_params))
        print(f"   Initial Params: {self.current_params}")
        print(f"   Initial Score: {self.current_score:.4f}")

        accepted_count = 0
        best_score = -np.inf
        best_params = None

        for i in range(MCMC_STEPS):
            prop_p = self.propose(self.current_params)
            daily_rets = self.run_simulation(prop_p)
            new_score = self.calculate_score(daily_rets)

            diff = new_score - self.current_score
            alpha = 1.0 if diff > 0 else np.exp(diff / TEMPERATURE)

            is_accepted = False
            if np.random.rand() < alpha:
                self.current_params = prop_p
                self.current_score = new_score
                accepted_count += 1
                is_accepted = True

                if new_score > best_score:
                    best_score = new_score
                    best_params = prop_p.copy()

            # Record State (Always record where the walker IS)
            rec = self.current_params.copy()
            rec['Score'] = self.current_score
            rec['Accepted'] = is_accepted # Just for debug
            self.history.append(rec)

            if i % 500 == 0:
                print(f"   Step {i} | Score: {self.current_score:.2f} | Acc Rate: {accepted_count/(i+1):.2f}")

        print("\n✅ MCMC Complete.")
        print(f"🏆 Best Score: {best_score:.4f}")
        print(f"🏆 Best Params: {best_params}")
        return pd.DataFrame(self.history)

# ==========================================
# 4. ANALYSIS & VISUALIZATION
# ==========================================
def analyze_results(df):
    if df.empty: return

    # Remove Burn-in
    df_clean = df.iloc[BURN_IN:].copy()
    print(f"\n🔬 Analysis on {len(df_clean)} samples (post burn-in)...")

    # 1. Parameter Importance (Random Forest)
    print("   Training Random Forest for Feature Importance...")
    features = list(PRIORS.keys())
    X = df_clean[features]
    y = df_clean['Score']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n📊 Feature Importance:")
    print(importances)

    # 2. Posterior Histograms
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 4, i+1) # Changed to 2x4 for 7 params
        plt.hist(df_clean[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"{col} Distribution")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mcmc_posteriors.png")
    print("✅ Saved mcmc_posteriors.png")

    # 3. Heatmap (Top 2 Features)
    top_2 = importances.index[:2]
    f1, f2 = top_2[0], top_2[1]

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df_clean[f1], df_clean[f2], c=df_clean['Score'], cmap='viridis', s=10, alpha=0.5)
    plt.colorbar(sc, label='Sortino Score')
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title(f"MCMC Landscape: {f1} vs {f2}")
    plt.grid(True, alpha=0.3)
    plt.savefig("mcmc_heatmap.png")
    print("✅ Saved mcmc_heatmap.png")

    # 4. Save Best Params to File
    best_row = df_clean.sort_values('Score', ascending=False).iloc[0]
    best_row.to_csv("best_params_mcmc.csv")

if __name__ == "__main__":
    pool = load_data()
    if not pool.empty:
        mcmc = Sea15_MCMC(pool)
        results = mcmc.run()
        analyze_results(results)
