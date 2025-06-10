import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader.data as web   # ← this is the only import you need from pdr

# -----------------------------------------------------------
# 1. Parameters
# -----------------------------------------------------------
start_date = '1996-01-01'
end_date   = '2024-10-01'
series_id  = 'NGDPRSAXDCBRQ'           # Brazil, real GDP, quarterly
lambdas    = [10, 100, 1600]           # smoothing parameters to compare

# -----------------------------------------------------------
# 2. Download series from FRED and take logs
# -----------------------------------------------------------
gdp_q   = web.DataReader(series_id, 'fred', start_date, end_date)
log_gdp = np.log(gdp_q[series_id]).rename('log_gdp')

# -----------------------------------------------------------
# 3. HP-filter for each λ, plot results
# -----------------------------------------------------------
for lam in lambdas:
    cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=lam)

    # ----- (a) Figure 1: log GDP + trend (one λ per chart)
    plt.figure(figsize=(10, 4))
    plt.plot(log_gdp, label='log real GDP', linewidth=2)
    plt.plot(trend,  label=f'Trend (λ = {lam})', linewidth=2)
    plt.title(f'Brazil: log GDP & HP-filter trend  (λ = {lam})')
    plt.xlabel('Year');  plt.ylabel('log level');  plt.legend()
    plt.tight_layout();  plt.show()

    # ----- (b) Figure 2: cyclical component (optional)
    plt.figure(figsize=(10, 4))
    plt.plot(cycle, label=f'Cycle (λ = {lam})')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.title(f'Brazil: HP-filter cyclical component  (λ = {lam})')
    plt.xlabel('Year');  plt.ylabel('Deviation from trend');  plt.legend()
    plt.tight_layout();  plt.show()
