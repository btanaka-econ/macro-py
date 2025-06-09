import pandas as pd
import numpy as np

pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

oecd_countries = [
    'Australia','Austria', 'Belgium', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Japan', 'Netherlands', 'New Zealand', 'Norway', 'Portugal', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'United States',
]

start = 1960
end = 2000
data = pwt90[
    pwt90['country'].isin(oecd_countries) &
    pwt90['year'].between(start, end)
]

relevant_cols = ['countrycode', 'country', 'year', 'rgdpna', 'rkna', 'pop', 'emp', 'avh', 'labsh', 'rtfpna']
data = data[relevant_cols].dropna()
# 'rgdpna' real GDP, national accounts
# 'rkna' real capital stock
# 'emp' number of persons engaged (workers)
# 'avh' average annual hours worked per worker
# 'labsh' labor share in GDP (1-alpha)

# Transform Y & K to y & k (per-worker)
data['y'] = data['rgdpna'] / data['emp'] # output per woker
data['k'] = data['rkna'] / data['emp'] # capital per woker
# Log transform
data['ln_y'] = np.log(data['y'])  # log (GDP per worker)
data['ln_k'] = np.log(data['k']) # log (Capital per worker)

# Calculate average annual growth rates
def annual_log_growth(group, col):
    t_start = group.loc[group['year'] == start, col].values[0]
    t_end = group.loc[group['year'] == end,   col].values[0]
    return 100.0 * (np.log(t_end) - np.log(t_start)) / (end - start)

results = []
for name, g in data.groupby('country'):
    g_y = annual_log_growth(g, 'y')     # total growth (output per worker)
    g_k = annual_log_growth(g, 'k')     # growth of capital per worker

    # capital-deepening component: α * g_k
    alpha_mean = g["labsh"].mean()          # 1 − labour share
    cap_deepen = (1 - alpha_mean) * g_k

    # TFP growth (compute residual)
    tfp_g = g_y - cap_deepen
    
    # shares of total growth
    tfp_share  = tfp_g / g_y if g_y != 0 else np.nan # to avoid dividing by zero
    cap_share  = cap_deepen / g_y if g_y != 0 else np.nan

    results.append(
        dict(Country=name,
             Growth_Rate=round(g_y, 2),
             TFP_Growth=round(tfp_g, 2),
             Capital_Deepening=round(cap_deepen, 2),
             TFP_Share=round(tfp_share, 2),
             Capital_Share=round(cap_share, 2))
    )
 
    

data['g_k'] = (data['ln_k']-data['ln_k'].shift(1))*100 # change in k %
data = data.sort_values('year').groupby('countrycode').apply(lambda x: x.assign(
    alpha=1 - x['labsh'],
    y_n_shifted=100 * x['y_n'] / x['y_n'].iloc[0],
    tfp_term_shifted=100 * x['tfp_term'] / x['tfp_term'].iloc[0],
    cap_term_shifted=100 * x['cap_term'] / x['cap_term'].iloc[0],
    lab_term_shifted=100 * x['lab_term'] / x['lab_term'].iloc[0]
)).reset_index(drop=True).dropna()



def calculate_growth_rates(country_data):
    
    start_year_actual = country_data['year'].min()
    end_year_actual = country_data['year'].max()

    start_data = country_data[country_data['year'] == start_year_actual].iloc[0]
    end_data = country_data[country_data['year'] == end_year_actual].iloc[0]

    years = end_data['year'] - start_data['year']

    g_y = ((end_data['y_n'] / start_data['y_n']) ** (1/years) - 1) * 100

    g_k = ((end_data['cap_term'] / start_data['cap_term']) ** (1/years) - 1) * 100

    g_a = ((end_data['tfp_term'] / start_data['tfp_term']) ** (1/years) - 1) * 100

    alpha_avg = (start_data['alpha'] + end_data['alpha']) / 2.0
    capital_deepening_contrib = alpha_avg * g_k
    tfp_growth_calculated = g_a
    
    tfp_share = (tfp_growth_calculated / g_y)
    cap_share = (capital_deepening_contrib / g_y)

    return {
        'Country': start_data['country'],
        'Growth Rate': round(g_y, 2),
        'TFP Growth': round(tfp_growth_calculated, 2),
        'Capital Deepening': round(capital_deepening_contrib, 2),
        'TFP Share': round(tfp_share, 2),
        'Capital Share': round(cap_share, 2)
    }


results_list = data.groupby('country').apply(calculate_growth_rates).dropna().tolist()
results_df = pd.DataFrame(results_list)

avg_row_data = {
    'Country': 'Average',
    'Growth Rate': round(results_df['Growth Rate'].mean(), 2),
    'TFP Growth': round(results_df['TFP Growth'].mean(), 2),
    'Capital Deepening': round(results_df['Capital Deepening'].mean(), 2),
    'TFP Share': round(results_df['TFP Share'].mean(), 2),
    'Capital Share': round(results_df['Capital Share'].mean(), 2)
}
results_df = pd.concat([results_df, pd.DataFrame([avg_row_data])], ignore_index=True)

print("\nGrowth Accounting in OECD Countries:", start,-  end, "period")
print("="*85)
print(results_df.to_string(index=False))

