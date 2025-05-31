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

# Transform Y to y
data['alpha'] = 1 - data['labsh'] # α (output elasticity of capital %)
print(data['alpha']) 
data['hours'] = data['emp'] * data['avh']  # L (annual hours h)
data['y_n'] = data['rgdpna'] / data['hours']  # y = Y/L
data['k_n'] = data['rkna'] / data['hours'] # k = K/L

# log transformation to compute growth rate 
logyn = np.log(data['y_n']) 
logkn = np.log(data['k_n'])

data['tfp_term'] = data['rtfpna'] ** (1 / (1 - data['alpha']))  # A^(1/(1-alpha))
data['cap_term'] = (data['rkna'] / data['rgdpna']) ** (data['alpha'] / (1 - data['alpha']))  # (K/Y)^(alpha/(1-alpha))
data['lab_term'] = data['hours'] / data['pop']  # L/N pop or emp
data = data.sort_values('year').groupby('countrycode').apply(lambda x: x.assign(
    alpha=1 - x['labsh'],
    y_h_shifted=100 * x['y_n'] / x['y_n'].iloc[0],
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

