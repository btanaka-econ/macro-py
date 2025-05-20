import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Load Penn World Table 9.0
# ------------------------------------------------------------
pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

oecd_countries = [
    'Australia','Austria','Belgium','Canada','Denmark','Finland','France',
    'Germany','Greece','Iceland','Ireland','Italy','Japan','Netherlands',
    'New Zealand','Norway','Portugal','Spain','Sweden','Switzerland',
    'United Kingdom','United States'
]

start_year, end_year = 1960, 2000

data = (
    pwt90.loc[
        pwt90['country'].isin(oecd_countries)
        & pwt90['year'].between(start_year, end_year)
    ,
        ['countrycode','country','year','rgdpna','rkna','pop',
         'emp','avh','labsh','rtfpna']
    ]
    .dropna()
)

# ------------------------------------------------------------
# 2. Build variables for growth accounting
# ------------------------------------------------------------
data['alpha']    = 1 - data['labsh']                    # capital share
data['y_n']      = data['rgdpna'] / data['emp']         # GDP per worker
data['hours']    = data['emp'] * data['avh']            # total hours
data['tfp_term'] = data['rtfpna']**(1/(1 - data['alpha']))
data['cap_term'] = (data['rkna']/data['rgdpna'])**(
                     data['alpha'] / (1 - data['alpha'])
                   )

# Sorting by time helps the .iloc[0] / .iloc[-1] calls later
data = data.sort_values('year')

# ------------------------------------------------------------
# 3. Growth-rate calculator
# ------------------------------------------------------------
def calculate_growth_rates(df):
    first, last = df.iloc[0], df.iloc[-1]
    years       = last['year'] - first['year']

    g_y = ((last['y_n']      / first['y_n'])      ** (1/years) - 1) * 100
    g_k = ((last['cap_term'] / first['cap_term']) ** (1/years) - 1) * 100
    g_a = ((last['tfp_term'] / first['tfp_term']) ** (1/years) - 1) * 100

    alpha_bar   = 0.5 * (first['alpha'] + last['alpha'])
    cap_contrib = alpha_bar * g_k
    tfp_contrib = g_a

    total = cap_contrib + tfp_contrib        # = “Growth Rate” in the table
    tfp_share = tfp_contrib / total
    cap_share = cap_contrib / total

    return {
        'Country'          : first['country'],
        'Growth Rate'      : round(total, 2),
        'TFP Growth'       : round(tfp_contrib, 2),
        'Capital Deepening': round(cap_contrib, 2),
        'TFP Share'        : round(tfp_share, 2),
        'Capital Share'    : round(cap_share, 2)
    }

# ------------------------------------------------------------
# 4. Apply to every country and build the output table
# ------------------------------------------------------------
rows = (
    data.groupby('country', group_keys=False)
        .apply(calculate_growth_rates)
        .tolist()
)
table51 = pd.DataFrame(rows)

# Add the across-country average row
table51.loc[len(table51)] = {
    'Country'          : 'Average',
    'Growth Rate'      : round(table51['Growth Rate'].mean(), 2),
    'TFP Growth'       : round(table51['TFP Growth'].mean(), 2),
    'Capital Deepening': round(table51['Capital Deepening'].mean(), 2),
    'TFP Share'        : round(table51['TFP Share'].mean(), 2),
    'Capital Share'    : round(table51['Capital Share'].mean(), 2)
}

# ------------------------------------------------------------
# 5. Print in Table 5-1 style
# ------------------------------------------------------------
print(f"\nGrowth Accounting in OECD Countries: {start_year}–{end_year}")
print("="*85)
print(table51.to_string(index=False))
