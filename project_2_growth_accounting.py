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
 
# Assemble the table and add average row of all countries
table = pd.DataFrame(results).sort_values("Country").reset_index(drop=True)

avg_row = {
    "Country": "Average",
    "Growth_Rate":  round(table["Growth_Rate"].mean(), 2),
    "TFP_Growth":   round(table["TFP_Growth"].mean(),  2),
    "Capital_Deepening": round(table["Capital_Deepening"].mean(), 2),
    "TFP_Share":    round(table["TFP_Share"].mean(),   2),
    "Capital_Share":round(table["Capital_Share"].mean(),2),
}
table = pd.concat([table, pd.DataFrame([avg_row])], ignore_index=True)

# print the result
print(f"Growth Accounting in OECD Countries: {start}-{end} period")
print(table.to_markdown(index=True)) # "pip install tabulate" if you have trouble using print()