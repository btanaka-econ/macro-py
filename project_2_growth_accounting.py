import pandas as pd
import numpy as np

pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

oecd_countries = [
    'Australia','Austria','Belgium','Canada','Denmark','Finland','France',
    'Germany','Greece','Iceland','Ireland','Italy','Japan','Netherlands',
    'New Zealand','Norway','Portugal','Spain','Sweden','Switzerland',
    'United Kingdom','United States',
]

start = 1990
end   = 2019

data = pwt90[
    pwt90['country'].isin(oecd_countries) &
    pwt90['year'].between(start, end)
]

relevant_cols = [
    'countrycode','country','year',
    'rgdpna','rkna','pop','emp','avh','labsh','rtfpna'
]
data = data[relevant_cols].dropna()

# Labor input in hours
L = data["emp"] * data["avh"]

# Output & capital per hour worked
data['y'] = data['rgdpna'] / L
data['k'] = data['rkna']   / L

# Log levels (not strictly needed for this version, but kept for consistency)
data['ln_y'] = np.log(data['y'])
data['ln_k'] = np.log(data['k'])

def annual_log_growth(group, col, start_year=start, end_year=end):
    years = group['year'].values
    # if both endpoints exist, use them; else use that country's min/max
    if (start_year in years) and (end_year in years):
        y0, yT = start_year, end_year
    else:
        y0, yT = years.min(), years.max()
    t0 = group.loc[group['year'] == y0, col].values[0]
    tT = group.loc[group['year'] == yT, col].values[0]
    return 100.0 * (np.log(tT) - np.log(t0)) / (yT - y0)

results = []
for name, g in data.groupby('country'):
    g_y = annual_log_growth(g, 'y')
    g_k = annual_log_growth(g, 'k')

    # fixed capital share
    alpha = 0.30
    cap_deepen = alpha * g_k
    tfp_g = g_y - cap_deepen

    # avoid division by zero
    tfp_share = tfp_g / g_y if g_y != 0 else np.nan
    cap_share = cap_deepen / g_y if g_y != 0 else np.nan

    results.append({
        'Country':            name,
        'Growth_Rate':        round(g_y, 2),
        'TFP_Growth':         round(tfp_g, 2),
        'Capital_Deepening':  round(cap_deepen, 2),
        'TFP_Share':          round(tfp_share, 2),
        'Capital_Share':      round(cap_share, 2),
    })

table = pd.DataFrame(results).sort_values('Country').reset_index(drop=True)

# add OECD average
avg = {
    'Country':          'Average',
    'Growth_Rate':      round(table['Growth_Rate'].mean(), 2),
    'TFP_Growth':       round(table['TFP_Growth'].mean(), 2),
    'Capital_Deepening':round(table['Capital_Deepening'].mean(), 2),
    'TFP_Share':        round(table['TFP_Share'].mean(), 2),
    'Capital_Share':    round(table['Capital_Share'].mean(), 2),
}
table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)

print(f"Growth Accounting in OECD Countries: {start}-{end} period")
print(table.to_markdown(index=True))
