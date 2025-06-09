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

def annual_log_growth(group, col):
    years = group['year'].values
    # choose 1990–2019 if available, otherwise use actual min/max years
    if (start in years) and (end in years):
        y0, yT = start, end
    else:
        y0, yT = years.min(), years.max()
    v0 = group.loc[group['year'] == y0, col].values[0]
    vT = group.loc[group['year'] == yT, col].values[0]
    g  = 100.0 * (np.log(vT) - np.log(v0)) / (yT - y0)
    return g, y0, yT

results = []
for country, grp in data.groupby('country'):
    # compute growth rates and actual start/end used
    g_y, y0, yT = annual_log_growth(grp, 'y')
    g_k, _, _   = annual_log_growth(grp, 'k')

    # compute time-varying alpha as average of endpoints
    row0     = grp.loc[grp['year'] == y0].iloc[0]
    rowT     = grp.loc[grp['year'] == yT].iloc[0]
    alpha0   = 1.0 - row0['labsh']
    alphaT   = 1.0 - rowT['labsh']
    alpha_av = (alpha0 + alphaT) / 2.0

    # decompose growth
    cap_deepen = alpha_av * g_k
    tfp_g      = g_y - cap_deepen

    # shares
    tfp_share = tfp_g / g_y if g_y != 0 else np.nan
    cap_share = cap_deepen / g_y if g_y != 0 else np.nan

    results.append({
        'Country':           country,
        'Start_Year':        y0,
        'End_Year':          yT,
        'Growth_Rate':       round(g_y, 2),
        'TFP_Growth':        round(tfp_g, 2),
        'Capital_Deepening': round(cap_deepen, 2),
        'TFP_Share':         round(tfp_share, 2),
        'Capital_Share':     round(cap_share, 2),
    })

# Build final table
table = pd.DataFrame(results).sort_values('Country').reset_index(drop=True)

# Add OECD average row
avg = {
    'Country':           'Average',
    'Start_Year':        '',
    'End_Year':          '',
    'Growth_Rate':       round(table['Growth_Rate'].mean(), 2),
    'TFP_Growth':        round(table['TFP_Growth'].mean(), 2),
    'Capital_Deepening': round(table['Capital_Deepening'].mean(), 2),
    'TFP_Share':         round(table['TFP_Share'].mean(), 2),
    'Capital_Share':     round(table['Capital_Share'].mean(), 2),
}
table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)

# Display
print(f"Growth Accounting in OECD Countries: {start}-{end} period")
print(table.to_markdown(index=True))