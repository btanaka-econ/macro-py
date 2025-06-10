# -----------------------------------------------------------------------------
# Re-create Table 5.1 from Aghion & Howitt (2009)
# -----------------------------------------------------------------------------
import pandas as pd
import numpy  as np

# -------------------------------------------------------------------
# 1. User-set parameters
# -------------------------------------------------------------------
OECD_COUNTRIES = [
    "Australia","Austria","Belgium","Canada","Denmark","Finland","France",
    "Germany","Greece","Iceland","Ireland","Italy","Japan","Netherlands",
    "New Zealand","Norway","Portugal","Spain","Sweden","Switzerland",
    "United Kingdom","United States",
]
START_YEAR   = 1960
END_YEAR     = 2000
PWT_URL      = "https://www.rug.nl/ggdc/docs/pwt90.dta"  # Penn World Table 9.0

# -------------------------------------------------------------------
# 2. Load the Penn World Table and keep only what we need
# -------------------------------------------------------------------
pwt = pd.read_stata(PWT_URL)

cols = ["countrycode","country","year",
        "rgdpna",    # real GDP (output, national accounts)
        "rkna",      # real capital stock
        "emp",       # persons engaged (number of workers)
        "labsh",     # labour-share of income (if you prefer a time-varying α)
        "rtfpna"]    # TFP index (optional, see below)

df = (
    pwt.loc[
        pwt["country"].isin(OECD_COUNTRIES)
        & pwt["year"].between(START_YEAR, END_YEAR),
        cols
    ]
    .dropna(subset=["rgdpna","rkna","emp"])
)

# -------------------------------------------------------------------
# 3. Construct per-worker series (output y, capital k) and log levels
# -------------------------------------------------------------------
df["y_pc"] = df["rgdpna"] / df["emp"]      # output per worker
df["k_pc"] = df["rkna"]  / df["emp"]      # capital per worker
df["ln_y"] = np.log(df["y_pc"])
df["ln_k"] = np.log(df["k_pc"])

# -------------------------------------------------------------------
# 4. Compute average annual (compound) growth rates
#    g = 100/(T) * (ln x_T – ln x_0)
# -------------------------------------------------------------------
def annual_log_growth(group, col):
    t0 = group.loc[group["year"] == START_YEAR, col].values[0]
    tT = group.loc[group["year"] == END_YEAR,   col].values[0]
    return 100.0 * (np.log(tT) - np.log(t0)) / (END_YEAR - START_YEAR)

results = []
for name, g in df.groupby("country"):
    G_y = annual_log_growth(g, "y_pc")     # total growth (output per worker)
    g_k = annual_log_growth(g, "k_pc")     # growth of capital per worker

    # --- capital-deepening component: α * g_k
    alpha_bar = g["labsh"].mean()          # 1 − labour share
    cap_deepen = (1 - alpha_bar) * g_k


    # --- TFP growth: residual OR directly from PWT's rtfpna index -----------
    # residual method (matches the book’s description)
    tfp_g = G_y - cap_deepen

    # If you prefer to take TFP growth directly from the TFP index, uncomment:
    # tfp_g = annual_log_growth(g, "rtfpna")

    # --- shares of total growth
    tfp_share  = tfp_g / G_y if G_y != 0 else np.nan
    cap_share  = cap_deepen / G_y if G_y != 0 else np.nan

    results.append(
        dict(Country=name,
             Growth_Rate=round(G_y, 2),
             TFP_Growth=round(tfp_g, 2),
             Capital_Deepening=round(cap_deepen, 2),
             TFP_Share=round(tfp_share, 2),
             Capital_Share=round(cap_share, 2))
    )

# -------------------------------------------------------------------
# 5. Assemble the table and add an OECD average row
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 6. Display or save ----------------------------------------------------------------
print(table.to_string(index=False))

# Want an Excel copy?
# table.to_excel("Table5_1_growth_accounting.xlsx", index=False)
# print("Saved to Table5_1_growth_accounting.xlsx")
