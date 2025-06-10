import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1955-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
country_id = 'NGDPRSAXDCBRQ' # Brazil, real GDP, quarterly
gdp = web.DataReader(country_id, 'fred', start_date, end_date)
log_gdp = np.log(gdp[country_id])

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
# use for loop to compute every lambda
lambdas = [10, 100, 1600]
for lamb in lambdas:
    cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb)

    # Plot the original time series data
    plt.plot(log_gdp, label="Original GDP (in log)")
    plt.title(f'Brazil: log GDP & HP-filter trend  (λ = {lamb})')

    # Plot the trend component
    plt.plot(trend, label="Trend")

    # Add a legend and show the plot
    plt.legend()
    plt.show()