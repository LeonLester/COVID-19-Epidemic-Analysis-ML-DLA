import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date, timedelta
import operator 
plt.style.use('fivethirtyeight')
#%matplotlib inline // cli for now
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import warnings
warnings.filterwarnings("ignore")



confirmed_global_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_global_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_global_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

confirmed_US_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
deaths_US_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

apple_mobility_df = pd.read_csv('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/apple_reports/apple_mobility_report.csv')


#yesterday = date.today() - timedelta(days=1)
#most_recent_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' + yesterday.strftime('%m-%d-%Y') + '.csv'
#most_recent_global_df = pd.read_csv(latest_url)
#print(latest_global_df.head())



confirmed_global_dates_df = confirmed_global_df.drop(columns=['Lat','Long'])
deaths_global_dates_df = deaths_global_df.drop(columns=['Lat','Long'])
recovered_global_dates_df = recovered_global_df.drop(columns=['Lat','Long'])

death_confirmed_dates_df = deaths_global_dates_df.copy()
recovered_confirmed_dates_df = recovered_global_dates_df.copy()

for j in range(2,len(confirmed_global_dates_df.columns)):
    death_confirmed_dates_df[confirmed_global_dates_df.columns[j]]=death_confirmed_dates_df[confirmed_global_dates_df.columns[j]].astype(float)
    recovered_confirmed_dates_df[confirmed_global_dates_df.columns[j]]=recovered_confirmed_dates_df[confirmed_global_dates_df.columns[j]].astype(float)
for i in range(len(confirmed_global_dates_df.index)):
    for j in range(2,len(confirmed_global_dates_df.columns)):
        if(confirmed_global_dates_df.iat[i,j]==0):
            death_confirmed_dates_df.at[i,death_confirmed_dates_df.columns[j]]=0
            recovered_confirmed_dates_df.at[i,recovered_confirmed_dates_df.columns[j]]=0
        else:
            death_confirmed_dates_df.at[i,death_confirmed_dates_df.columns[j]]=deaths_global_dates_df.iat[i,j]/confirmed_global_dates_df.iat[i,j]
            recovered_confirmed_dates_df.at[i,recovered_confirmed_dates_df.columns[j]]=recovered_confirmed_dates_df.iat[i,j]/confirmed_global_dates_df.iat[i,j]

confirmed_latest = confirmed_global_dates_df[confirmed_global_dates_df.columns[(len(confirmed_global_dates_df.columns)-1)]].sum()
death_latest = deaths_global_dates_df[deaths_global_dates_df.columns[(len(deaths_global_dates_df.columns)-1)]].sum()
recovered_latest = recovered_global_dates_df[recovered_global_dates_df.columns[(len(recovered_global_dates_df.columns)-1)]].sum()
active_latest = confirmed_latest-death_latest-recovered_latest

death_to_confirmed_ratio_global = death_latest/confirmed_latest
recovered_to_confirmed_ratio_global = recovered_latest/confirmed_latest



# daily change confirmed,death,recoverd


for i in range(2,len(confirmed_global_dates_df.columns))
    daily_increase_confirmed[i-1] = confirmed_global_dates_df[i-1]-confirmed_global_dates_df[i-2]
    daily_increase_deaths[i-1] = deaths_global_dates_df[i-1]-deaths_global_dates_df[i-2]
    daily_increase_recoverd[i-1] = recovered_global_dates_df[i-1]-recovered_global_dates_df[i-2]
    
    #average
    avg_confirmed[i-1] = confirmed_global_dates_df[i-1].sum()/(i-1)
    avg_deaths[i-1] = deaths_global_dates_df[i-1].sum()/(i-1)
    avg_recoverd[i-1] = recovered_global_dates_df[i-1].sum()/(i-1)
    
    #moving average
    if(i>2):
        moving_avg_confirmed[i-1] = moving_avg_confirmed[i-1].sum()/(i-1)
        moving_avg_deaths[i-1] = moving_avg_deaths[i-1].sum()/(i-1)
        moving_avg_recoverd[i-1] = moving_avg_recoverd[i-1].sum()/(i-1)
        
    if(i==2):
        moving_avg_confirmed[i-1] = avg_confirmed[i-1]
        moving_avg_deaths[i-1] = avg_deaths[i-1]
        moving_avg_recoverd[i-1] = avg_recoverd[i-1] 