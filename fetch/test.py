import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date, timedelta
import operator
import warnings
from IPython.display import set_matplotlib_formats
plt.style.use('fivethirtyeight')
# %matplotlib inline


#set_matplotlib_formats('retina')
#warnings.filterwarnings("ignore")δ

# This version doesn't utilize num_crunch.py
# It's more of an "all in one" pack
#
# -----------To Do----------------
# >Improve naming scheme
# >Enable parameter tuning
# >Add a more interactive "UI"
# >Integrate with num_crunch.py
# >...
# --------------------------------


confirmed_global_df = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_global_df = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_global_df = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

# ---------------------useless (for now)-----------------------------
# confirmed_US_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
# deaths_US_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
# apple_mobility_df = pd.read_csv('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/apple_reports/apple_mobility_report.csv')
# -------------------------------------------------------------------

confirmed_global_dates_df = confirmed_global_df.drop(columns=['Lat', 'Long'])
deaths_global_dates_df = deaths_global_df.drop(columns=['Lat', 'Long'])
recovered_global_dates_df = recovered_global_df.drop(columns=['Lat', 'Long'])

death_confirmed_dates_df = deaths_global_dates_df.copy()
recovered_confirmed_dates_df = recovered_global_dates_df.copy()

for j in range(2, len(confirmed_global_dates_df.columns)):
    death_confirmed_dates_df[confirmed_global_dates_df.columns[j]] = death_confirmed_dates_df[
        confirmed_global_dates_df.columns[j]].astype(float)
    recovered_confirmed_dates_df[confirmed_global_dates_df.columns[j]] = recovered_confirmed_dates_df[
        confirmed_global_dates_df.columns[j]].astype(float)
for i in range(len(confirmed_global_dates_df.index)):
    for j in range(2, len(confirmed_global_dates_df.columns)):
        if (confirmed_global_dates_df.iat[i, j] == 0):
            death_confirmed_dates_df.at[i, death_confirmed_dates_df.columns[j]] = 0
            recovered_confirmed_dates_df.at[i, recovered_confirmed_dates_df.columns[j]] = 0
        else:
            death_confirmed_dates_df.at[i, death_confirmed_dates_df.columns[j]] = deaths_global_dates_df.iat[i, j] / \
                                                                                  confirmed_global_dates_df.iat[i, j]
            recovered_confirmed_dates_df.at[i, recovered_confirmed_dates_df.columns[j]] = \
                recovered_confirmed_dates_df.iat[i, j] / confirmed_global_dates_df.iat[i, j]

confirmed_per_day = [0 for y in range(len(confirmed_global_dates_df.columns) - 2)]
deaths_per_day = [0 for y in range(len(deaths_global_dates_df.columns) - 2)]
recovered_per_day = [0 for y in range(len(recovered_global_dates_df.columns) - 2)]

# -------those are kinda useless, will remove later-------------------v
days = [0 for y in range(len(confirmed_global_dates_df.columns) - 2)]
x = [[0 for i in range(2)] for y in range(len(confirmed_global_dates_df.columns) - 3)]
y = [0 for y in range(len(confirmed_global_dates_df.columns) - 3)]
# --------------------------------------------------------------------^
for i in range(len(confirmed_global_dates_df.columns) - 2):
    days[i] = i

for i in range(len(confirmed_global_dates_df.columns) - 2):  # well the dates are the same, so...
    confirmed_per_day[i] = confirmed_global_dates_df[confirmed_global_dates_df.columns[i + 2]].sum()
    deaths_per_day[i] = deaths_global_dates_df[deaths_global_dates_df.columns[i + 2]].sum()
    recovered_per_day[i] = recovered_global_dates_df[recovered_global_dates_df.columns[i + 2]].sum()

for i in range(len(confirmed_global_dates_df.columns) - 3):
    x[i][0] = i
    x[i][1] = confirmed_global_dates_df[confirmed_global_dates_df.columns[i + 3]].sum()

for i in range(1, len(confirmed_global_dates_df.columns) - 2):
    y[i - 1] = ((confirmed_per_day[i] - confirmed_per_day[i - 1]) / confirmed_per_day[i - 1])

prediction_days = 15

xx = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]
yy = [[0 for i in range(1)] for y in range(len(confirmed_per_day) + prediction_days)]

xx_deaths = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]
xx_recovered = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]

for i in range(len(confirmed_per_day)):
    xx[i][0] = confirmed_per_day[i]
    xx_deaths[i][0] = deaths_per_day[i]
    xx_recovered[i][0] = recovered_per_day[i]

for i in range(len(confirmed_per_day) + prediction_days):
    yy[i][0] = i

# --------------------for later use---------------------------------------------------v
x_train = confirmed_per_day[0:round(len(confirmed_per_day) * 0.8)]
x_test = confirmed_per_day[round(len(confirmed_per_day) * 0.8):len(confirmed_per_day)]
y_train = days[0:round(len(days) * 0.8)]
y_test = days[round(len(days) * 0.8):len(days)]
# ------------------------------------------------------------------------------------^
# -----------------------------------------------------------------------------------------------v
confirmed_latest = confirmed_global_dates_df[
    confirmed_global_dates_df.columns[(len(confirmed_global_dates_df.columns) - 1)]].sum()
death_latest = deaths_global_dates_df[deaths_global_dates_df.columns[(len(deaths_global_dates_df.columns) - 1)]].sum()
recovered_latest = recovered_global_dates_df[
    recovered_global_dates_df.columns[(len(recovered_global_dates_df.columns) - 1)]].sum()
active_latest = confirmed_latest - death_latest - recovered_latest
death_to_confirmed_ratio_global = death_latest / confirmed_latest
recovered_to_confirmed_ratio_global = recovered_latest / confirmed_latest
# -----------------------------------------------------------------------------------------------^

# poly = polynomial degree 4
#
x_scaler = StandardScaler()  # scales values appropriately (float64)
y_scaler = StandardScaler()  # >>
x_deaths_scaler = StandardScaler()  # >>
x_recovered_scaler = StandardScaler()  # >>

X = x_scaler.fit_transform(xx)
X_deaths = x_deaths_scaler.fit_transform(xx_deaths)
X_recovered = x_recovered_scaler.fit_transform(xx_recovered)
Y = y_scaler.fit_transform(yy)

YY = Y[:len(confirmed_per_day)][:]  # subset of Y (total-prediction)

c = [0.0001, 0.001, 0.01, 0.1, 1]  # added more values for experimentation
gamma = [0.0001, 0.001, 0.01, 0.1, 1]  # >>
epsilon = [0.0001, 0.001, 0.01, 0.1, 1]  # >>
shrinking = [True, False]
svm_grid = {'C': c, 'gamma': gamma, 'epsilon': epsilon, 'shrinking': shrinking}
svm = SVR(
    kernel='rbf')  # poly kernel didnt really work out, but you're free to try. just replace rbf with poly and add degree=3
regr = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1,
                          n_iter=30, verbose=0)
regr_deaths = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                 n_jobs=-1, n_iter=30, verbose=0)
regr_recovered = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                    n_jobs=-1, n_iter=30, verbose=0)
regr.fit(YY, X.ravel())
regr_deaths.fit(YY, X_deaths.ravel())
regr_recovered.fit(YY, X_recovered.ravel())
print(
    regr.best_params_)  # nice to know (what worked for me {'shrinking': True, 'gamma': 1, 'epsilon': 0.001, 'C': 1} (for confirmed))

yyy = yy[:len(confirmed_per_day)][:]  # i mean, it works

predictions = x_scaler.inverse_transform(regr.predict(Y))
predictions_deaths = x_deaths_scaler.inverse_transform(regr_deaths.predict(Y))
predictions_recovered = x_recovered_scaler.inverse_transform(regr_recovered.predict(Y))

fig, axs = plt.subplots(3)
fig.suptitle('Covid-19 Confirmed, Deaths & Recovered Cases')
axs[0].scatter(yyy, xx, color='red')
axs[0].plot(yy, predictions, color='blue')
axs[1].scatter(yyy, xx_deaths, color='red')
axs[1].plot(yy, predictions_deaths, color='blue')
axs[2].scatter(yyy, xx_recovered, color='red')
axs[2].plot(yy, predictions_recovered, color='blue')

axs[0].set_ylabel('# of cases')
axs[0].set_title('Confirmed cases (global)')

axs[1].set_ylabel('# of deaths')
axs[1].set_title('Deaths (global)')

axs[2].set_xlabel('# of days')
axs[2].set_ylabel('# of cases')
axs[2].set_title('Recovered cases (global)')
plt.show()
