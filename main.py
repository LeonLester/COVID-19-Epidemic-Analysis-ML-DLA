# file imports
from links import link_dict

# regular imports
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

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
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date, timedelta
import operator
from IPython.display import set_matplotlib_formats
import warnings

plt.style.use('fivethirtyeight')
# %matplotlib inline
# set_matplotlib_formats('retina')

warnings.filterwarnings("ignore")


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


# auxiliary functions

# zitoumeno 2
def getData():
    df1 = pd.read_csv(link_dict["positive_global"])
    df2 = pd.read_csv(link_dict["deaths_global"])
    df3 = pd.read_csv(link_dict["recovered_global"])

    df4 = pd.read_csv(link_dict["positive_US"])
    df5 = pd.read_csv(link_dict["deaths_US"])

    df6 = pd.read_csv(link_dict["apple_mobility"])

    df7 = pd.read_csv(link_dict["daily_reports_global"])
    df8 = pd.read_csv(link_dict["daily_reports_US"])

    return df1, df2, df3, df4, df5, df6, df7, df8


# get the index of the country from the dataframe
def getCountryIndex(country):
    global confirmed_global_df
    return confirmed_global_df.index[confirmed_global_df['Country/Region'] == country].tolist()


# get the data
print("11111111111")
confirmed_global_df, deaths_global_df, recovered_global_df, confirmed_US_df, deaths_US_df, apple_mobility_df, latest_data_df, us_medical_data_df = getData()
print("222222222222")
# ---------------------useless (for now)-----------------------------
# confirmed_US_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
# deaths_US_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
# apple_mobility_df = pd.read_csv('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/apple_reports/apple_mobility_report.csv')
# -------------------------------------------------------------------


confirmed_global_dates_df = confirmed_global_df.drop(columns=['Lat', 'Long'])
deaths_global_dates_df = deaths_global_df.drop(columns=['Lat', 'Long'])
recovered_global_dates_df = recovered_global_df.drop(columns=['Lat', 'Long'])

# default country
selected = 'Greece'

# get the unique countries in a list
countries = confirmed_global_df['Country/Region'].unique().tolist()


# -------------------------------------GUI STUFF--------------------------------------v

# GUI auxiliary functions
def on_closing():
    app.destroy()
    exit()


def callbackFunc():
    global selected
    selected = comboExample.get()
    app.destroy()


def GlobalcallbackFunc():
    global selected
    selected = comboExample.get()
    app.destroy()


app = tk.Tk()
app.resizable(False, False)
windowWidth = app.winfo_reqwidth()
windowHeight = app.winfo_reqheight()
positionRight = int(app.winfo_screenwidth() / 2 - windowWidth / 2)
positionDown = int(app.winfo_screenheight() / 2 - windowHeight / 2)

app.geometry("+{}+{}".format(positionRight, positionDown))
app.title("  SARS-CoV-2")

helv36 = tkFont.Font(family="Helvetica", size=36, weight="bold")

labelTop = tk.Label(app, text="Choose Country", font='helv36')
labelTop.grid(column=0, row=0)

comboExample = ttk.Combobox(app, values=countries, font='helv36', state="readonly")
comboExample.grid(column=0, row=1)
comboExample.current(1)

resultButton = tk.Button(app, text='Get Result', command=callbackFunc, font='helv36')

resultButton.grid(column=0, row=2, pady=10, padx=80, sticky=tk.W)

globalButton = tk.Button(app, text='Get Global Results', command=GlobalcallbackFunc, font='helv36')
checkbox = tk.Checkbutton(app, text="Custom Date")
checkbox.grid(column=0, row=4, pady=10, padx=80)
globalButton.grid(column=0, row=3, pady=10, padx=50, sticky=tk.W)
app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()

# -------------------------------------GUI STUFF--------------------------------------^


sel_confirmed_df = confirmed_global_dates_df[confirmed_global_dates_df["Country/Region"] == selected]
sel_deaths_df = deaths_global_dates_df[deaths_global_dates_df["Country/Region"] == selected]
sel_recovered_df = recovered_global_dates_df[recovered_global_dates_df["Country/Region"] == selected]

death_confirmed_dates_df = deaths_global_dates_df.copy()
recovered_confirmed_dates_df = recovered_global_dates_df.copy()

# make to float
for j in range(2, len(confirmed_global_dates_df.columns)):
    death_confirmed_dates_df[confirmed_global_dates_df.columns[j]] = death_confirmed_dates_df[
        confirmed_global_dates_df.columns[j]].astype(float)

    recovered_confirmed_dates_df[confirmed_global_dates_df.columns[j]] = recovered_confirmed_dates_df[
        confirmed_global_dates_df.columns[j]].astype(float)

for i in range(len(confirmed_global_dates_df.index)):

    for j in range(2, len(confirmed_global_dates_df.columns)):
        if confirmed_global_dates_df.iat[i, j] == 0:
            death_confirmed_dates_df.at[i, death_confirmed_dates_df.columns[j]] = 0
            recovered_confirmed_dates_df.at[i, recovered_confirmed_dates_df.columns[j]] = 0
        else:
            death_confirmed_dates_df.at[i, death_confirmed_dates_df.columns[j]] = deaths_global_dates_df.iat[i, j] / \
                                                                                  confirmed_global_dates_df.iat[i, j]
            recovered_confirmed_dates_df.at[i, recovered_confirmed_dates_df.columns[j]] = \
                recovered_confirmed_dates_df.iat[i, j] / confirmed_global_dates_df.iat[i, j]

# get the data for global variables

# get total for up to each day
latest_data_df.head()
confirmed_global_df.head()
us_medical_data_df.head()
cols = confirmed_global_df.keys()

confirmed = confirmed_global_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_global_df.loc[:, cols[4]:cols[-1]]
recoveries = recovered_global_df.loc[:, cols[4]:cols[-1]]

dates = confirmed.keys()


# code for global ---------------------------------------------V
def getGlobalStatistics():
    global confirmed
    global deaths
    global recoveries
    global dates

    confirmed_per_day = []
    deaths_per_day = []
    mortality_rate = []
    recovery_rate = []
    recovered_per_day = []
    total_active = []

    for i in dates:
        # sums of the column of date i
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()

        # confirmed, deaths, recovered, and active
        confirmed_per_day.append(confirmed_sum)
        deaths_per_day.append(death_sum)
        recovered_per_day.append(recovered_sum)
        total_active.append(confirmed_sum - death_sum - recovered_sum)

        # calculate rates
        mortality_rate.append(death_sum / confirmed_sum)
        recovery_rate.append(recovered_sum / confirmed_sum)

    return confirmed_per_day, deaths_per_day, recovered_per_day, mortality_rate, recovery_rate, total_active


def getCountryStatistics(country):
    global confirmed
    global deaths
    global recoveries
    global dates

    indexList = getCountryIndex(country)

    country_confirmed = []
    country_deaths = []
    country_mortality_rate = []
    country_recovery_rate = []
    country_recovered = []
    country_active = []
    confirmedSum, deathSum, recoveredSum = 0, 0, 0

    for region in indexList:
        for i in dates:
            # sums of the column of date i
            confirmedSum += confirmed.iloc[region][i]
            deathSum = deaths.iloc[region][i]
            recoveredSum = recoveries.iloc[region][i]

            # confirmed, deaths, recovered, and active
            country_confirmed.append(confirmedSum)
            country_deaths.append(deathSum)
            country_recovered.append(recoveredSum)
            country_active.append(confirmedSum - deathSum - recoveredSum)

            # calculate rates
            country_mortality_rate.append(deathSum / confirmedSum)
            country_recovery_rate.append(recoveredSum / confirmedSum)

    return country_confirmed, country_deaths, country_recovered, country_mortality_rate, country_recovery_rate, country_active


# -------------------------------------Global--------------------------------------v
confirmed_per_day, deaths_per_day, recovered_per_day, global_mortality_rate, global_recovery_rate, active_per_day = getGlobalStatistics()
# ---------------------------------------------------------------------------------^

# -------------------------------------Selected------------------------------------v
sel_confirmed_per_day, sel_deaths_per_day, sel_recovered_per_day, country_mortality_rate, country_recovery_rate, country_active = getCountryStatistics(selected)
# ---------------------------------------------------------------------------------^

# -------those are kinda useless, will remove later----------------------------------v
days = [0 for y in range(len(confirmed_global_dates_df.columns) - 2)]
x = [[0 for i in range(2)] for y in range(len(confirmed_global_dates_df.columns) - 3)]
y = [0 for y in range(len(confirmed_global_dates_df.columns) - 3)]
# -----------------------------------------------------------------------------------^

for i in range(len(confirmed_global_dates_df.columns) - 2):
    days[i] = i

# get the sums for global and for selected country
for i in range(len(confirmed_global_dates_df.columns) - 2):  # well the dates are the same, so...
    confirmed_per_day[i] = confirmed_global_dates_df[confirmed_global_dates_df.columns[i + 2]].sum()
    deaths_per_day[i] = deaths_global_dates_df[deaths_global_dates_df.columns[i + 2]].sum()
    recovered_per_day[i] = recovered_global_dates_df[recovered_global_dates_df.columns[i + 2]].sum()
    sel_confirmed_per_day[i] = sel_confirmed_df[sel_confirmed_df.columns[i + 2]].sum()
    sel_deaths_per_day[i] = sel_deaths_df[sel_deaths_df.columns[i + 2]].sum()
    sel_recovered_per_day[i] = sel_recovered_df[sel_recovered_df.columns[i + 2]].sum()

for i in range(len(confirmed_global_dates_df.columns) - 3):
    x[i][0] = i
    x[i][1] = confirmed_global_dates_df[confirmed_global_dates_df.columns[i + 3]].sum()

for i in range(1, len(confirmed_global_dates_df.columns) - 2):
    y[i - 1] = ((confirmed_per_day[i] - confirmed_per_day[i - 1]) / confirmed_per_day[i - 1])

data_days = len(confirmed_per_day)
prediction_days = 15

# -----------------------------------------------------------------------------------v
data_days_plus_prediction_days = [[0 for i in range(1)] for y in range(len(confirmed_per_day) + prediction_days)]

yyy = data_days_plus_prediction_days[:len(confirmed_per_day)][:]  # could be better

xx = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]
xx_deaths = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]
xx_recovered = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]

xx_sel = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]
xx_sel_deaths = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]
xx_sel_recovered = [[0 for i in range(1)] for y in range(len(confirmed_per_day))]
# -----------------------------------------------------------------------------------^

for i in range(data_days):
    xx[i][0] = confirmed_per_day[i]
    xx_deaths[i][0] = deaths_per_day[i]
    xx_recovered[i][0] = recovered_per_day[i]

for i in range(data_days):
    xx_sel[i][0] = sel_confirmed_per_day[i]
    xx_sel_deaths[i][0] = sel_deaths_per_day[i]
    xx_sel_recovered[i][0] = sel_recovered_per_day[i]

for i in range(data_days + prediction_days):
    data_days_plus_prediction_days[i][0] = i

# --------------------Training set splittng (for later use)---------------------------v
x_train = confirmed_per_day[0:round(len(confirmed_per_day) * 0.8)]
x_test = confirmed_per_day[round(len(confirmed_per_day) * 0.8):len(confirmed_per_day)]
y_train = days[0:round(len(days) * 0.8)]
y_test = days[round(len(days) * 0.8):len(days)]
# ------------------------------------------------------------------------------------^


# ---------------------------------------------------------------------------------------------------------------------------------v
# Some global stats
confirmed_latest = confirmed_global_dates_df[
    confirmed_global_dates_df.columns[(len(confirmed_global_dates_df.columns) - 1)]].sum()
death_latest = deaths_global_dates_df[deaths_global_dates_df.columns[(len(deaths_global_dates_df.columns) - 1)]].sum()
recovered_latest = recovered_global_dates_df[
    recovered_global_dates_df.columns[(len(recovered_global_dates_df.columns) - 1)]].sum()

active_latest = confirmed_latest - death_latest - recovered_latest
death_to_confirmed_ratio_global = death_latest / confirmed_latest
recovered_to_confirmed_ratio_global = recovered_latest / confirmed_latest
# ---------------------------------------------------------------------------------------------------------------------------------^

# ----------------------------------------------------------------------------v
# Scales values appropriately (float64)
y_scaler = StandardScaler()
# Global
x_scaler = StandardScaler()
x_deaths_scaler = StandardScaler()
x_recovered_scaler = StandardScaler()
# Selected
x_sel_scaler = StandardScaler()
x_sel_deaths_scaler = StandardScaler()
x_sel_recovered_scaler = StandardScaler()
# ----------------------------------------------------------------------------^


# ----------------------------------------------------------------------------v
Y = y_scaler.fit_transform(data_days_plus_prediction_days)  # with prediction days
YY = Y[:len(confirmed_per_day)][:]  # subset of Y (total-prediction)
# Global
X = x_scaler.fit_transform(xx)
X_deaths = x_deaths_scaler.fit_transform(xx_deaths)
X_recovered = x_recovered_scaler.fit_transform(xx_recovered)
# Selected
X_sel = x_sel_scaler.fit_transform(xx_sel)
X_sel_deaths = x_sel_deaths_scaler.fit_transform(xx_sel_deaths)
X_sel_recovered = x_sel_recovered_scaler.fit_transform(xx_sel_recovered)
# ----------------------------------------------------------------------------^


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------v
# SVM Regression
c = [0.0001, 0.001, 0.01, 0.1, 1]  # added more values for experimentation
gamma = [0.0001, 0.001, 0.01, 0.1, 1]  # >>
epsilon = [0.0001, 0.001, 0.01, 0.1, 1]  # >>
shrinking = [True, False]
svm_grid = {'C': c, 'gamma': gamma, 'epsilon': epsilon, 'shrinking': shrinking}
svm = SVR(
    kernel='rbf')  # poly kernel didnt really work out, but you're free to try. just replace rbf with poly and add degree=3

regr = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1,
                          n_iter=30, verbose=1)
regr_deaths = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                 n_jobs=-1, n_iter=30, verbose=0)
regr_recovered = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                    n_jobs=-1, n_iter=30, verbose=0)

regr_sel = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1,
                              n_iter=30, verbose=1)
regr_sel_deaths = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                     n_jobs=-1, n_iter=30, verbose=1)
regr_sel_recovered = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                        n_jobs=-1, n_iter=30, verbose=1)

regr.fit(YY, X.ravel())
regr_deaths.fit(YY, X_deaths.ravel())
regr_recovered.fit(YY, X_recovered.ravel())

regr_sel.fit(YY, X_sel.ravel())
regr_sel_deaths.fit(YY, X_sel_deaths.ravel())
regr_sel_recovered.fit(YY, X_sel_recovered.ravel())

# print(regr.best_params_) #nice to know (what worked for me {'shrinking': True, 'gamma': 1, 'epsilon': 0.001, 'C': 1} (for confirmed))

predictions = x_scaler.inverse_transform(regr.predict(Y))
predictions_deaths = x_deaths_scaler.inverse_transform(regr_deaths.predict(Y))
predictions_recovered = x_recovered_scaler.inverse_transform(regr_recovered.predict(Y))

predictions_sel = x_sel_scaler.inverse_transform(regr_sel.predict(Y))
predictions_sel_deaths = x_sel_deaths_scaler.inverse_transform(regr_sel_deaths.predict(Y))
predictions_sel_recovered = x_sel_recovered_scaler.inverse_transform(regr_sel_recovered.predict(Y))

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------^


# ------------------------------------------------------------------------------------------------------------------------------------------------------v
# ------------------------------------------------------------------------------------------------------------------------------------------------------v
# ------------------------------------------------------------------------------------------------------------------------------------------------------v

#                                                                  Bayesian Ridge Test

bayes_setup = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
alpha_1 = [0.000000001, 0.0000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
           100000, 1000000, 1000000]
alpha_2 = [0.000000001, 0.0000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
           100000, 1000000, 1000000]
lambda_1 = [0.000000001, 0.0000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
            100000, 1000000, 1000000]
lambda_2 = [0.000000001, 0.0000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
            100000, 1000000, 1000000]
tol = [0.000000000001, 0.00000000001, 0.0000000001, 0.000000001, 0.0000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001,
       0.01, 0.1, 1]
# too many values?
fit_intercept = [True, False]
bayes_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
              'fit_intercept': fit_intercept}
bayes = RandomizedSearchCV(bayes_setup, bayes_grid, scoring='neg_mean_squared_error', cv=None, return_train_score=True,
                           n_jobs=-1, n_iter=1000, verbose=1)
bayes.fit(YY, X.ravel())
# print(bayes.best_params_)
degree = 3
bayes_poly = BayesianRidge()  # not really poly..
bayes_poly.fit(YY, X.ravel())

# predictions_bayesian = x_scaler.inverse_transform(bayes.predict(Y))
# predictions_bayesian_poly = x_scaler.inverse_transform(bayes_poly.predict(Y))
# ------------------------------------------------------------------------------------------------------------------------------------------------------^
# ------------------------------------------------------------------------------------------------------------------------------------------------------^
# ------------------------------------------------------------------------------------------------------------------------------------------------------^


# ---------------------------------------------------------------------------------------------------------V
# Selected
linear_reg_sel = LinearRegression().fit(YY, X_sel.ravel())
linear_reg_deaths_sel = LinearRegression().fit(YY, X_sel_deaths.ravel())
linear_reg_recovered_sel = LinearRegression().fit(YY, X_sel_recovered.ravel())

poly_reg_sel = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_reg_deaths_sel = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_reg_recovered_sel = make_pipeline(PolynomialFeatures(3), LinearRegression())

poly_reg_sel.fit(YY, X_sel.ravel())
poly_reg_deaths_sel.fit(YY, X_sel_deaths.ravel())
poly_reg_recovered_sel.fit(YY, X_sel_recovered.ravel())

linear_reg_sel.score(YY, X_sel.ravel())
linear_reg_deaths_sel.score(YY, X_sel_deaths.ravel())
linear_reg_recovered_sel.score(YY, X_sel_recovered.ravel())

predictions_linear_sel = x_scaler.inverse_transform(linear_reg_sel.predict(Y))
predictions_linear_deaths_sel = x_deaths_scaler.inverse_transform(linear_reg_deaths_sel.predict(Y))
predictions_linear_recovered_sel = x_recovered_scaler.inverse_transform(linear_reg_recovered_sel.predict(Y))

predictions_poly_sel = x_sel_scaler.inverse_transform(poly_reg_sel.predict(Y))
predictions_poly_deaths_sel = x_sel_deaths_scaler.inverse_transform(poly_reg_deaths_sel.predict(Y))
predictions_pol_recovered_sel = x_sel_recovered_scaler.inverse_transform(poly_reg_recovered_sel.predict(Y))

# ---------------------------------------------------------------------------------------------------------^

# ---------------------------------------------------------------------------------------------------------V
# Global
linear_reg = LinearRegression().fit(YY, X.ravel())
linear_reg_deaths = LinearRegression().fit(YY, X_deaths.ravel())
linear_reg_recovered = LinearRegression().fit(YY, X_recovered.ravel())

poly_reg = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_reg_deaths = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_reg_recovered = make_pipeline(PolynomialFeatures(3), LinearRegression())

poly_reg.fit(YY, X.ravel())
poly_reg_deaths.fit(YY, X_deaths.ravel())
poly_reg_recovered.fit(YY, X_recovered.ravel())

linear_reg.score(YY, X.ravel())
linear_reg_deaths.score(YY, X_deaths.ravel())
linear_reg_recovered.score(YY, X_recovered.ravel())

predictions_linear = x_scaler.inverse_transform(linear_reg.predict(Y))
predictions_linear_deaths = x_deaths_scaler.inverse_transform(linear_reg_deaths.predict(Y))
predictions_linear_recovered = x_recovered_scaler.inverse_transform(linear_reg_recovered.predict(Y))

predictions_poly = x_scaler.inverse_transform(poly_reg.predict(Y))
predictions_poly_deaths = x_deaths_scaler.inverse_transform(poly_reg_deaths.predict(Y))
predictions_pol_recovered = x_recovered_scaler.inverse_transform(poly_reg_recovered.predict(Y))

# ---------------------------------------------------------------------------------------------------------^


# ------------------------------------------------------------------------------------------v
fig, axes = plt.subplots(4)
fig.suptitle('SVR vs Linear Regr. vs Poly. Regr. vs Bayesian Ridge')

axes[0].scatter(yyy, xx, color='red')
axes[0].plot(data_days_plus_prediction_days, predictions, color='blue')
axes[1].scatter(yyy, xx, color='red')
axes[1].plot(data_days_plus_prediction_days, predictions_linear, color='blue')
axes[2].scatter(yyy, xx, color='red')
axes[2].plot(data_days_plus_prediction_days, predictions_poly, color='blue')
axes[3].scatter(yyy, xx_sel, color='red')
axes[3].plot(data_days_plus_prediction_days, predictions_sel, color='blue')
plt.show()
# ------------------------------------------------------------------------------------------^
