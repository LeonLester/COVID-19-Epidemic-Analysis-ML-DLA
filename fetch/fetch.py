from links import link_dict
import data
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
import datetime
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# from datetime import date, timedelta, datetime

plt.style.use('fivethirtyeight')
# %matplotlib inline // cli for now
# set_matplotlib_formats('retina')
warnings.filterwarnings("ignore")

# yesterday = date.today() - timedelta(days=1)
# most_recent_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' + yesterday.strftime('%m-%d-%Y') + '.csv'
# most_recent_global_df = pd.read_csv(latest_url)
# print(latest_global_df.head())

'''
positive_global_dates_df = positive_global_df.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])
deaths_global_dates_df = deaths_global_df.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])
recovered_global_dates_df = recovered_global_df.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

death_confirmed_dates_df = deaths_global_dates_df.copy()
recovered_confirmed_dates_df = recovered_global_dates_df.copy()

for j in range(2, len(positive_global_dates_df.columns)):
    death_confirmed_dates_df[positive_global_dates_df.columns[j]] = death_confirmed_dates_df[
        positive_global_dates_df.columns[j]].astype(float)
    recovered_confirmed_dates_df[positive_global_dates_df.columns[j]] = recovered_confirmed_dates_df[
        positive_global_dates_df.columns[j]].astype(float)
for i in range(len(positive_global_dates_df.index)):
    for j in range(2, len(positive_global_dates_df.columns)):
        if positive_global_dates_df.iat[i, j] == 0:
            death_confirmed_dates_df.at[i, death_confirmed_dates_df.columns[j]] = 0
            recovered_confirmed_dates_df.at[i, recovered_confirmed_dates_df.columns[j]] = 0
        else:
            death_confirmed_dates_df.at[i, death_confirmed_dates_df.columns[j]] = deaths_global_dates_df.iat[i, j] / \
                                                                                  positive_global_dates_df.iat[i, j]
            recovered_confirmed_dates_df.at[i, recovered_confirmed_dates_df.columns[j]] = \
                recovered_confirmed_dates_df.iat[i, j] / positive_global_dates_df.iat[i, j]

confirmed_latest = positive_global_dates_df[
    positive_global_dates_df.columns[(len(positive_global_dates_df.columns) - 1)]].sum()
death_latest = deaths_global_dates_df[
    deaths_global_dates_df.columns[(len(deaths_global_dates_df.columns) - 1)]].sum()
recovered_latest = recovered_global_dates_df[
    recovered_global_dates_df.columns[(len(recovered_global_dates_df.columns) - 1)]].sum()
active_latest = confirmed_latest - death_latest - recovered_latest

death_to_confirmed_ratio_global = death_latest / confirmed_latest
recovered_to_confirmed_ratio_global = recovered_latest / confirmed_latest
'''


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


# functions that work with dataframes
# zitoumeno 3
def getDates(df):
    dates = df.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])
    return dates


# zitoumeno 3
def getSums(sumglobal, sumUS):
    totalConfirmed = sumglobal.iloc[:, 7].sum()
    totalDeaths = sumglobal.iloc[:, 8].sum()
    totalRecovered = sumglobal.iloc[:, 9].sum()
    totalActive = sumglobal.iloc[:, 10].sum()

    totalConfirmedUS = sumUS.iloc[:, 5].sum()
    totalDeathsUS = sumUS.iloc[:, 6].sum()
    totalRecoveredUS = sumUS.iloc[:, 7].sum()
    totalActiveUS = sumUS.iloc[:, 8].sum()

    return [totalConfirmed, totalDeaths, totalRecovered, totalActive,
            totalConfirmedUS, totalDeathsUS, totalRecoveredUS, totalActiveUS]


def get_daily_increase(data):
    d = data.copy()
    cols = len(data.columns)
    for i in range(5, cols):
        d.iloc[:, i] = data.iloc[:, i] - data.iloc[:, i - 1]
    return d


def get_average(data):
    d = data.copy()
    cols = len(data.columns)
    for i in range(5, cols):
        d.iloc[:, i] = data.iloc[:, i].div(i - 3)
    return d


def get_moving_average(data, window_size):
    moving_average = data.copy()
    cols = len(data.columns)
    for i in range(10, cols):
        moving_average.iloc[:, i] = data.iloc[:, (i - window_size):i].sum(axis=1) / window_size
    return moving_average


def get_daily_total(data):
    d = data.copy()
    cols = len(data.columns)
    for i in range(5, cols):
        d.iloc[:, i] = data.iloc[:, i:]
    return d


# functions that work with lists
def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i] - data[i - 1])
    return d


def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i + window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average

def getDiscreteCountry(dataframe,country):
    countrydata = dataframe.copy()
    cols = len(dataframe.columns)
    for i in range(5, cols):
        countrydata.iloc[:, i] = dataframe.loc[dataframe['Country/Region'] == country].sum(axis=[:, i])
    return countrydata

# get the index of the country from the dataframe
def getCountryIndex(dataframe, country):
    return dataframe.index[dataframe['Country/Region'] == country].tolist()

# get the data from a list
def getCountryData(list, countryindex):
    return list[countryindex]


def plotCountryData(country):
    data = getCountryData


def (dataframe,country):


# Main

window = 7

# get the data
confirmed_global_df, deaths_global_df, recovered_global_df, confirmed_US_df, deaths_US_df, apple_mobility_df, latest_data_df, us_medical_data_df = getData()

# dataframes that show the daily increase instead of the sum up until that day

countryList = confirmed_global_df['Country/Region'].unique().tolist()

countryindices = []

for country in countryList:
    countryindices.append(getCountryIndex(confirmed_global_df, country))

# confirmed cases
daily_increase_confirmed = get_daily_increase(confirmed_global_df)
daily_increase_deaths = get_daily_increase(deaths_global_df)
daily_increase_recovered = get_daily_increase(recovered_global_df)

# average
avg_confirmed = get_average(confirmed_global_df)
avg_deaths = get_average(deaths_global_df)
avg_recovered = get_average(recovered_global_df)

# moving average
moving_avg_confirmed = get_moving_average(daily_increase_confirmed, window)
moving_avg_deaths = get_moving_average(daily_increase_deaths, window)
moving_avg_recovered = get_moving_average(daily_increase_recovered, window)

print("Test")

################ kodikas gounaridis ###############################


# get total for up to each day
latest_data_df.head()
confirmed_global_df.head()
us_medical_data_df.head()
cols = confirmed_global_df.keys()
confirmed = confirmed_global_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_global_df.loc[:, cols[4]:cols[-1]]
recoveries = recovered_global_df.loc[:, cols[4]:cols[-1]]

dates = confirmed.keys()
world_cases = []
total_deaths = []
mortality_rate = []
recovery_rate = []
total_recovered = []
total_active = []

for i in dates:
    # sums of the column of date i
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()

    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum - death_sum - recovered_sum)

    # calculate rates
    mortality_rate.append(death_sum / confirmed_sum)
    recovery_rate.append(recovered_sum / confirmed_sum)

# confirmed cases
world_daily_increase = daily_increase(world_cases)
# every day, the average of the total global confirmed cases of the last window days
world_confirmed_avg = moving_average(world_cases, window)
# every day, the average of the daily global confirmed cases of the last window days
world_daily_increase_avg = moving_average(world_daily_increase, window)

# deaths
world_daily_death = daily_increase(total_deaths)
world_death_avg = moving_average(total_deaths, window)
world_daily_death_avg = moving_average(world_daily_death, window)

# recoveries
world_daily_recovery = daily_increase(total_recovered)
world_recovery_avg = moving_average(total_recovered, window)
world_daily_recovery_avg = moving_average(world_daily_recovery, window)

# active
world_active_avg = moving_average(total_active, window)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)

# zitoumeno 3
# confirmed, dead, recovered and active for global and US
# mylist = [totalConfirmed, totalDeaths, totalRecovered, totalActive,
#         totalConfirmedUS, totalDeathsUS, totalRecoveredUS, totalActiveUS] = getSums(latest_data_df,
#                                                                                      us_medical_data_df)


#################################### predicting stuff ##########################


days_in_future = 10
future_forcast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

[X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed] = train_test_split(days_since_1_22[50:],
                                                                                              world_cases[50:],
                                                                                              test_size=0.05,
                                                                                              shuffle=False)
poly = PolynomialFeatures(degree=4)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=5)
bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)

linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:', mean_squared_error(test_linear_pred, y_test_confirmed))

# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
                 'normalize': normalize}

bayesian = BayesianRidge(fit_intercept=False)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3,
                                     return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
