from links import link_dict
import update_df
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta, datetime

plt.style.use('fivethirtyeight')
# %matplotlib inline // cli for now
# set_matplotlib_formats('retina')
warnings.filterwarnings("ignore")

# last added column minus today's column in a list of dates to append


# getting the links from a different file for readability
positive_global_df = pd.read_csv(link_dict["positive_global"])
deaths_global_df = pd.read_csv(link_dict["deaths_global"])
recovered_global_df = pd.read_csv(link_dict["recovered_global"])

confirmed_US_df = pd.read_csv(link_dict["positive_US"])
deaths_US_df = pd.read_csv(link_dict["deaths_US"])

apple_mobility_df = pd.read_csv(link_dict["apple_mobility"])

'''
positive_global_dates_df = positive_global_df.drop(columns=['12/26/20'])
temp_positive_global = pd.read_csv(link_dict["positive_global"])
sdate = datetime.strptime(positive_global_dates_df.columns[-1], '%m/%d/%y')
edate = datetime.strptime(temp_positive_global.columns[-1], '%m/%d/%y')

    delta = edate - sdate  # as timedelta

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        print(day)

def update_df():
    temp_positive_global = pd.read_csv(link_dict["positive_global"])
    temp_deaths_global = pd.read_csv(link_dict["deaths_global"])
    temp_recovered_global = pd.read_csv(link_dict["recovered_global"])
    temp_confirmed_US = pd.read_csv(link_dict["positive_US"])
    temp_deaths_US = pd.read_csv(link_dict["deaths_US"])
    temp_apple_mobility = pd.read_csv(link_dict["apple_mobility"])

    temp = [temp_positive_global, temp_deaths_global, temp_recovered_global, temp_confirmed_US, temp_deaths_US,
            temp_apple_mobility]


    sdate = datetime.strptime(temp_positive_global.columns[-1], '%m/%d/%y')
    edate = datetime.strptime(temp_positive_global.columns[-1], '%m/%d/%y')

    delta = edate - sdate  # as timedelta

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        print(day)


def days_between(d1, d2):  # get the difference of two dates as an integer
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


# yesterday = date.today() - timedelta(days=1)
# most_recent_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' + yesterday.strftime('%m-%d-%Y') + '.csv'
# most_recent_global_df = pd.read_csv(latest_url)
# print(latest_global_df.head())

def main():
    # initial_fetch()
    # update_df()

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

# dataframes that show the daily increase instead of the sum up until that day


'''

# daily change confirmed,death,recoverd
daily_increase_confirmed = positive_global_df.copy()
daily_increase_deaths = deaths_global_df.copy()
daily_increase_recovered = recovered_global_df.copy()

avg_confirmed = positive_global_df.copy()
avg_deaths = deaths_global_df.copy()
avg_recovered = recovered_global_df.copy()

moving_avg_confirmed = positive_global_df.copy()
moving_avg_deaths = deaths_global_df.copy()
moving_avg_recovered = recovered_global_df.copy()

# initialization to fit moving average calculation indexes
for i in range(5, len(positive_global_df.columns)):
    daily_increase_confirmed.iloc[:, i] = positive_global_df.iloc[:, i] - positive_global_df.iloc[:, i - 1]
    daily_increase_deaths.iloc[:, i] = deaths_global_df.iloc[:, i] - deaths_global_df.iloc[:, i - 1]
    daily_increase_recovered.iloc[:, i] = recovered_global_df.iloc[:, i] - recovered_global_df.iloc[:, i - 1]

    # average
    avg_confirmed.iloc[:, i] = positive_global_df.iloc[:, i].div(i - 3)
    avg_deaths.iloc[:, i] = deaths_global_df.iloc[:, i].div(i - 3)
    avg_recovered.iloc[:, i] = recovered_global_df.iloc[:, i].div(i - 3)

    ################### moving average ########################
    '''
    if i > 5:  # fix the moving average indexes
        moving_avg_confirmed.iloc[:, i] = moving_avg_confirmed.iloc[:, i - 1].sum() / (i - 1)
        moving_avg_deaths.iloc[:, i] = moving_avg_deaths.iloc[:, i - 1].sum() / (i - 1)
        moving_avg_recovered.iloc[:, i] = moving_avg_recovered.iloc[:, i - 1].sum() / (i - 1)

    elif i == 5:
        moving_avg_confirmed.iloc[:, i-1] = avg_confirmed.iloc[:, i-1]
        moving_avg_deaths.iloc[:, i-1] = avg_deaths.iloc[:, i-1]
        moving_avg_recovered.iloc[:, i-1] = avg_recovered.iloc[:, i-1]    
    '''


