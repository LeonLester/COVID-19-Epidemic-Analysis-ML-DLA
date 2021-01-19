

def update_df():
    temp_positive_global = pd.read_csv(link_dict["positive_global"])
    temp_deaths_global = pd.read_csv(link_dict["deaths_global"])
    temp_recovered_global = pd.read_csv(link_dict["recovered_global"])

    temp_confirmed_US = pd.read_csv(link_dict["positive_US"])
    temp_deaths_US = pd.read_csv(link_dict["deaths_US"])

    temp_apple_mobility = pd.read_csv(link_dict["apple_mobility"])

    sdate = datetime.strptime(positive_global_df.columns[-1], '%m/%d/%y')
    edate = datetime.strptime(temp_positive_global.columns[-1], '%m/%d/%y')

    delta = edate - sdate  # as timedelta

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        print(day)


def days_between(d1, d2):  # get the difference of two dates as an integer
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)
