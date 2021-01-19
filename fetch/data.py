from links import link_dict
import pandas as pd
from datetime import date, datetime
import requests
import urllib

'''
def getFiles(link_dict):
    file1 = link_dict['positive_global']
    file2 = link_dict['deaths_global']
    file3 = link_dict['recovered_global']
    file4 = link_dict['positive_US']
    file5 = link_dict['deaths_US']
    file6 = link_dict['apple_mobility']

    localDestination = "/csvfiles/"
    resultFilePath, responseHeaders = urllib.urlretrieve(file1, localDestination)


def updateDates(lastUpdatedCols): # NOT READY
    # get the new dataframes
    # get the first and check if it's the same as lastUpdated
    positive_global_df = pd.read_csv(link_dict["positive_global"])
    #             # of columns in new dataframes - # of columns in last update
    newColumns = len(positive_global_df.columns) - lastUpdatedCols
    if newColumns == 0: print("Data is up to date")
    if newColumns > 0:  # if not, get the rest

        deaths_global_df = pd.read_csv(link_dict["deaths_global"])
        recovered_global_df = pd.read_csv(link_dict["recovered_global"])
        confirmed_US_df = pd.read_csv(link_dict["positive_US"])
        deaths_US_df = pd.read_csv(link_dict["deaths_US"])
        apple_mobility_df = pd.read_csv(link_dict["apple_mobility"])

        # keep only the new columns
        update_pos_global = positive_global_df.iloc[:, -newColumns:]
        update_deaths_global = deaths_global_df.iloc[:, -newColumns:]
        update_recovered_global = recovered_global_df.iloc[:, -newColumns:]
        update_confirmed_US = confirmed_US_df.iloc[:, -newColumns:]
        update_deaths_US = deaths_US_df.iloc[:, -newColumns:]
        apple_mobility_df = apple_mobility_df.iloc[:, -newColumns:]

        # return the dataframes that contain only the new data to append to the main ones
        return [update_pos_global, update_deaths_global, update_recovered_global, update_confirmed_US,
                update_deaths_US, apple_mobility_df]
    '''
