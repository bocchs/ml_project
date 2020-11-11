import numpy as np
import pandas as pd
import sqlite3
import sys

# download dataset here:
# https://www.kaggle.com/rtatman/188-million-us-wildfires

if __name__ == "__main__":
    db_file = 'FPA_FOD_20170508.sqlite'
    conn = sqlite3.connect(db_file) # create a database connection to the SQLite database specified by the db_file
    # select features
    # df = pd.read_sql_query("SELECT FIRE_YEAR, STAT_CAUSE_CODE, STAT_CAUSE_DESCR, LATITUDE, LONGITUDE, STATE, DISCOVERY_DATE, FIRE_SIZE FROM 'Fires'", conn)
    # df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE FROM 'Fires'", conn)
    df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, FIRE_SIZE FROM 'Fires'", conn)
    # print(df.head()) #check the data
    causes = df.STAT_CAUSE_CODE.unique()
    # print(causes)
    # print(len(causes))

    num_features = df.shape[1] - 1

    # collect features and label from entire dataset
    X = np.zeros((0,num_features))
    y = np.zeros(0)
    for label in causes:
        label_data = df.loc[df['STAT_CAUSE_CODE']==label] # collect all examples of a specific label

        lab = label_data['STAT_CAUSE_CODE'].to_numpy() # extract labels
        y = np.concatenate((y,lab))

        feature_data = label_data.drop(columns=['STAT_CAUSE_CODE']).to_numpy() # extract features
        X = np.vstack((X, feature_data))

