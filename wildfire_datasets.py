import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import sys

"""
download wildfire dataset here:
https://www.kaggle.com/rtatman/188-million-us-wildfires

download weather dataset here (and store in folder named "weather_data"):
https://www.kaggle.com/selfishgene/historical-hourly-weather-data


STAT_CAUSE_CODE to description:
    1: Lightning
    2: Equipment Use
    3: Smoking
    4: Campfire
    5: Debris burning
    6: Railroad
    7: Arson
    8: Children
    9: Miscellaneous
    10: Fireworks
    11: Powerline
    12: Structure
    13: Missing/undefined

"""

# read wildfire dataset (no weather data)
# if y_label=='STAT_CAUSE_CODE' then predicting cause of fire
# else predicting fire size
def get_wildfire_dataset(y_label='STAT_CAUSE_CODE'):
    db_file = 'FPA_FOD_20170508.sqlite'
    conn = sqlite3.connect(db_file) # create a database connection to the SQLite database specified by the db_file
    # select features
    df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, FIRE_SIZE, STATE FROM 'Fires'", conn)
    df = df.dropna() # remove rows with missing entries
    df = df[df['STAT_CAUSE_CODE'] != 13] # remove missing/undefined causes
    df = df[df['STATE'] == 'CA'] # data only in California so that dataset is not too large to process

    causes = np.sort(df.STAT_CAUSE_CODE.unique())
    states = df.STATE.unique()
    df = df.drop('STATE', 1) # remove state from features

    num_features = df.shape[1] - 1

    # collect features and label from entire dataset
    X = np.zeros((0,num_features))
    y = np.zeros(0)
    # print(df)
    if y_label == 'STAT_CAUSE_CODE': # predicting the cause of the fire
        cause_to_label = {} # map causes 1...13 to labels 0...num_causes-1
        df['STAT_CAUSE_CODE'] = df['STAT_CAUSE_CODE'].astype('int32')
        # extract features and labels
        for i, cause in enumerate(causes):
            cause_to_label[cause] = i
            label_data = df.loc[df[y_label]==cause] # collect all examples of a specific label
            lab = label_data[y_label].to_numpy() # extract labels
            y = np.concatenate((y,lab))
            feature_data = label_data.drop(columns=[y_label]).to_numpy() # extract features
            X = np.vstack((X, feature_data))
        new_y = [cause_to_label[cause] for cause in y] # map labels to 0...num_causes-1
        y = np.asarray(new_y)
    else: # predicting fire size
        df = df.drop('STAT_CAUSE_CODE',1)
        y = df['FIRE_SIZE'].to_numpy()
        X = df.drop(columns=['FIRE_SIZE']).to_numpy()

    # shuffle data
    N, d = X.shape
    indices = list(range(N))
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(12)
        np.random.shuffle(indices)
    # indices = indices[:dataset_size]
    X = X[indices]
    y = y[indices]
    return X, y

# read wildfire data and relevant weather data for a city
def get_dataset_with_weather(city="Los Angeles"):
    print("Getting data for " + city + "...")
    db_file = 'FPA_FOD_20170508.sqlite'
    conn = sqlite3.connect(db_file) # create a database connection to the SQLite database specified by the db_file
    # select features
    df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE, FIRE_YEAR, datetime(DISCOVERY_DATE) as DISCOVERY_DATE, DISCOVERY_TIME  FROM 'Fires'", conn)
    df = df.dropna() # remove rows with missing entries
    df = df[df['STAT_CAUSE_CODE'] != 13] # remove missing/undefined causes
    causes = np.sort(df['STAT_CAUSE_CODE'].unique()) # list of wildfire causes (1-13)
    df = df.sort_values(by=['DISCOVERY_DATE'], ascending=True) # sort by date of discovery
    df['DISCOVERY_DATE'] = df['DISCOVERY_DATE'].str.replace(' 00:00:00', '') # remove time from date column


    # get city latitude/longitude
    df_cities = pd.read_csv('weather_data/city_attributes.csv')
    city_row = df_cities.loc[df_cities['City'] == city] # get city's csv entry
    city_lat = float(city_row['Latitude']) # get city's latitude and longitude
    city_long = float(city_row['Longitude'])
    # collect all examples of a specific city where wildfire data and weather data overlap in time and location
    # wildfire's latitude and longitude must be close to city's latitude and longitude
    lat_long_bound = 1 # wildfire lat/long must be with +- this bound from the city's lat/long
    # weather data exists for 2012-10-01 to 2017-11-30
    df = df.loc[(df['DISCOVERY_DATE'].str.replace("-","").astype('int32')>=20121001) & (df['DISCOVERY_DATE'].str.replace("-","").astype('int32')<=20171130)]
    df = df.loc[(df['LATITUDE']>=city_lat-lat_long_bound) & (df['LATITUDE']<=city_lat+lat_long_bound)]
    df = df.loc[(df['LONGITUDE']>=city_long-lat_long_bound) & (df['LONGITUDE']<=city_long+lat_long_bound)]
    df = df.sort_values(by=['FIRE_SIZE'], ascending=False)

    # add new weather columns into original wildfire dataframe
    df_temp = pd.read_csv('weather_data/temperature.csv')
    df_temp = df_temp[df_temp[city].notna()] # drop rows that have a missing entry in city's data
    df_temp['date'] = df_temp['datetime'].str[:11] # make new column of only dates
    df_temp['time'] = df_temp['datetime'].str[-8:-3].str.replace(":", "").astype('int32') # make new col of times in 24-hour without ":"
    df["Temperature"] = "" # add empty column to entire dataset dataframe

    df_wind_speed = pd.read_csv('weather_data/wind_speed.csv')
    df_wind_speed = df_wind_speed[df_wind_speed[city].notna()] # drop rows that have a missing entry in city's data 
    df_wind_speed['date'] = df_wind_speed['datetime'].str[:11] # make new column of only dates
    df_wind_speed['time'] = df_wind_speed['datetime'].str[-8:-3].str.replace(":", "").astype('int32') # make new col of times in 24-hour without ":"
    df["Wind_Speed"] = "" # add empty column to entire dataset dataframe

    df_humidity = pd.read_csv('weather_data/humidity.csv')
    df_humidity = df_humidity[df_humidity[city].notna()] # drop rows that have a missing entry in city's data
    df_humidity['date'] = df_humidity['datetime'].str[:11] # make new column of only dates
    df_humidity['time'] = df_humidity['datetime'].str[-8:-3].str.replace(":", "").astype('int32') # make new col of times in 24-hour without ":"
    df["Humidity"] = "" # add empty column to entire dataset dataframe

    df_pressure = pd.read_csv('weather_data/pressure.csv')
    df_pressure = df_pressure[df_pressure[city].notna()] # drop rows that have a missing entry in city's data
    df_pressure['date'] = df_pressure['datetime'].str[:11] # make new column of only dates
    df_pressure['time'] = df_pressure['datetime'].str[-8:-3].str.replace(":", "").astype('int32') # make new col of times in 24-hour without ":"
    df["Pressure"] = "" # add empty column to entire dataset dataframe

    # iterate over each wildfire entry and find corresponding weather entry with same date and closest time
    for index, row in df.iterrows():
        date = row['DISCOVERY_DATE']
        time = int(row['DISCOVERY_TIME'])

        df_temp_date_mask = df_temp['date'].str.contains(date)
        temp_idx = (df_temp['time'][df_temp_date_mask] - time).abs().idxmin()
        temp = df_temp.iloc[temp_idx][city]
        df.at[index, 'Temperature'] = temp

        df_wind_speed_date_mask = df_wind_speed['date'].str.contains(date)
        wind_speed_idx = (df_wind_speed['time'][df_wind_speed_date_mask] - time).abs().idxmin()
        wind_speed = df_wind_speed.iloc[wind_speed_idx][city]
        df.at[index, 'Wind_Speed'] = wind_speed

        df_humidity_date_mask = df_humidity['date'].str.contains(date)
        humidity_idx = (df_humidity['time'][df_humidity_date_mask] - time).abs().idxmin()
        humidity = df_humidity.iloc[humidity_idx][city]
        df.at[index, 'Humidity'] = humidity

        df_pressure_date_mask = df_pressure['date'].str.contains(date)
        pressure_idx = (df_pressure['time'][df_pressure_date_mask] - time).abs().idxmin()
        pressure = df_pressure.iloc[pressure_idx][city]
        df.at[index, 'Pressure'] = pressure

    df_orig = df.copy() # save copy of full dataset

    # remove certain features
    df = df.drop('STATE', 1)
    df = df.drop('FIRE_YEAR', 1)
    df = df.drop('DISCOVERY_DATE', 1)
    df = df.drop('DISCOVERY_TIME', 1)

    # df = df.drop('LATITUDE', 1)
    # df = df.drop('LONGITUDE', 1)

    # df = df.drop('Temperature', 1)
    # df = df.drop('Wind_Speed', 1)
    # df = df.drop('Humidity', 1)
    # df = df.drop('Pressure', 1)

    num_features = df.shape[1] - 1

    # collect features and label from entire dataset
    X = np.zeros((0,num_features))
    y = np.zeros(0)
    y_label = 'STAT_CAUSE_CODE' # predicting the cause of the fire
    df['STAT_CAUSE_CODE'] = df['STAT_CAUSE_CODE'].astype('int32')
    # print(df)
    causes = [1, 4] # select labels to train/test on
    # extract features and labels
    for label in causes:
        label_data = df.loc[df[y_label]==label] # collect all examples of a specific label
        lab = label_data[y_label].to_numpy() # extract labels
        y = np.concatenate((y,lab))
        feature_data = label_data.drop(columns=[y_label]).to_numpy() # extract features
        X = np.vstack((X, feature_data))

    # shuffle data
    N, d = X.shape
    indices = list(range(N))
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(12)
        np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y, df_orig

# combine wildfire and weather data over multiple cities (this limits the wildire data to lat/long near cities)
def get_dataset_with_weather_multi_cities(cities, csv_filename=None):
    X, y, df = get_dataset_with_weather(cities[0])
    for city in cities[1:]:
        X1, y1, df1 = get_dataset_with_weather(city)
        X = np.concatenate((X, X1))
        y = np.concatenate((y, y1))
        df = df.append(df1)

    if not csv_filename is None:
        df.to_csv(csv_filename) # save entire dataset to csv

    # shuffle dataset
    N, d = X.shape
    indices = list(range(N))
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(12)
        np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y

# load dataset from csv
# if y_label=='STAT_CAUSE_CODE' then predicting cause of fire and can select a subset of causes (labels) to train/test on
# else predicting fire size
def get_dataset_from_csv(csv_file, features=['latitude','longitude','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE'):
    df = pd.read_csv(csv_file, index_col=0)

    if 'STATE' not in (feat.upper() for feat in features):
        df = df.drop('STATE', 1)
    if 'FIRE_YEAR' not in (feat.upper() for feat in features):
        df = df.drop('FIRE_YEAR', 1)
    if 'DISCOVERY_DATE' not in (feat.upper() for feat in features):
        df = df.drop('DISCOVERY_DATE', 1)
    if 'DISCOVERY_TIME' not in (feat.upper() for feat in features):
        df = df.drop('DISCOVERY_TIME', 1)

    if 'LATITUDE' not in (feat.upper() for feat in features):
        df = df.drop('LATITUDE', 1)
    if 'LONGITUDE' not in (feat.upper() for feat in features):
        df = df.drop('LONGITUDE', 1)

    if 'TEMPERATURE' not in (feat.upper() for feat in features):
        df = df.drop('Temperature', 1)
    if 'WIND_SPEED' not in (feat.upper() for feat in features):
        df = df.drop('Wind_Speed', 1)
    if 'HUMIDITY' not in (feat.upper() for feat in features):
        df = df.drop('Humidity', 1)
    if 'PRESSURE' not in (feat.upper() for feat in features):
        df = df.drop('Pressure', 1)

    # print(df)

    num_features = df.shape[1] - 1
    # collect features and label from entire dataset
    X = np.zeros((0,num_features))
    y = np.zeros(0)
    if y_label == 'STAT_CAUSE_CODE': # predicting the cause of the fire
        cause_to_label = {} # map causes 1...13 to labels 0...num_causes-1
        df['STAT_CAUSE_CODE'] = df['STAT_CAUSE_CODE'].astype('int32')
        # extract features and labels
        for i, cause in enumerate(causes):
            cause_to_label[cause] = i
            label_data = df.loc[df[y_label]==cause] # collect all examples of a specific label
            lab = label_data[y_label].to_numpy() # extract labels
            y = np.concatenate((y,lab))
            feature_data = label_data.drop(columns=[y_label]).to_numpy() # extract features
            X = np.vstack((X, feature_data))
        new_y = [cause_to_label[cause] for cause in y] # map labels to 0...num_causes-1
        y = np.asarray(new_y)
    else: # predicting fire size
        df = df.drop('STAT_CAUSE_CODE',1)
        y = df['FIRE_SIZE'].to_numpy()
        X = df.drop(columns=['FIRE_SIZE']).to_numpy()

    # print(df)

    # shuffle data
    # print(df)
    N, d = X.shape
    indices = list(range(N))
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(12)
        np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y
