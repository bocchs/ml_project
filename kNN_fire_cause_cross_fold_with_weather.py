import numpy as np
import pandas as pd
import sqlite3
import math
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

db_uri = './wildfire_data'

"""
    STAT_CAUSE_CODE to description
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
def get_dataset_with_weather(city="Los Angeles"):
    db_file = db_uri + '/FPA_FOD_20170508.sqlite'
    conn = sqlite3.connect(db_file) # create a database connection to the SQLite database specified by the db_file
    # select features
    # df = pd.read_sql_query("SELECT FIRE_YEAR, STAT_CAUSE_CODE, STAT_CAUSE_DESCR, LATITUDE, LONGITUDE, STATE, DISCOVERY_DATE, FIRE_SIZE FROM 'Fires'", conn)
    # df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE FROM 'Fires'", conn)
    # df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME  FROM 'Fires'", conn)
    df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE, FIRE_YEAR, datetime(DISCOVERY_DATE) as DISCOVERY_DATE, DISCOVERY_TIME  FROM 'Fires'", conn)
    df = df.dropna() # remove rows with missing entries
    df = df[df['STAT_CAUSE_CODE'] != 13] # remove missing/undefined causes
    causes = df['STAT_CAUSE_CODE'].unique() # list of wildfire causes (1-13)
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
    df["Tempature"] = "" # add empty column to entire dataset dataframe

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
        df.at[index, 'Tempature'] = temp

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


    # remove certain features
    df = df.drop('STATE', 1)
    df = df.drop('FIRE_YEAR', 1)
    df = df.drop('DISCOVERY_DATE', 1)
    df = df.drop('DISCOVERY_TIME', 1)

    # df = df.drop('LATITUDE', 1)
    # df = df.drop('LONGITUDE', 1)

    # df = df.drop('Tempature', 1)
    df = df.drop('Wind_Speed', 1)
    df = df.drop('Humidity', 1)
    df = df.drop('Pressure', 1)

    num_features = df.shape[1] - 1

    # collect features and label from entire dataset
    X = np.zeros((0,num_features))
    y = np.zeros(0)
    y_label = 'STAT_CAUSE_CODE' # predicting the cause of the fire
    if y_label == 'STAT_CAUSE_CODE':
        df['STAT_CAUSE_CODE'] = df['STAT_CAUSE_CODE'].astype('int32')
    print(df)
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
    return X, y

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('expand_frame_repr', False)

    X, y = get_dataset_with_weather()

    # K Nearest Neighbors
    neighbors_list = [ num * 5 for num in range( 1, 21 ) ]
    accuracy = []

    for num_neighbors in neighbors_list:
        print( 'Neighbors: ', num_neighbors )
        clasifier = neighbors.KNeighborsClassifier( num_neighbors, weights='distance' )

        # Cross fold validation    
        k = 3
        cross_fold = KFold( n_splits=k, random_state=None )

        avg_accuracy = sum( cross_val_score( clasifier, X, y, cv=cross_fold, n_jobs=-1 ) ) / k
        accuracy.append( avg_accuracy )

    # Plot accuracy for varying n
    plt.plot( neighbors_list, accuracy )
    plt.ylabel( 'Accuracy, %' )
    plt.xlabel( 'Neighbors, n' )
    plt.savefig( 'output4.png' )
