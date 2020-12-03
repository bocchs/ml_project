import numpy as np
import pandas as pd
import sqlite3
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta


# download wildfire dataset here:
# https://www.kaggle.com/rtatman/188-million-us-wildfires

# download weather dataset here:
# https://www.kaggle.com/selfishgene/historical-hourly-weather-data


class LinearNN(nn.Module):
    def __init__(self, in_features, out_classes):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, out_classes)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_nn(model, X_train, y_train):
    model.train()
    X_train = torch.from_numpy(X_train).float()
    print(X_train.dtype)
    y_train = torch.from_numpy(y_train).long()
    print(y_train.dtype)
    N = X_train.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_steps = int(1e3)
    for step in range(num_steps):
        inds = np.random.choice(N, batch_size, replace=False) # sample batch 
        X_batch = X_train[inds]
        y_batch = y_train[inds]
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Step ' + str(step) + ' loss = ' + str(loss.item()))
    return model



# perform cross validation using classifier clf
def cross_val(X, y, K, clf):
    # make the number of examples a multiple of K
    num_extra_examples = len(y) % K
    if num_extra_examples != 0:
        X = X[:-num_extra_examples]
        y = y[:-num_extra_examples]
    N = len(y)
    examples_per_group = N // K
    accuracies = np.zeros(K)
    for k in range(K):
        # get train and test groups
        test_idx_start = k * examples_per_group
        test_idx_end = test_idx_start + examples_per_group
        test_X = X[test_idx_start:test_idx_end]
        test_y = y[test_idx_start:test_idx_end]

        train_idx_start = test_idx_end
        if train_idx_start == N:
            train_idx_start = 0
        train_idx_end = train_idx_start + examples_per_group * (K - 1)
        if train_idx_end > N:
            train_idx_end = train_idx_end % N
            train_X = np.vstack((X[train_idx_start:], X[:train_idx_end]))
            train_y = np.concatenate((y[train_idx_start:], y[:train_idx_end]))
        else:
            train_X = X[train_idx_start:train_idx_end]
            train_y = y[train_idx_start:train_idx_end]

        # train on current training data split
        clf = clf.fit(train_X, train_y)
        # model = train_nn(LinearNN(train_X.shape[1], 13), train_X, train_y)
        # model.eval()
        # fig = plt.figure()
        # tree.plot_tree(clf)
        # fig.savefig("tree.png")
        # sys.exit()

        # get predictions for current test data split
        test_N = len(test_y)
        predictions = np.zeros(test_N)
        for i in range(test_N):
            sample = np.reshape(test_X[i],(1,-1))
            pred = clf.predict(sample)
            # pred = model(torch.from_numpy(sample).float()).argmax()
            predictions[i] = pred
        accuracies[k] = np.sum(predictions == test_y) / test_N
    return accuracies, accuracies.mean()


def get_dataset(dataset_size=int(1e5)):
    db_file = 'FPA_FOD_20170508.sqlite'
    conn = sqlite3.connect(db_file) # create a database connection to the SQLite database specified by the db_file
    # select features
    # df = pd.read_sql_query("SELECT FIRE_YEAR, STAT_CAUSE_CODE, STAT_CAUSE_DESCR, LATITUDE, LONGITUDE, STATE, DISCOVERY_DATE, FIRE_SIZE FROM 'Fires'", conn)
    # df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE FROM 'Fires'", conn)
    # df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME  FROM 'Fires'", conn)
    df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE, FIRE_YEAR, datetime(DISCOVERY_DATE) as DISCOVERY_DATE, DISCOVERY_TIME  FROM 'Fires'", conn)
    df = df.dropna() # remove rows with missing entries
    print(df)#.head())
    df = df.sort_values(by=['DISCOVERY_DATE'], ascending=True) # sort by date of discovery
    df['DISCOVERY_DATE'] = df['DISCOVERY_DATE'].str.replace(' 00:00:00', '') # remove time from date column

    causes = df.STAT_CAUSE_CODE.unique()
    states = df.STATE.unique()
    selected_state = 'CA'
    df = df.loc[df['STATE']==selected_state] # collect all examples of a specific city
    df = df.loc[(df['FIRE_YEAR']>=2012) & (df['FIRE_YEAR']<=2017)]
    df = df.loc[(df['LATITUDE']>=33.05) & (df['LATITUDE']<=35.05)]
    df = df.loc[(df['LONGITUDE']>=-117.24) & (df['LONGITUDE']<=119.24)]
    # df = df.drop('STATE', 1) # remove state from features
    df = df.sort_values(by=['FIRE_SIZE'], ascending=False)
    print(df)#.head())

    sys.exit()

    num_features = df.shape[1] - 1

    # collect features and label from entire dataset
    X = np.zeros((0,num_features))
    y = np.zeros(0)
    y_label = 'STAT_CAUSE_CODE'
    # y_label = 'FIRE_SIZE'
    if y_label == 'STAT_CAUSE_CODE':
        df['STAT_CAUSE_CODE'] -= 1 # predict class 0-12
        df['STAT_CAUSE_CODE'] = df['STAT_CAUSE_CODE'].astype('int32')
    print(df)
    for label in causes:
        label_data = df.loc[df[y_label]==label] # collect all examples of a specific label
        lab = label_data[y_label].to_numpy() # extract labels
        y = np.concatenate((y,lab))
        feature_data = label_data.drop(columns=[y_label]).to_numpy() # extract features
        X = np.vstack((X, feature_data))

    # take subset of shuffled data
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
    db_file = 'FPA_FOD_20170508.sqlite'
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

    # max_depth_array = [3, 5, 10, 20, 25, 50, 100]
    max_depth_array = [10]
    max_mean_acc = 0
    best_depth = 0
    for max_depth in max_depth_array:
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)#, min_samples_split=50)
        print(max_depth)
        # clf = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
        accs, mean_acc_crossval = cross_val(X, y, K=10, clf=clf)
        if mean_acc_crossval > max_mean_acc:
            best_depth = max_depth
            max_mean_acc = mean_acc_crossval
            best_accs = accs
    print("Best max depth = " + str(best_depth))
    print("Best avg crossval accuracy = " + str(max_mean_acc))
    print("Best crossval accuracies = " + str(best_accs))
