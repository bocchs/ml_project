import numpy as np
import pandas as pd
import sqlite3
import math
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

db_uri = './wildfire_data'

if __name__ == "__main__":
    # DB file url
    db_file = db_uri + '/FPA_FOD_20170508.sqlite'
    
    # create a database connection to the SQLite database specified by the db_file
    conn = sqlite3.connect( db_file ) 
    
    # select features
    features = [
        'FIRE_YEAR', 
        'DISCOVERY_TIME', 
        'LATITUDE', 
        'LONGITUDE', 
        'FIRE_SIZE'
    ]
    output_feature = 'STAT_CAUSE_CODE'

    df = pd.read_sql_query( "SELECT " + ', '.join( features ) + ", " + output_feature + " FROM 'Fires'", conn )

    # Discovery and containment date data
    fire_duration_df = pd.read_sql_query( "SELECT DISCOVERY_DATE, CONT_DATE FROM 'Fires'", conn )
    fire_duration_df['DATE'] = pd.to_datetime( fire_duration_df['DISCOVERY_DATE'], unit='D', origin='julian' )

    df['DURATION'] = fire_duration_df['CONT_DATE'] - fire_duration_df['DISCOVERY_DATE']
    features.append( 'DURATION' )

    df['MONTH'] = pd.DatetimeIndex( fire_duration_df['DATE'] ).month
    features.append( 'MONTH' )
    
    df['WEEK'] = pd.DatetimeIndex( fire_duration_df['DATE'] ).dayofweek
    features.append( 'WEEK' )

    df['DAY'] = pd.DatetimeIndex( fire_duration_df['DATE'] ).day
    features.append( 'DAY' )

    # Remove samples with NaN data
    df = df.dropna()

    # STAT Cause Description Reference
    stat_cause_df = pd.read_sql_query( "SELECT STAT_CAUSE_CODE, STAT_CAUSE_DESCR FROM 'Fires'", conn )
    stat_cause_desc = { cause_pair[0] : cause_pair[1] for cause_pair in stat_cause_df.to_numpy() }

    # Set X and y from data file
    X = df.loc[:, features]
    y = df.loc[:, output_feature]

    num_samples = len( y )
    num_features = len( features )

    # K Nearest Neighbors
    n_neighbors = math.floor( 1 * ( num_samples ** ( 4 / ( 4 + num_features ) ) ) ) 
    clasifier = neighbors.KNeighborsClassifier( n_neighbors, weights='distance' )

    # Cross fold validation    
    k = 10
    cross_fold = KFold( n_splits=k, random_state=None )

    avg_accuracy = cross_val_score( clasifier, X, y, cv=cross_fold )

    print( 'Avg accuracy: ', avg_accuracy )