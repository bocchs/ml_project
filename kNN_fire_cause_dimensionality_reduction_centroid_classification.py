import numpy as np
import pandas as pd
import sqlite3
import math
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

db_uri = './wildfire_data'

if __name__ == "__main__":
    # DB file url
    db_file = db_uri + '/FPA_FOD_20170508.sqlite'
    
    # create a database connection to the SQLite database specified by the db_file
    conn = sqlite3.connect( db_file ) 
    
    # select features
    features = [
        'FIRE_YEAR',
        #'DISCOVERY_TIME',
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
    stat_cause_desc = { cause_pair[1] : cause_pair[0] for cause_pair in stat_cause_df.to_numpy() }

    # Only use lightning and campfire causes to try and improve accuracy for just two causes
    df = df.loc[df['STAT_CAUSE_CODE'].isin( [stat_cause_desc['Lightning'], stat_cause_desc['Campfire']] )]

    # Set X and y from data file
    X = df.loc[:, features]
    y = df.loc[:, output_feature]

    num_samples = len( y )
    num_features = len( features )

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, stratify=y, random_state=0 )

    # Reduce dimensionality to 3 with Incremental Principal Component Analysis
    pca = pipeline.make_pipeline( preprocessing.StandardScaler(), decomposition.IncrementalPCA( n_components=3, batch_size=10000000 ) )
    pca.fit( X, y )

    # K Nearest Neighbors
    neighbors_list = [ num * 5 for num in range( 1, 21 ) ]
    accuracy = []

    for num_neighbors in neighbors_list:
        print( 'Neighbors: ', num_neighbors )

        clasifier = neighbors.KNeighborsClassifier( num_neighbors, weights='distance' )
        #clasifier = neighbors.NearestCentroid()
        pca.fit( X_train, y_train )
        clasifier.fit( pca.transform( X_train ), y_train )

        accuracy.append( clasifier.score( pca.transform( X_test ), y_test ) )

    # Plot accuracy for varying n
    plt.plot( neighbors_list, accuracy )
    plt.ylabel( 'Accuracy, %' )
    plt.xlabel( 'Neighbors, n' )
    plt.savefig( 'output2.png' )
