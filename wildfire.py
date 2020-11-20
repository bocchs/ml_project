import numpy as np
import pandas as pd
import sqlite3
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import sys
import matplotlib.pyplot as plt

# download dataset here:
# https://www.kaggle.com/rtatman/188-million-us-wildfires

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
        # fig = plt.figure()
        # tree.plot_tree(clf)
        # fig.savefig("tree.png")
        # sys.exit()

        # get predictions for current test data split
        test_N = len(test_y)
        predictions = np.zeros(test_N)
        for i in range(test_N):
            sample = np.reshape(test_X[i],(1,-1))
            predictions[i] = clf.predict(sample)
        accuracies[k] = np.sum(predictions == test_y) / test_N
    return accuracies, accuracies.mean()


if __name__ == "__main__":
    db_file = 'FPA_FOD_20170508.sqlite'
    conn = sqlite3.connect(db_file) # create a database connection to the SQLite database specified by the db_file
    # select features
    # df = pd.read_sql_query("SELECT FIRE_YEAR, STAT_CAUSE_CODE, STAT_CAUSE_DESCR, LATITUDE, LONGITUDE, STATE, DISCOVERY_DATE, FIRE_SIZE FROM 'Fires'", conn)
    # df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, STATE, FIRE_SIZE FROM 'Fires'", conn)
    df = pd.read_sql_query("SELECT STAT_CAUSE_CODE, LATITUDE, LONGITUDE, FIRE_SIZE, DISCOVERY_DOY, CONT_DOY FROM 'Fires'", conn)
    df = df.dropna() # remove rows with missing entries
    duration = df['CONT_DOY'] - df['DISCOVERY_DOY']
    df = df.drop(columns=['DISCOVERY_DOY', 'CONT_DOY'])
    df['DURATION'] = duration

    print(df.head()) # check the data

    df.to_csv('dataset.csv')
    causes = df.STAT_CAUSE_CODE.unique()

    num_features = df.shape[1] - 1

    # collect features and label from entire dataset
    X = np.zeros((0,num_features))
    y = np.zeros(0)
    y_label = 'STAT_CAUSE_CODE'
    # y_label = 'FIRE_SIZE'
    for label in causes:
        label_data = df.loc[df[y_label]==label] # collect all examples of a specific label

        lab = label_data[y_label].to_numpy() # extract labels
        y = np.concatenate((y,lab))

        feature_data = label_data.drop(columns=[y_label]).to_numpy() # extract features
        X = np.vstack((X, feature_data))

    # X: Nxd
    # y: N

    test_split = .3
    N, d = X.shape
    indices = list(range(N))
    # split = int(np.floor(test_split * N))
    dataset_size = int(1e4)
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(12)
        np.random.shuffle(indices)
    indices = indices[:dataset_size]
    X = X[indices]
    y = y[indices]

    max_depth_array = [3, 5, 10, 20, 25, 50, 100]
    max_acc = 0
    best_depth = 0
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(16, 8, 4), max_iter=10000, random_state=1)
    for max_depth in max_depth_array:
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)#, min_samples_split=50)
        print(max_depth)
        # clf = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
        accs, mean_acc_crossval = cross_val(X, y, K=10, clf=clf)
        if mean_acc_crossval > max_acc:
            best_depth = max_depth
            max_acc = mean_acc_crossval
            best_accs = accs
    print("Best max depth = " + str(best_depth))
    print("Best avg crossval accuracy = " + str(max_acc))
    print("crossval accuracies = " + str(best_accs))
