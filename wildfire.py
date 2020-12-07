import numpy as np
import pandas as pd
import sqlite3
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from wildfire_datasets import *


# download wildfire dataset here:
# https://www.kaggle.com/rtatman/188-million-us-wildfires

# download weather dataset here (and store in folder named "weather_data"):
# https://www.kaggle.com/selfishgene/historical-hourly-weather-data


class BigNet(nn.Module):
    def __init__(self, in_features, out_classes):
        super(BigNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, out_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SmallNet(nn.Module):
    def __init__(self, in_features, out_classes):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 20)
        self.fc2 = nn.Linear(20, out_classes)   

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_nn(model, X_train, y_train):
    model.train()
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    N = X_train.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    num_steps = int(2e3)
    for step in range(num_steps):
        inds = np.random.choice(N, batch_size, replace=False) # sample batch 
        X_batch = X_train[inds]
        y_batch = y_train[inds]
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Step ' + str(step) + ' loss = ' + str(loss.item()))
    return model



# perform cross validation using classifier clf
def cross_val_tree(X, y, K, clf):
    new_clf = sklearn.base.clone(clf) # save untrained classifier
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

        # reset to untrained classifier for each crossval fold
        new_clf = sklearn.base.clone(clf)

        # train on current training data split
        new_clf = new_clf.fit(train_X, train_y)

        # get predictions for current test data split
        test_N = len(test_y)
        predictions = np.zeros(test_N)
        for i in range(test_N):
            sample = np.reshape(test_X[i],(1,-1))
            pred = new_clf.predict(sample)
            predictions[i] = pred
        accuracies[k] = np.sum(predictions == test_y) / test_N
    return accuracies, accuracies.mean()


# perform cross validation using neural network
def cross_val_neural_network(X, y, K, net):
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

        # reset model weights for each crossval fold
        for layer in net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # train on current training data split
        model = train_nn(net, train_X, train_y)


        # get predictions for current test data split
        test_N = len(test_y)
        predictions = np.zeros(test_N)
        for i in range(test_N):
            sample = np.reshape(test_X[i],(1,-1))
            pred = model(torch.from_numpy(sample).float()).argmax()
            predictions[i] = pred
        accuracies[k] = np.sum(predictions == test_y) / test_N
    return accuracies, accuracies.mean()


def test_decision_tree(X, y, K=10):
    # max_depth_array = [3, 5, 10, 20, 25, 50, 100]
    max_depth_array = [10]
    max_mean_acc = 0
    best_depth = 0
    for max_depth in max_depth_array:
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)#, min_samples_split=50)
        print(max_depth)
        # clf = svm.SVC()
        # clf = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
        accs, mean_acc_crossval = cross_val_tree(X, y, K=K, clf=clf)
        if mean_acc_crossval > max_mean_acc:
            best_depth = max_depth
            max_mean_acc = mean_acc_crossval
            best_accs = accs
    print("Decision Tree best max depth = " + str(best_depth))
    print("Decision Tree best avg crossval accuracy = " + str(max_mean_acc))
    print("Decision Tree best crossval accuracies = " + str(best_accs))


def test_big_net(X, y, K=10):
    num_classes = len(np.unique(y))
    net = BigNet(X.shape[1], num_classes)
    accs, mean_acc_crossval = cross_val_neural_network(X, y, K=10, net=net)
    print("Big Net average crossval accuracy = " + str(mean_acc_crossval))
    print("Big Net crossval accuracies = " + str(accs))


def test_small_net(X, y, K=10):
    num_classes = len(np.unique(y))
    net = BigNet(X.shape[1], num_classes)
    accs, mean_acc_crossval = cross_val_neural_network(X, y, K=K, net=net)
    print("Small Net average crossval accuracy = " + str(mean_acc_crossval))
    print("Small Net crossval accuracies = " + str(accs))



if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('expand_frame_repr', False)

    csv_file = 'fire_weather_cities.csv'

    cities = ['Los Angeles', 'San Francisco', 'San Diego']
    X, y = get_dataset_with_weather_multi_cities(cities, csv_file) # save combined wildfire and weather dataset to csv file

    X, y = get_wildfire_dataset(y_label='STAT_CAUSE_CODE') # for classifying cause of fire (only using wildfire dataset)

    X, y = get_wildfire_dataset(y_label='FIRE_SIZE') # for predicting size of fire (only iusing wildfire dataset)

    # # for classifying cause of fire (wildfire and weather dataset combined)
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')

    # for classifying size of fire (wildfire and weather dataset combined)
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','temperature','wind_speed','humidity','pressure'], causes=[], y_label='FIRE_SIZE')


    # test_decision_tree(X, y) # perform cross validation using decision tree classifier

    test_big_net(X, y, K=2)
    test_small_net(X, y, K=2)

