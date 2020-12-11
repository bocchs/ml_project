import numpy as np
import pandas as pd
import sqlite3
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn import tree
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
import sys
import os
from os import path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import argparse
from wildfire_datasets import *


# download wildfire dataset here:
# https://www.kaggle.com/rtatman/188-million-us-wildfires

# download weather dataset here (and store in folder named "weather_data"):
# https://www.kaggle.com/selfishgene/historical-hourly-weather-data


# determine best regularization coefficient using cross validation (for ridge regression and SVM)
alpha_array = [0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 500, 1000]

# determine best decision tree depths using cross validation
max_depth_array = [3, 5, 10, 20, 25, 50, 100] 

# determine best number of neighbors using cross validation (5 to 100 increments of 5)
neighbors_list = [ num * 5 for num in range( 1, 21 ) ]

# filename of csv for saving combined wildfire and weather data to
csv_file = 'fire_weather_cities.csv'

# save test result files in this folder
test_folder = 'tests/'



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


def train_nn(model, X_train, y_train, display_train_loss=False):
    model.train()
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    N = X_train.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_steps = 1000
    loss_ra = np.zeros(num_steps)
    for step in range(num_steps):
        inds = np.random.choice(N, batch_size, replace=False) # sample batch 
        X_batch = X_train[inds]
        y_batch = y_train[inds]
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ra[step] = loss.item()
        if step % 100 == 0 and display_train_loss:
            print('Step ' + str(step) + ' loss = ' + str(loss.item()))
    print("Finished training neural net: total loss = " + str(loss_ra.sum()) + "   avg loss = " + str(loss_ra.mean()))
    return model



# perform cross validation using classifier clf
# return test accuracy of each fold in K-fold cross validation (length-K vector)
# return average of test accuracies (average of length-K vector)
def cross_val_classification(X, y, K, clf):
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

# perform cross validation using classifier clf and pca pipeline
# return test accuracy of each fold in K-fold cross validation (length-K vector)
# return average of test accuracies (average of length-K vector)
def cross_val_classification_with_reduction(X, y, K, clf, pipe):
    new_clf = sklearn.base.clone(clf) # save untrained classifier
    new_pipe = sklearn.base.clone(pipe) # save untrained pipeline for dimension reduction
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

        # reset to untrained classifier and pca pipeline for each crossval fold
        new_clf = sklearn.base.clone(clf)
        new_pipe = sklearn.base.clone(pipe)

        # train on current training data split
        new_pipe.fit(train_X, train_y)
        new_clf = new_clf.fit(new_pipe.transform(train_X), train_y)

        # get predictions for current test data split
        test_N = len(test_y)
        predictions = np.zeros(test_N)
        for i in range(test_N):
            sample = np.reshape(test_X[i],(1,-1))
            pred = new_clf.predict(new_pipe.transform(sample))
            predictions[i] = pred
        accuracies[k] = np.sum(predictions == test_y) / test_N
    return accuracies, accuracies.mean()

# perform cross validation using neural network
# return test accuracy of each fold in K-fold cross validation (length-K vector)
# return average of test accuracies (average of length-K vector)
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


# perform cross validation using regressor reg
# return MSE of each fold in K-fold cross validation (length-K vector)
# return average of average MSE (average of length-K vector)
def cross_val_regression(X, y, K, reg):
    new_reg = sklearn.base.clone(reg) # save untrained classifier
    # make the number of examples a multiple of K
    num_extra_examples = len(y) % K
    if num_extra_examples != 0:
        X = X[:-num_extra_examples]
        y = y[:-num_extra_examples]
    N = len(y)
    examples_per_group = N // K
    mse_ra = np.zeros(K)
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
        new_reg = sklearn.base.clone(reg)

        # train on current training data split
        new_reg = new_reg.fit(train_X, train_y)

        # get predictions for current test data split
        test_N = len(test_y)
        predictions = np.zeros(test_N)
        for i in range(test_N):
            sample = np.reshape(test_X[i],(1,-1))
            pred = new_reg.predict(sample)
            predictions[i] = pred
            # print('truth = ' + str(test_y[i]) + ',  pred = ' + str(pred))
        mse = np.sum(np.square(predictions - test_y)) / test_N
        mse_ra[k] = mse
    return mse_ra, mse_ra.mean()



def test_nearest_neighbor(X, y, neighbors_list, K=10):
    X = normalize_features(X)
    print("Testing nearest neighbor for predicting wildfire cause...")
    max_mean_acc = 0
    best_num_neighbors = 0
    for num_neighbors in neighbors_list:
        clf = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
        print("Testing nearest neighbors with k = " + str(num_neighbors))
        accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=clf)
        print("Obtained avg crossval accuracy = " + str(mean_acc_crossval))
        if mean_acc_crossval > max_mean_acc:
            best_num_neighbors = num_neighbors
            max_mean_acc = mean_acc_crossval
            best_accs = accs
    print("Nearest Neighbor best crossval accuracies = " + str(best_accs))
    print("Nearest Neighbor best avg crossval accuracy = " + str(max_mean_acc))
    print("Nearest Neighbor best number of neighbors = " + str(best_num_neighbors))

def test_nearest_neighbor_reduction(X, y, neighbors_list, K=10, d=3):
    X = normalize_features(X)
    print("Testing nearest neighbor with dimension reduction for predicting wildfire cause...")
    max_mean_acc = 0
    best_num_neighbors = 0
    for num_neighbors in neighbors_list:
        clf = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
        pca = make_pipeline( StandardScaler(), IncrementalPCA(n_components=d,batch_size=1000000 ) )
        print("Testing nearest neighbors with k = " + str(num_neighbors) + " and dimension = " + str(d))
        accs, mean_acc_crossval = cross_val_classification_with_reduction(X, y, K=K, clf=clf, pipe=pca)
        print("Obtained avg crossval accuracy = " + str(mean_acc_crossval))
        if mean_acc_crossval > max_mean_acc:
            best_num_neighbors = num_neighbors
            max_mean_acc = mean_acc_crossval
            best_accs = accs
    print("Nearest Neighbor dimension reduction best crossval accuracies = " + str(best_accs))
    print("Nearest Neighbor dimension reduction best avg crossval accuracy = " + str(max_mean_acc))
    print("Nearest Neighbor dimension reduction best number of neighbors = " + str(best_num_neighbors))

def test_decision_tree_max_depth(X, y, max_depth_array, K=10):
    X = normalize_features(X)
    print("Testing decision tree best max depth for predicting wildfire cause...")
    max_mean_acc = 0
    best_depth = 0
    for max_depth in max_depth_array:
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
        print("Testing decision tree max_depth = " + str(max_depth))
        accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=clf)
        print("Obtained avg crossval accuracy = " + str(mean_acc_crossval))
        if mean_acc_crossval > max_mean_acc:
            best_depth = max_depth
            max_mean_acc = mean_acc_crossval
            best_accs = accs
    print("Decision Tree best crossval accuracies = " + str(best_accs))
    print("Decision Tree best avg crossval accuracy = " + str(max_mean_acc))
    print("Decision Tree best max depth = " + str(best_depth))


def test_decision_tree_criterion(X, y, K=10):
    X = normalize_features(X)
    print("Testing decision tree mutual info vs Gini feature split for predicting wildfire cause...")

    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=10)
    print("Testing decision tree with Gini criterion")
    accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=clf)
    print("Gini criterion obtained avg crossval accuracy = " + str(mean_acc_crossval))

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
    print("Testing decision tree with entropy criterion")
    accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=clf)
    print("entropy criterion obtained avg crossval accuracy = " + str(mean_acc_crossval))


def test_big_net(X, y, K=10):
    X = normalize_features(X)
    print("Testing Big Net for predicting wildfire cause...")
    num_classes = len(np.unique(y))
    net = BigNet(X.shape[1], num_classes)
    accs, mean_acc_crossval = cross_val_neural_network(X, y, K=K, net=net)
    print("Big Net crossval accuracies = " + str(accs))
    print("Big Net average crossval accuracy = " + str(mean_acc_crossval))


def test_small_net(X, y, K=10):
    X = normalize_features(X)
    print("Testing Small Net for predicting wildfire cause...")
    num_classes = len(np.unique(y))
    net = BigNet(X.shape[1], num_classes)
    accs, mean_acc_crossval = cross_val_neural_network(X, y, K=K, net=net)
    print("Small Net crossval accuracies = " + str(accs))
    print("Small Net average crossval accuracy = " + str(mean_acc_crossval))


def test_svm_L2_coeff(X, y, alpha_array, K=10):
    X = normalize_features(X)
    print("Testing L2 coeffs in SVM with linear kernel for predicting wildfire cause...")
    max_mean_acc = 0
    best_alpha = 0
    for alpha in alpha_array:
        svm_linear_kernel = SVC(kernel='linear', C=alpha)
        print("Testing L2 regularization coeff = " + str(alpha))
        accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=svm_linear_kernel)
        print("Obtained avg crossval accuracy = " + str(mean_acc_crossval))
        if mean_acc_crossval > max_mean_acc:
            best_alpha = alpha
            max_mean_acc = mean_acc_crossval
            best_accs = accs
    print("SVM with linear kernel best crossval accuracies = " + str(best_accs))
    print("SVM with linear kernel best avg crossval accuracy = " + str(max_mean_acc))
    print("SVM with linear kernel best L2 regularization coefficient = " + str(best_alpha))


def test_svm_kernel(X, y, K=10):
    X = normalize_features(X)
    print("Testing SVM with Gaussian vs linear kernel for predicting wildfire cause...")
    svm_gauss_kernel = SVC(kernel='rbf')
    print("Testing Gaussian kernel")
    accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=svm_gauss_kernel)
    print("Gaussian kernel obtained avg crossval accuracy = " + str(mean_acc_crossval))

    svm_linear_kernel = SVC(kernel='linear')
    print("Testing linear kernel")
    accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=svm_linear_kernel)
    print("Linear kernel obtained avg crossval accuracy = " + str(mean_acc_crossval))


# alpha is the L2 regularization coefficient in the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2
def test_ridge_regression(X, y, alpha_array, K=10):
    X = normalize_features(X)
    print("Testing ridge regression for predicting wildfire size...")
    mask = y < 1
    X = X[mask]
    y = y[mask]
    min_mean_mse = 1e30
    best_alpha = 0
    for alpha in alpha_array:
        print("Testing L2 regularization coeff  = " + str(alpha))
        regressor = Ridge(alpha=alpha)
        mse_ra, mean_mse = cross_val_regression(X, y, K=K, reg=regressor)
        print("Obtained avg crossval MSE = " + str(mean_mse))
        if mean_mse < min_mean_mse:
            best_alpha = alpha
            min_mean_mse = mean_mse
    print("Ridge regression best average MSE = " + str(min_mean_mse))
    print("Ridge regression best L2 regularization coefficient = " + str(best_alpha))


# X is Nxd, normalize each feature to zero-mean, unit-variance
def normalize_features(X):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    normalized_X = (X - mean) / stdev
    return normalized_X



def run_nearest_neighbor_tests():
    if print_tests_to_files:
        sys.stdout = open(test_folder + 'nearest_neighbor_tests.txt', 'w')
    # for classifying cause of fire (wildfire and weather dataset combined)
    # select which causes to train/test on by setting the causes list (e.g. causes=[1,4])
    print(" --------- Nearest Neighbors: testing classification of 12 causes of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor(X, y, neighbors_list, K=10)
    print('\n')

    print(" --------- Nearest Neighbors: testing classification of 12 causes of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor(X, y, neighbors_list, K=10)
    print('\n')

    print(" --------- Nearest Neighbors: testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor(X, y, neighbors_list, K=10)
    print('\n')

    print(" --------- Nearest Neighbors: testing classification of lightning vs campfire as cause of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor(X, y, neighbors_list, K=10)
    print('\n\n\n')

def run_nearest_neighbor_dimensional_reduction_tests():
    if print_tests_to_files:
        sys.stdout = open(test_folder + 'nearest_neighbor_dim_reduc_tests.txt', 'w')
    print(" --------- Nearest Neighbors Dimension 3: testing classification of 12 causes of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor_reduction(X, y, neighbors_list, K=10, d=3)
    print('\n')

    print(" --------- Nearest Neighbors Dimension 3: testing classification of 12 causes of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor_reduction(X, y, neighbors_list, K=10, d=3)
    print('\n')

    print(" --------- Nearest Neighbors Dimension 3: testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor_reduction(X, y, neighbors_list, K=10, d=3)
    print('\n')

    print(" --------- Nearest Neighbors Dimension 3: testing classification of lightning vs campfire as cause of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_nearest_neighbor_reduction(X, y, neighbors_list, K=10, d=3)
    print('\n\n\n')

def run_decision_tree_tests():
    if print_tests_to_files:
        sys.stdout = open(test_folder + 'decision_tree_tests.txt', 'w')
    print(" --------- Decision Tree (max depth): testing classification of 12 causes of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_decision_tree_max_depth(X, y, max_depth_array, K=10)
    print('\n')

    print(" --------- Decision Tree (max depth): testing classification of 12 causes of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_decision_tree_max_depth(X, y, max_depth_array, K=10)
    print('\n')

    print(" --------- Decision Tree (max depth): testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_decision_tree_max_depth(X, y, max_depth_array, K=10)
    print('\n')

    print(" --------- Decision Tree (max depth): testing classification of lightning vs campfire as cause of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_decision_tree_max_depth(X, y, max_depth_array, K=10)
    print('\n')

    print(" --------- Decision Tree (criterion): testing classification of 12 causes of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_decision_tree_criterion(X, y, K=10)
    print('\n')

    print(" --------- Decision Tree (criterion): testing classification of lightning vs campfire as cause of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_decision_tree_criterion(X, y, K=10)
    print('\n\n\n')

    print(" --------- Decision Tree (max depth): testing classification of 12 causes with all 176,945 wildfire-only samples in California --------- ")
    X, y = get_wildfire_dataset()
    test_decision_tree_max_depth(X, y, max_depth_array, K=10)
    print('\n\n\n')


def run_neural_network_tests():
    if print_tests_to_files:
        sys.stdout = open(test_folder + 'neural_net_tests.txt', 'w')
    print(" --------- Neural Network: testing classification of 12 causes of wildfire using only latitude, longitude, and fire size (no weather features) --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_big_net(X, y, K=5)
    print('\n')
    test_small_net(X, y, K=5)
    print('\n')

    print(" --------- Neural Network: testing classification of 12 causes of wildfire including weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_big_net(X, y, K=5)
    print('\n')
    test_small_net(X, y, K=5)
    print('\n')

    print(" --------- Neural Network: testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features) --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_big_net(X, y, K=5)
    print('\n')
    test_small_net(X, y, K=5)
    print('\n')

    print(" --------- Neural Network: testing classification of lightning vs campfire as cause of wildfire including weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_big_net(X, y, K=5)
    print('\n')
    test_small_net(X, y, K=5)
    print('\n\n\n')

    print(" --------- Neural Network: testing classification of 12 causes with all 176,945 wildfire-only samples in California --------- ")
    X, y = get_wildfire_dataset()
    test_big_net(X, y, K=5)
    print('\n')
    test_small_net(X, y, K=5)
    print('\n\n\n')


def run_svm_tests():
    if print_tests_to_files:
        sys.stdout = open(test_folder + 'svm_tests.txt', 'w')
    print(" --------- SVM (kernel): testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features) --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_svm_kernel(X, y, K=10)
    print('\n')

    print(" --------- SVM (kernel): testing classification of lightning vs campfire as cause of wildfire including weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_svm_kernel(X, y, K=10)
    print('\n')

    print(" --------- SVM (L2 coeff): testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features) --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_svm_L2_coeff(X, y, alpha_array, K=10)
    print('\n')

    print(" --------- SVM (L2 coeff): testing classification of lightning vs campfire as cause of wildfire including weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_svm_L2_coeff(X, y, alpha_array, K=10)
    print('\n\n\n')


def run_ridge_regression_tests():
    if print_tests_to_files:
        sys.stdout = open(test_folder + 'rdige_regression_tests.txt', 'w')
    # for classifying size of fire (wildfire and weather dataset combined)
    print(" --------- Ridge Regression: testing prediction of wildfire size using only weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['temperature','wind_speed','humidity','pressure'], causes=[], y_label='FIRE_SIZE')
    test_ridge_regression(X, y, alpha_array)
    print('\n')

    print(" --------- Ridge Regression: testing prediction of wildfire size using latitude, longitude, and weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','temperature','wind_speed','humidity','pressure'], causes=[], y_label='FIRE_SIZE')
    test_ridge_regression(X, y, alpha_array)
    print('\n\n\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true', default=False, help='Flag for printing test results to console rather than to files.')
    args = parser.parse_args()
    return args

"""
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
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('expand_frame_repr', False)

    args = get_args()

    print_tests_to_files = not args.display

    cities = ['Los Angeles', 'San Francisco', 'San Diego'] # cities for combining wildfire and weather data
    # save combined wildfire and weather dataset to csv file
    # can then read dataset faster from csv file rather than SQLite database
    if ( not path.exists(csv_file) ):
        X, y = get_dataset_with_weather_multi_cities(cities, csv_file) # <-- run this to combine wildfire data with weather data and save to csv

    if ( print_tests_to_files and not path.exists(test_folder) ):
        os.mkdir(test_folder) # store test output files here

    run_nearest_neighbor_tests()
    run_nearest_neighbor_dimensional_reduction_tests()
    run_decision_tree_tests()
    run_neural_network_tests()
    run_svm_tests()
    run_ridge_regression_tests()
    