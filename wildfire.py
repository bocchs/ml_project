import numpy as np
import pandas as pd
import sqlite3
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
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


# determine best regularization coefficient using cross validation (for ridge regression and SVM)
alpha_array = [0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 500, 1000]

# determine best decision tree depths using cross validation
max_depth_array = [3, 5, 10, 20, 25, 50, 100] 

# filename of csv for saving combined wildfire and weather data to
csv_file = 'fire_weather_cities.csv'



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
    batch_size = 64
    num_steps = int(5e3)
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




def test_decision_tree(X, y, max_depth_array, K=10):
    X = normalize_features(X)
    print("Testing decision tree for predicting wildfire cause...")
    max_mean_acc = 0
    best_depth = 0
    for max_depth in max_depth_array:
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)
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


def test_linear_svm(X, y, alpha_array, K=10):
    X = normalize_features(X)
    print("Testing SVM with linear kernel for predicting wildfire cause...")
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


def test_gauss_svm(X, y, alpha_array, K=10):
    X = normalize_features(X)
    print("Testing SVM with Gaussian kernel for predicting wildfire cause...")
    max_mean_acc = 0
    best_alpha = 0
    for alpha in alpha_array:
        svm_gauss_kernel = SVC(kernel='rbf', C=alpha)
        print("Testing L2 regularization coeff = " + str(alpha))
        accs, mean_acc_crossval = cross_val_classification(X, y, K=K, clf=svm_gauss_kernel)
        print("Obtained avg crossval accuracy = " + str(mean_acc_crossval))
        if mean_acc_crossval > max_mean_acc:
            best_alpha = alpha
            max_mean_acc = mean_acc_crossval
            best_accs = accs
    print("SVM with Gaussian kernel best crossval accuracies = " + str(best_accs))
    print("SVM with Gaussian kernel best avg crossval accuracy = " + str(max_mean_acc))
    print("SVM with Gaussian kernel best L2 regularization coefficient = " + str(best_alpha))


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



def run_decision_tree_tests():
    # for classifying cause of fire (wildfire and weather dataset combined)
    # select which causes to train/test on by setting the causes list (e.g. causes=[1,4])
    print(" --------- Decision Tree: testing classification of 12 causes of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_decision_tree(X, y, max_depth_array, K=10)
    print('\n')

    print(" --------- Decision Tree: testing classification of 12 causes of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=list(range(1,13)), y_label='STAT_CAUSE_CODE')
    test_decision_tree(X, y, max_depth_array, K=10)
    print('\n')

    print(" --------- Decision Tree: testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features)  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_decision_tree(X, y, max_depth_array, K=10)
    print('\n')

    print(" --------- Decision Tree: testing classification of lightning vs campfire as cause of wildfire using latitude, longitude, fire size, and weather features  --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1,4], y_label='STAT_CAUSE_CODE')
    test_decision_tree(X, y, max_depth_array, K=10)
    print('\n\n\n')


def run_neural_network_tests():
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


def run_svm_tests():
    print(" --------- SVM: testing classification of lightning vs campfire as cause of wildfire using only latitude, longitude, and fire size (no weather features) --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_linear_svm(X, y, alpha_array, K=10)
    print('\n')
    test_gauss_svm(X, y, alpha_array, K=10)
    print('\n')

    print(" --------- SVM: testing classification of lightning vs campfire as cause of wildfire including weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','fire_size','temperature','wind_speed','humidity','pressure'], causes=[1, 4], y_label='STAT_CAUSE_CODE')
    test_linear_svm(X, y, alpha_array, K=10)
    print('\n')
    test_gauss_svm(X, y, alpha_array, K=10)
    print('\n\n\n')


def run_ridge_regression_tests():
    # for classifying size of fire (wildfire and weather dataset combined)
    print(" --------- Ridge Regression: testing prediction of wildfire size using only weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['temperature','wind_speed','humidity','pressure'], causes=[], y_label='FIRE_SIZE')
    test_ridge_regression(X, y, alpha_array)
    print('\n')

    print(" --------- Ridge Regression: testing prediction of wildfire size using latitude, longitude, and weather features --------- ")
    X, y = get_dataset_from_csv(csv_file, features=['latitude','longitude','temperature','wind_speed','humidity','pressure'], causes=[], y_label='FIRE_SIZE')
    test_ridge_regression(X, y, alpha_array)
    print('\n\n\n')


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


    cities = ['Los Angeles', 'San Francisco', 'San Diego'] # cities for combining wildfire and weather data
    # save combined wildfire and weather dataset to csv file
    # can then read dataset faster from csv file rather than SQLite database
    # X, y = get_dataset_with_weather_multi_cities(cities, csv_file) # <-- run this to combine wildfire data with weather data and save to csv

    run_decision_tree_tests()
    run_neural_network_tests()
    run_svm_tests()
    run_ridge_regression_tests()
    