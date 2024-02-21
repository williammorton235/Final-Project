import pandas
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.preprocessing as test
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import resample
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def selectExperiment(data, experiment):
    if experiment == 1:
        data = data
    elif experiment == 2.1:  # simple sum
        data = data.drop(['EXUSEU_Avg', 'US_GDP', 'US_CPI', 'US_BOP', 'US_DM3Index', 'US_UC', 'EU_GDP', 'EU_CPI', 'EU_BOP', 'EU_DM3Index', 'EU_UC', 'OIL_Avg'], axis=1)
    elif experiment == 2.2:  # divisia
        data = data.drop(['EXUSEU_Avg', 'US_GDP', 'US_IR', 'US_CPI', 'US_M3', 'US_BOP', 'EU_GDP', 'EU_OR_Avg', 'EU_CPI', 'EU_M3', 'EU_BOP', 'OIL_Avg'], axis=1)
    elif experiment == 3:
        data = data.drop(['US_GDP', 'US_CPI', 'US_BOP', 'EU_GDP', 'EU_CPI', 'EU_BOP', 'OIL_Avg'], axis=1)
    return data


def createRollingWindow(window, offset, X, y, names):
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    X_train = X[:18]
    X_val = X[18:-12]
    X_test = X[-12:]
    # scale factors
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    # create window
    X_train = np.array([X_train[j - window:j, :].flatten() for j in range(window, len(X_train))])
    X_val = np.array([X_val[j - window:j, :].flatten() for j in range(window, len(X_val))])
    X_test = np.array([X_test[j - window:j, :].flatten() for j in range(window, len(X_test))])
    y_train = y[:18]
    y_val = y[18:-12]
    y_test = y[-12:]
    y_train = y_train[window:]
    y_val = y_val[window:]
    y_test = y_test[window:]

    return X_train, X_test, X_val, y_train, y_test, y_val


def downsample(X_train, y_train):
    traind = pd.DataFrame(X_train)
    traind['y'] = y_train
    # Separate majority and minority classes
    majority_val = int(traind.y.mode()[0])
    majority = traind[traind['y'] == majority_val]
    minority = traind[traind['y'] != majority_val]
    if len(minority) < len(traind) / 10:
        pass
        print("Insufficient Data")
    # Downsample majority class
    df_majority_downsampled = resample(majority, replace=False, n_samples=len(minority))
    downsample = pd.concat([df_majority_downsampled, minority])
    X_train = downsample.drop(['y'], axis=1).values
    y_train = downsample['y'].values
    return X_train, y_train


def main():
    experiment = 1
    horizon = 12
    window = 6
    data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
    data = data.dropna()

    # training, horizon, offset, testing
    X = data.iloc[170:170+18+12+12+12, 1:]
    X = selectExperiment(X, experiment)
    names = X.columns
    X = X.values
    # training, horizonx2, offset, testing
    y = data.iloc[170:170+18+12+12+12+12, 1].values
    y = np.sign([y[i + horizon] - y[i] for i in range(len(y) - horizon)])
    y[y == -1] = 2
    # make differences
    X = X[12:] - X[:-12]
    y = y[12:]
    print(X, len(X), "\n", y, len(y))
    X_train, X_test, X_val, y_train, y_test, y_val = createRollingWindow(window, 12, X, y, names)
    print(X_train, X_test, y_train, y_test)
    X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    validation_results = dict((el, []) for el in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    print(X_val, y_val)
    for repeat in range(20):
        for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
            svmodel = svm.LinearSVC(C=c, dual='auto', max_iter=100000)
            svmodel.fit(X_train_downsample, y_train_downsample)
            pred = svmodel.predict(X_val)
            score = accuracy_score(y_val, pred)
            validation_results[c].append(score)
    list = []
    for key, value in validation_results.items():
        print(key, value)
        list.append([key, sum(value) / len(value)])
    list = sorted(list, key=lambda x: (x[1]))
    print(list[-1])


if __name__ == "__main__":
    main()