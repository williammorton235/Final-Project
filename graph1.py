def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE

from joblib import Parallel, delayed

import mat4py as m4p
# test



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


def downsample(X_train, y_train):
    traind = pd.DataFrame(X_train)
    traind['y'] = y_train
    # Separate majority and minority classes
    majority_val = int(traind.y.mode()[0])
    majority = traind[traind['y'] == majority_val]
    minority = traind[traind['y'] != majority_val]
    if len(minority) < len(traind) / 10:
        pass
        #print("Insufficient Data")
    # Downsample majority class
    df_majority_downsampled = resample(majority, replace=False, n_samples=len(minority))
    """try:
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    except:
        oversample = RandomOverSampler()
        X_train, y_train = oversample.fit_resample(X_train, y_train)"""

    downsample = pd.concat([df_majority_downsampled, minority])
    X_train = downsample.drop(['y'], axis=1).values
    y_train = downsample['y'].values
    return X_train, y_train


def createRollingWindow(window, offset, X, y, names):
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    #X = np.diff(X, axis=0)
    #print(X, len(X))
    #X = X[12:] - X[:-12]
    #print(X, len(X))
    #print(y, len(y))
    #y = y[12:]
    #print(y, len(y))
    #X_train = X[:-offset]
    #X_test = X[-window:]
    #print(X_train)
    #print(X_test)
    # scale data
    #print(X_train, '\n', X_test)
    #print(X_train.shape)
    """temp_train = pd.DataFrame(X_train, columns=names)
    temp_test = pd.DataFrame(X_test, columns=names)
    print(temp_train[[names[0]]])
    print(temp_test[[names[0]]])
    temp_scaler = MinMaxScaler()
    print(temp_scaler.fit_transform(temp_train[[names[0]]]))
    print(temp_scaler.transform(temp_test[[names[0]]]))
    input()
    for name in names:
        temp_scaler = StandardScaler()
        temp_train[name] = temp_scaler.fit(temp_train[[name]])
        temp_train[name] = temp_scaler.transform(temp_train[[name]])
        temp_test[name] = temp_scaler.transform(temp_test[[name]])
    X_train = temp_train.values
    X_test = temp_test.values
    print(X_train, '\n', X_test)
    input()"""


    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    #print(scaler.get_params())
    #print(X_train, '\n', X_test)
    #input()
    X = np.array([X[j - window:j, :].flatten() for j in range(window, len(X))])
    X = X[12:] - X[:-12]
    y = y[12:]
    #print(X)
    #print(len(X))
    #input()
    X_train = X[:-offset]
    X_test = X[-1]
    #X_train = np.array([X_train[j - window:j, :].flatten() for j in range(window, len(X_train))])
    #X_test = np.array(X_test.flatten())
    #print(X_train)
    #print(X_test)
    y_roll = y[window:]
    y_train = y_roll[:-offset]
    y_test = y_roll[-1]
    X_test = [X_test]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #print(y_train)
    #print(y_test)
    #input()
    return X_train, X_test, y_train, y_test


def createRollingWindow1(window, offset, X, y):
    X = X[window:] - X[:-window]
    X_train = X[:-2*offset]
    X_test = X[-1:]
    y = y[window:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y[:-offset]
    y_test = y[-1]
    #X_test = [X_test]
    return X_train, X_test, y_train, y_test


def crossValidation(offset, X, y):
    for lag in range(1,7):
        scores = {}
        X_train, X_test, y_train, y_test = createRollingWindow(lag, offset, X, y)
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']}
        grid = GridSearchCV(svm.SVC(), param_grid, scoring='accuracy', cv=3)
        grid.fit(X_train, y_train)
        c = grid.best_params_['C']
        clf = svm.SVC(kernel='linear', C=c)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
        scores[lag] = [sum(cv_scores) / len(cv_scores), c]
    window = max(scores, key=scores.get)
    return window, scores[window][1]


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def train_lr(X_train, y_train, X_test, y_test, features, c):
    X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    svmodel = linear_model.LogisticRegression(C=c, solver='lbfgs')
    try:
        """opt = BayesSearchCV(
            svmodel,
            {
                'C': (c/4, c*4, 'log-uniform'),
            },
            n_iter=32,
            cv=3
        )

        opt.fit(X_train_downsample, y_train_downsample)"""
        #svmodel.fit(X_train_downsample, y_train_downsample)
        #param_grid = {'C': np.arange(c / 4, c * 4, c / 4).tolist(), 'kernel': ['linear']}
        #grid = GridSearchCV(svm.SVC(), param_grid, scoring='accuracy', cv=3)
        #grid.fit(X_train, y_train)
        svmodel.fit(X_train_downsample, y_train_downsample)
        pred = svmodel.predict(X_test)
        return np.linalg.norm(svmodel.coef_[0]), pred, y_test, svmodel.coef_[0], features, c
    except Exception as e:
        print(e)
        return None


def logisticRegression(window, c, offset, X, y, names):
    features = names * window
    X_train, X_test, y_train, y_test = createRollingWindow(window, offset, X, y, names)

    if len(set(y_train)) == 1:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    # X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    # list1 = np.concatenate([X_train_downsample, X_test])
    # list2 = np.concatenate([y_train_downsample, [y_test]])
    # data = {'fvec': list1.tolist(), 'lbl': list2.tolist()}
    # m4p.savemat('datafile' + str(n) + '.mat', data)
    # input()
    results = Parallel(n_jobs=-1)(
        delayed(train_lr)(X_train, y_train, X_test, y_test, features, c)
        for _ in range(100)
    )

    try:
        results_df = pd.DataFrame(results, columns=['w', 'pred', 'test', 'coef', 'features', 'c'])
        results_df = results_df.sort_values(by=['w'], ascending=False).reset_index(drop=True)[:10]
    except:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    return results_df


def objective(params, X, y):
    c = params[0]
    clf = svm.LinearSVC(C=c, penalty='l2', loss='squared_hinge', dual='auto', fit_intercept=True)
    score = cross_val_score(clf, X, y, cv=3).mean()
    return -score


def train_svm(X_train, y_train, X_test, y_test, features, c, test):
    X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    try:
        """opt = BayesSearchCV(
            svmodel,
            {
                'C': (c/4, c*4, 'log-uniform'),
            },
            n_iter=32,
            cv=3
        )

        opt.fit(X_train_downsample, y_train_downsample)"""
        #svmodel.fit(X_train_downsample, y_train_downsample)
        if test == False:
            grid = svm.LinearSVC(C=c, dual='auto', max_iter=100000)
            #loo = LeaveOneOut()
            #C_range = np.logspace(-5, 5, 20)
            #param_grid = dict(C=C_range)
            #grid = GridSearchCV(svm.LinearSVC(dual='auto', max_iter=100000), param_grid, scoring='accuracy', cv=loo)

            grid.fit(X_train_downsample, y_train_downsample)
            pred = grid.predict(X_test)
            return 1, pred, y_test, [1] * len(features), features, c#grid.best_params_['C']
        else:

            svmodel = svm.LinearSVC(C=c, dual='auto', max_iter=100000)
            svmodel.fit(X_train_downsample, y_train_downsample)
            pred = svmodel.predict(X_test)
            return 1, pred, y_test, [1]*len(features), features, c
        return np.linalg.norm(svmodel.coef_[0]), pred, y_test, svmodel.coef_[0], features
    except Exception as e:
        print(e)
        return None


def SVM(window, c, offset, X, y, names, test, n):
    features = names * window
    X_train, X_test, y_train, y_test = createRollingWindow(window, offset, X, y, names)


    if len(set(y_train)) == 1:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    #X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    #list1 = np.concatenate([X_train_downsample, X_test])
    #list2 = np.concatenate([y_train_downsample, [y_test]])
    #data = {'fvec': list1.tolist(), 'lbl': list2.tolist()}
    #m4p.savemat('datafile' + str(n) + '.mat', data)
    #input()
    results = Parallel(n_jobs=-1)(
        delayed(train_svm)(X_train, y_train, X_test, y_test, features, c, test)
        for _ in range(100)
    )

    try:
        results_df = pd.DataFrame(results, columns=['w', 'pred', 'test', 'coef', 'features', 'c'])
        results_df = results_df.sort_values(by=['w'], ascending=False).reset_index(drop=True)[:10]
    except:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    return results_df


def main():
    training = 18
    horizon = 12
    offset = 12
    window = 6
    c = 1
    experiment = 1
    data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
    data = data.dropna()
    #data = data[:int(len(data)/2)]
    prognoza = []
    test = []
    importances = []
    clist = []
    w = []

    for i in range(len(data)-(training+horizon)):
        print(i)
        X = data.iloc[i:i+training+offset+horizon, 1:]
        X = selectExperiment(X, experiment)
        names = X.columns
        X = X.values
        y = data.iloc[i:i+training+horizon+offset+horizon, 1].values
        y = np.sign([y[i + horizon] - y[i] for i in range(len(y) - horizon)])
        y[y == -1] = 2
        #print(len(X))
        #input()
        if len(X) != len(y):
            continue

        results = SVM(window, c, offset, X, y, names, False, i)
        #results = logisticRegression(window, c, offset, X, y, names)
        if results.empty:
            #print("1x")
            continue
        #print(results["test"])
        for index, row in results.iterrows():
            weights = [abs(x) for x in row['coef']]
            total = sum(weights)
            #results['coef'][index] = [x/total for x in weights]
            importances.append([x/total for x in weights])
            prognoza.append(row['pred'])
            test.append(row['test'])
            w.append(results['w'])
            clist.append(results['c'])
        #print(prognoza, test)
        #input()

        clist1 = []
        for list in clist:
            clist1.append(list.mean())
        c = sum(clist1) / len(clist1)
        # find best constraints
        bestconstraints = pandas.DataFrame(columns=['accuracy', 'window', 'c'])
        for testwindow in [1, 2, 3, 4, 5, 6]:
            for testc in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:#, 10000, 100000]:0.00001, 0.0001,
                results = SVM(testwindow, testc, offset, X, y, names, True, i)
                #results = logisticRegression(window, c, offset, X, y, names)
                if results.empty:
                    continue
                testprognoza = []
                testtest = []
                for index, row in results.iterrows():
                    testprognoza.append(row['pred'])
                    testtest.append(row['test'])
                    bestconstraints.loc[len(bestconstraints)] = [metrics.accuracy_score(testtest, testprognoza), testwindow, testc]
        bestconstraints = bestconstraints.sort_values('accuracy', ascending=False, ignore_index=True)
        window = int(bestconstraints.iloc[0]['window'])
        c = bestconstraints.iloc[0]['c']

    prognoza = [x[0] for x in prognoza]
    print(sum(w)/len(w))
    print(prognoza.count(1.0))
    print("test", test, "\n", "preds", prognoza)
    print(metrics.classification_report(test, prognoza))
    #print(results.iloc[0]['coef'])
    importances = np.array(importances)
    print(importances)
    culm_weights = np.mean(importances, axis=0)
    print(culm_weights, names)
    f_importances(culm_weights, names)


if __name__ == "__main__":
    main()