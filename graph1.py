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
from sklearn_lvq import GmlvqModel

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
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_train = X[:18]
    #X_val = X[18:-12]
    X_test = X[-12:]
    # scale factors
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    # create window
    X_train = np.array([X_train[j - window:j, :].flatten() for j in range(window, len(X_train))])
    #X_val = np.array([X_val[j - window:j, :].flatten() for j in range(window, len(X_val))])
    X_test = np.array([X_test[j - window:j, :].flatten() for j in range(window, len(X_test))])
    y_train = y[:18]
    #y_val = y[18:-12]
    y_test = y[-12:]
    y_train = y_train[window:]
    #y_val = y_val[window:]
    y_test = y_test[window:]

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
    try:

        svmodel = svm.LinearSVC(C=c, dual='auto', max_iter=100000)
        svmodel.fit(X_train_downsample, y_train_downsample)
        pred = svmodel.predict(X_test)
        return np.linalg.norm(svmodel.coef_[0]), pred, y_test, [1] * len(features), features, c
        return np.linalg.norm(svmodel.coef_[0]), pred, y_test, svmodel.coef_[0], features
    except Exception as e:
        print(e)
        return None


def logisticRegression1(window, c, offset, X, y, names):
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


def logisticRegression(window, c, offset, X, y, names):
    try:
        features = names * window
        validation_results = []
        for window in [1, 2, 3, 4, 5, 6]:
            X_train, X_test, y_train, y_test = createRollingWindow(window, 12, X, y, names)
            # print(X_train, X_test, y_train, y_test)
            # X_train_downsample, y_train_downsample = downsample(X_train, y_train)

            C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
            svmodel = linear_model.LogisticRegression(solver='lbfgs')
            grid = dict(C=C)
            loo = LeaveOneOut()
            grid_search = GridSearchCV(estimator=svmodel, param_grid=grid, n_jobs=-1, cv=loo, scoring='accuracy',
                                       error_score=0)
            grid_result = grid_search.fit(X_train, y_train)
            #print(grid_result.best_params_)
            #print(grid_result.best_score_)
            validation_results.append([grid_result.best_score_, grid_result.best_params_['C'], window])

        validation_results = sorted(validation_results, key=lambda x: (x[0]))
        best_params = validation_results[-1]
        #print(y_train)

        #print(validation_results[-1])
        window = best_params[2]
        c = best_params[1]
        X_train, X_test, y_train, y_test = createRollingWindow(window, offset, X, y, names)
    except:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    if len(set(y_train)) == 1:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    #X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    #list1 = np.concatenate([X_train_downsample, X_test])
    #list2 = np.concatenate([y_train_downsample, [y_test]])
    #data = {'fvec': list1.tolist(), 'lbl': list2.tolist()}
    #m4p.savemat('datafile' + str(n) + '.mat', data)
    #input()
    results = Parallel(n_jobs=-1)(
        delayed(train_lr)(X_train, y_train, X_test, y_test, features, c)
        for _ in range(100)
    )

    try:
        results_df = pd.DataFrame(results, columns=['w', 'pred', 'test', 'coef', 'features', 'c'])
        results_df = results_df.sort_values(by=['w'], ascending=False).reset_index(drop=True)[:10]
        return results_df
    except:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])


def objective(params, X, y):
    c = params[0]
    clf = svm.LinearSVC(C=c, penalty='l2', loss='squared_hinge', dual='auto', fit_intercept=True)
    score = cross_val_score(clf, X, y, cv=3).mean()
    return -score


def train_svm(X_train, y_train, X_test, y_test, features, c):
    X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    try:

        svmodel = svm.LinearSVC(C=c, dual='auto', max_iter=100000)
        svmodel.fit(X_train_downsample, y_train_downsample)
        pred = svmodel.predict(X_test)
        return 1, pred, y_test, [1] * len(features), features, c

        svmodel = GmlvqModel(prototypes_per_class=c)
        svmodel.fit(X_train_downsample, y_train_downsample)
        pred = svmodel.predict(X_test)
        return 1, pred, y_test, [1] * len(features), features, c
        return np.linalg.norm(svmodel.coef_[0]), pred, y_test, svmodel.coef_[0], features, c
    except Exception as e:
        print(e)
        return None


def SVM(window, c, offset, X, y, names, test, n):

    try:
        features = names * window
        validation_results = []
        for window in [1, 2, 3, 4, 5, 6]:
            X_train, X_test, y_train, y_test = createRollingWindow(window, 12, X, y, names)
            # print(X_train, X_test, y_train, y_test)
            # X_train_downsample, y_train_downsample = downsample(X_train, y_train)

            C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
            svmodel = svm.LinearSVC(dual='auto', max_iter=100000)
            #svmodel = GmlvqModel()
            #grid = dict(prototypes_per_class=[1, 2])
            grid = dict(C=C)
            loo = LeaveOneOut()
            grid_search = GridSearchCV(estimator=svmodel, param_grid=grid, n_jobs=-1, cv=loo, scoring='accuracy',
                                       error_score=0)
            grid_result = grid_search.fit(X_train, y_train)
            #print(grid_result.best_params_)
            #print(grid_result.best_score_)
            #validation_results.append([grid_result.best_score_, grid_result.best_params_['prototypes_per_class'], window])
            validation_results.append([grid_result.best_score_, grid_result.best_params_['C'], window])

        validation_results = sorted(validation_results, key=lambda x: (x[0]))
        best_params = validation_results[-1]
        #print(y_train)

        #print(validation_results[-1])
        window = best_params[2]
        c = best_params[1]
        X_train, X_test, y_train, y_test = createRollingWindow(window, offset, X, y, names)
    except Exception as e:
        print(e)
        input()
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    if len(set(y_train)) == 1:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])

    #X_train_downsample, y_train_downsample = downsample(X_train, y_train)
    #list1 = np.concatenate([X_train_downsample, X_test])
    #list2 = np.concatenate([y_train_downsample, [y_test]])
    #data = {'fvec': list1.tolist(), 'lbl': list2.tolist()}
    #m4p.savemat('datafile' + str(n) + '.mat', data)
    #input()
    results = Parallel(n_jobs=-1)(
        delayed(train_svm)(X_train, y_train, X_test, y_test, features, c)
        for _ in range(100)
    )

    try:
        results_df = pd.DataFrame(results, columns=['w', 'pred', 'test', 'coef', 'features', 'c'])
        results_df = results_df.sort_values(by=['w'], ascending=False).reset_index(drop=True)[:10]
        return results_df
    except:
        return pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features', 'c'])


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
        """if i == 62:
            X = X[window:] - X[:-window]
            X_train = X[:-2 * offset]
            X_test = X[-1:]
            y = y[window:]
            #scaler = StandardScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_test = scaler.transform(X_test)
            y_train = y[:-offset]
            y_test = y[-1]
            # X_test = [X_test]
            print(X_train, X_test, y_train, y_test)
            input()"""
        # make differences
        X = X[12:] - X[:-12]
        y = y[12:]
        if len(X) != len(y):
            print("XXX")
            continue

        results = SVM(window, c, offset, X, y, names, False, i)
        #results = logisticRegression(window, c, offset, X, y, names)
        if results.empty:
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