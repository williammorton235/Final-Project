import pandas
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as test
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import resample
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE


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
    #df_majority_downsampled = resample(majority, replace=False, n_samples=len(minority))
    try:
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    except:
        oversample = RandomOverSampler()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    #downsample = pd.concat([df_majority_downsampled, minority])
    #X_train = downsample.drop(['y'], axis=1).values
    #y_train = downsample['y'].values
    return X_train, y_train


def createRollingWindow(window, offset, X, y):
    X_train = X[:-offset]
    X_test = X[-window:]
    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.array([X_train[j - window:j, :].flatten() for j in range(window, len(X_train))])
    X_test = np.array(X_test.flatten())
    y_roll = y[window:]
    y_train = y_roll[:-offset]
    y_test = y_roll[-1]
    X_test = [X_test]
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


def SVM(window, c, offset, X, y, names):
    features = names*window
    # create rolling window of size 6
    X_train, X_test, y_train, y_test = createRollingWindow(window, offset, X, y)
    # print(X_train, X_test, y_train, y_test)
    results = pd.DataFrame(columns=['w', 'pred', 'test', 'coef', 'features'])
    for i in range(100):
        if len(set(y_train)) == 1:
            return results
        else:
            X_train_downsample, y_train_downsample = downsample(X_train, y_train)
            # print(X_train_downsample, y_train_downsample)
            # print(X_train, X_test, y_train, y_test)
            #param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
            #grid = GridSearchCV(svm.SVC(), param_grid, scoring='accuracy', cv=3)
            #grid.fit(X_train, y_train)
            #best_params = grid.best_params_
            svmodel = svm.SVC(kernel='linear', C=c)
            # svmodel = linear_model.LogisticRegression()
            svmodel.fit(X_train_downsample, y_train_downsample)
            pred = svmodel.predict(X_test)
        results.loc[len(results)] = [np.linalg.norm(svmodel.coef_[0]), pred, y_test, svmodel.coef_[0], features]
    results = results.sort_values(by=['w'], ascending=False).reset_index(drop=True)[:10]
    return results


def main():
    training = 18
    horizon = 12
    offset = 11
    window = 3
    c = 1
    experiment = 1
    data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
    data = data.dropna()
    prognoza = []
    test = []
    importances = []

    for i in range(len(data)-(training+horizon)):
        X = data.iloc[i:i+training+offset, 1:]
        X = selectExperiment(X, experiment)
        names = X.columns
        X = X.values
        y = data.iloc[i:i+training+horizon+offset, 1].values
        y = np.sign([y[i + horizon] - y[i] for i in range(len(y) - horizon)])

        try:
            pass
            #window, c = crossValidation(offset, X, y)
        except:
            continue


        results = SVM(window, c, offset, X, y, names)
        if results.empty:
            continue
        for index, row in results.iterrows():
            weights = [abs(x) for x in row['coef']]
            total = sum(weights)
            #results['coef'][index] = [x/total for x in weights]
            importances.append([x/total for x in weights])
            prognoza.append(row['pred'])
            test.append(row['test'])

        # find best constraints
        bestconstraints = pandas.DataFrame(columns=['accuracy', 'window', 'c'])
        for testwindow in [1, 2, 3, 4, 5, 6]:
            for testc in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                results = SVM(testwindow, testc, offset, X, y, names)
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

    print(prognoza, "\n", test)
    print(metrics.classification_report(test, prognoza))
    #print(results.iloc[0]['coef'])
    importances = np.array(importances)
    print(importances)
    culm_weights = np.mean(importances, axis=0)
    print(culm_weights, names)
    f_importances(culm_weights, names)


if __name__ == "__main__":
    main()