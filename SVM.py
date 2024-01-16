import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import feature_selection
from sklearn import metrics
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn import datasets
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV


features = ['Date', 'EXUSEU_Avg', 'US_GDP', 'US_IR', 'US_CPI', 'US_M3', 'US_BOP',
       'US_DM3Index', 'US_UC', 'EU_GDP', 'EU_OR_Avg', 'EU_CPI', 'EU_M3',
       'EU_BOP', 'EU_DM3Index', 'EU_UC', 'OIL_Avg', 'Diff']


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def data_preprocessing1(data, horizon):
    data["Diff"] = data.EXUSEU_Avg.diff(-horizon)
    data["y"] = data["Diff"].apply(lambda x: -1 if x > 0 else 1)
    data = data.dropna()
    print(data.head(20))
    return data


def split_data(data, experiment):
    if experiment == 1:
        data = data.drop(['Date', 'Diff'], axis=1)
    elif experiment == 2.1:  # simple sum
        data = data.drop(['Date', 'EXUSEU_Avg', 'US_GDP', 'US_CPI', 'US_BOP', 'US_DM3Index', 'US_UC', 'EU_GDP', 'EU_CPI', 'EU_BOP', 'EU_DM3Index', 'EU_UC', 'OIL_Avg', 'Diff'], axis=1)
    elif experiment == 2.2:  # divisia
        data = data.drop(['Date', 'EXUSEU_Avg', 'US_GDP', 'US_IR', 'US_CPI', 'US_M3', 'US_BOP', 'EU_GDP', 'EU_OR_Avg', 'EU_CPI', 'EU_M3', 'EU_BOP', 'OIL_Avg', 'Diff'], axis=1)
    elif experiment == 3:
        data = data.drop(['Date', 'EXUSEU_Avg', 'US_GDP', 'US_CPI', 'US_BOP', 'EU_GDP', 'EU_CPI', 'EU_BOP', 'OIL_Avg', 'Diff'], axis=1)
    #print(data.head(20))
    #print(data.columns)
    return data


def data_preprocessing2(data, window):
    print(data.columns)
    target = data['y']
    data = data.drop(['y'], axis=1)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    print(data)
    output = pd.DataFrame(columns=['data', 'y'])
    for i in range(window, len(data), 1):
        hold = data.iloc[i-window:i, :].to_numpy()
        holdl = numpy.concatenate(hold, axis=0)
        print(holdl, "***")
        y = target[i]
        output.loc[len(output)] = [holdl, y]
    return output


def function1(data, horizon):
    data["Diff"] = data.EXUSEU_Avg.diff(-horizon)
    data["y"] = data["Diff"].apply(lambda x: -1 if x > 0 else 1)
    data.to_csv('Data//testdf1.csv')
    data = data.dropna()
    target = data["y"]
    data.to_csv('Data//testdf.csv')
    data = data.drop(["Diff"], axis=1)
    return data, target


def function2(data, experiment):
    if experiment == 1:
        data = data.drop(['Date'], axis=1)
    elif experiment == 2.1:  # simple sum
        data = data.drop(['Date', 'EXUSEU_Avg', 'US_GDP', 'US_CPI', 'US_BOP', 'US_DM3Index', 'US_UC', 'EU_GDP', 'EU_CPI', 'EU_BOP', 'EU_DM3Index', 'EU_UC', 'OIL_Avg'], axis=1)
    elif experiment == 2.2:  # divisia
        data = data.drop(['Date', 'EXUSEU_Avg', 'US_GDP', 'US_IR', 'US_CPI', 'US_M3', 'US_BOP', 'EU_GDP', 'EU_OR_Avg', 'EU_CPI', 'EU_M3', 'EU_BOP', 'OIL_Avg'], axis=1)
    elif experiment == 3:
        data = data.drop(['Date', 'US_GDP', 'US_CPI', 'US_BOP', 'EU_GDP', 'EU_CPI', 'EU_BOP', 'OIL_Avg'], axis=1)
    return data


def function31(train, test, target, window):
    print(train)
    print(test)
    cols = train.columns.tolist()
    new_cols = []
    for i in range(window):
        print(i)
        new_cols = new_cols + [s + str(i) for s in cols]
    print(new_cols)
    scaler = preprocessing.StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    train = pd.DataFrame(train, columns=[cols])
    test = pd.DataFrame(test, columns=[cols])
    new_train = pd.DataFrame(columns=['vector', 'y'])
    new_test = pd.DataFrame(columns=['vector', 'y'])
    for i in range(window, len(train), 1):
        vector = train.iloc[i - window:i, :].to_numpy()
        vector = numpy.concatenate(vector, axis=0)
        y = target[i]
        new_train.loc[len(new_train)] = [vector, y]
    new_train[new_cols] = pd.DataFrame(new_train.vector.tolist(), index=new_train.index)
    new_train = new_train.drop(['vector'], axis=1)
    for i in range(window, len(test), 1):
        vector = test.iloc[i - window:i, :].to_numpy()
        vector = numpy.concatenate(vector, axis=0)
        print(vector, "***")
        y = target[i]
        new_test.loc[len(new_test)] = [vector, y]
    new_test[new_cols] = pd.DataFrame(new_test.vector.tolist(), index=new_test.index)
    new_test = new_test.drop(['vector'], axis=1)
    return new_train, new_test


def function32(train, test, target, window):
    cols = train.columns.tolist()
    new_cols = []
    for i in range(window):
        new_cols = new_cols + [s + str(i) for s in cols]

    scaler = preprocessing.StandardScaler()
    train_values = scaler.fit_transform(train.values)
    test_values = scaler.transform(test.values)
    #train_values = train.values
    #test_values = test.values

    train = pd.DataFrame(train_values, columns=cols)
    test = pd.DataFrame(test_values, columns=cols)

    new_train = pd.DataFrame(columns=['vector', 'y'])
    new_test = pd.DataFrame(columns=['vector', 'y'])
    for i in range(window, len(train), 1):
        vector = train.iloc[i - window:i, :].to_numpy()
        vector = numpy.concatenate(vector, axis=0)
        y = target[i-1]
        new_train.loc[len(new_train)] = [vector, y]
    new_train[new_cols] = pd.DataFrame(new_train.vector.tolist(), index=new_train.index)
    new_train = new_train.drop(['vector'], axis=1)
    for j in range(window, len(test), 1):
        vector = test.iloc[j - window:j, :].to_numpy()
        vector = numpy.concatenate(vector, axis=0)
        y = target[j+i]
        new_test.loc[len(new_test)] = [vector, y]
    new_test[new_cols] = pd.DataFrame(new_test.vector.tolist(), index=new_test.index)
    new_test = new_test.drop(['vector'], axis=1)
    return new_train, new_test


def function3(data, target, window):
    cols = data.columns.tolist()[:-1]
    print(cols)
    for i in range(1, window, 1):
        for col in cols:
            data[col+str(i+1)] = data[col].shift(-i)
    data["y"] = data["y"].shift(-window)
    data = data.dropna()
    print(data.head(20))
    data.to_csv('Data//testdf.csv')
    return data


def recursive_feature_selection():
    test = 0
    data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
    print(data.columns)
    data = data_preprocessing(data)
    data = split_data(data)
    results = pd.DataFrame(columns=['w', 'coef', 'names', 'accuracy'])
    # Split data
    for start in range(0, len(data) - (len(data) % train), train):
        set = data[start:start + (train + test)]
        traind = data[start:start + train]
        testd = data[start + train:start + (train + test)]
        for i in range(100):
            # print(traind)
            # Separate majority and minority classes
            majority_val = int(traind.y.mode()[0])
            majority = traind[traind['y'] == majority_val]
            minority = traind[traind['y'] != majority_val]
            if len(minority) < len(traind) / 4:
                continue
            # Downsample majority class
            df_majority_downsampled = resample(majority, replace=False, n_samples=len(minority))
            downsample = pd.concat([df_majority_downsampled, minority])
            X_train = downsample.drop(['y'], axis=1)
            y_train = downsample['y']
            # X_test = testd.drop(['y'], axis=1)
            # y_test = testd['y']
            # Scale inputs
            cols = X_train.columns
            scaler = preprocessing.StandardScaler()
            X_train = scaler.fit_transform(X_train)
            # X_test = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=[cols])
            # X_test = pd.DataFrame(X_test, columns=[cols])
            # Run FS
            estimator = svm.SVC(kernel='linear')
            selector = feature_selection.RFE(estimator, n_features_to_select=1, step=1)
            selector = selector.fit(X_train, y_train)
            print(X_train.columns)
            print(selector.support_)
            print(selector.ranking_)


def logistic_regression(experiment, train, test, horizon, window):
    data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
    data, target = function1(data, horizon)
    data = function2(data, experiment)
    results = pd.DataFrame(columns=['w', 'coef', 'names', 'accuracy', 'different'])
    # Split data
    print(data)
    n = 0
    for start in range(0, len(data) - (len(data) % test), test):
        # set = data[start:start+(train+test)]
        traind = data[start:start + train]
        testd = data[start + train:start + (train + test)]
        targetd = target[start:start + (train + test)].reset_index(drop=True)
        traind, testd = function3(traind, testd, targetd, window)
        print(traind)
        for i in range(100):
            # Separate majority and minority classes
            majority_val = int(traind.y.mode()[0])
            majority = traind[traind['y'] == majority_val]
            minority = traind[traind['y'] != majority_val]
            if len(minority) < len(traind) / 10:
                print("Insufficient Data")
                continue
            # Downsample majority class
            df_majority_downsampled = resample(majority, replace=False, n_samples=len(minority))
            downsample = pd.concat([df_majority_downsampled, minority])
            X_train = downsample.drop(['y'], axis=1)
            y_train = downsample['y']
            X_test = testd.drop(['y'], axis=1)
            y_test = testd['y']
            print(X_train, y_train, X_test, y_test)
            # Run SVM
            lr = linear_model.LogisticRegression()
            print(X_train, y_train)
            lr.fit(X_train, y_train)
            predictions = lr.predict(X_test)
            print("Actual Values:", y_test)
            print("Predicted Values:", predictions)
            # print(grid.best_params_)
            print(metrics.classification_report(y_test, predictions))
            #input()
            #results.loc[len(results)] = [np.linalg.norm(lr.coef_[0]), lr.coef_[0], X_train.columns, metrics.accuracy_score(y_test, predictions) * 100]
        input()
    #results = results.sort_values(by=['w'], ascending=False).reset_index(drop=True)
    #print(results['accuracy'].mean())
    f_importances(results.iloc[0]['coef'], results.iloc[0]['names'])


def vector(experiment, train, test, horizon, window):
    data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
    data, target = function1(data, horizon)
    data = function2(data, experiment)
    results = pd.DataFrame(columns=['w', 'coef', 'names', 'accuracy', 'different'])
    # Split data
    print(data)
    n = 0
    for start in range(0, len(data)-(len(data) % test), test):
        #set = data[start:start+(train+test)]
        traind = data[start:start+train]
        testd = data[start+train:start+(train+test)]
        targetd = target[start:start+(train+test)].reset_index(drop=True)
        traind, testd = function3(traind, testd, targetd, window)
        print(traind)
        for i in range(100):
            # Separate majority and minority classes
            majority_val = int(traind.y.mode()[0])
            majority = traind[traind['y'] == majority_val]
            minority = traind[traind['y'] != majority_val]
            if len(minority) < len(traind) / 10:
                print("Insufficient Data")
                break
            # Downsample majority class
            df_majority_downsampled = resample(majority, replace=False, n_samples=len(minority))
            downsample = pd.concat([df_majority_downsampled, minority])
            X_train = downsample.drop(['y'], axis=1)
            y_train = downsample['y']
            X_test = testd.drop(['y'], axis=1)
            y_test = testd['y']
            print(X_train, y_train, X_test, y_test)
            # defining parameter range
            param_grid = {'C': [x for x in range(1,15,4)], 'kernel': ['linear']}
            grid = GridSearchCV(svm.SVC(), param_grid, scoring='accuracy', cv=3)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            svc = svm.SVC(kernel='linear', C=best_params['C'], class_weight='balanced')
            svc.fit(X_train, y_train)
            predictions = svc.predict(X_test)
            different = len(set(predictions))
            print("Actual Values:", y_test)
            print("Predicted Values:", predictions)
            print(grid.best_params_)
            print(metrics.classification_report(y_test, predictions))
            results.loc[len(results)] = [np.linalg.norm(svc.coef_[0]), svc.coef_[0], X_train.columns, metrics.accuracy_score(y_test, predictions)*100, different]
        results = results.sort_values(by=['w'], ascending=False).reset_index(drop=True)[0:10]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(results)
        input()
    results = results.sort_values(by=['w'], ascending=False).reset_index(drop=True)
    print(results)
    print(results['accuracy'].mean())
    return results
    f_importances(results.iloc[0]['coef'], results.iloc[0]['names'])


def test_svm(experiment, train, test, horizon, window):
    data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
    data, target = function1(data, horizon)
    data = function2(data, experiment)
    data = function3(data, target, window)
    results = pd.DataFrame(columns=['w', 'coef', 'names', 'accuracy', 'different'])
    # Split data
    print(data)
    n = 0
    for start in range(0, len(data)-(len(data) % test), test):
        #results = pd.DataFrame(columns=['w', 'coef', 'names', 'accuracy', 'different'])
        #set = data[start:start+(train+test)]
        traind = data[start:start+train]
        testd = data[start+train:start+(train+test)]
        #traind = data[55:84]
        #testd = data[84:101]
        print(traind)
        print(testd)
        print(traind['y'])
        print(testd['y'])
        if traind.empty or testd.empty:
            break
        for i in range(100):
            # Separate majority and minority classes
            majority_val = int(traind.y.mode()[0])
            majority = traind[traind['y'] == majority_val]
            minority = traind[traind['y'] != majority_val]
            if len(minority) < len(traind) / 10:
                print("Insufficient Data")
                break
            # Downsample majority class
            df_majority_downsampled = resample(majority, replace=False, n_samples=len(minority))
            downsample = pd.concat([df_majority_downsampled, minority])
            X_train = downsample.drop(['y'], axis=1)
            y_train = downsample['y']
            X_test = testd.drop(['y'], axis=1)
            y_test = testd['y']
            # scale inputs
            cols = X_train.columns.to_list()
            scaler = preprocessing.StandardScaler()
            train_values = scaler.fit_transform(X_train.values)
            test_values = scaler.transform(X_test.values)
            # train_values = X_train.values
            # test_values = X_test.values
            X_train = pd.DataFrame(train_values, columns=cols)
            X_test = pd.DataFrame(test_values, columns=cols)
            print(X_train, y_train, X_test, y_test)
            # defining parameter range
            param_grid = {'C': [0.1, 1, 5, 10, 100, 1000], 'class_weight': ['balanced']}
            #grid = GridSearchCV(svm.LinearSVC(), param_grid, scoring='accuracy', cv=3)
            #grid.fit(X_train, y_train)
            #best_params = grid.best_params_
            svc = svm.LinearSVC()
            svc.fit(X_train, y_train)
            predictions = svc.predict(X_test)
            different = len(set(predictions))
            print("Actual Values:", y_test)
            print("Predicted Values:", predictions)
            #print(grid.best_params_)
            print(metrics.classification_report(y_test, predictions))
            #results.loc[len(results)] = [np.linalg.norm(svc.coef_[0]), svc.coef_[0], X_train.columns, metrics.accuracy_score(y_test, predictions)*100, different]
            results.loc[len(results)] = [1, 1, X_train.columns, metrics.accuracy_score(y_test, predictions)*100, different]
        #results = results.sort_values(by=['w'], ascending=False).reset_index(drop=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(results)
    return
    #results = results.sort_values(by=['w'], ascending=False).reset_index(drop=True)
    print(results)
    print(results['accuracy'].mean())
    print(results['different'].value_counts())
    return results
    f_importances(results.iloc[0]['coef'], results.iloc[0]['names'])


def main():
    test_svm(3, 24, 18, 12, 6)
    return
    experiment = 1
    train = [30]
    test = 24
    horizon = [6, 12, 24, 36]
    results = pd.DataFrame(columns=['w', 'coef', 'names', 'accuracy'])
    for x in train:
        for y in horizon:
            results = pd.concat([results, vector(experiment, x, test, y)])
    f_importances(results.iloc[0]['coef'], results.iloc[0]['names'])


if __name__ == "__main__":
    main()
