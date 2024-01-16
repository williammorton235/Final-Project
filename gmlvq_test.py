# prepare sample data in the form of data frame with cols of timesteps (x) and values (y)
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Data//Data_EuroDollar_Avg.csv')
print(data.rolling(window=18).apply(lambda x: pd.DataFrame(x[-18:])))
input()
X = data.iloc[:50, 1:-1].values
y = data.iloc[:56, 1].values
y = np.sign(np.diff(y, 6))
print(X, y)

# create rolling window of size 3
window = 6
X_roll = np.array([X[i-window:i, :].flatten() for i in range(window, len(X))])
y_roll = y[window:]
X_train, X_test, y_train, y_test = train_test_split(X_roll, y_roll, test_size=0.33)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# train an svm model, consider further tuning parameters for lower MSE
svmodel = svm.SVC(kernel='linear', C=10)
print(X_train, y_train)
svmodel.fit(X_train, y_train)

# specify timesteps for forecast, eg for all series + 12 months ahead
nd = np.arange(len(X), len(X)+18)

prognoza = svmodel.predict(X_test)
print(prognoza, "\n", y_test)
print(metrics.classification_report(y_test, prognoza))

# compute forecast for all the 12 months
"""prognoza = []
for i in range(0, len(nd)):
    X_test = (X[-window:, :].flatten())
    print(X_test)
    yhat = svmodel.predict([X_test])
    prognoza.append(yhat[0])
print(prognoza, "\n", y_roll[24:36])
# plot the results
import matplotlib.pyplot as plt
plt.plot(y_roll, color = 'blue', label = 'Real data')
plt.plot(prognoza, color = 'red', label = 'Predicted data')
plt.title('SVM-based time series forecasting')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()"""
