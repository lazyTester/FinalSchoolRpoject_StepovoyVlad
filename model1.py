import numpy as np
import pandas as pd

df = pd.read_csv('2019.csv', sep=',')

df["1"] = 1
X = df.iloc[:, 3:].values
# print(X)

Y = df['Score'].values.reshape((df.shape[0], 1))
# print(Y)

beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

Yhat = np.dot(X, beta)
uhat = Y - Yhat
MSE = (1/X.shape[0]) * np.sum(uhat**2)
MAPE = (100/X.shape[0]) * np.sum(np.abs(uhat)/Y)
print(beta)
print(MSE)
print(MAPE)