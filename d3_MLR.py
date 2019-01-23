# encoding: UTF-8
# 1 data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])  # 自变量第4列转为连续的数值变量
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()  # 自变量进行独热编码

# Avoiding Dummy Variable Trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 2 Fitting Multiple Linear Regression Mode training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 3 Predecting the Result
Y_pred = regressor.predict(X_test)

# 4 Visualization
# plt.scatter(X_train, Y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
#
# plt.scatter(X_test, Y_test, color='red')
# plt.plot(X_test, regressor.predict(X_test), color='blue')

# plt.show()
print('X_train')
print(X_train)
print('Y_train')
print(Y_train)
print('X_test')
print(X_test)
print('Y_test')
print(Y_test)
