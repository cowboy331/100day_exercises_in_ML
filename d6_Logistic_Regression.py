# encoding: UTF-8
# 1 data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
# labelEncoder = LabelEncoder()
# X[:, 3] = labelEncoder.fit_transform(X[:, 3])  # 自变量第4列转为连续的数值变量
# onehotencoder = OneHotEncoder(categorical_features=[3])
# X = onehotencoder.fit_transform(X).toarray()  # 自变量进行独热编码
#
# # Avoiding Dummy Variable Trap
# X = X[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(X_train)
x_test=sc.fit_transform(X_test)

# 2 Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

# 3 Predicting
y_pred=classifier.predict(x_test)

# 4 Evaluating the Predection
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

