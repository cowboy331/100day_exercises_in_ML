# encoding: UTF-8
# 1 data preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
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

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# 2 Logistic Regression Model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# 3 Predicting
y_pred = classifier.predict(x_test)

# 4 Evaluating the Predection
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# 5 Visualization
from matplotlib.colors import ListedColormap

x_set, y_set = x_train, y_train
# 画网格
X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
# contour和contourf都是画三维等高线图的，不同点在于contourf会对等高线间的区域进行填充
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
# 画坐标
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# 画散点图
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('LOGISTIC(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                     np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(X1, X2,classifier.predict(np.array([X1.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)

plt.title('LOGISTIC(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

