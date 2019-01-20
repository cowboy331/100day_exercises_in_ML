#encoding: UTF-8
# 1 importing the libraries
import pandas as pd

# 2 Importing dataset
dataset = pd.read_csv('..\datasets\Data.csv')
X = dataset.iloc[:, :-1].values     #自变量选取到倒数第2列的所有数据
Y = dataset.iloc[:, 3].values       #因变量选取第四列的所有数据
print("step 2: Importing dataset")
print("X")
print(X)
print("Y")
print(Y)

# 3 Handing the missing data
from sklearn.preprocessing import Imputer

Imputer = Imputer(missing_values='NaN', strategy="mean", axis=0)    #对于缺失值，用均值替换，指定数轴为列
Imputer = Imputer.fit(X[:, 1:3])    #对2-4列，应用imputer
X[:, 1:3] = Imputer.transform(X[:, 1:3])    #对2-4列，将填补后的数据代替原空缺数据
print("--------------------------------------")
print("step 3:Handing the missing data")
print("step 2")
print("X")
print(X)


# 4 Encoding categorical data对分类特征的数据进行处理
'''
独热编码（哑变量 dummy variable）
是因为大部分算法是基于向量空间中的度量来进行计算的，为了使非偏序关系的变量取值不具有偏序性，并且到圆点是等距的。使用one-hot编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。将离散型特征使用one-hot编码，会让特征之间的距离计算更加合理。离散特征进行one-hot编码后，编码后的特征，其实每一维度的特征都可以看做是连续的特征。就可以跟对连续型特征的归一化方法一样，对每一维特征进行归一化。比如归一化到[-1,1]或归一化到均值为0,方差为1。
为什么特征向量要映射到欧式空间？
将离散特征通过one-hot编码映射到欧式空间，是因为，在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的，而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间。
LabelEncoder()
将转换成连续的数值型变量。即是对不连续的数字或者文本进行编号。
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])

# Creating a dummy variable
OnehotEncoder = OneHotEncoder(categorical_features=[0])
X = OnehotEncoder.fit_transform(X).toarray()
LabelEncoder_Y = LabelEncoder()
Y = LabelEncoder_Y.fit_transform(Y)
print("----------------------------------------")
print("step 4:encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

# 5 Splitting the datasets into training sets and Test sets
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print('-----------------------------------------')
print('step 5:Splitting the datasets into training sets and Test sets')
print('X train:')
print(X_train)
print("X test")
print(X_test)
print('Y train:')
print(Y_train)
print("Y test")
print(Y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print('--------------------------------')
print('Step 6:Feature Scaling')
print('X train:')
print(X_train)
print("X test")
print(X_test)
