#1 importing the libraries
import numpy as np
import pandas as pd

#2 Importing dataset
dataset=pd.read_csv('..\datasets\Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#3 Handing the missing data
from sklearn.preprocessing import Imputer
Imputer=Imputer(missing_values='NaN',strategy="mean",axis=0)
Imputer=Imputer.fit(X[:,1:3])
X[:,1:3]=Imputer.transform(X[:,1:3])

#4 Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder_X=LabelEncoder()
X[:,0]=LabelEncoder_X.fit_transform(X[:,0])

#Creating a dummy variable
OnehotEncoder=OneHotEncoder(categorical_features=[0])
X=OnehotEncoder.fit_transform(X).toarray()
LabelEncoder_Y=LabelEncoder()
Y=LabelEncoder_Y.fit_transform(Y)

#5 Splitting the datasets into training sets and Test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
