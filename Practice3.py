from numpy import *
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
iris=datasets.load_iris()
data=iris["data"]
#print(data)
iris=pd.read_csv("Iris.csv")
#print(iris.head(5))
#print(iris.info)
train,test=train_test_split(iris,test_size=0.3)
#print(train.shape)
#print(test.shape)
x_train=train[['SepalLengthCm','SepalWidthCm']]
y_train=train.Species
#print(x_train)
#print(y_train)
x_test=test[['SepalLengthCm','SepalWidthCm']]
y_test=test.Species
#print(x_test)
#print(y_test)
#odel=LinearRegression()
#odel.fit(x_train,y_train)
#rediction=model.predict(x_test)

