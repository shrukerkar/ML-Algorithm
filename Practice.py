from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib
# matplotlib.use("agg")
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Load dataset
iris = load_iris()
print(iris.data)
#print(iris.data.shape)
#print(iris.feature_names)
#print(iris.target)

featuresAll=[]
features= iris.data[:,[0,1,2,3]]
print(features.shape)

targets=iris.target
targets.reshape(targets.shape[0],-1)
print(targets.shape)
for obs in features:
    featuresAll.append([obs[0]+obs[1]+obs[2]+obs[3]])
print(featuresAll)
#X=iris.data[:,0]
#y=iris.target[:,0:1]
# split data into training and test data.
#train_X, test_X, train_y, test_y = train_test_split(X, y,train_size=0,test_size=0.5,random_state=123)
#print(train_y)
#print(test_y)
plt.scatter(featuresAll,targets,color="blue",alpha='0.1')
plt.rcParams['figure.figsize']=[10,8]
plt.title('iris dataset plot')
plt.xlabel('features')
plt.ylabel('targets')
plt.show()
featuresAll=[]
targets=[]
for feature in features:
    featuresAll.append(feature[2])
    targets.append(feature[3])
try:
    groups=('Iris-setosa','Iris-versicolor','Iris-virginica')
    colors=('blue','green','red')
    data=((featuresAll[:50],targets[:50]),(featuresAll[50:100],targets[50:100]),(featuresAll[100:150],targets[100:150]))
    #print(data)
    for item,color,groups in zip(data,colors,groups):
        x0,y0=item
        plt.scatter(x0,y0,color=color,alpha=1)
        plt.title('iris dataset plot')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
#    plt.draw()
    plt.show()
    plt.show(block=True)
except Exception as e:
    print(e)

model=LinearRegression(fit_intercept=True)
#print(model)
X=np.asarray(featuresAll)
X1=X[:,np.newaxis]
#print(X1)
#print(X1.shape)
y1=iris.target
#print(y1)
#print(y1.shape)
model.fit(X1,y1)
model.coef_
model.intercept_
Xfit=np.random.randint(8,size=(150))
Xfit.astype(float)
Xfit=Xfit[:,np.newaxis]
#print(Xfit.shape)
yfit=(model.predict(Xfit))
yfit.shape
plt.scatter(X1,y1)
plt.plot(Xfit,yfit)
plt.show()
iris=pd.read_csv('Iris.csv')
iris.head(5)
iris.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')
plt.show()
iris.plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm')
plt.show()
X2=iris.iloc[:,0:5]
y2=iris['Species']
X_train,X_test,y_train,y_test=train_test_split(X2,y2,test_size=0.3,random_state=0)
print("X_train",X_train)
print("X_test",X_test)
print("y_train",y_train)
print("y_test",y_test)
model=GaussianNB()
model=model.fit(X_train,y_train)
y_model=model.predict(X_test)
#print(y_model)
print("accuracy score is %f"%accuracy_score(y_test, y_model))


#import PyQt4
#import matplotlib
#matplotlib.use('qt4agg')
#import matplotlib.pyplot as plt
#plt.show()