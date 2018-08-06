import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Iris.csv')
#print(dataset)
dataset = dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
#print(dataset.head())
#print(dataset[:10])
#plt.scatter(dataset.SepalLengthCm, dataset.SepalWidthCm)
#plt.title('SepalLengthCm Vs SepalWidthCm');
#plt.xlabel('SepalLengthCm');
#plt.ylabel('SepalWidthCm');
#plt.show()
#plt.scatter(dataset.PetalLengthCm, dataset.PetalWidthCm)
#plt.title('PetalLengthCm Vs PetalWidthCm');
#plt.xlabel('PetalLengthCm');
#plt.ylabel('PetalWidthCm');
#plt.show()
x = dataset.iloc[:,:2].values
print(x)
y = dataset.iloc[:,3:].values
print(y)

