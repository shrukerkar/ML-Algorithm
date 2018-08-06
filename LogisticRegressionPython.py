import numpy as np

from sklearn.linear_model import LogisticRegression
import h5py

#Load dataset

trainfile = '/home/shruti/Downloads/train_catvnoncat.h5'
testfile = '/home/shruti/Downloads/test_catvnoncat.h5'
def load_dataset():
    train_dataset = h5py.File(trainfile, "r")
    test_dataset = h5py.File(testfile, "r")

    x_train=np.array(train_dataset['train_set_x'][:])     #train set feature
    y_train=np.array(train_dataset['train_set_y'][:])             #train set label
    x_test=np.array(test_dataset['test_set_x'][:])             #test set feature
    y_test=np.array(test_dataset['test_set_y'][:])  # test set label

    classes=np.array(test_dataset["list_classes"][:]) #list of classes
    y_train=y_train.reshape((1,y_train.shape[0]))
    y_test=y_test.reshape((1,y_test.shape[0]))
    return y_train,y_test,x_train,x_test,classes





#x_train=np.reshape(np.array(train_dataset['train_set_x'][:]), (np.array(train_dataset['train_set_x'][:]).shape[0], -1))
#x_data_train = np.transpose(x_train)
#y_data_train = (np.array([train_dataset['train_set_y'][:]]))
#x_test=np.reshape(np.array(test_dataset['test_set_x'][:]), (np.array(test_dataset['test_set_x'][:]).shape[0], -1))
#x_data_test = np.transpose(x_test)
#y_data_test = (np.array([test_dataset['test_set_y'][:]]))

#x_data_train = x_data_train/255.
#print(x_data_train)
#x_data_test = x_data_test/255.
#print(x_data_test)
#LR = LogisticRegression(C=1000.0, random_state=0)

#LR.fit(x_data_train.T, y_data_train.T.ravel())

#Y_prediction = LR.predict(x_data_test.T)
#Y_prediction_train = LR.predict(x_data_train.T)

#print(LR.coef_)
#print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_data_train)) * 100))
#print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - y_data_test)) * 100))

