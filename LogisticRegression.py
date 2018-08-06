import numpy as np
from LogisticRegressionPython import load_dataset
import h5py
import matplotlib.pyplot as plt
import scipy
x_train,x_test,test_set_y,train_set_y,classes=load_dataset()
m_train=x_train.shape[0]
m_test=x_test.shape[0]
num_px=x_train.shape[1]
print("number of training examples:"+str(m_train))
print("number of test examples:"+str(m_test))
print("size of each image:"+str(num_px))
print("x_train:"+str(x_train))
print("x_test :"+str(x_test))
print("train_set_y:"+str(train_set_y.shape))
print("test_set_y:"+str(test_set_y.shape))
print("train_set_x:"+str(x_train.shape))
print("test_set_x:"+str(x_test.shape))

x_train_flatten=x_train.reshape(x_train.shape[0],-1).T
x_test_flatten=x_test.reshape(x_test.shape[0],-1).T
print("x_train_flatten shape:"+str(x_train_flatten.shape))
print("x_test_flatten shape:"+str(x_test_flatten.shape))
print("train_set_y shape:"+str(train_set_y.shape))
print("test_set_y shape:"+str(test_set_y.shape))

train_set_x=x_train_flatten/255.

print("train_set_x shape:"+str(train_set_x.shape))

test_set_x=x_test_flatten/255.

print("test_set_x shape:"+str(test_set_x.shape))
print("sanity check after reshaping:"+str(x_train_flatten[0:5,0]))
print("train_set_x shape"+str(train_set_x.shape))





def sigmoid(z):
    s=None
    s=1/(1+np.exp(-z))
    return s
print("sigmoid([0,2])="+str(sigmoid(np.array([0,2]))))

def intialize_zeros(dim):
    w = np.zeros((dim,1))
    b=0.0
    assert(w.shape ==(dim,1))
    assert (isinstance(b,float) or isinstance(b,int))
    return w,b
dim=2
w,b=intialize_zeros(dim)
print("w="+str(w))
print("b="+str(b))

def propagate(w,b,X,Y):
    m=X.shape[1]
    a=sigmoid(np.dot(w.T, X)+b)
    cost= - 1/m *np.sum ((Y*np.log(a))+(1-Y)* np.log(1-a))
    dw = 1/m * np.dot(X,(a-Y).T)
    db = 1/m * np.sum(a-Y)
    assert(dw.shape==w.shape)
    assert (db.dtype==float)
    cost= np.squeeze(cost)
    assert(cost.shape==())
    grad ={"dw": dw,
               "db":db}
    return grad,cost
w,b,X,Y=np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
grad, cost=propagate(w,b,X,Y)
print("dw="+str(grad["dw"]))
print("db="+str(grad["db"]))
print("cost="+str(cost))
def update(w,b,X,Y,iterations,learning_rate,print_cost=False):
    costs=[]
    for i in range(iterations):
         grad, cost = propagate(w, b, X, Y)
         dw = grad["dw"]
         db = grad["db"]

         w = w - learning_rate * dw
         b = b - learning_rate * db
         # cost
         if i % 100 == 0:
             costs.append(cost)
         # After every 100 training set examples
         if print_cost and i % 100 == 0:
             print("cost after every 100 training example %i : %f"%(i,cost))
    param = {"w":w ,
             "b":b}

    grad={"dw":dw , "db":db}
    return param,grad,costs
param, grads, costs = update(w, b, X, Y,iterations= 1000, learning_rate = 0.009, print_cost = False)

print ("w = " + str(param["w"]))
print ("b = " + str(param["b"]))
print ("dw = " + str(grad["dw"]))
print ("db = " + str(grad["db"]))


def pred(w, b, X):
    m= X.shape[1]
    Y_pred= np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A= sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0,i]>=0.5:
            Y_pred[0,i]=1
        else:
            Y_pred[0,i]=0
    assert(Y_pred.shape==(1,m))
    return Y_pred
print ("predictions = " + str(pred(w, b, X)))

def model(X_train,X_test,Y_train,Y_test,iterations=1000,learning_rate=0.5,print_cost=False):
    w,b = intialize_zeros(X_train.shape[0])
    param,grad,costs=update(w,b,X_train,Y_train,iterations,learning_rate,print_cost)
    w= param["w"]
    b=param["b"]
    Y_pred_test= pred(w,b,X_test)
    Y_pred_train=pred(w,b,X_train)
    print("train accuracy:{}%".format(100-np.mean(np.abs(Y_pred_train - Y_train))*100))
    print("test accuracy:{}%".format(100-np.mean(np.abs(Y_pred_test - Y_test))* 100))
    d = {"costs": costs,
         "Y_pred_test": Y_pred_test,
         "Y_pred_train": Y_pred_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "iterations":iterations}
    return d
d=model(train_set_x,test_set_x,train_set_y,test_set_y,iterations=2000,learning_rate=0.005,print_cost=True)

learning_rates=[0.1,0.001]
models={}
for i in learning_rates:
    print("learning rate is:"+str(i))
    models[str(i)]=model(train_set_x,test_set_x,train_set_y,test_set_y,iterations=1500,learing_rate=0.005,print_cost=False)
    print('\n'+-------------------+'\n')

for i in learning_rates:
    plt.plot(np.sqeeze(models[str(i)]["costs"]),label=str(models[str(i)]["learning_rate"]))
plt.ylabel('costs')
plt.xlabel('iterations')

legend=plt.legend(loc='upper center',shadow=True)
frame=legend.get_frame()
frame.set_facecolor('0.90')
plt.show()



#my_image='/home/shruti/Downloads/train_catvnoncat.h5'

#fname="/image" +my_image
#image = np.array(ndimage.imread(fname, flatten=False))
#my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
#my_predicted_image = pred(d["w"], d["b"], my_image)

#plt.imshow(image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")









