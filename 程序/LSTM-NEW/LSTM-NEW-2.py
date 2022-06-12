import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import pandas as pd
import os
import keras.callbacks
import matplotlib.pyplot as plt
import pandas as pd

f = '11.csv' #读取数据文件名
N = 2  #训练次数
I = 100
O = 20
# #设定为自增长    强制使用gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# KTF.set_session(session)

def create_dataset(data,n_predictions,n_next): #n_predictions：输入数据个数    n_next：输出数据个数
    '''
    对数据进行处理
    '''
    dim = data.shape[1]  #读取data的列数 （[0]行数）     data是一个二维数组——n行两列
    train_X, train_Y = [], []  #创建X Y
    for i in range(data.shape[0]-n_predictions-n_next-1):
        a = data[i:(i+n_predictions),:]
        train_X.append(a)
        tempb = data[(i+n_predictions):(i+n_predictions+n_next),:]
        b = []
        for j in range(dim):
            for k in range(len(tempb)):
                b.append(tempb[k,j])
        train_Y.append(b)
    train_X = np.array(train_X,dtype='float64')
    train_Y = np.array(train_Y,dtype='float64')

    return train_X, train_Y


def NormalizeMult(data):
    '''
    归一化 适用于单维和多维
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)

    for i in range(0,data.shape[1]):

        list = data[:,i]
        listlow,listhigh = np.percentile(list, [0, 100])

        normalize[i,0] = listlow
        normalize[i,1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i] = (data[j,i] - listlow)/delta

    return data,normalize


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def trainModel(train_X, train_Y):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(LSTM(
        140,  #输出维度？
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True))   #return_sequences神将网络的层数，不是最后一层=true，最后一层=false
    model.add(Dropout(0.3))  #正则化

    model.add(LSTM(
        140,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam')
    history = LossHistory()
    model.fit(train_X, train_Y, epochs=N, batch_size=I, verbose=2, callbacks=[history])   #verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("训练loss值")
    plt.plot(history.losses)
    plt.show()
    return model

def reshape_y_hat(y_hat,dim):
    re_y = np.zeros(len(y_hat),dtype='float64')
    length =int(len(y_hat)/dim)
    re_y = re_y.reshape(length,dim)

    for curdim in range(dim):
        for i in range(length):
            re_y[i,curdim] = y_hat[i + curdim*length]

    return re_y

#多维反归一化
def FNormalizeMult(data,normalize):

    data = np.array(data,dtype='float64')
    #列
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        #行
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i] = data[j,i]*delta + listlow

    return data

#使用训练数据的归一化
def NormalizeMultUseData(data,normalize):

    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i] = (data[j,i] - listlow)/delta

    return data

#读取数据
data_csv_0 = pd.read_csv(f, usecols=[0])    #读取第一列数据
t0 = pd.read_csv(f, usecols=[1])
T0 = pd.read_csv(f, usecols=[2])
Data0 = data_csv_0.values.flatten()
T0 = T0.values.flatten()
t0 = t0.values.flatten()
#进行测试
data = np.zeros(len(Data0)*2)
data.dtype = 'float64'
data = data.reshape(len(Data0),2)

data[:,0] = T0   #时间
data[:,1] = Data0   #位移

plt.plot(t0,data[:,1])
plt.show()
plt.plot(t0,data[:,0])
plt.show()


#归一化的加入
data,normalize = NormalizeMult(data)
train_X,train_Y = create_dataset(data,I,O)
model = trainModel(train_X,train_Y)
# params = model.get_weights()
# print(params)
params = model.get_weights() #返回模型权重张量的列表，类型为numpy

# np.save("./MultiSteup2II.npy",normalize)
# model.save("./MultiSteup2II.h5")

data_csv_1 = pd.read_csv(f, usecols=[3]).dropna()    #读取第一列数据
T1 = pd.read_csv(f, usecols=[5]).dropna()
t1 = pd.read_csv(f, usecols=[4]).dropna()
# data_csv_1 = data_csv_1.dropna()
# T1 = T1.dropna()
Data1 = data_csv_1.values.flatten()
T1 = T1.values.flatten()
t1 = t1.values.flatten()

#进行测试
data0 = np.zeros((len(Data1)-O)*2)
data0.dtype = 'float64'
data0 = data0.reshape((len(Data1)-O),2)

data0[:,0] = T1[:I]
data0[:,1] = Data1[:I]

#归一化
# normalize = np.load("./MultiSteup2II.npy")
# data0 = NormalizeMultUseData(data0, normalize)

# model = load_model("./MultiSteup2II.h5")
# a = len(data0)//10

data01 = NormalizeMultUseData(data0, normalize)
test_X = data01.reshape(1, I, 2)
y_hat = model.predict(test_X)
# 重组
y_hat = y_hat.reshape(y_hat.shape[1])
y_hat = reshape_y_hat(y_hat, 2)
# 反归一化
y_hat = FNormalizeMult(y_hat, normalize)


a=np.sum(T1[:I])
y1 = []
for i in y_hat[:,0]:
    a = a + i
    y1.append(a)

z=[]
for i in range(I+O):
    z.append(i)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("预测")
plt.plot(y1,y_hat[:,1],'r',label='prediction')
plt.plot(t1,Data1,'b',label='real')
plt.legend(loc='upper left')
plt.xlabel("时间")
plt.show()

plt.title("预测")
plt.plot(y1,y_hat[:,1],'r',label='prediction')
plt.plot(t1[I:(I+O)],Data1[I:(I+O)],'b',label='real')
plt.legend(loc='upper left')
plt.xlabel("时间")
plt.show()

plt.title("时间差")
plt.plot(z[I:(I+O)],y_hat[:,0],'r')
plt.plot(z,T1,'b')
plt.show()

plt.title("时间")
plt.plot(z[I:(I+O)],y1,'r')
plt.plot(z,t1,'b')
plt.show()
