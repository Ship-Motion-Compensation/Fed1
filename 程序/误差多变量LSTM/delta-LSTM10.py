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
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from time import time
import math
from PyEMD import EMD, Visualisation

'''
本地训练学习率为自适应(学习率起始值=0.01)；
分段函数求误差(预测点位不变，实际值做分段处理)；
实时预测。
单变量——双变量
'''

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

f1 = '海况三.csv'  # A数据
f3 = '海况三.csv'     # 测试集数据
F = r"C:\Users\LDK\Desktop\E-LSTM\DAN-SHUANG-1"  # 图片保存位置
f_1 = r"C:\Users\LDK\Desktop\E-LSTM\DAN-SHUANG-1\rmse.txt"  # print输出保存位置
N1 = 5  # 两变量LSTM训练次数
N2 = 1  # 三变量LSTM训练次数
M = 32  # 训练步长
I = 100  # 输入
O = 10  # 输出

# #设定为自增长    强制使用gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# KTF.set_session(session)

def create_dataset(data, n_predictions, n_next):  # n_predictions：输入数据个数    n_next：输出数据个数
    '''
    对数据进行处理
    '''
    dim = data.shape[1]  # 读取data的列数 （[0]行数）     data是一个二维数组——n行两列
    train_X, train_Y = [], []  # 创建X Y
    for i in range(data.shape[0] - n_predictions - n_next - 1):
        a = data[i:(i + n_predictions), :]
        train_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        for j in range(dim):
            for k in range(len(tempb)):
                b.append(tempb[k, j])
        train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    return train_X, train_Y


def NormalizeMult(data):
    '''
    归一化 适用于单维和多维
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2*data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)

    for i in range(0, data.shape[1]):

        list = data[:, i]
        listlow,listhigh = np.percentile(list, [0, 100])

        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j, i] = (data[j, i] - listlow)/delta

    return data, normalize


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def trainModel(train_X, train_Y, N):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(LSTM(
        140,  # 输出维度？
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True))  # return_sequences神将网络的层数，不是最后一层=true，最后一层=false
    model.add(Dropout(0.3))  # 正则化

    model.add(LSTM(
        140,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))

    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='mse', optimizer=adam)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto', factor=0.5)
    history = model.fit(train_X, train_Y, epochs=N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录

    plt.title("LSTM模型训练loss值")
    plt.plot(history.history['loss'], label="loss", color='k',linewidth=1.0)
    plt.xlabel("次数")
    # plt.savefig(F + "\本地loss.png", dpi=600)
    plt.show()
    return model


def reshape_y_hat(y_hat, dim):
    re_y = np.zeros(len(y_hat), dtype='float64')
    length = int(len(y_hat) / dim)
    re_y = re_y.reshape(length, dim)

    for curdim in range(dim):
        for i in range(length):
            re_y[i, curdim] = y_hat[i + curdim * length]
    return re_y


# 多维反归一化
def FNormalizeMult(data,normalize):

    data = np.array(data,dtype='float64')
    # 列
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        # 行
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i]*delta + listlow
    return data

def train_test(f):
    # 读取数据(训练)
    data_csv0 = pd.read_csv(f, usecols=[0]).dropna()  # 读取第一列数据
    data_csv1 = pd.read_csv(f, usecols=[1]).dropna()
    Data = data_csv0.values.flatten()
    t = data_csv1.values.flatten()

    plt.plot(t, Data)
    plt.show()

    # 提取imfs和剩余
    emd = EMD()
    emd.emd(Data)
    imfs, res = emd.get_imfs_and_residue()

    # 创建训练数组
    data0 = np.zeros(len(imfs) * len(imfs[0]))
    data0.dtype = 'float64'
    data0 = data0.reshape(len(imfs[0]), len(imfs))
    for i in range(len(imfs)):
        data0[:, i] = imfs[i]
        plt.subplot(len(imfs), 1, i+1)
        plt.plot(data0[:, i], color='k', linewidth=1.0)
    plt.show()

    # 训练
    data0, normalize0 = NormalizeMult(data0)
    train_X0, train_Y0 = create_dataset(data0, I, O)
    model0 = trainModel(train_X0, train_Y0, N1)
    data0 = FNormalizeMult(data0, normalize0)
    # params = model.get_weights()  # 读取权值
    # 创建测试数组
    y01_2 = []
    for i in range((len(imfs[0]) - I)):
        data01 = np.zeros(I*len(imfs))
        data01.dtype = 'float64'
        data01 = data01.reshape(I, len(imfs))
        for i0 in range(len(imfs)):
            data01[:, i0] = imfs[i0][i:i + I]
        # 测试
        data01, normalize01 = NormalizeMult(data01)
        test_X01 = data01.reshape(1, I, len(imfs))
        y_hat01 = model0.predict(test_X01)
        y_hat01 = y_hat01.reshape(y_hat01.shape[1])  # 重组
        y_hat01 = reshape_y_hat(y_hat01, len(imfs))
        y_hat01 = FNormalizeMult(y_hat01, normalize01)  # 反归一化
        y01_2.append(y_hat01[-1])

    for i in range(len(imfs)):
        if i == 0:
            y01_20 = np.array(y01_2)[:, 0]
        else:
            y01_20 += np.array(y01_2)[:, i]

    D01 = Data[I+O:] - y01_20[:-O]

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D01]) / len(D01)))
    print("单变量训练数据预测rmse："+str(rmse))

    plt.plot(t[I+O:], Data[I+O:])
    plt.plot(t[I+O:], y01_20[:-O])
    plt.show()

    for i in range(len(imfs)):
        plt.subplot(len(imfs), 1, i+1)
        plt.plot(t[I+O:], imfs[i][I+O:])
        plt.plot(t[I+O:], np.array(y01_2)[:, i][:-O])
    plt.show()


    # plt.subplot(3, 1, 1)
    # plt.title("训练集预测")
    # plt.plot(y01_2, color='gray', linewidth=1.0)
    # plt.plot(Data0[I:], color='k', linewidth=1.0)
    # plt.xlabel("时间/s")
    # plt.ylabel("波高/m")
    # plt.subplot(3, 1, 3)
    # plt.title("训练集预测误差")
    # plt.plot(D01, color='k', linewidth=1.0)
    # plt.xlabel("时间/s")
    # plt.ylabel("波高/m")
    # plt.savefig(F + "\单变量训练数据预测图.png", dpi=600)
    # plt.show()
    return D01, model0

train_test(f1)