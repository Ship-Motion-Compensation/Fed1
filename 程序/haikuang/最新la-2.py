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
'''
只归一化位移；
本地训练学习率为自适应(学习率起始值=0.01)；
联邦训练学习率为自适应(学习率起始值=0.01)；
分段函数求误差(预测点位不变，实际值做分段处理)；
实时预测。
'''

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

f1 = 'HK4.csv'  # A数据
f2 = 'HK40.csv'  # B数据
F = r"C:\Users\LDK\Desktop\fed-data-pic\new5"  # 图片保存位置
f_1 = r"C:\Users\LDK\Desktop\fed-data-pic\new5\rmse.txt"  #print输出保存位置
N = 2  # 训练次数
F_N = 2  #联邦训练次数
M = 50  #训练步长
m = 2  # 联邦次数
I = 50  # 输入
O = 5  # 输出

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

    list = data[:, 1]
    max = np.max(list)
    min = np.min(list)

    delta = max -min
    if delta != 0:
        data[:, 1] = (data[:, 1] - min) / delta

    return data, delta, min


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def trainModel(train_X, train_Y, name):
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

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='mse', optimizer=adam)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto')
    history = model.fit(train_X, train_Y, epochs=N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录

    plt.title("本地训练loss值——" + name)
    plt.plot(history.history['loss'], label="loss")
    plt.savefig(F + "\本地loss" + name + ".png", dpi=600)
    plt.show()
    return model


def FedtrainModel(train_X, train_Y, weights, name):
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

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='mse', optimizer=adam)
    model.set_weights(weights)

    reduce_lr0 = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto')
    history = model.fit(train_X, train_Y, epochs=F_N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr0])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录

    plt.title("第" + str(i0 + 1) + "轮训练loss值——" + name)
    plt.plot(history.history['loss'], label="loss")
    plt.savefig(F + "\联邦loss" + name + "第" + str(i0 + 1) + "轮.png", dpi=600)
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
def FNormalizeMult(data, delta, min):
    data = np.array(data, dtype='float64')
    # 列
    min1 = min
    delta1 = delta
    # 行
    if delta1 != 0:
        data[:, 1] = data[:, 1] * delta + min1
    return data

# 使用训练数据的归一化
def NormalizeMultUseData(data, delta, min):
    min0 = min
    delta0 = delta

    if delta != 0:
        data[:, 1] = (data[:, 1] - min) / delta

    return data


def train_test(f, name):
    # 读取数据(训练)
    data_csv_0 = pd.read_csv(f, usecols=[0])  # 读取第一列数据
    t0 = pd.read_csv(f, usecols=[1])
    T0 = pd.read_csv(f, usecols=[2])
    Data0 = data_csv_0.values.flatten()
    T0 = T0.values.flatten()
    t0 = t0.values.flatten()
    # 读取数据(预测)
    data_csv_1 = pd.read_csv(f, usecols=[3]).dropna()  # 读取第一列数据
    T1 = pd.read_csv(f, usecols=[5]).dropna()
    t1 = pd.read_csv(f, usecols=[4]).dropna()
    Data1 = data_csv_1.values.flatten()
    T1 = T1.values.flatten()
    t1 = t1.values.flatten()
    # 创建训练数组
    data = np.zeros(len(Data0) * 2)
    data.dtype = 'float64'
    data = data.reshape(len(Data0), 2)
    data[:, 0] = T0  # 时间
    data[:, 1] = Data0  # 位移
    # 画图
    plt.subplot(2, 1, 1)
    plt.title("训练数据位移" + name)
    plt.plot(t0, data[:, 1])
    plt.subplot(2, 1, 2)
    plt.title("训练数据时间差" + name)
    plt.plot(t0[1:], data[:, 0][1:])
    plt.savefig(F + "\本地训练" + name + ".png", dpi=600)
    plt.show()
    # 训练
    data, normalize, min = NormalizeMult(data)
    train_X, train_Y = create_dataset(data, I, O)
    model = trainModel(train_X, train_Y, name)
    params = model.get_weights()  # 读取权值
    # 创建测试数组
    y = [[0, 0]]
    y2 = []
    for i in range((len(Data1) - I) // O):
        data0 = np.zeros(I*2)
        data0.dtype = 'float64'
        data0 = data0.reshape(I, 2)
        data0[:, 0] = T1[(i * O):((i * O) + I)]
        data0[:, 1] = Data1[(i * O):((i * O) + I)]
        a = t1[(i * O):((i * O) + I)][-1]
        # 测试
        data01, normalize1, min1 = NormalizeMult(data0)
        test_X = data01.reshape(1, I, 2)
        y_hat = model.predict(test_X)
        y_hat = y_hat.reshape(y_hat.shape[1])  # 重组
        y_hat = reshape_y_hat(y_hat, 2)
        y_hat = FNormalizeMult(y_hat,normalize1,min1)  # 反归一化

        y1 = []
        for i2 in y_hat[:, 0]:
            y2.append(i2)
            a = a + i2
            y1.append(a)
        y_hat[:, 0] = y1
        y0 = np.vstack((y, y_hat))
        y = y0[np.lexsort(y0[:, ::-1].T)]

    # 分段求误差
    D1 = []
    for s in range(len(y)):
        for s1 in range(len(Data1)):
            delta = y[:, 0][s] - t1[s1]
            if delta <= 0:
                w1 = (y[:, 0][s] - t1[s1 - 1]) / (t1[s1] - t1[s1 - 1]) * (Data1[s1] - Data1[s1 - 1]) + Data1[s1 - 1]
                D1.append(w1)
                break
    mm = np.min([len(y[:, 1][1:]),len(D1[1:])])
    err = np.array(y[:, 1][1:][:mm]) - np.array(D1[1:][:mm])
    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in err]) / len(err)))
    print("本地" + name + "--rmse:" + str(rmse), file=f0)

    plt.title("本地预测——" + name + "误差\n" + "rmse:" + str(rmse))
    plt.plot(err, 'b', label='误差')
    plt.legend(loc='upper left')
    plt.savefig(F + "\本地预测" + name + "误差.png", dpi=600)
    plt.show()

    # 处理时间
    z = []
    for i in range(len(y) + I):
        z.append(i)
    # 画图
    plt.figure()
    plt.suptitle("本地预测——" + name)

    plt.subplot(2, 2, 1)
    plt.title("总预测")
    plt.plot(y[:, 0][1:], y[:, 1][1:], 'r', label='prediction')
    plt.plot(t1, Data1, 'b', label='real')
    plt.legend(loc='upper left')
    plt.xlabel("时间/s")
    plt.ylabel("电压/v")

    plt.subplot(2, 2, 2)
    plt.title("位移预测")
    plt.plot(y[:, 1][1:], 'r', label='prediction')
    plt.plot(Data1[I:], 'b', label='real')
    plt.legend(loc='upper left')
    plt.xlabel("时间/s")
    plt.ylabel("电压/v")

    plt.subplot(2, 2, 3)
    plt.title("时间差预测")
    plt.plot(z[I + 1:], y2, 'r', label='prediction')
    plt.plot(z[2:], T1[1:], 'b', label='real')
    plt.legend(loc='upper left')
    plt.xlabel("次数")
    plt.ylabel("时间/s")

    plt.subplot(2, 2, 4)
    plt.title("时间预测")
    plt.plot(z[I + 1:], y[:, 0][1:], 'r', label='prediction')
    plt.plot(z[1:], t1, 'b', label='real')
    plt.legend(loc='upper left')
    plt.xlabel("次数")
    plt.ylabel("时间/s")
    plt.savefig(F + "\本地预测" + name + ".png", dpi=600)
    plt.show()

    return params


def fed_train_test(f, name):
    # 读取数据(训练)
    data_csv_0 = pd.read_csv(f, usecols=[0])  # 读取第一列数据
    t0 = pd.read_csv(f, usecols=[1])
    T0 = pd.read_csv(f, usecols=[2])
    Data0 = data_csv_0.values.flatten()
    T0 = T0.values.flatten()
    t0 = t0.values.flatten()
    # 读取数据(预测)
    data_csv_1 = pd.read_csv(f, usecols=[3]).dropna()  # 读取第一列数据
    T1 = pd.read_csv(f, usecols=[5]).dropna()
    t1 = pd.read_csv(f, usecols=[4]).dropna()
    Data1 = data_csv_1.values.flatten()
    T1 = T1.values.flatten()
    t1 = t1.values.flatten()
    # 创建训练数组
    data = np.zeros(len(Data0) * 2)
    data.dtype = 'float64'
    data = data.reshape(len(Data0), 2)
    data[:, 0] = T0  # 时间
    data[:, 1] = Data0  # 位移

    # 训练
    data, normalize, min = NormalizeMult(data)
    train_X, train_Y = create_dataset(data, I, O)
    model = FedtrainModel(train_X, train_Y, weights, name)
    params = model.get_weights()  # 读取权值

    # 创建测试数组
    y = [[0, 0]]
    y2 = []
    for i in range((len(Data1) - I) // O):
        data0 = np.zeros(I * 2)
        data0.dtype = 'float64'
        data0 = data0.reshape(I, 2)
        data0[:, 0] = T1[(i * O):((i * O) + I)]
        data0[:, 1] = Data1[(i * O):((i * O) + I)]
        a = t1[(i * O):((i * O) + I)][-1]
        # 测试
        data01, normalize1, min1 = NormalizeMult(data0)
        test_X = data01.reshape(1, I, 2)
        y_hat = model.predict(test_X)
        y_hat = y_hat.reshape(y_hat.shape[1])  # 重组
        y_hat = reshape_y_hat(y_hat, 2)
        y_hat = FNormalizeMult(y_hat, normalize1,min1)  # 反归一化
        y1 = []
        for i2 in y_hat[:, 0]:
            y2.append(i2)
            a = a + i2
            y1.append(a)
        y_hat[:, 0] = y1
        y0 = np.vstack((y, y_hat))
        y = y0[np.lexsort(y0[:, ::-1].T)]

    # 分段求误差
    D1 = []
    for s in range(len(y)):
        for s1 in range(len(Data1)):
            delta1 = y[:,0][s] - t1[s1]
            if delta1 <= 0:
                w1 = (y[:,0][s] - t1[s1 - 1]) / (t1[s1] - t1[s1 - 1]) * (Data1[s1] - Data1[s1 - 1]) + Data1[s1 - 1]
                D1.append(w1)
                break
    mm = np.min([len(y[:, 1][1:]), len(D1[1:])])
    err = np.array(y[:,1][1:][:mm]) - np.array(D1[1:][:mm])
    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in err]) / len(err)))
    print("第" + str(i0 + 1) + "轮联邦" + name + "--rmse:" + str(rmse), file=f0)

    plt.title("第" + str(i0 + 1) + "轮联邦预测——" + name + "误差\n" + "rmse:" + str(rmse))
    plt.plot(err, 'b', label='误差')
    plt.legend(loc='upper left')
    plt.savefig(F + "\联邦预测" + name + "第" + str(i0 + 1) + "轮误差.png", dpi=600)
    plt.show()

    z = []
    for i in range(len(y) + I):
        z.append(i)

    # 画图
    plt.figure()
    plt.suptitle("第" + str(i0 + 1) + "轮联邦预测——" + name)

    plt.subplot(2, 2, 1)
    plt.title("总预测")
    plt.plot(y[:, 0][1:], y[:, 1][1:], 'r', label='prediction')
    plt.plot(t1, Data1, 'b', label='real')
    plt.legend(loc='upper left')
    plt.xlabel("时间/s")
    plt.ylabel("电压/v")

    plt.subplot(2, 2, 2)
    plt.title("位移预测")
    plt.plot(y[:, 1][1:], 'r', label='prediction')
    plt.plot(Data1[I:], 'b', label='real')
    plt.legend(loc='upper left')
    plt.xlabel("时间/s")
    plt.ylabel("电压/v")

    plt.subplot(2, 2, 3)
    plt.title("时间差预测")
    plt.plot(z[I + 1:], y2, 'r', label='prediction')
    plt.plot(z[2:], T1[1:], 'b', label='real')
    plt.xlabel("次数")
    plt.ylabel("时间/s")

    plt.subplot(2, 2, 4)
    plt.title("时间预测")
    plt.plot(z[I + 1:], y[:, 0][1:], 'r', label='prediction')
    plt.plot(z[1:], t1, 'b', label='real')
    plt.legend(loc='upper left')
    plt.xlabel("次数")
    plt.ylabel("时间/s")
    plt.savefig(F + "\联邦预测" + name + "第" + str(i0 + 1) + "轮.png", dpi=600)
    plt.show()

    return params


def AVG(A, B):
    weigth = A
    weigth[0] = (A[0] + B[0]) / 2
    weigth[1] = (A[1] + B[1]) / 2
    weigth[2] = (A[2] + B[2]) / 2
    weigth[3] = (A[3] + B[3]) / 2
    weigth[4] = (A[4] + B[4]) / 2
    weigth[5] = (A[5] + B[5]) / 2
    weigth[6] = (A[6] + B[6]) / 2
    weigth[7] = (A[7] + B[7]) / 2
    weights = weigth
    return weights

f0 = open(f_1, "w")
print("本地A")
A = train_test(f1, "A")
print("本地B")
B = train_test(f2, "B")

for i0 in range(m):
    print("第" + str(i0 + 1) + "轮联邦")
    weights = AVG(A, B)
    print("联邦A")
    A = fed_train_test(f1, "A")
    print("联邦B")
    B = fed_train_test(f2, "B")

f0.close()

