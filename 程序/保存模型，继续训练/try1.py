from vmdpy import VMD
from scipy.fftpack import fft

import keras.backend as K
from keras.layers import Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import keras.callbacks
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from time import time
import math

from keras.layers import *
'''
学习率为自适应(学习率起始值=0.01)；
单注意力机制LSTM+误差注意力机制LSTM
可运行（数据调整过）
训练步长调整
输入调整

注意力放在时间步上
'''

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# f1 = r'C:\Users\LDK\Desktop\数据\6\5-15-6.csv'  # A数据
# f3 = r'C:\Users\LDK\Desktop\数据\6\5-15-6.csv'     # 测试集数据
# F = r"C:\Users\LDK\Desktop\da-tu\try\new\6"  # 图片保存位置
# f_1=r"C:\Users\LDK\Desktop\da-tu\try\new\6\rmse.txt"  # print输出保存位置
N1 = 30  # 两变量LSTM训练次数
N2 = 30  # 三变量LSTM训练次数
N3 = 3
M = 5  # 训练步长
I = 4  # 输入
O = 10  # 输出

K_IMF = 6  # 3 modes
INPUT_DIM = K_IMF + 1  # 3 modes
SINGLE_ATTENTION_VECTOR = False



# #设定为自增长    强制使用gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# KTF.set_session(session)

def vmd(f):
    alpha = len(f) * 1.7  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)

    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7

    """  
    alpha、tau、K、DC、init、tol 六个输入参数的无严格要求； 
    alpha 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍； 
    tau 噪声容限 ；
    K 分解模态（IMF）个数； 
    DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1； 
    init 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数； 
    tol 控制误差大小常量，决定精度与迭代次数
    """

    u, u_hat, omega = VMD(f, alpha, tau, K_IMF, DC, init, tol)
    b = []
    for i in range(len(u[0])):
        b.append(0)

    for i in range(len(u)):
        b = b + u[i]

    res = f - b

    return u, u_hat, omega, res

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = inputs
    # a = Reshape((input_dim, I))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(I, activation='softmax')(a)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def slice(inputs, h):
    """ Define a tensor slice function
    """
    x = tf.slice(inputs, [0, 0, h], [-1, I, 1])
    return x

def attention_3d_block0(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int((inputs.shape[2])/2)
    c = Lambda(slice, arguments={'h': 0})(inputs)
    d = Lambda(slice, arguments={'h': 1})(inputs)
    # c = tf.slice(inputs, [0, 0, 0], [-1, 100, 1])
    # d = tf.slice(inputs, [0, 0, 1], [-1, 100, 1])
    # a = Permute((2, 1))(c)
    # b = Permute((2, 1))(d)
    a = c
    b = d
    # a = Reshape((input_dim, I))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(I, activation='softmax')(a)
    b = Dense(I, activation='softmax')(b)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    b_probs = Permute((1, 2))(b)
    output_attention_mul = Multiply()([c, a_probs])
    output_attention_mul0 = Multiply()([output_attention_mul, b_probs])
    return output_attention_mul0


def model_attention_applied_before_lstm():
    K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(I, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = I*2
    # attention_mul = LSTM(lstm_units, return_sequences=True)(attention_mul)
    # attention_mul = Dropout(0.5)(attention_mul)
    attention_mul = Bidirectional(LSTM(lstm_units, return_sequences=False))(attention_mul)
    output = Dense(O*(K_IMF+1), activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_attention_applied_before_lstm0():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(I, 1,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = I * 2
    # attention_mul = LSTM(lstm_units, return_sequences=True)(attention_mul)
    # attention_mul = Dropout(0.5)(attention_mul)
    attention_mul = Bidirectional(LSTM(lstm_units, return_sequences=False))(attention_mul)
    output = Dense(O, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_after_lstm():
    K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(I, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(K_IMF+1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def create_dataset(data, n_predictions, n_next):  # n_predictions：输入数据个数    n_next：输出数据个数
    '''
    对数据进行处理
    '''
    dim = data.shape[1]  # 读取data的列数 （[0]行数）     data是一个二维数组——n行两列
    train_X, train_Y = [], []  # 创建X Y
    for i in range(data.shape[0] - n_predictions - n_next - 1):
        a = data[i:(i + n_predictions), :]
        train_X.append(a)
        # tempb =data[(i + n_predictions + n_next - 1), :]
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
        listlow, listhigh = np.percentile(list, [0, 100])

        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow)/delta

    return data, normalize


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def trainModel(train_X, train_Y, N, mm):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(model_attention_applied_before_lstm())
    # model.add(LSTM(
    #     140,  # 输出维度？
    #     input_shape=(train_X.shape[1], train_X.shape[2]),
    #     return_sequences=True))  # return_sequences神将网络的层数，不是最后一层=true，最后一层=false
    # model.add(Dropout(0.3))  # 正则化
    #
    # model.add(LSTM(
    #     140,
    #     return_sequences=False))
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(train_Y.shape[1]))
    # model.add(Activation("relu"))
    # adam = optimizers.Adam(lr=0.01)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # adam = optimizers.Adam(lr=0.01)
    # model.compile(loss='mse', optimizer=adam)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto', factor=0.5)
    history = model.fit(train_X, train_Y, epochs=N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录
    model.save(mm)

    plt.title("LSTM模型训练loss值")
    plt.plot(history.history['loss'], label="loss", color='k', linewidth=1.0)
    plt.xlabel("次数")
    # plt.savefig(F + "\本地loss.png", dpi=600)
    plt.show()
    return model

def loadtrainModel(train_X, train_Y, N, mm):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = load_model('model0.h5')
    # model = Sequential()
    # model.add(model_attention_applied_before_lstm())
    # model.add(LSTM(
    #     140,  # 输出维度？
    #     input_shape=(train_X.shape[1], train_X.shape[2]),
    #     return_sequences=True))  # return_sequences神将网络的层数，不是最后一层=true，最后一层=false
    # model.add(Dropout(0.3))  # 正则化
    #
    # model.add(LSTM(
    #     140,
    #     return_sequences=False))
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(train_Y.shape[1]))
    # model.add(Activation("relu"))
    # adam = optimizers.Adam(lr=0.01)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # adam = optimizers.Adam(lr=0.01)
    # model.compile(loss='mse', optimizer=adam)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto', factor=0.5)
    history = model.fit(train_X, train_Y, epochs=N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录
    model.save('model0.h5')

    plt.title("LSTM模型训练loss值")
    plt.plot(history.history['loss'], label="loss", color='k', linewidth=1.0)
    plt.xlabel("次数")
    # plt.savefig(F + "\本地loss.png", dpi=600)
    plt.show()
    return model

def trainModel0(train_X, train_Y, N, mm):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(model_attention_applied_before_lstm0())
    # model.add(LSTM(
    #     140,  # 输出维度？
    #     input_shape=(train_X.shape[1], train_X.shape[2]),
    #     return_sequences=True))  # return_sequences神将网络的层数，不是最后一层=true，最后一层=false
    # model.add(Dropout(0.3))  # 正则化
    #
    # model.add(LSTM(
    #     140,
    #     return_sequences=False))
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(train_Y.shape[1]))
    # model.add(Activation("relu"))
    # adam = optimizers.Adam(lr=0.01)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # adam = optimizers.Adam(lr=0.01)
    # model.compile(loss='mse', optimizer=adam)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto', factor=0.5)
    history = model.fit(train_X, train_Y, epochs=N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录
    model.save(mm)

    plt.title("LSTM模型训练loss值")
    plt.plot(history.history['loss'], label="loss", color='k', linewidth=1.0)
    plt.xlabel("次数")
    # plt.savefig(F + "\本地loss.png", dpi=600)
    plt.show()
    return model

def loadtrainModel0(train_X, train_Y, N, mm):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = load_model('model1.h5')
    # model = Sequential()
    # model.add(model_attention_applied_before_lstm0())
    # model.add(LSTM(
    #     140,  # 输出维度？
    #     input_shape=(train_X.shape[1], train_X.shape[2]),
    #     return_sequences=True))  # return_sequences神将网络的层数，不是最后一层=true，最后一层=false
    # model.add(Dropout(0.3))  # 正则化
    #
    # model.add(LSTM(
    #     140,
    #     return_sequences=False))
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(train_Y.shape[1]))
    # model.add(Activation("relu"))
    # adam = optimizers.Adam(lr=0.01)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # adam = optimizers.Adam(lr=0.01)
    # model.compile(loss='mse', optimizer=adam)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto', factor=0.5)
    history = model.fit(train_X, train_Y, epochs=N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录
    model.save(mm)

    plt.title("LSTM模型训练loss值")
    plt.plot(history.history['loss'], label="loss", color='k', linewidth=1.0)
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
def FNormalizeMult(data, normalize):

    data = np.array(data, dtype='float64')
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


def train_test(f, i00):
    # 读取数据(训练)
    data_csv_0 = pd.read_csv(f, usecols=[4])  # 读取第一列数据  位移
    data_csv_011 = pd.read_csv(f, usecols=[0])

    Data00 = data_csv_0.values.flatten()[500:2500]
    t = data_csv_011.values.flatten()[500:2500]
    Data0, u_hat, omega, res = vmd(Data00)

    # plt.figure(figsize=(K_IMF, K_IMF), dpi=200)
    for ii in range(K_IMF+1):
        if ii <= K_IMF-1:
            plt.subplot(K_IMF+1, 1, ii+1)
            plt.plot(t, Data0[ii, :], linewidth=0.8, c='r')
            plt.ylabel('IMF{}'.format(ii + 1))
        else:
            plt.subplot(K_IMF + 1, 1, K_IMF + 1)
            plt.plot(t, res, linewidth=0.8, c='r')
            plt.ylabel('res')
    plt.show()


    # 创建训练数组
    data0 = np.zeros(len(Data0[0]) * (K_IMF+1))
    data0.dtype = 'float64'
    data0 = data0.reshape(len(Data0[0]), K_IMF+1)
    for i0 in range(K_IMF):
        data0[:, i0] = Data0[i0]  # 位移
    data0[:, K_IMF] = res
    # 画图
    plt.subplot(3, 1, 1)
    plt.title("训练数据")
    plt.plot(t, Data00, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.subplot(3, 1, 3)
    plt.title("训练数据时间差")
    plt.plot(t, Data00, color='k', linewidth=1.0)
    plt.xlabel("个数")
    plt.ylabel("倾角/°")
    plt.savefig(F + "\训练数据图.png", dpi=600)
    plt.show()

    # 训练
    data0, normalize0 = NormalizeMult(data0)
    train_X0, train_Y0 = create_dataset(data0, I, O)
    if i00 == 0:
        model0 = trainModel(train_X0, train_Y0, N1, 'model0.h5')
    if i00 >= 1:
        model0 = loadtrainModel(train_X0, train_Y0, N3, 'model0.h5')
    data0 = FNormalizeMult(data0, normalize0)
    # params = model.get_weights()  # 读取权值
    # 创建测试数组
    b = []
    y01_2 = []
    for i in range((len(Data0[0]) - I)):
        data01 = np.zeros(I*(K_IMF + 1))
        data01.dtype = 'float64'
        data01 = data01.reshape(I, (K_IMF + 1))
        for i1 in range(K_IMF):
            data01[:, i1] = Data0[i1][i:i + I]
        data01[:, K_IMF] = res[i:i + I]

        # 测试
        data01, normalize01 = NormalizeMult(data01)
        test_X01 = data01.reshape(1, I, K_IMF + 1)
        y_hat01 = model0.predict(test_X01)
        y_hat01 = y_hat01.reshape(y_hat01.shape[1])  # 重组
        y_hat01 = reshape_y_hat(y_hat01, K_IMF + 1)
        y_hat01 = FNormalizeMult(y_hat01, normalize01)  # 反归一化
        b.append(y_hat01[-1])


        a = 0
        for i2 in range(K_IMF + 1):
            a = a + y_hat01[-1][i2]
        y01_2.append(a)

    ss_rmse = []
    for iii in range(len(data0[0])):
        ss = np.array(data0[:-I])[:,iii] - np.array(b)[:,iii]
        ss_r = '%.5f' % (math.sqrt(sum([x ** 2 for x in ss]) / len(ss)))
        ss_rmse.append(ss_r)
    print("单变量训练数据预测vmd_rmse：" + str(ss_rmse), file=f0)
    print(ss_rmse)


    D01 = Data00[:-I] - y01_2

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D01]) / len(D01)))
    print("单变量训练数据预测rmse："+str(rmse), file=f0)
    print("单变量训练数据预测rmse：" + str(rmse))

    # plt.figure(figsize=(K_IMF, K_IMF), dpi=200)
    for ii0 in range(K_IMF+1):
        c = [x[ii0] for x in b]
        if ii0 <= K_IMF-1:
            plt.subplot(K_IMF + 1, 1, ii0 + 1)
            plt.plot(t, Data0[ii0, :], linewidth=0.8, c='r')
            plt.plot(t[:-I], c, linewidth=0.8, c='b')
            plt.ylabel('IMF{}'.format(ii0 + 1))
        else:
            plt.subplot(K_IMF + 1, 1, K_IMF + 1)
            plt.plot(t, res, linewidth=0.8, c='r')
            plt.plot(t[:-I], c, linewidth=0.8, c='b')
            plt.ylabel('res')
    plt.show()

    plt.subplot(3, 1, 1)
    plt.title("训练集预测")
    plt.plot(t[:-I], y01_2, color='gray', linewidth=1.0)
    plt.plot(t, Data00, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.subplot(3, 1, 3)
    plt.title("训练集预测误差")
    plt.plot(t[:-I], D01, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.savefig(F + "\单变量训练数据预测图.png", dpi=600)
    plt.show()
    return D01, model0, Data00, y01_2,t

def text(f3, model0):
    # 读取数据(预测)
    data_csv_02 = pd.read_csv(f3, usecols=[4]).dropna()  # 读取第一列数据
    data_csv_0211 = pd.read_csv(f3, usecols=[0]).dropna()

    Data002 = data_csv_02.values.flatten()[500:2500]
    t = data_csv_0211.values.flatten()[500:2500]
    Data02, u_hat, omega, res = vmd(Data002)

    data020 = np.zeros(len(Data02[0]) * (K_IMF + 1))
    data020.dtype = 'float64'
    data020 = data020.reshape(len(Data02[0]), K_IMF + 1)
    for i0 in range(K_IMF):
        data020[:, i0] = Data02[i0]
    data020[:, K_IMF] = res

    # 创建测试数组
    b02 = []
    y02_2 = []
    for i in range((len(data020) - I)):
        data02 = np.zeros(I * (K_IMF + 1))
        data02.dtype = 'float64'
        data02 = data02.reshape(I, K_IMF + 1)
        for i1 in range(K_IMF):
            data02[:, i1] = Data02[i1][i:i + I]
        data02[:, K_IMF] = res[i:i + I]

        # 测试
        data02, normalize02 = NormalizeMult(data02)
        test_X02 = data02.reshape(1, I, (K_IMF + 1))
        y_hat02 = model0.predict(test_X02)
        y_hat02 = y_hat02.reshape(y_hat02.shape[1])  # 重组
        y_hat02 = reshape_y_hat(y_hat02, K_IMF + 1)
        y_hat02 = FNormalizeMult(y_hat02, normalize02)  # 反归一化
        b02.append(y_hat02[-1])
        a = 0
        for i2 in range(K_IMF+1):
            a = a + y_hat02[-1][i2]
        y02_2.append(a)

    ss_rmse = []
    for iii in range(len(data020[0])):
        ss = np.array(data020[:-I])[:, iii] - np.array(b02)[:, iii]
        ss_r = '%.5f' % (math.sqrt(sum([x ** 2 for x in ss]) / len(ss)))
        ss_rmse.append(ss_r)
    print("单变量预测数据预测vmd_rmse：" + str(ss_rmse), file=f0)
    print(ss_rmse)

    D02 = Data002[:-I] - y02_2

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D02]) / len(D02)))
    print("单变量测试数据预测rmse："+str(rmse), file=f0)
    print("单变量测试数据预测rmse：" + str(rmse))

    plt.subplot(3, 1, 1)
    plt.title("测试集预测")
    plt.plot(t[:-I], y02_2, color='gray', linewidth=1.0)
    plt.plot(t, Data002, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.subplot(3, 1, 3)
    plt.title("测试集预测误差")
    plt.plot(t[:-I], D02, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.savefig(F + "\单变量测试数据预测图.png", dpi=600)
    plt.show()

    return D02, Data002, y02_2,t

def train_text2(D01,t, i00):
    # 读取数据(训练)
    Data1 = D01
    # T1 = D_T01
    # t1 = D_t01
    # delta_t1 = D01

    # 创建训练数组
    data1 = np.zeros(len(Data1) * 1)
    data1.dtype = 'float64'
    data1 = data1.reshape(len(Data1), 1)
    # data1[:, 0] = T1  # 时间
    data1[:, 0] = Data1  # 位移
    # data1[:, 1] = delta_t1[:,0]    # 误差
    # 画图
    plt.subplot(3, 1, 1)
    plt.title("本地训练数据")
    plt.plot(t[:-I], data1[:, 0],color='k',linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.subplot(3, 1, 3)
    plt.title("训练数据误差")
    # plt.plot(data1[:, 1],color='k',linewidth=1.0)
    # plt.xlabel("个数")
    # plt.ylabel("时间/s")
    plt.savefig(F + "\双变量训练数据图.png", dpi=600)
    plt.show()

    # 训练
    data1, normalize1 = NormalizeMult(data1)
    train_X1, train_Y1 = create_dataset(data1, I, O)
    # train_Y10 = np.split(train_Y1, 2, axis=1)[0]
    # model1 = trainModel0(train_X1, train_Y1, N2, 'model1.h5')
    if i00 == 0:
        model1 = trainModel0(train_X1, train_Y1, N2, 'model1.h5')
    if i00 >= 1:
        model1 = loadtrainModel0(train_X1, train_Y1, N3, 'model1.h5')
    data1 = FNormalizeMult(data1, normalize1)
    # 创建测试数组
    y11 = [[0]]
    for i in range((len(Data1) - I)):
        data11 = np.zeros(I*1)
        data11.dtype = 'float64'
        data11 = data11.reshape(I, 1)
        data11[:, 0] = Data1[i:i + I]
        # data11[:, 1] = delta_t1[i:i + I][:, 0]

        # 测试
        data11, normalize11 = NormalizeMult(data11)
        test_X11 = data11.reshape(1, I, 1)
        y_hat11 = model1.predict(test_X11)
        y_hat11 = y_hat11.reshape(y_hat11.shape[1])  # 重组
        y_hat11 = reshape_y_hat(y_hat11, 1)
        y_hat11 = FNormalizeMult(y_hat11, np.array([normalize11[0]]))  # 反归一化

        y11 = np.vstack((y11, y_hat11[-1]))

    D11 = data1[:, 0][:-I] - y11[1:][:, 0]
    # D11_1 = data1[:, 0][I+O:] - (y11[1:][:,0][:-O]+y11[1:][:,1][:-O])

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D11]) / len(D11)))
    print("双变量训练数据预测（位移）rmse：" + str(rmse), file=f0)
    # rmse1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in D11_1]) / len(D11_1)))
    # print("双变量训练数据预测（位移+误差）rmse：" + str(rmse1), file=f0)

    # plt.subplot(3, 1, 1)
    # plt.title("训练集预测位移+误差")
    # # plt.plot(y11[1:][:-O]+y11[1:][:-O], color='gray', linewidth=1.0)
    # plt.plot(t, Data1[I:], color='k', linewidth=1.0)
    # plt.xlabel("时间/s")
    # plt.ylabel("波高/m")
    plt.subplot(3, 1, 1)
    plt.title("训练集预测仅位移")
    plt.plot(t[:-2*I], y11[1:][:, 0], color='gray', linewidth=1.0)
    plt.plot(t[:-2*I], Data1[:-I], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.subplot(3, 1, 3)
    plt.title("训练集预测实际误差")
    plt.plot(t[:-2*I], D11, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.savefig(F + "\双变量训练数据预测图.png", dpi=600)
    plt.show()

    return model1, y11[1:][:, 0]

def text2(D02, model1,t):
    # 读取数据(预测)
    Data12 = D02
    # delta_t12 = D02

    data120 = np.zeros(len(Data12) * 1)
    data120.dtype = 'float64'
    data120 = data120.reshape(len(Data12), 1)
    data120[:, 0] = Data12  # 位移
    # data120[:, 1] = delta_t12[:, 0]

    # 创建测试数组
    y12 = [[0]]
    for i in range((len(Data12) - I)):
        data12 = np.zeros(I * 1)
        data12.dtype = 'float64'
        data12 = data12.reshape(I, 1)
        data12[:, 0] = Data12[i:i + I]
        # data12[:, 1] = delta_t12[i:i + I][:, 0]
        # 测试
        data12, normalize12 = NormalizeMult(data12)
        test_X12 = data12.reshape(1, I, 1)
        y_hat12 = model1.predict(test_X12)
        y_hat12 = y_hat12.reshape(y_hat12.shape[1])  # 重组
        y_hat12 = reshape_y_hat(y_hat12, 1)
        y_hat12 = FNormalizeMult(y_hat12, np.array([normalize12[0]]))  # 反归一化

        y12 = np.vstack((y12, y_hat12[-1]))

    D12 = data120[:, 0][:-I] - y12[1:][:, 0]
    # D12_1 = data120[:, 0][I+O:] - (y12[1:][:, 0][:-O] + y12[1:][:, 1][:-O])

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D12]) / len(D12)))
    print("双变量测试数据预测（位移）rmse：" + str(rmse), file=f0)
    # rmse1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in D12_1]) / len(D12_1)))
    # print("双变量测试数据预测（位移+误差）rmse：" + str(rmse1), file=f0)

    # plt.subplot(3, 1, 1)
    # plt.title("测试集预测位移+误差")
    # # plt.plot((y12[1:][:, 0][:-O] + y12[1:][:, 1][:-O]), color='gray', linewidth=1.0)
    # plt.plot(t[:-I], Data12[I:], color='k', linewidth=1.0)
    # plt.xlabel("时间/s")
    # plt.ylabel("波高/m")
    plt.subplot(3, 1, 1)
    plt.title("测试集预测")
    plt.plot(t[:-2*I], y12[1:][:, 0], color='gray', linewidth=1.0)
    plt.plot(t[:-2*I], Data12[:-I], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.subplot(3, 1, 3)
    plt.title("测试集预测实际误差")
    plt.plot(t[:-2*I], D12, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("倾角/°")
    plt.savefig(F + "\双变量测试数据预测图.png", dpi=600)
    plt.show()

    # plt.title("测试集预测预测误差")
    # # plt.plot(y12[:,1], color='k')
    # plt.savefig(F + "\双变量测试集预测预测误差图.png", dpi=600)
    # plt.show()
    return y12[1:][:, 0]


def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mae -- MAE 评价指标
    """

    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

def gyh(A):
    Min = min(A)
    Max = max(A)
    D = (A - Min) / (Max - Min)
    return D


if __name__=="__main__":


    path1 = r'C:\Users\LDK\Desktop\数据\6'
    path_list1 = os.listdir(path1)

    path2 = r'C:\Users\LDK\Desktop\da-tu\try\继续训练\二'
    path_list2 = os.listdir(path2)

    # f11 = r'C:\Users\LDK\Desktop\da-tu\try\new\rmse1.txt'
    # f110 = open(f11, "w")

    for i0 in range(len(path_list1)):

        f1 = os.path.join(path1, path_list1[i0])
        # f3 = os.path.join(path1, path_list1[i0])
        F = os.path.join(path2, path_list2[i0])
        f_1 = os.path.join(F, 'rmse.txt')
        f_110 = os.path.join(F, 'rmse1.txt')

        f0 = open(f_1, "w")
        f110 = open(f_110, "w")

        D01, model0, Data00, y01_2, t1 = train_test(f1, i0)
        model1, res_1 = train_text2(D01, t1, i0)

        for i9 in range(len(path_list1)):
            print(path_list1[i9], file=f0)
            f3 = os.path.join(path1, path_list1[i9])
            # D01, model0, Data00, y01_2, t1 = train_test(f1)
            D02, Data002, y02_2, t2 = text(f3, model0)

            # model1, res_1 = train_text2(D01,t1)
            res_2 = text2(D02, model1, t2)

            result0 = y01_2[:-I] + res_1
            result1 = y02_2[:-I] + res_2

            plt.subplot(3, 1, 1)
            plt.title("训练集预测")
            plt.plot(t2[:-2 * I], result1, color='gray', linewidth=1.0)
            plt.plot(t2[:-2 * I], Data002[:-2*I], color='k', linewidth=1.0)
            plt.xlabel("时间/s")
            plt.ylabel("倾角/°")
            plt.subplot(3, 1, 3)
            plt.title("训练集预测实际误差")
            plt.plot(t2[:-2 * I], result1, color='gray', linewidth=1.0)
            plt.plot(t2[:-2 * I], Data002[:-2*I], color='k', linewidth=1.0)
            plt.xlabel("时间/s")
            plt.ylabel("倾角/°")
            plt.savefig(F + "\预测结果图.png", dpi=600)
            plt.show()

            d_1 = Data00[:-2*I] - result0
            d_2 = Data002[:-2*I] - result1

            d_10 = gyh(d_1)
            d_20 = gyh(d_2)

            rmse_1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in d_1]) / len(d_1)))
            mae_1 = mae_value(Data00[:-2*I], result0)
            rmse_2 = '%.5f' % (math.sqrt(sum([x ** 2 for x in d_2]) / len(d_2)))
            mae_2 = mae_value(Data002[:-2*I], result1)

            rmse_10 = '%.5f' % (math.sqrt(sum([x ** 2 for x in d_10]) / len(d_10)))
            rmse_20 = '%.5f' % (math.sqrt(sum([x ** 2 for x in d_20]) / len(d_20)))

            print("rmse:", rmse_1, rmse_2, "归一rmse:", rmse_10, rmse_20, "mae:", mae_1, mae_2, file=f0)
            print("rmse:",rmse_2, "归一rmse:", rmse_20, "mae:", mae_2, file=f110)


        f0.close()
        f110.close()