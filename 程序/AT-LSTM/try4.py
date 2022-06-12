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
'''
本地训练学习率为自适应(学习率起始值=0.01)；
分段函数求误差(预测点位不变，实际值做分段处理)；
实时预测。
单变量——双变量
'''

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

f1 = '5.csv'  # A数据
f3 = '5-0.csv'     # 测试集数据
F = r"C:\Users\LDK\Desktop\E-LSTM\DAN-SHUANG-1"  # 图片保存位置
f_1 = r"C:\Users\LDK\Desktop\E-LSTM\DAN-SHUANG-1\rmse.txt"  # print输出保存位置
N1 = 50 # 两变量LSTM训练次数
N2 = 5  # 三变量LSTM训练次数
M = 5  # 训练步长
I = 10  # 输入
O = 10  # 输出
INPUT_DIM = 1
SINGLE_ATTENTION_VECTOR = False

# #设定为自增长    强制使用gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# KTF.set_session(session)

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, I))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(I, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs[:, 0], a_probs[:, 1]])
    return output_attention_mul


def model_attention_applied_before_lstm():
    K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(I, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    # attention_mul = LSTM(lstm_units, return_sequences=True)(attention_mul)
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(10, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs[:, 0]], output=output)
    return model


def model_attention_applied_after_lstm():
    K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(I, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
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
    normalize = normalize.reshape(data.shape[1],2)

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

def trainModel(train_X, train_Y, N):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    # model.add(model_attention_applied_before_lstm())
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
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()
    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='mse', optimizer=adam)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto', factor=0.5)
    history = model.fit(train_X, train_Y, epochs=N, batch_size=M, verbose=2, shuffle=False, callbacks=[reduce_lr])  # verbose = 1 为输出进度条记录,日志显示；verbose = 2 为每个epoch输出一行记录

    plt.title("LSTM模型训练loss值")
    plt.plot(history.history['loss'], label="loss", color='k', linewidth=1.0)
    plt.xlabel("次数")
    # plt.savefig(F + "\本地loss.png", dpi=600)
    plt.show()
    return model

def trainModel_at(train_X, train_Y, N):
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



def train_test(f):
    # 读取数据(训练)
    data_csv_0 = pd.read_csv(f, usecols=[0])  # 读取第一列数据  位移

    Data0 = data_csv_0.values.flatten()


    # 创建训练数组
    data0 = np.zeros(len(Data0) * 1)
    data0.dtype = 'float64'
    data0 = data0.reshape(len(Data0), 1)
    data0[:, 0] = Data0  # 位移
    # 画图
    plt.subplot(3, 1, 1)
    plt.title("训练数据")
    plt.plot(data0[:, 0], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 3)
    plt.title("训练数据时间差")
    plt.plot(data0[:, 0], color='k', linewidth=1.0)
    plt.xlabel("个数")
    plt.ylabel("时间/s")
    plt.savefig(F + "\训练数据图.png", dpi=600)
    plt.show()

    # 训练
    data0, normalize0 = NormalizeMult(data0)
    train_X0, train_Y0 = create_dataset(data0, I, O)
    model0 = trainModel(train_X0, train_Y0, N1)
    data0 = FNormalizeMult(data0, normalize0)
    # params = model.get_weights()  # 读取权值
    # 创建测试数组
    y01_2 = []
    for i in range((len(Data0) - I)):
        data01 = np.zeros(I*1)
        data01.dtype = 'float64'
        data01 = data01.reshape(I, 1)
        data01[:, 0] = Data0[i:i + I]
        # 测试
        data01, normalize01 = NormalizeMult(data01)
        test_X01 = data01.reshape(1, I, 1)
        y_hat01 = model0.predict(test_X01)
        y_hat01 = y_hat01.reshape(y_hat01.shape[1])  # 重组
        y_hat01 = reshape_y_hat(y_hat01, 1)
        y_hat01 = FNormalizeMult(y_hat01, normalize01)  # 反归一化
        y01_2.append(y_hat01[-1])

    D01 = data0[I:] - y01_2

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D01]) / len(D01)))
    print("单变量训练数据预测rmse："+str(rmse), file=f0)

    plt.subplot(3, 1, 1)
    plt.title("训练集预测")
    plt.plot(y01_2, color='gray', linewidth=1.0)
    plt.plot(Data0[I:], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 3)
    plt.title("训练集预测误差")
    plt.plot(D01, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.savefig(F + "\单变量训练数据预测图.png", dpi=600)
    plt.show()
    return Data0[I:], D01, model0

def text(f3, model0):
    # 读取数据(预测)
    data_csv_02 = pd.read_csv(f3, usecols=[0]).dropna()  # 读取第一列数据

    Data02 = data_csv_02.values.flatten()

    data020 = np.zeros(len(Data02) * 1)
    data020.dtype = 'float64'
    data020 = data020.reshape(len(Data02), 1)
    data020[:, 0] = Data02  # 位移

    # 创建测试数组
    y02_2 = []
    for i in range((len(data020) - I)):
        data02 = np.zeros(I * 1)
        data02.dtype = 'float64'
        data02 = data02.reshape(I, 1)
        data02[:, 0] = Data02[i:i + I]

        # 测试
        data02, normalize02 = NormalizeMult(data02)
        test_X02 = data02.reshape(1, I, 1)
        y_hat02 = model0.predict(test_X02)
        y_hat02 = y_hat02.reshape(y_hat02.shape[1])  # 重组
        y_hat02 = reshape_y_hat(y_hat02, 1)
        y_hat02 = FNormalizeMult(y_hat02, normalize02)  # 反归一化
        y02_2.append(y_hat02[-1])

    D02 = data020[I:] - y02_2

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D02]) / len(D02)))
    print("单变量测试数据预测rmse："+str(rmse), file=f0)

    plt.subplot(3, 1, 1)
    plt.title("测试集预测")
    plt.plot(y02_2, color='gray', linewidth=1.0)
    plt.plot(data020[I:], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 3)
    plt.title("测试集预测误差")
    plt.plot(D02, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.savefig(F + "\单变量测试数据预测图.png", dpi=600)
    plt.show()

    return Data02[I:], D02

def train_text2(D_d01,D01):
    # 读取数据(训练)
    Data1 = D_d01
    # T1 = D_T01
    # t1 = D_t01
    delta_t1 = D01

    # 创建训练数组
    data1 = np.zeros(len(Data1) * 2)
    data1.dtype = 'float64'
    data1 = data1.reshape(len(Data1), 2)
    # data1[:, 0] = T1  # 时间
    data1[:, 0] = Data1  # 位移
    data1[:, 1] = delta_t1[:,0]    # 误差
    # 画图
    plt.subplot(3, 1, 1)
    plt.title("本地训练数据")
    plt.plot(data1[:, 0], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 3)
    plt.title("训练数据误差")
    plt.plot(data1[:, 1], color='k', linewidth=1.0)
    plt.xlabel("个数")
    plt.ylabel("时间/s")
    plt.savefig(F + "\双变量训练数据图.png", dpi=600)
    plt.show()

    # 训练
    data1, normalize1 = NormalizeMult(data1)
    train_X1, train_Y1 = create_dataset(data1, I, O)
    model1 = trainModel_at(train_X1, train_Y1, N2)
    data1 = FNormalizeMult(data1, normalize1)
    # 创建测试数组
    y11 = [[0, 0]]
    for i in range((len(Data1) - I)):
        data11 = np.zeros(I*2)
        data11.dtype = 'float64'
        data11 = data11.reshape(I, 2)
        data11[:, 0] = Data1[i:i + I]

        # 测试
        data11, normalize11 = NormalizeMult(data11)
        test_X11 = data11.reshape(1, I, 2)
        y_hat11 = model1.predict(test_X11)
        y_hat11 = y_hat11.reshape(y_hat11.shape[1])  # 重组
        y_hat11 = reshape_y_hat(y_hat11, 2)
        y_hat11 = FNormalizeMult(y_hat11, normalize11)  # 反归一化

        y11 = np.vstack((y11, y_hat11[-1]))

    D11 = data1[:, 0][I+O:] - y11[1:][:, 0][:-O]
    D11_1 = data1[:, 0][I+O:] - (y11[1:][:, 0][:-O]+y11[1:][:, 1][:-O])

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D11]) / len(D11)))
    print("双变量训练数据预测（位移）rmse：" + str(rmse), file=f0)
    rmse1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in D11_1]) / len(D11_1)))
    print("双变量训练数据预测（位移+误差）rmse：" + str(rmse1), file=f0)

    plt.subplot(3, 1, 1)
    plt.title("训练集预测位移+误差")
    plt.plot(y11[1:][:, 0][:-O]+y11[1:][:, 1][:-O], color='gray', linewidth=1.0)
    plt.plot(Data1[I+O:], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 2)
    plt.title("训练集预测仅位移")
    plt.plot(y11[1:][:, 0][:-O], color='gray', linewidth=1.0)
    plt.plot(Data1[I+O:], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 3)
    plt.title("训练集预测实际误差")
    plt.plot(D11, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.savefig(F + "\双变量训练数据预测图.png", dpi=600)
    plt.show()

    return model1

def text2(D_d02, D02, model1):
    # 读取数据(预测)
    Data12 = D_d02
    delta_t12 = D02

    data120 = np.zeros(len(Data12) * 2)
    data120.dtype = 'float64'
    data120 = data120.reshape(len(Data12), 2)
    data120[:, 0] = Data12  # 位移
    data120[:, 1] = delta_t12[:, 0]

    # 创建测试数组
    y12 = [[0, 0]]
    for i in range((len(Data12) - I)):
        data12 = np.zeros(I * 2)
        data12.dtype = 'float64'
        data12 = data12.reshape(I, 2)
        data12[:, 0] = Data12[i:i + I]
        data12[:, 1] = delta_t12[i:i + I][:, 0]
        # 测试
        data12, normalize12 = NormalizeMult(data12)
        test_X12 = data12.reshape(1, I, 2)
        y_hat12 = model1.predict(test_X12)
        y_hat12 = y_hat12.reshape(y_hat12.shape[1])  # 重组
        y_hat12 = reshape_y_hat(y_hat12, 2)
        y_hat12 = FNormalizeMult(y_hat12, normalize12)  # 反归一化

        y12 = np.vstack((y12, y_hat12[-1]))

    D12 = data120[:, 0][I+O:] - y12[1:][:, 0][:-O]
    D12_1 = data120[:, 0][I+O:] - (y12[1:][:, 0][:-O] + y12[1:][:, 1][:-O])

    rmse = '%.5f' % (math.sqrt(sum([x ** 2 for x in D12]) / len(D12)))
    print("双变量测试数据预测（位移）rmse：" + str(rmse), file=f0)
    rmse1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in D12_1]) / len(D12_1)))
    print("双变量测试数据预测（位移+误差）rmse：" + str(rmse1), file=f0)

    plt.subplot(3, 1, 1)
    plt.title("测试集预测位移+误差")
    plt.plot((y12[1:][:, 0][:-O] + y12[1:][:, 1][:-O]), color='gray', linewidth=1.0)
    plt.plot(Data12[I+O:], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 2)
    plt.title("测试集预测仅位移")
    plt.plot(y12[1:][:, 0][:-O], color='gray', linewidth=1.0)
    plt.plot(Data12[I+O:], color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.subplot(3, 1, 3)
    plt.title("测试集预测实际误差")
    plt.plot(D12, color='k', linewidth=1.0)
    plt.xlabel("时间/s")
    plt.ylabel("波高/m")
    plt.savefig(F + "\双变量测试数据预测图.png", dpi=600)
    plt.show()

    plt.title("测试集预测预测误差")
    plt.plot(y12[:,1], color='k')
    plt.savefig(F + "\双变量测试集预测预测误差图.png", dpi=600)
    plt.show()

if __name__=="__main__":
    f0 = open(f_1, "w")

    D_d01, D01, model0 = train_test(f1)
    D_d02, D02 = text(f3, model0)
    # model1 = train_text2(D_d01, D01)
    # text2(D_d02, D02, model1)

    f0.close()
