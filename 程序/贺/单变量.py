import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers

warnings.filterwarnings('ignore')

dataset = pd.read_csv('4156 2510带标题.csv')

# 将字段Datetime设置为索引列
# 目的：后续基于索引来进行数据集的切分

dataset.index = dataset.time

# 可视化显示DOM_MW的数据分布情况

dataset['high'].plot(figsize=(16, 8))
plt.show()

# 数据进行归一化
# 均值为0，标准差为1
scaler = MinMaxScaler()

dataset['high'] = scaler.fit_transform(dataset['high'].values.reshape(-1, 1))

# 可视化显示归一化后的数据分布情况
#还有一种方法？

dataset['high'].plot(figsize=(16, 8))
plt.show()


# 功能函数：构造特征数据集和标签集

def create_new_dataset(dataset, seq_len=12):
    #seq_len是滑动窗口的大小
    '''基于原始数据集构造新的序列特征数据集
    Params:
        dataset : 原始数据集
        seq_len : 序列长度（时间跨度）

    Returns:
        X, y
    '''
    X = []  # 初始特征数据集为空列表
    y = []  # 初始标签数据集为空列表

    start = 0  # 初始位置
    end = dataset.shape[0] - seq_len  # 截止位置

    for i in range(start, end):  # for循环构造特征数据集
        sample = dataset[i: i + seq_len]  # 基于时间跨度seq_len创建样本
        label = dataset[i + seq_len]  # 创建sample对应的标签
        X.append(sample)  # 保存sample
        y.append(label)  # 保存label
        # 返回特征数据集和标签集
    return np.array(X), np.array(y)



# 功能函数：基于新的特征的数据集和标签集，切分：X_train, X_test

def split_dataset(X, y, train_ratio=0.8):
    '''基于X和y，切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例

    Returns:
        X_train, X_test, y_train, y_test
    '''
    X_len = len(X)  # 特征数据集X的样本数量
    train_data_len = int(X_len * train_ratio)  # 训练集的样本数量

    X_train = X[:train_data_len]  # 训练集
    y_train = y[:train_data_len]  # 训练标签集

    X_test = X[train_data_len:]  # 测试集
    y_test = y[train_data_len:]  # 测试集标签集

    # 返回值
    return X_train, X_test, y_train, y_test


# 功能函数：基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)

def create_batch_data(X, y, batch_size=4, data_type=1):
    #batch是使数据一批一批的训练，缩短训练时间
    '''基于训练集和测试集，创建批数据
    Params:
        X : 特征数据集
        y : 标签数据集
        batch_size : batch的大小，即一个数据块里面有几个样本
        data_type : 数据集类型（测试集表示1，训练集表示2）

    Returns:
        train_batch_data 或 test_batch_data
    '''
    if data_type == 1:  # 测试集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        test_batch_data = dataset.batch(batch_size)  # 构造批数据
        # 返回
        return test_batch_data
    else:  # 训练集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        train_batch_data = dataset.cache().shuffle(1000).batch(batch_size)  # 构造批数据，使用shuffle可以使模型泛化能力更强
        # 返回
        return train_batch_data


# ① 原始数据集

dataset_original = dataset[['high']]

# ② 构造特征数据集和标签集，seq_len序列长度为12小时

SEQ_LEN = 20  # 序列长度

X, y = create_new_dataset(dataset_original.values, seq_len=SEQ_LEN)

# ③ 数据集切分

X_train, X_test, y_train, y_test = split_dataset(X, y, train_ratio=0.9)

# ④ 基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)

# 测试批数据

test_batch_dataset = create_batch_data(X_test, y_test, batch_size=4, data_type=1)

# 训练批数据

train_batch_dataset = create_batch_data(X_train, y_train, batch_size=4, data_type=2)
# 构建模型
model = Sequential([
    layers.LSTM(8, input_shape=(SEQ_LEN, 1)),
    layers.Dense(1)
])

# 定义 checkpoint，保存权重文件

file_path = "best_checkpoint.hdf5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                         monitor='loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_weights_only=True)

# 模型编译

model.compile(optimizer='adam', loss="mae")

# 模型训练

history = model.fit(train_batch_dataset,
                    epochs=10,
                    validation_data=test_batch_dataset,
                    callbacks=[checkpoint_callback])

# 显示 train loss 和 val loss

plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("LOSS")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

# 模型验证

test_pred = model.predict(X_test, verbose=1)


# 计算r2

score = r2_score(y_test, test_pred)
print("r^2 的值： ", score)

# 绘制模型验证结果

plt.figure(figsize=(16, 8))
plt.plot(y_test, label="True label")
plt.plot(test_pred, label="Pred label")
plt.title("True vs Pred")
plt.legend(loc='best')
plt.show()

# 绘制test中前100个点的真值与预测值

y_true = y_test[:100]
y_pred = test_pred[:100]

fig, axes = plt.subplots(2, 1, figsize=(16, 8))
axes[0].plot(y_true, marker='o', color='red')
axes[1].plot(y_pred, marker='*', color='blue')
plt.show()
