import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
warnings.filterwarnings('ignore')

from numpy import *
import numpy as np
import openpyxl
workbook = openpyxl.load_workbook("6906 2569.xlsx")
column=int(input("请输入要读取的表格列索引="))#5

column = column-1
worksheet=workbook.worksheets[0]
beginIndex = 1
endIndex = 2569
lines = endIndex-beginIndex+1

data=mat(zeros((lines,1)))

for i in range(2, lines+1):
    data[i-2] = float(worksheet.cell(row=i+beginIndex, column=column+1).value)

weishu=int(input("请输入维数="))#3
jiange=int(input("请输入间隔="))#5

dataNew = mat(zeros((lines, weishu)))
#
index = 0
for i in range(lines):
    if i+jiange >= lines:
        break
    for j in range(weishu):
        index = i+j*jiange
        if index < lines:
            dataNew[i, j] = data[index, 0]
        else:
            #dataNew[i, j] = 0
            break
dataNew = np.delete(dataNew, [i, lines-1], axis=0)
dataNew = np.array(dataNew)
# 特征数据集
X = np.delete(dataNew, 0, axis=1)
# 标签数据集
y =dataNew[:,0]

# 1 数据集分离： X_train, X_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)

# 2 构造特征数据集


def create_dataset(X, y, seq_len=10):
    features = []
    targets = []

    for i in range(0, len(X) - seq_len, 1):
        data = X[i:i + seq_len]  # 序列数据
        label = y[i + seq_len]  # 标签数据

        # 保存到features和labels
        features.append(data)
        targets.append(label)

    # 返回
    return np.array(features), np.array(targets)
# ① 构造训练特征数据集

train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=10)

# ② 构造测试特征数据集

test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=10)
# 3 构造批数据

def create_batch_dataset(X, y, train=True, buffer_size=1000, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 数据封装，tensor类型
    if train:  # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:  # 测试集
        return batch_data.batch(batch_size)
# 训练批数据
train_batch_dataset = create_batch_dataset(train_dataset, train_labels)

# 测试批数据
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)
# 从测试批数据中，获取一个batch_size的样本数据
list(test_batch_dataset.as_numpy_iterator())[0]

### 第5步：模型搭建、编译、训练
# 模型搭建--版本1
model = Sequential([
    layers.LSTM(units=256, input_shape=train_dataset.shape[-2:], return_sequences=True),
    layers.Dropout(0.4),
    layers.LSTM(units=256, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(units=128, return_sequences=True),
    layers.LSTM(units=32),
    layers.Dense(1)
])

# 模型编译
model.compile(optimizer='adam',
              loss='mse')

'''
# 保存模型权重文件和训练日志

!rm - rf
logs
'''
log_file = os.path.join('logs', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

tensorboard_callback = TensorBoard(log_file)

checkpoint_file = "best_model.hdf5"

checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                      monitor='loss',
                                      mode='min',
                                      save_best_only=True,
                                      save_weights_only=True)

# 模型训练
history = model.fit(train_batch_dataset,
                    epochs=30,
                    validation_data=test_batch_dataset,
                    callbacks=[tensorboard_callback, checkpoint_callback])

# # 显示 train loss 和 val loss
plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(loc='best')
plt.show()

'''
% tensorboard - -logdir
logs
'''

### 第6步：模型验证

test_preds = model.predict(test_dataset, verbose=1)

# 计算r2值
score = r2_score(test_labels, test_preds)
print("r^2 值为： ", score)

# 绘制 预测与真值结果
plt.figure(figsize=(16, 8))
plt.plot(test_labels[:300], label="True value")
plt.plot(test_preds[:300], label="Pred value")
plt.legend(loc='best')
plt.show()

