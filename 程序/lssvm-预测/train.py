import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from LSSVMRegression import LSSVMRegression


import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

# 加载数据
df = pd.read_csv("dsjjjg1.csv")
data = df.iloc[:,1]

data_csv = data.dropna()  # 滤除缺失数据
dataset = data_csv.values  # 获得csv的值
dataset = dataset.astype('float32')  #确定数据精度，内存位32bits
max_value = np.max(dataset)  # 获得最大值
min_value = np.min(dataset)  # 获得最小值
scalar = max_value - min_value  # 获得间隔数量
dataset = list(map(lambda x: (x-min_value) / scalar, dataset))

  #数据归一化

#%%

# 利用滑动窗口构造训练样本
# data: 输入的序列
# seq_length: 切分的时间间隔
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


#%%

# 历史序列长度
seq_length = 16

# 构建时间序列
x, y = sliding_windows(dataset, seq_length)

# 切分训练集和预测集
train_size = int(len(y) * 0.9)
test_size = len(y) - train_size
trainX = x[0:train_size]
trainY = y[0:train_size]
testX = x[train_size:len(x)]
testY = y[train_size:len(y)]
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

#%%

# 构建LS-SVM回归模型
clf=LSSVMRegression(
        gamma=100,
        kernel='rbf', # 使用RBF核
        sigma=1.0,
        c=0.01,
        d=2,
)

# 训练
clf.fit(trainX, trainY)

# 打印最小二乘回归系数
print("b = ",clf.intercept_)
print("a_i = ",clf.coef_)


#%%

# 预测
testy_pred=clf.predict(testX)

# 画图
time_idx=df.iloc[:, 0]
plt.style.use('ggplot')
plt.figure(figsize=(10,6))
plt.plot(time_idx[-len(testy_pred):],testy_pred, 'b-', linewidth=2, label='predict')
plt.plot(time_idx[-len(testy_pred):],testY, 'r--' , linewidth=2, label='real')
plt.legend()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# 准确率
print("测试集准确率",testy_pred / testY)

