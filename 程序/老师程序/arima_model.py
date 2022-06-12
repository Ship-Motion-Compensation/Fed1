#!/usr/bin/env python
# coding: utf-8
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf # 自相关
from statsmodels.tsa.stattools import adfuller as ADF # 平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf  # 偏自相关
from statsmodels.stats.diagnostic import acorr_ljungbox # 白噪声检验
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm  # D-W检验,差分自相关检验
from statsmodels.graphics.api import qqplot  # QQ图,检验一组数据是否服从正态分布

# 参数初始化
discfile = 'data(1).csv'
forecastnum = 5

# 读取数据，指定日期列为指标，pandas自动将“日期”列识别为Datetime格式
data = pd.read_csv(discfile)
data.head()
print(data.describe())
print(data.info())

print(data.duplicated(subset=['日期', '销量']))
data = data.drop_duplicates(subset=['日期', '销量'], keep='first')
data = data.dropna()
print(data.describe())
print(data.info())
data = data.set_index('日期')
print(data.head())

# 时序图

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
plt.ylabel('销量')
plt.title("销量量时间序列分析图")
plt.show()

# 自相关图

plot_acf(data)
plt.title("原始序列的自相关图")
plt.show()

# 平稳性检测

print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 差分后的结果
D_data = data.diff().dropna()
D_data.columns = [u'销量差分']
D_data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)  # 时序图
plt.title("一阶差分之后序列的时序图")
plt.ylabel('销量')
plt.show()
plot_acf(D_data)  # 自相关图
plt.title("一阶差分之后序列的自相关图")
plt.show()

print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分']))  # 平稳性检测

# 白噪声检验

print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

plot_pacf(D_data)  # 偏自相关图
plt.title("一阶差分后序列的偏自相关图")
plt.show()

# 定阶
data[u'销量'] = data[u'销量'].astype(float)
pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
print('pmax：')
print(pmax)
print('qmax：')
print(qmax)
bic_matrix = []  # BIC矩阵
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:  # 存在部分报错，所以用try来跳过报错。
            tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值
print('BIC矩阵：')
print(bic_matrix)

tmp_data = bic_matrix.values
tmp_data = tmp_data.flatten()
s = pd.DataFrame(tmp_data, columns=['value'])
s = s.dropna()
print('BIC最小值：', s.min())
s.to_csv('tmp.csv')

#p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
p, q = bic_matrix.stack().astype('float64').idxmin()
print(u'BIC最小的p值和q值为：%s、%s' % (p, q))

model = ARIMA(data, (p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型
print('模型报告为：\n', model.summary2())
print('模型报告为：\n', model.summary())
resid = model.resid
# 自相关图
plot_acf(resid)
plt.title("残差自相关图")
plt.show()
# 偏自相关图
plot_pacf(resid)
plt.title("残差偏自相关图")
plt.show()
# 线性即正态分布
qqplot(resid, line='q', fit=True)
plt.title("残差Q-Q图")
plt.show()
# 解读：残差服从正态分布，均值为零，方差为常数
print('D-W检验的结果为：', sm.stats.durbin_watson(resid.values))
print('残差序列的白噪声检验结果为：', acorr_ljungbox(resid, lags=1))  # 返回统计量、P值

print('预测未来5天，其预测结果、标准误差、置信区间如下：\n', model.forecast(5))
