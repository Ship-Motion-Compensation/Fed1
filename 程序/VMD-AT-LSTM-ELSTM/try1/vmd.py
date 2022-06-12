import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft
from time import time

filename = r'仿真6-60-2.csv'
f = pd.read_csv(filename, usecols=[4])[:1000].values
f = f.reshape(1000)
filename1 = r'仿真5-60-6.csv'
f1 = pd.read_csv(filename1, usecols=[4])[:1000].values
f1 = f1.reshape(1000)

from sklearn import metrics
if __name__ == '__main__':
    A = [1, 1, 1, 2, 3, 3]
    B = [1, 2, 3, 1, 2, 3]
    result_NMI=metrics.normalized_mutual_info_score(f, f1)
    print("result_NMI:",result_NMI)




# plt.plot(f.values)
#
# alpha = len(f)*1.5  # moderate bandwidth constraint
# tau = 0.  # noise-tolerance (no strict fidelity enforcement)
# K = 4  # 3 modes
# DC = 0  # no DC part imposed
# init = 1  # initialize omegas uniformly
# tol = 1e-7
#
# """
# alpha、tau、K、DC、init、tol 六个输入参数的无严格要求；
# alpha 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍；
# tau 噪声容限 ；
# K 分解模态（IMF）个数；
# DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1；
# init 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数；
# tol 控制误差大小常量，决定精度与迭代次数
# """
# s = time()
# u, u_hat, omega = VMD(f.values, alpha, tau, K, DC, init, tol)
# e = time()
# print(e-s)
# plt.figure()
#
# plt.plot(u.T)
# plt.title('Decomposed modes')
#
# fig1 = plt.figure()
# plt.plot(f.values)
#
# fig1.suptitle('Original input signal and its components')
#
#
# plt.figure(figsize=(7, 7), dpi=200)
# for i in range(K):
#     plt.subplot(K, 1, i + 1)
#     plt.plot(u[i, :], linewidth=0.8, c='r')
#     plt.ylabel('IMF{}'.format(i + 1))
#
# # 中心模态
# plt.figure(figsize=(7, 7), dpi=200)
# for i in range(K):
#     plt.subplot(K, 1, i + 1)
#     plt.plot(abs(fft(u[i, :])))
#     plt.ylabel('IMF{}'.format(i + 1))
#
# plt.show()

# # 保存子序列数据到文件中
# for i in range(K):
#     a = u[i, :]
#     dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
#     dataframe.to_csv(r"D:\研究生文件\代码区\2020.10.12\41046\VMDban-%d.csv" % (i + 1), index=False, sep=',')
