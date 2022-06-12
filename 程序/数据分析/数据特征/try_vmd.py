import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import os
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft
import math


def vvmmdd(f):
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 1  # 3 modes
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

    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
    o = omega[-1]*2*1000
    n = np.argsort(o)
    m = []
    for i2 in range(len(n)-1):
        z = (o)[n[i2+1]] - (o)[n[i2]]
        m.append(z)
    # print(z)

    return o


path = r'C:\Users\LDK\Desktop\da-tu\try\fed'
path_list = os.listdir(path)
# path_list.remove('.DS_Store')  # macos中的文件管理文件，默认隐藏，这里可以忽略
print(path_list)
filename = os.path.join(path, path_list[0])
f = os.path.join(filename, 'rmse.txt')
f0 = open(f, "w")
print(123,file=f0)
f0.close()
# a = []
# for i0 in range(len(path_list)):
#     filename = os.path.join(path, path_list[i0])
#
#     f = pd.read_csv(filename, usecols=[4]).values.flatten()[500:2500]
#     # t = pd.read_csv(filename, usecols=[0]).values.flatten()
#     # t = pd.read_csv(filename, usecols=[0])
#     # f = pd.read_excel(filename, usecols=[4]).values.flatten()
#     print(path_list[i0])
#     # print("最小值："+str(min(f)))
#     # print("最大值："+str(max(f)))
#     # print("均值："+str(f.mean()))
#     # print("方差："+str(f.var()))
#     # print("标准差："+str(f.std()))
#     # print('\n')
#
#
#     z = vvmmdd(f)
#     a.append(z)
#     print(z)
#     print('\n')
#
# print(a)
# for i_0 in range(len(a)):
#     cc = []
#     for i_1 in range(len(a)):
#         C = a[i_0] - a[i_1]
#         rmse_1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in C]) / len(C)))
#         cc.append(rmse_1)
#     print(i_0+1)
#     print(cc)
#     print('\n')




