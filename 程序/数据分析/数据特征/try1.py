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


def vvmmdd(f, K):
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    # K = 6  # 3 modes
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
    o = omega[-1]*2
    n = np.argsort(o)
    m = []
    for i2 in range(len(n)-1):
        z = (o)[n[i2+1]] - (o)[n[i2]]
        m.append(z)
    # print(z)

    return min(m)


path = r'C:\Users\LDK\Desktop\数据\6'
path_list = os.listdir(path)
# path_list.remove('.DS_Store')  # macos中的文件管理文件，默认隐藏，这里可以忽略
# print(path_list)


for i0 in range(len(path_list)):
    filename = os.path.join(path, path_list[i0])

    f = pd.read_csv(filename, usecols=[4]).values.flatten()[500:2500]
    # t = pd.read_csv(filename, usecols=[0]).values.flatten()
    # t = pd.read_csv(filename, usecols=[0])
    # f = pd.read_excel(filename, usecols=[4]).values.flatten()
    print(path_list[i0])
    print("最小值："+str(min(f)))
    print("最大值："+str(max(f)))
    print("均值："+str(f.mean()))
    print("方差："+str(f.var()))
    print("标准差："+str(f.std()))
    print('\n')

    K = 2
    while True:
        z = vvmmdd(f, K)
        if z <= 0.02:
            print(z)
            print(K-1)
            print('\n')
            break
        else:
            K = K+1

