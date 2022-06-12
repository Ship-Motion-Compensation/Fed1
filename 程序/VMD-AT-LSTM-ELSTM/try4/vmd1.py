import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft
import math

f1 = r'仿真5-60-6.csv'
f2 = r'仿真6-60-2.csv'


def vvmmdd(filename):
    f = pd.read_csv(filename, usecols=[4])
    t = pd.read_csv(filename, usecols=[0])
    # plt.plot(f.values)

    alpha = len(f) * 1.5  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 5  # 3 modes
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

    u, u_hat, omega = VMD(f.values, alpha, tau, K, DC, init, tol)

    # a = []
    # for i in range(len(f)):
    #     a.append(f.values[i][0])
    #
    # b = []
    # for i in range(len(u[0])):
    #     b.append(0)
    #
    # for i in range(len(u)):
    #     b = b + u[i]
    #
    # c = a - b
    # print(c)
    # plt.plot(c)
    # plt.show()
    # print(np.max(c))
    # print(np.min(c))

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
    for i in range(K):
        plt.subplot(K, 1, i + 1)
        plt.plot(t, u[i, :], linewidth=0.8, c='k')
        plt.ylabel('IMF{}'.format(i + 1))
    plt.savefig(r"C:\Users\LDK\Desktop\da-tu\VMD\0" + "\imf.png", dpi=600)

    #
    # 中心模态
    plt.figure(figsize=(7, 7), dpi=200)
    for i in range(K):
        plt.subplot(K, 1, i + 1)
        plt.plot(t, abs(fft(u[i, :])), c='k')
        plt.ylabel('IMF{}'.format(i + 1))
    plt.savefig(r"C:\Users\LDK\Desktop\da-tu\VMD\0" + "\zhongmo.png", dpi=600)
    #
    plt.show()

    # # 保存子序列数据到文件中
    # for i in range(K):
    #     a = u[i, :]
    #     dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
    #     dataframe.to_csv(r"D:\研究生文件\代码区\2020.10.12\41046\VMDban-%d.csv" % (i + 1), index=False, sep=',')
    # u0, u_hat0, omega0 = VMD(u[1], alpha, tau, K, DC, init, tol)
    # for i in range(K):
    #     plt.subplot(K, 1, i + 1)
    #     plt.plot(u0[i, :], linewidth=0.8, c='r')
    #     plt.ylabel('IMF{}'.format(i + 1))
    # plt.show()

    print(omega[-1])
    return omega[-1]

if __name__ == "__main__":
    A = vvmmdd(f1)*1000*2
    B = vvmmdd(f2)*1000*2
    C = A-B
    print(C)
    rmse_1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in C]) / len(C)))
    print(rmse_1)