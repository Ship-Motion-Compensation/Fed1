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
from torch.autograd import Variable

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

def MMMVS(filename):
    f = pd.read_csv(filename, usecols=[4]).values.flatten()[200:-200]
    t = pd.read_csv(filename, usecols=[0]).values.flatten()[200:-200]
    MM = []
    T = []
    MIN = []
    MAX = []
    MEAN = []
    VAR = []
    STD = []
    for i in range(int(len(f)/50)-1):
        # t = pd.read_csv(filename, usecols=[0]).values.flatten()
        # t = pd.read_csv(filename, usecols=[0])
        # f = pd.read_excel(filename, usecols=[4]).values.flatten()
        f0 = f[50*i:50*(i+2)]
        t0 = t[50*i]
        MIN.append(min(f0))
        MAX.append(max(f0))
        MEAN.append(f0.mean())
        VAR.append(f0.var())
        STD.append(f0.std())
        T.append(t0)
        # print("最小值："+str(min(f0)))
        # print("最大值："+str(max(f0)))
        # print("均值："+str(f0.mean()))
        # print("方差："+str(f0.var()))
        # print("标准差："+str(f0.std()))
        # print('\n')
    MM.append(MIN)
    MM.append(MAX)
    # MM.append(MEAN)
    # MM.append(VAR)
    # MM.append(STD)

    return MM, T

import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


path = r'C:\Users\刘敦康\Desktop\数据\6'
path_list = os.listdir(path)
# path_list.remove('.DS_Store')  # macos中的文件管理文件，默认隐藏，这里可以忽略
# print(path_list)

f1 = r'C:\Users\刘敦康\Desktop\数据\拼接\四_all.csv'  # A数据
x, t = MMMVS(f1)
x1 = [x[0][:4],x[1][:4]]
X = torch.Tensor(x1)
X = Variable(X)
print(len(x[0]))
for i in range(len(x[0])-5):
    y1 = [x[0][i:4+i], x[1][i:4+i]]
    Y = torch.Tensor(y1)
    Y = Variable(Y)
    print(t[i], t[4+i])
    print(mmd_rbf(X, Y))


# for i0 in range(len(path_list)):
#     filename = os.path.join(path, path_list[i0])
#     # print(path_list[i0])
#     y = MMMVS(filename)
#     Y = torch.Tensor(y)
#     Y = Variable(Y)
#     print(mmd_rbf(X, Y))


