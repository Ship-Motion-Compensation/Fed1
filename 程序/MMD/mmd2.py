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


import numpy as np

def MaxMinNormalization(x):
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x


import random
import matplotlib
import matplotlib.pyplot as plt
#
# SAMPLE_SIZE = 500
# buckets = 50
#
# #第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
# plt.subplot(1,2,1)
# plt.xlabel("random.lognormalvariate")
# mu = -0.6
# sigma = 0.15#将输出数据限制到0-1之间
# res1 = [random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
# plt.hist(res1, buckets)
#
# #第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
# plt.subplot(1,2,2)
# plt.xlabel("random.betavariate")
# alpha = 1
# beta = 10
# res2 = [random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)]
# plt.hist(res2, buckets)
#
# plt.savefig('data.jpg')
# plt.show()
#
from torch.autograd import Variable
#
# #参数值见上段代码
# #分别从对数正态分布和beta分布取两组数据
# diff_1 = []
# for i in range(10):
#     diff_1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])
#
# diff_2 = []
# for i in range(10):
#     diff_2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])
#
# X = torch.Tensor(diff_1)
# Y = torch.Tensor(diff_2)
# X,Y = Variable(X), Variable(Y)
# print(mmd_rbf(X,Y))
import pandas as pd



f1 = r'C:\Users\刘敦康\Desktop\数据\6\4-15-6.csv'  # A数据
f3 = r'C:\Users\刘敦康\Desktop\数据\6\5-15-6.csv'     # 测试集数据

data_csv_0 = pd.read_csv(f1, usecols=[4])  # 读取第一列数据  位移
# Data00 = data_csv_0.values.flatten()[500:1000]

data_csv_01 = pd.read_csv(f3, usecols=[4])  # 读取第一列数据  位移
# Data001 = data_csv_01.values.flatten()[600:1000]

# plt.hist(Data00, 50)
# plt.show()
# plt.hist(Data001, 50)
# plt.show()
D0 = []
D1 = []
for i in range(5):
    D00 = data_csv_0.values.flatten()[200*(i+1):200*(i+2)]
    D0.append(D00)
    D01 = data_csv_01.values.flatten()[200 * (i + 1):200 * (i + 2)]
    D1.append(D01)
# D0 = MaxMinNormalization(Data00)
# D1 = MaxMinNormalization(Data001)

# print(D0)
# print(D1)

A = torch.Tensor(D0)
B = torch.Tensor(D1)

A,B = Variable(A), Variable(B)
print(mmd_rbf(A,B))


