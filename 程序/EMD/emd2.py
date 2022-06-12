# 导入工具库
import numpy as np
from PyEMD import EMD, Visualisation
import pandas as pd
import matplotlib.pyplot as plt

f = '仿真5-60-6.csv'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 构建信号
# t = np.arange(0,1, 0.01)
# S = 2*np.sin(2*np.pi*15*t) +4*np.sin(2*np.pi*10*t)*np.sin(2*np.pi*t*0.1)+np.sin(2*np.pi*5*t)

data_csv0 = pd.read_csv(f, usecols=[4]).dropna()  # 读取第一列数据
data_csv1 = pd.read_csv(f, usecols=[0]).dropna()
Data = data_csv0.values.flatten()
t = data_csv1.values.flatten()

plt.plot(t, Data,color='k')
plt.xlabel('时间/s')
plt.show()

# 提取imfs和剩余
emd = EMD()
emd.emd(Data)
imfs, res = emd.get_imfs_and_residue()
# print(len(imfs))
# 绘制 IMF
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
# 绘制并显示所有提供的IMF的瞬时频率
vis.plot_instant_freq(t, imfs=imfs)
vis.show()

# for ii in range(len(imfs) + 1):
#     if ii <= len(imfs) - 1:
#         plt.subplot(len(imfs) + 1, 1, ii + 1)
#         plt.plot(t,imfs[ii], linewidth=0.8, c='k')
#         plt.ylabel('IMF{}'.format(ii + 1))
#     else:
#         plt.subplot(len(imfs) + 1, 1, len(imfs) + 1)
#         plt.plot(t,res, linewidth=0.8, c='k')
#         plt.ylabel('res')
# # plt.savefig(F + "\分解图——" + name + ".png", dpi=600)
# plt.show()
# print(imfs[0,:])
# print(imfs)
#
# plt.plot(t, imfs[0])
# plt.show()
# plt.plot(t, imfs[0]+imfs[1]+imfs[2]+imfs[3]+imfs[4]+imfs[5]+imfs[6]+res)
# plt.show()