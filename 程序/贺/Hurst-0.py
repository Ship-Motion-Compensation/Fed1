from time import time #计时开始
# from numpy import *#导入numpy的库函数
import numpy as np
import pandas as pd
import math
import openpyxl
start=time()


def hurst_rs(data: list):
    """
    使用rs法计算Hurst指数值
    """
    t_i = []
    for r in range(2, (len(data)) // 2 + 1):
        g = len(data) // r
        x_i_j = [data[i * r: (i + 1) * r] for i in range(g)]
        x_i_mean = [sum(x_i) / r for x_i in x_i_j]
        y_i_j = [[x_i_j[i][j] - x_i_mean[i] for j in range(r)] for i in range(g)]
        z_i_j = [[sum(y_i_j[i][: j + 1]) for j in range(r)] for i in range(g)]
        r_i = [max(z_i_j[i]) - min(z_i_j[i]) for i in range(g)]
        s_i = [math.sqrt(sum([(x_i_j[i][j] - x_i_mean[i]) ** 2 for j in range(r)]) / (r - 1)) for i in range(g)]

        #简单的操作，直接将分母等于0的值舍掉，用下一个数据
        a = 0
        SUM = 0
        for i in range(g):
            if s_i[i] != 0:  #直接加判断语句，等于0的直接去点
                rs_i = r_i[i] / s_i[i]
                SUM += rs_i   #总和
                a += 1   #计数    有多少个数据加入了运算
        rs_mean = SUM / a
        # rs_i = [r_i[i] / s_i[i] for i in range(g)]
        # rs_mean = sum(rs_i) / g
        # t_i.append( math.sqrt(sum([(rs_i[i] - rs_mean)**2 for i in range(g)])/(g-1)) )
        t_i.append(rs_mean)
    return np.polyfit(np.log(np.arange(2, len(data) // 2 + 1)), np.log(np.array(t_i)), 1)[0]


if __name__ == "__main__":
    df0 = pd.read_csv("升沉横纵总.csv")
    data0 = df0.iloc[:, 0]

    data_csv_0 = data0.dropna()  # 滤除缺失数据
    dataset0 = data_csv_0.values  # 获得csv的值
    dataset0 = dataset0.astype('float32')  # 确定数据精度，内存位32bits

    # 你这里只用了归一化，最后的结果还要做反归一化的，挺麻烦的（你这里用的不对，就不用了）
    # max_value0 = np.max(dataset0)  # 获得最大值
    # min_value0 = np.min(dataset0)  # 获得最小值
    # scalar0 = max_value0 - min_value0  # 获得间隔数量
    # dataset0 = list(map(lambda x: (x - min_value0) / scalar0, dataset0))  # 数据归一化
    data_test = dataset0
    print(hurst_rs(data_test))
    end = time()# 运行的时间
    t = end - start  #结束时间要减去开始时间才是程序运行时间
    print("程序运行时间为：" + str(t)) #最后要打印出来，并且t的格式要调整成str