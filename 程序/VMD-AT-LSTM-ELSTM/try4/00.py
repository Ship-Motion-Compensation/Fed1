# a = [1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,4,3]
# print(a[2:])


#
# b = []
# for i in range(9):
#     b.append(i)
# print(b)

# def daee(a, b):
#     d = []
#     for i in range(5):
#
#         d.append('model{}'.format(i))
#     print(d)
#     d[0] = 'gpsojogvijrosivjois'
#     d[1] = a + a
#     return d
#
# print(daee(1, 3))
import numpy as np
# a = np.array([np.array([1,2,3,4,5,6,7,8]), np.array([1,2,3,4,5,6,7,8])])
# b = np.array([np.array([9,0,9,8,7,6,5,4]), np.array([9,0,9,8,7,6,5,4])])
# c = (a+b)/2
# print(c)

# def ji(a,b,c):
#     d = a + b
#     if a == b:
#         print(c)
#     print(d)
#
# ji(1,1,[])

# a = [1,3,54,2,4,6,4,76,5,7,8,5,3]
# for i in range(len(a)):
#
#     if i >= 2:
#
#       b = a[i]

def NormalizeMult(data):
    '''
    归一化 适用于单维和多维
    返回归一化后的数据和最大最小值
    '''
def mm(data):
    d = max(data) - min(data)
    data0 = (data - min(data)) / d
    return data0


from sklearn import metrics
import math
# if __name__ == '__main__':
#     A = np.array([0.00556637, 0.07657283, 0.04631159, 0.02592296])
#     B = np.array([0.00504931, 0.07296714, 0.08609844, 0.02812859])
#     C = np.array([0.0056797,  0.0466765,  0.07926354, 0.02383219])
#     # ab = mm(A-B)
#     # ac = mm(A-C)
#     # bc = mm(B-C)
#
#     ab = (A - B)
#     ac = (A - C)
#     bc = (B - C)
#
#     print(ab, ac, bc)
#     # print(abs(ab))
#     print(A-B)
#     rmse_0 = '%.5f' % (math.sqrt(sum([x ** 2 for x in ab]) / len(ab)))
#     rmse_1 = '%.5f' % (math.sqrt(sum([x ** 2 for x in ac]) / len(ac)))
#     rmse_2 = '%.5f' % (math.sqrt(sum([x ** 2 for x in bc]) / len(bc)))
#     print(rmse_0, rmse_1, rmse_2)
    # result_NMI = metrics.normalized_mutual_info_score(A, B)
    # result_NMI1 = metrics.normalized_mutual_info_score(A, C)
    # result_NMI2 = metrics.normalized_mutual_info_score(C, B)
    # print("result_NMI:", result_NMI)
    # print("result_NMI:", result_NMI1)
    # print("result_NMI:", result_NMI2)
    # print(A)

# import pandas as pd
# # from sklearn import metrics as mr
# if __name__ == '__main__':
#     filename = r'ew.csv'
#     A = pd.read_csv(r'仿真5-30-6.csv', usecols=[4]).values[:2000].reshape(2000)
#     B = pd.read_csv(r'仿真5-60-6.csv', usecols=[4]).values[:2000].reshape(2000)
#     print(A.reshape(2000))
#     # A = [1, 1, 1, 2, 3, 3]
#     # B = [1, 2, 3, 1, 2, 3]
#     result_NMI=metrics.normalized_mutual_info_score(A, B)
#     print("result_NMI:",result_NMI)
# A = np.array([[2, 3, 4, 5],[2, 3, 4, 5],[2, 3, 4, 5],[2, 3, 4, 5]])
# m = 1
# n = 3
# b = A[m:n]
# print(b)
A = np.array([1, 3, 6, 4, 5, 3])
# for i in range(len(A)):
#     if i == 0:
#         b = A[i]
#     if i >= 1:
#         A[0] = 2
min = min(A)
max = max(A)
print(min)
print(max)
print(A-min)
c = (A - min) / (max - min)
print(c)


