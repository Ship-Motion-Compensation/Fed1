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
a = np.array([1,2,3,4,5,6,7,8])
b = np.array([9,0,9,8,7,6,5,4])
c = a+b
print(c)