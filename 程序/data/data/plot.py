import matplotlib.pyplot as plt
from xlrd import open_workbook

'"画三组数"'

x_data = []
y0_data = []
y1_data = []
y2_data = []

workbook = open_workbook('jg1.xlsx')   #打开指定文件
table = workbook.sheets()[0]   #打开第一张表格
n = table.nrows  #获取表格行数
x = list(range(n-1))  #定义x为次数，从0到n-2

cap0 = table.col_values(0)    #读取1，2，3列的数
cap1 = table.col_values(1)
cap2 = table.col_values(2)
cap3 = table.col_values(3)

for i in range(1, n):      #将数值添加到y里
    y0_data.append(float(cap0[i]))
    y1_data.append(float(cap1[i]))
    y2_data.append(float(cap2[i]))
    x_data.append(float(cap3[i]))

plt.rcParams['font.sans-serif']=['SimHei']   #显示中文标签，不加的话，中文没法显示
plt.rcParams['axes.unicode_minus']=False   #用来显示负号
plt.subplot(3,1,1)  #创建子图
plt.plot(x_data, y0_data, 'r-', linewidth=1)  #画图
plt.title('位移')

plt.subplot(3,1,3)
plt.plot(x_data, y1_data, 'g-', linewidth=1)
plt.title('加速度')

plt.subplot(3,1,2)
plt.plot(x_data, y2_data, 'b-', linewidth=1)
plt.title('激光')

plt.show()

