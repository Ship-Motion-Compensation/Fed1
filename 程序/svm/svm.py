# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:23:40 2020

@author: 刘敦康
"""

#引入相关程序包
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

#构造一个class
class Support_Vector_machine:
    #定义一个类
    def __init__(self,visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}  #定义两个标签，1为红，-1为蓝
        if self.visualization:
            self.fig = plt.figure()  #画图
            self.ax = self.fig.add_subplot(1,1,1)
    #训练出一个模型
    def fit(self,data):
        self.data = data  #将传入的data赋值给self.data，使其在整个类的范围内可以访问

        opt_dict = {}   #用来存储w，b的组合的值  

        rotMatrix = lambda theta: np.array([np.sin(theta),np.cos(theta)])  #从sin转到cos

        thetaStep = np.pi/10  #定义步长  为了确保精确度，可以设的很小
        transforms = [np.array(rotMatrix(theta))
                      for theta in np.arange(0,np.pi,thetaStep)]  #从0到180，每次增加步长，循环，生成theta里，然后将theta添加到rotMatrix里，最后生成array
        #对数据集进行预处理。将数据集拉平到一个list当中，方便处理
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        #找到最大值和最小值
        self.max_feature_value = max(all_data)  #8
        self.min_feature_value = min(all_data)  #-4
        #定义步长
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
        #寻找b的准备工作
        b_rang_multiple = 5  #定义b的间隔值

        b_multiple = 5    #另一个间隔值

        latest_optimum = self.max_feature_value * 10  #以此为起点来找w的值

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])   #80*80
            optimized = False  #未找到优化值，定义为false

            while not optimized:   #未找到就一直执行，直到找到为止
                for b in np.arange(-1*(self.max_feature_value*b_rang_multiple),   #从-40到+40开始循环
                                  self.max_feature_value*b_rang_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation   #
                        found_option = True   #找到符合条件的值，在到处之前，定义为ture

                        for i in self.data:  #循环  i的值为1；-1
                            for xi in self.data[i]:  # 然后开始循环xi，第一个为[1，8]，，，，
                                yi = i       #这里对应着yi为-1
                                if not yi*(np.dot(w_t,xi)+b) >= 1:   #dot  内积函数
                                    found_option = False   #找不着，定义为false
                                    break   #跳出for xi ，，循环
                            if not found_option:
                                break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]   #以w-t的模（范数）作为K

                if w[0] < 0:   #减步长，一直减，减没了，代表所有的都找完了
                    optimized = True
                else:
                    w = w-step  #向量减单值，原本不能实现的，但这里能运行，这里代表两个数分别减去步长

            norms = sorted([n for n in opt_dict])   #sorteg函数将值按从小到大的顺序排列
            opt_choice = opt_dict[norms[0]]    #取出第一个值，也就是最小值
            self.w = opt_choice[0]    #分别将w和b的值从opt_choice中取出。[w_t,b]————0为w；1为b
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]*step*2   #取出w的第一个值

    def predict(self,features):
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker = '*',c=self.colors[classification])
        return classification

    def visualize(self):   #可视化分割平面
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]   #打点：s是点的大小，color。从后往前看，i为1或-1；x为坐标

        def hyperlane(x,w,b,v):     #打直线   相当于  cy=ax+b   w => w[0]=a,  w[1]=c,  v => y
            return (-w[0]*x-b+v)/w[1]

        datarange = (self.min_feature_value,self.max_feature_value)   #定义一个数据范围

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        psv1 = hyperlane(hyp_x_min,self.w,self.b, 1)    #两点法，直接带入hyperlane
        psv2 = hyperlane(hyp_x_max,self.w,self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')

        nsv1 = hyperlane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperlane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        db1 = hyperlane(hyp_x_min, self.w, self.b, 0)      #分割平面
        db2 = hyperlane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')  #y--：颜色+线形

        plt.show()

#定义一个数据集
data_dict = {-1:np.array([[1,8],
                          [2,3],
                          [3,6]],),
             1:np.array([[1,-2],
                         [3,-4],
                         [3,0]])}
#构造svm,并调用相关函数
svm = Support_Vector_machine()
svm.fit(data=data_dict)

predict_us = [[0,1],
             [1,3],
             [3,4],
             [3,5],
             [5,5],
             [5,6],
             [6,-5],
             [5,8],
             [-1,0],
             [-2,1]]

for p in predict_us:
    svm.predict(p)

svm.visualize()