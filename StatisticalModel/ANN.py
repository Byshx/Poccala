# -*-coding:utf-8-*-
"""
人工神经网络
Artificial Neural Network

Author:Byshx

Date:2017.03.30

算法包括：1、自组织映射(Self-Organizing Map, SOM)

        未完待续...
"""

import math
import sys

import numpy as np
from StatisticalModel.DataInitialization import DataInitialization
from StatisticalModel.Distance import Distance
from StatisticalModel.EA import EA

datapath = '/home/luochong/PycharmProjects/MachineLearning/EMDataSet3.csv'


class ANN(DataInitialization):
    def __init__(self, data=None, dimension=None):
        super().__init__()
        '''传递数据维度信息'''
        self.__dimension = dimension
        '''无标注数据'''
        if data is not None:
            self.__data = data
        else:
            self.__data = self.mess_data

    """
       *** Self Organizing Maps (SOM): 一种基于神经网络的聚类算法***

        参数 dimensions 为神经元维度，Example [3,3] 神经元数量为 3*3=9
        参数 *args 包括 σ0, τ1, η0, τ2 分别为Learning Rate Function 和 Neighbour Function 的初始参数
        参数 T 为迭代次数 T >= 500
        参数method为距离计算函数 默认欧氏距离
    """

    def som(self, neural_num, args, T=500, weight=None, method=Distance.euclidean_metric):
        """"""
        '''初始化参数'''
        σ0, τ1, η0, τ2 = args[0], args[1], args[2], args[3]
        """初始化神经元"""
        neural = [np.random.random([1, self.__dimension]) for _ in range(neural_num)]
        """初始化权值向量weight"""
        if weight is None:
            weight = [np.random.random([1, self.__dimension]) for _ in range(neural_num)]
        """开始迭代"""
        '''训练次数'''
        t = 0
        σ = σ0 * math.exp(-t / τ1)
        '''Learning Rate Function'''
        η = η0 * math.exp(-t / τ2)
        while t < T:
            t += 1
            for index in range(len(self.__data)):
                """Competition"""
                '''最短距离'''
                min_distance = sys.maxsize
                '''胜出神经元'''
                win_index = -1
                for n in range(neural_num):
                    distance = method(weight[n], self.__data[index])
                    if distance < min_distance:
                        min_distance = distance
                        win_index = n
                """Cooperation"""
                for index_ in range(neural_num):
                    if index == win_index:
                        continue
                    '''神经元距离'''
                    neural_distance = method(neural[index_], neural[win_index])
                    '''Neighbour Function'''
                    neighbour = math.exp(-neural_distance ** 2 / (2 * σ ** 2))
                    weight[index_] += η * neighbour * (self.__data[index] - weight[win_index])
        print(weight)

    """
            *** 基于粒子群的SOM神经网络 ***

        传统SOM算法问题引入：
            SOM神经网络训练时，权值向量和输入模式相差过大，某些神经元不断获胜，导致
        一些神经元不能获胜成为“死神经元”，另外，某些神经元获胜次数过多，导致对其
        过度利用。
        参数：
            dimensions,T,*args,method为som参数
            T_为pso迭代次数
            scopev为粒子飞行速度范围
        Example:
            ann.p_som([1, 2], (0.6, 20, 0.6, 20), [-0.005, 0.005], T=500, T_=1000, method=Distance.euclidean_metric)
    """

    def p_som(self, neural_num, args, scopev, T=500, T_=500, method=Distance.euclidean_metric):
        """"""
        '''定义适应函数(欧氏距离之和)'''

        def func(*param):
            index, _data, data_ = param[0], param[1], param[2]
            f = 0.
            for d in data_:
                f += Distance.euclidean_metric(_data[index], d)
            return f

        '''寻找数据边界'''
        if self.__dimension is None:
            if self.dimension is None:
                raise ValueError('Error: Dimension Unknown.')
            else:
                self.__dimension = self.dimension
        scopex = [[sys.maxsize, 0] for _ in range(self.__dimension)]
        for d in self.__data:
            for _ in range(len(d)):
                if d[_] < scopex[_][0]:
                    scopex[_][0] = d[_]
                if d[_] > scopex[_][1]:
                    scopex[_][1] = d[_]
        '''使用粒子群算法'''
        ea = EA(data=self.__data)
        p, v = ea.init_particle(neural_num, self.__dimension, scopex, scopev)
        '''调整后的权重向量'''
        weight = ea.pso(p, v, T_, func, scopev, maxvalue=False)
        print('粒子最优位置：', weight)
        return self.som(neural_num, args, T, weight=weight, method=method)


if __name__ == '__main__':
    ann = ANN()
    ann.init_data(datapath=datapath)
    # ann.som(2, (0.6, 20, 0.6, 20), T=500, method=Distance.euclidean_metric)
    ann.p_som(2, (0.6, 20, 0.6, 20), [-1, 1], T=500, T_=500, method=Distance.euclidean_metric)
