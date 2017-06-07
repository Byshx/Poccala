# -*-coding:utf-8-*-

"""
进化算法集合
Evolutionary Algorithm

Author:Byshx

Date:2017.03.31

算法：粒子群算法(Particle Swarm Optimization,PSO)
"""

import copy
import sys

import numpy as np
from StatisticalModel.DataInitialization import DataInitialization

datapath = '/home/luochong/PycharmProjects/MachineLearning/EMDataSet3.csv'


class EA(DataInitialization):
    def __init__(self, data=None):
        super().__init__()
        if data is not None:
            self.__data = data
        else:
            self.__data = self.mess_data

    """
            *** 粒子群初始化 ***

         输入参数:quantity粒子数量
                 dimension维度
                 scope各维取值范围(min~max) Example:二维粒子各维范围:[[1,2],[2,3]]
    """

    @staticmethod
    def init_particle(quantity, dimension, scopex, scopev):
        if len(scopex) != dimension:
            raise 'Error: 某些维度没有限定范围'
        '''粒子位置'''
        location = []
        '''初始化粒子速度'''
        v = [(scopev[0] + (scopev[1] - scopev[0]) * np.random.rand(1, dimension)) for _ in range(quantity)]
        for _ in range(quantity):
            tmp = []
            for index in range(dimension):
                tmp.append(scopex[index][0] + (scopex[index][1] - scopex[index][0]) * np.random.rand())
            location.append(np.array([tmp]))
        return location, v

    """
        ***粒子群算法(Particle Swarm Optimization,PSO)***

        输入参数：
            ω, c1, c2, r, function, particle
            1、particle:粒子群
            2、T为迭代次数
            3、function为适应函数
            4、max_x限制粒子位置的最大和最小
            5、max_v限制粒子的速度(速度过小收敛过慢，速度过大容易冲出最优点)
            6、ω为保持原来速度的系数，叫惯性权重 采用线性递减方法: (ω_min + (T - t) * (ω_max - ω_min) / T)
            7、c1为跟踪自己历史最优的权重系数，一般为2
            8、c2为跟踪群体历史最优的权重系数，一般为2
            9、r为约束因子，控制更新速度，一般为1
            10、maxvalue为取适应函数最大值为最优值 False为最小值为最优值
        1、当C1＝0时，则粒子没有了认知能力，变为只有社会的模型(social-only)：
        被称为全局PSO算法。粒子有扩展搜索空间的能力，具有较快的收敛速度，但由于缺少局部搜索，对于复杂问题
        比标准PSO 更易陷入局部最优。
        2、当C2＝0时，则粒子之间没有社会信息，模型变为只有认知(cognition-only)模型：
        被称为局部PSO算法。由于个体之间没有信息的交流，整个群体相当于多个粒子进行盲目的随机搜索，收敛速度慢，因而得到最优解的可能性小。
    """

    def pso(self, p_location, p_v, T, function, scopev, ω=(0.01, 1), args=(2, 2, 1), maxvalue=True):
        """"""
        '''ξ, η为在[0,1]区间的相互独立的随机数'''
        ξ, η = np.random.rand(), np.random.rand()
        '''惯性因子的最小值和最大值'''
        ω_min, ω_max = ω[0], ω[1]
        '''更新公式系数'''
        c1, c2, r = args[0], args[1], args[2]
        '''当前迭代次数'''
        t = 0
        '''粒子历史最优位置'''
        p = copy.deepcopy(p_location)
        '''粒子历史最优值'''
        p_value = [sys.maxsize for _ in range(len(p_location))]
        '''群历史最优位置'''
        g_location = None
        '''群体历史最优值'''
        if maxvalue:
            g_ = 0.
        else:
            g_ = sys.maxsize
        while t < T:
            t += 1
            for index in range(len(p_location)):
                '''计算各粒子适应值'''
                _p = function(index, p_location, self.__data)
                '''更新历史最优值'''
                if maxvalue:
                    for _ in range(len(p_location)):
                        if p_value[_] < _p:
                            p_value[_] = _p
                    if _p > g_:
                        g_ = _p
                        g_location = p_location[index]
                else:
                    for _ in range(len(p_location)):
                        if p_value[_] > _p:
                            p_value[_] = _p
                    if _p < g_:
                        g_ = _p
                        g_location = p_location[index]
                '''更新粒子飞行速度及位置'''
                p_v[index] = (ω_min + (T - t) * (ω_max - ω_min) / T) * p_v[index] + c1 * ξ * (
                    p[index] - p_location[index]) + c2 * η * (g_location - p_location[index])
                '''纠正超速'''
                for v in range(len(p_v[index][0])):
                    if p_v[index][0][v] < scopev[0]:
                        p_v[index][0][v] = scopev[0]
                    if p_v[index][0][v] > scopev[1]:
                        p_v[index][0][v] = scopev[1]
                p_location[index] += r * p_v[index]
        return p_location


if __name__ == '__main__':
    ea = EA()
    ea.init_data(datapath=datapath)
