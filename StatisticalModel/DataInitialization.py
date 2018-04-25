# -*-coding:utf-8-*-

"""

数据初始化模块
Data Initialization Module

Author:Byshx

Date:2017.03.30

"""

import sys
import random
import numpy as np


class DataInitialization(object):
    def __init__(self):
        """原数据"""
        self.__markdata = {}
        """数据维度"""
        self.__dimension = 0
        """数据类别"""
        self.__k = 0
        """乱序数据"""
        self.__data = []
        """数据量"""
        self.__data_size = 0

    def init_data(self, data=None, datapath=None, shuffle=False, continuous=True, matrix=True, hasmark=True):
        """
        从文本中获取数据
        (做实验时用的方法)
        :param data: 从参数传入数据
        :param datapath: 数据文件路径
        :param shuffle: 对数据进行乱序处理，默认为False
        :param continuous: 连续数值（非离散数据），默认为True
        :param matrix: 导入数据是否矩阵化，默认为True
        :param hasmark: 数据是否标注，默认为True
        :return:
        
        """
        if data is not None:
            self.__data = data
            self.__data_size = len(data)
        elif datapath is not None:
            with open(datapath) as f:
                """打印数据说明"""
                print(f.readline())
                """数据大小、维度和类别数"""
                s = f.readline().strip('\n').split(' ')
                self.__dimension = int(s[1])
                self.__k = int(s[2])
                print('数据大小和类别数:%d %d\n' % (int(s[0]), int(s[2])))
                if hasmark:
                    """创建字典"""
                    for i in range(self.__k):
                        self.__markdata[s[i + 3]] = []
                """收集数据"""
                s = f.readline()
                while s:
                    s = s.strip('\n').split(',')
                    tmpdata = []
                    if continuous:
                        for i in range(self.__dimension):
                            tmpdata.append(float(s[i]))
                    else:
                        for i in range(self.__dimension):
                            tmpdata.append(s[i])
                    """将数据添加并转换为矩阵"""
                    if hasmark:
                        if matrix:
                            self.__markdata[s[self.__dimension]].append(np.array(tmpdata))
                        else:
                            self.__markdata[s[self.__dimension]].append(tmpdata)
                    if matrix:
                        self.__data.append(np.array(tmpdata))
                    else:
                        self.__data.append(tmpdata)
                    s = f.readline()
                self.__data_size = len(self.__data)
        else:
            sys.stderr.write('Error: 未传入数据')
            raise Exception

        if shuffle:
            """对_data做乱序处理"""
            random.shuffle(self.__data)

    def add_data(self, data):
        """添加新数据"""
        self.__data.extend(data)
        self.__data_size += len(data)

    def clear_data(self):
        """清空数据内存"""
        self.__data = []
        self.__data_size = 0

    @property
    def markdata(self):
        return self.__markdata

    @property
    def dimension(self):
        return self.__dimension

    @property
    def classes(self):
        return self.__k

    @property
    def data(self):
        return self.__data

    @property
    def datasize(self):
        return self.__data_size
