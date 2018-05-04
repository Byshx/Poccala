# -*-coding:utf-8-*-
"""
    工具模块

Author: Byshx

Date: 2018.04.09
"""
import math
import numpy as np

#############
# 高斯函数常量
G_VALUE_1 = np.log10(2 * math.pi)
G_VALUE_2 = 2 * math.pi


#############

def gaussian_function(y, mean, cov, dimension, log=False, standard=False):
    """根据高斯模型计算估计值 y为数据(行向量) mean为模型参数(mean为行向量) cov为协方差矩阵,log为是否计算对数值,standard为是否规范化"""
    x = y - mean
    diag = cov.diagonal()
    if standard:
        x *= diag ** 0.5
        diag = np.ones_like(diag)
    np.seterr(all='ignore')  # 取消错误提示
    if log:
        func = - dimension / 2 * G_VALUE_1 - 0.5 * np.sum(diag)
        exp = -0.5 * np.dot(x * (1. / diag), x)
        return func + exp
    else:
        covariance = G_VALUE_2 ** (dimension / 2) * np.prod(diag) ** 0.5
        func = 1. / covariance
        exp = np.exp(-0.5 * np.dot(x * (1. / diag), x))
        return func * exp


def matrix_dot(data1, data2, axis=0):
    """矩阵相乘"""
    p = []
    i, j = data2.shape
    if axis == 0:
        for index in range(i):
            tmp_p_list = data1 + data2[index, :]
            p.append(log_sum_exp(tmp_p_list))
    elif axis == 1:
        for index in range(j):
            tmp_p_list = data1 + data2[:, index]
            p.append(log_sum_exp(tmp_p_list))
    return np.array(p)


def log_sum_exp(p_list, vector=False):
    """
    计算和的对数
    :param p_list: 一维/多维数组(Array)或列表(List)
    :param vector: 是否为向量运算
    :return:
    """

    def cal(v):
        max_p = np.max(v)
        if abs(max_p) == float('inf'):
            return max_p
        r = max_p + np.log10(np.sum(np.exp(v - max_p)))
        return r

    if vector:
        result = []
        for index in range(len(p_list)):
            result.append(cal(p_list[index]))
        if type(p_list) is np.ndarray:
            return np.array(result)
        return result
    else:
        return cal(p_list)


def matrix_log_sum_exp(array_list, axis_x):
    """
    计算矩阵和的对数
    :param array_list: 矩阵列表
    :param axis_x: 纵向矩阵高度值
    """
    array = []
    for i in range(axis_x):
        index_list = [i] * len(array_list)
        array_axis_x = log_sum_exp(np.array(list(map(lambda x, y: x[y, :], array_list, index_list))).T,
                                   vector=True)
        array.append(array_axis_x)
    return np.array(array)
