# -*-coding:utf-8-*-
"""
    各种距离计算

Author:Byshx

Date:2017.03.31

距离计算方式：欧氏距离、余弦相似计算、马氏距离
"""

import numpy as np


class Distance(object):
    """"""
    """
        ***欧氏距离***
        欧拉距离是最经典的一种距离算法，适用于求解两点之间直线的距离，适用于各个向量标准统一的情况，
        如各种药品的使用量、商品的售销量等。
    """

    @staticmethod
    def euclidean_metric(d1, d2):
        d = d1 - d2
        return np.dot(d.T, d)[0][0] ** 0.5

    """
        ***余弦相似度计算***
        余弦距离不关心向量的长度，而只关心向量的夹角余弦。应用场景，如文本分类时，两文本之间距离计算。
    """

    @staticmethod
    def cosine_similarity(d1, d2):
        cosθ = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        return cosθ

    """
        ***马氏距离***
        马氏距离表示数据的协方差距离。马氏距离与欧氏距离的不同之处在于，马氏距离考虑了各个维度之间的
        联系(如身高与体重)，并且独立于测量尺度（米、千克等不同量纲）。
    """

    @staticmethod
    def mahalanobis_distance(d1, d2):
        pass
