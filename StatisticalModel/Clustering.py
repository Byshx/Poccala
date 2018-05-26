# -*-coding:utf-8-*-

"""
Clustering 聚类算法

高斯混合模型(GMM,基于 Expectation Maximization Algorithm)、K-means聚类算法(EM算法特例)、层次聚类、自组织映射神经网络(SOM)

Author:Byshx

Date:2017.03.21

"""
import os
import sys
import time
import copy
import random
import configparser
from Exceptions import *
from StatisticalModel.util import *
from StatisticalModel.ANN import ANN
from StatisticalModel.Distance import Distance
from StatisticalModel.DataInitialization import DataInitialization


class Clustering(DataInitialization):
    def __init__(self):
        """"""
        super().__init__()

    """
        高斯混合模型
        初始化参数需要聚类分析算法
    """

    class GMM(object):
        def __init__(self, log=None, dimension=1, mix_level=1, data=None, alpha=None, mean=None, variance=None,
                     covariance=None, differentiation=True, gmm_id=0):
            """
                初始化参数
            :param log: 记录日志
            :param dimension: 数据维度
            :param mix_level: 高斯混合度
            :param data: 载入的数据
            :param alpha: 高斯加权系数
            :param mean: 均值
            :param variance: 特征向量各维度的方差 矩阵shape=(mix_level,dimension)
            :param covariance: 协方差矩阵 矩阵shape=(mix_level,dimension,dimension)
            :param differentiation: 初始值差异化
            :param gmm_id: gmm模型id，用于区分
            """
            if data:
                self.__data = np.array(data)  # shape = (数据量，数据维度)
            else:
                self.__data = None
            """数据维度"""
            self.__dimension = dimension
            """模型数量"""
            self.__mix_level = mix_level
            """记录训练日志"""
            if log:
                self.log = log
            else:
                self.log = Clustering.LogInfoPrint()
            """初始参数"""
            if mean is None:
                if differentiation:
                    self.__mean = np.random.random((mix_level, dimension))
                else:
                    self.__mean = np.zeros((mix_level, dimension))
            else:
                self.__mean = mean
            '''生成对角矩阵'''
            if covariance is not None:
                self.__covariance = covariance
            elif variance is not None:
                self.__covariance = np.array([np.diag(variance[_, :]).reshape((1, dimension, dimension)) for _ in
                                              range(mix_level)])
            else:
                if differentiation:
                    self.__covariance = np.diag(np.random.random((dimension,))).reshape(
                        (1, dimension, dimension)).repeat(self.__mix_level, axis=0)
                else:
                    self.__covariance = 1e-3 * np.eye(dimension).reshape((1, dimension, dimension)).repeat(
                        self.__mix_level, axis=0)

            if alpha is None:
                self.__alpha = 1. / mix_level * np.ones((mix_level,))
            else:
                self.__alpha = alpha
            if data:
                """gamma的期望"""
                self.__gamma = np.ones((mix_level, len(data)))
            '''在计算一个数据的观测概率密度时，将各高斯分量的输出值(v = alpha[i] * N(mean[i],covariance[i]))保存在record'''
            self.__record = []  # 结构：[数据1：[g_value1,g_value2,...g_valueN]，数据2：[g_value1...] ...] 按数据出现的时间顺序保存
            '''Accumulator'''
            np.seterr(divide='ignore')
            self.__alpha_acc = -np.inf
            self.__mean_acc = np.log(np.zeros_like(self.__mean))
            self.__covariance_acc = [np.log(np.zeros((self.__dimension,))) for _ in range(self.__mix_level)]
            self.__acc = np.log(np.zeros_like(self.__alpha))
            '''使mean值非负的偏移值'''
            self.__bias = 100.
            self.__gmm_id = gmm_id

        def add_data(self, data):
            """
            加入新数据
            :param data:
            :return:
            """
            if self.__data:
                self.__data = np.append(self.__data, np.array(data), axis=0)
            else:
                self.__data = np.array(data)
            self.__gamma = np.ones((self.__mix_level, len(self.__data)))

        def clear_data(self):
            """清空所有数据"""
            self.__data = None

        @property
        def mean(self):
            return self.__mean

        @mean.setter
        def mean(self, mean):
            self.__mean = mean

        @property
        def covariance(self):
            return self.__covariance

        @covariance.setter
        def covariance(self, covariance):
            self.__covariance = covariance

        @property
        def alpha(self):
            return self.__alpha

        @alpha.setter
        def alpha(self, alpha):
            self.__alpha = alpha

        @property
        def dimension(self):
            """数据维度"""
            return self.__dimension

        @property
        def mixture(self):
            """高斯混合度"""
            return self.__mix_level

        @mixture.setter
        def mixture(self, mix_level):
            self.__mix_level = mix_level
            self.__gamma = np.ones((mix_level, len(self.__data)))

        @property
        def data(self):
            """获得数据"""
            return self.__data

        @data.setter
        def data(self, data):
            """
            载入新数据
            :param data:
            :return:
            """
            self.__data = np.array(data)
            self.__gamma = np.ones((self.__mix_level, len(self.__data)))

        @property
        def acc(self):
            """获取辅助变量累加器"""
            return self.__acc

        @acc.setter
        def acc(self, acc):
            """设置新的辅助变量累加器"""
            self.__acc = acc

        @property
        def alpha_acc(self):
            """获取高斯权重累加器"""
            return self.__alpha_acc

        @alpha_acc.setter
        def alpha_acc(self, alpha_acc):
            """设置新的高斯权重累加器"""
            self.__alpha_acc = alpha_acc

        @property
        def mean_acc(self):
            """获取均值累加器"""
            return self.__mean_acc

        @mean_acc.setter
        def mean_acc(self, mean_acc):
            """设置新的均值累加器"""
            self.__mean_acc = mean_acc

        @property
        def covariance_acc(self):
            """获取协方差累加器"""
            return self.__bias

        @covariance_acc.setter
        def covariance_acc(self, covariance_acc):
            """设置新的协方差累加器"""
            self.__covariance_acc = covariance_acc

        @property
        def bias(self):
            """获取偏移值"""
            return self.__bias

        @bias.setter
        def bias(self, bias):
            """设置新的偏移值"""
            self.__bias = bias

        @property
        def gmm_id(self):
            """模型编号"""
            return self.__gmm_id

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """参数操作"""

        def save_parameter(self, path):
            """
            参数保存
            :param path: 参数保存地址
            :type path: str
            :return:
            """
            path.strip('/')
            path = path + '/GMM_%d' % self.__gmm_id
            if not os.path.exists(path):
                os.mkdir(path)
            np.save(path + '/GMM_means.npy', self.__mean)  # 保存均值
            np.save(path + '/GMM_covariance.npy', self.__covariance)  # 保存协方差矩阵
            np.save(path + '/GMM_weight.npy', self.__alpha)  # 保存权重矩阵
            with open(path + '/GMM_config.ini', 'w+') as gmm_config_file:
                '''保存其他配置参数'''
                gmm_config = configparser.ConfigParser()
                gmm_config.add_section('Configuration')
                gmm_config.set('Configuration', 'MIXTURE', value=str(self.__mix_level))
                gmm_config.set('Configuration', 'DIMENSION', value=str(self.__dimension))
                gmm_config.set('Configuration', 'BIAS', value=str(self.__bias))
                gmm_config.write(gmm_config_file)

        def save_acc(self, path):
            """
            累加器保存
            :param path: 累加器保存地址
            :type path: str
            :return:
            """
            path.strip('/')
            path = path + '/GMM_%d' % self.__gmm_id
            if not os.path.exists(path):
                os.mkdir(path)
            path_acc = path + '/acc'
            path_alpha_acc = path + '/alpha-acc'
            path_mean_acc = path + '/mean-acc'
            path_covariance_acc = path + '/covariance-acc'
            try:
                os.mkdir(path_acc)
                os.mkdir(path_alpha_acc)
                os.mkdir(path_mean_acc)
                os.mkdir(path_covariance_acc)
            except FileExistsError:
                pass
            localtime = int(time.time())  # 时间戳
            np.save(path_acc + '/GMM_acc_%d.npy' % localtime, self.__acc)  # 保存輔助变量累加器
            np.save(path_alpha_acc + '/GMM_alpha_acc_%d.npy' % localtime, self.__alpha_acc)  # 保存高斯权重累加器
            np.save(path_mean_acc + '/GMM_mean_acc_%d.npy' % localtime, self.__mean_acc)  # 保存均值累加器
            np.save(path_covariance_acc + '/GMM_covariance_acc_%d.npy' % localtime, self.__covariance_acc)  # 保存协方差累加器

        def init_parameter(self, path):
            """
            参数读取
            :param path: 参数读取地址
            :type path: str
            :return:
            """
            path.strip('/')
            path = path + '/GMM_%d' % self.__gmm_id
            if not os.path.exists(path):
                raise ParameterFileExistsError(self.log)
            '''读取均值、协方差、权重矩阵'''
            mean = np.load(path + '/GMM_means.npy')
            covariance = np.load(path + '/GMM_covariance.npy')
            alpha = np.load(path + '/GMM_weight.npy')
            '''将数据初始化到GMM'''
            self.__mean = mean
            self.__covariance = covariance
            self.__alpha = alpha
            with open(path + '/GMM_config.ini', 'r+') as gmm_config_file:
                gmm_config = configparser.ConfigParser()
                gmm_config.read(gmm_config_file)
                sections = gmm_config.sections()
                for section in sections:
                    items = dict(gmm_config.items(section))
                    self.__mix_level = int(items['mixture'])
                    self.__dimension = int(items['dimension'])
                    self.__bias = float(items['bias'])

        def init_acc(self, path):
            """
            收集基元参数目录下的所有acc文件，并初始化到GMM中
            :param path: 累加器读取地址
            :type path: str
            :return:
            """
            path.strip('/')
            path = path + '/GMM_%d' % self.__gmm_id
            path_acc = path + '/acc'
            path_alpha_acc = path + '/alpha-acc'
            path_mean_acc = path + '/mean-acc'
            path_covariance_acc = path + '/covariance-acc'
            '''读取acc文件'''
            self.__acc = self.__acc.reshape(-1, 1)
            for dir in os.walk(path_acc):
                for filename in dir[2]:
                    file = open(path_acc + '/' + filename, 'rb')
                    data = np.load(file)
                    self.__acc = np.append(self.__acc, data.reshape(-1, 1), axis=1)
                    file.close()
            '''累加acc'''
            self.__acc = log_sum_exp(self.__acc, vector=True)
            '''读取alpha_acc文件'''
            for dir in os.walk(path_alpha_acc):
                for filename in dir[2]:
                    file = open(path_alpha_acc + '/' + filename, 'rb')
                    data = np.load(file)
                    self.__alpha_acc = np.append(self.__alpha_acc, data)
                    file.close()
            '''累加alpha_acc'''
            self.__alpha_acc = log_sum_exp(self.__alpha_acc)
            '''读取mean_acc文件'''
            mean_acc = [np.log(np.zeros((self.__dimension, 1))) for _ in range(self.__mix_level)]
            for dir in os.walk(path_mean_acc):
                for filename in dir[2]:
                    file = open(path_mean_acc + '/' + filename, 'rb')
                    data = np.load(file)
                    for index in range(self.__mix_level):
                        mean_acc[index] = np.append(mean_acc[index], data[index].reshape(-1, 1), axis=1)
                    file.close()
            for index in range(self.__mix_level):
                self.__mean_acc[index] = log_sum_exp(mean_acc[index], vector=True)
            '''读取covariance_acc文件'''
            covariance_acc = [np.log(np.zeros((self.__dimension, 1))) for _ in range(self.__mix_level)]
            for dir in os.walk(path_covariance_acc):
                for filename in dir[2]:
                    file = open(path_covariance_acc + '/' + filename, 'rb')
                    data = np.load(file)
                    for index in range(self.__mix_level):
                        covariance_acc[index] = np.append(covariance_acc[index], data[index].reshape(-1, 1), axis=1)
                    file.close()
            for index in range(self.__mix_level):
                self.__covariance_acc[index] = log_sum_exp(covariance_acc[index], vector=True)

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        """     ***************     Split and Merge     *****************     """

        def __J_merge(self):
            """
            计算合并得分并排序
            :return: 
            """
            j_merge_list = []
            for i in range(self.__mix_level):
                for j in range(i + 1, self.__mix_level):
                    j_point = np.dot(self.__gamma[i, :], self.__gamma[j, :].reshape((len(self.__gamma[j, :]), 1))) / \
                              (np.linalg.norm(self.__gamma[i, :], axis=0) * np.linalg.norm(self.__gamma[j, :], axis=0))
                    j_merge_list.append([i, j, j_point])
            '''排序'''
            j_merge_list.sort(key=lambda x: x[2], reverse=True)
            return j_merge_list

        def __J_split(self):
            """
            计算分裂得分并排序
            :return: 
            """

            def cal_p(x, data_index, k):
                p_gamma = self.__gamma[k, :]
                p_1 = 0.
                p_2 = np.sum(p_gamma)
                d_distance_list = []

                for index in range(len(data_index[k])):
                    sub = x - self.__data[data_index[k][index]]
                    distance = np.dot(sub, sub.reshape((self.__dimension, 1)))
                    d_distance_list.append([data_index[k][index], distance])

                d_distance_list.sort(key=lambda y: y[1])
                for index in range(len(d_distance_list)):
                    p_1 += index / len(d_distance_list) * p_gamma[d_distance_list[index][0]]
                return p_1 / p_2

            '''将属于不同GMM的数据分类'''
            data = [[] for _ in range(self.__mix_level)]
            for i in range(len(self.__data)):
                max_p = np.max(self.__gamma[:, i])
                loc = np.where(self.__gamma[:, i] == max_p)[0]  # 最大值所在位置
                max_index = loc[0]  # 若有多个点(特殊情况)，则选取第一个点
                data[max_index].append(i)

            '''计算分裂分数'''
            j_split_list = []
            for i in range(self.__mix_level):
                j_point = 0.
                for j in range(len(self.__data)):
                    p = cal_p(self.__data[j], data, i)
                    j_point += p * np.log(
                        p / gaussian_function(self.__data[j], self.__mean[i], self.__covariance[i], self.__dimension,
                                              standard=True))
                j_split_list.append([i, j_point])
            j_split_list.sort(key=lambda y: y[1], reverse=True)
            return j_split_list

        def __merge(self, x, y):
            """
            模型合并
            :return: 
            """
            new_alpha = self.__alpha[x] + self.__alpha[y]
            new_mean = (self.__mean[x] * self.__alpha[x] + self.__mean[y] * self.__alpha[y]) / new_alpha
            new_covariance = (self.__covariance[x] * self.__alpha[x] + self.__covariance[y] * self.__alpha[
                y]) / new_alpha
            return new_mean, new_covariance, new_alpha

        def __split(self, x):
            """
            模型分裂
            :return: 
            """
            data = []
            for index in range(len(self.__data)):
                max_p = np.max(self.__gamma[:, index])
                max_index = np.where(self.__gamma[:, index] == max_p)[0][0]
                if max_index == x:
                    data.append(self.__data[index])

            if len(data) < self.__mix_level:
                return False

            cluster = Clustering.ClusterInitialization(data, 2, self.__dimension, self.log)
            mean_, variance_, alpha_, cluster_data = cluster.kmeans(algorithm=1)

            new_mean_1 = mean_[0] + np.random.rand(self.__dimension) * 1e-2
            new_mean_2 = mean_[1] + np.random.rand(self.__dimension) * 1e-2
            new_covariance_1 = np.eye(self.__dimension) * np.linalg.det(self.__covariance[x]) ** (1 / self.__dimension)
            new_covariance_2 = np.copy(new_covariance_1)
            new_alpha_1 = self.__alpha[x] * 0.5
            new_alpha_2 = self.__alpha[x] * 0.5
            new_sp = [[new_mean_1, new_mean_2], [new_covariance_1, new_covariance_2], [new_alpha_1, new_alpha_2]]
            return new_sp

        def __reestimate(self, new_mean, new_covariance, new_alpha, gamma_sum):
            """重估"""
            new_gamma = np.ones((len(new_mean), len(self.__data)))
            for i in range(len(self.__data)):
                tmp_gamma_list = []
                for j in range(len(new_mean)):
                    new_gamma[j][i] = gaussian_function(self.__data[i], new_mean[j], new_covariance[j],
                                                        self.__dimension,
                                                        log=True) + np.log(new_alpha[j])
                    tmp_gamma_list.append(new_gamma[j][i])
                """得到列之和，更新gamma表"""
                new_gamma[:, i] = np.exp(new_gamma[:, i] - log_sum_exp(tmp_gamma_list)) * gamma_sum[i]
            return new_gamma

        def __SMEM(self, q_value, c_max=5, c_covariance=1e-3):
            """
            执行SMEM算法
            :param q_value: 终止迭代时的Q值
            :param c_max: 候选模型数
            :param c_covariance: 纠正协方差矩阵值过小问题
            """

            if self.__mix_level < 3:
                self.log.note('聚类数目小于3，结束SMEM算法', cls='w')
                return False

            '''保存先前的值'''
            save_mean = np.copy(self.__mean)
            save_covariance = np.copy(self.__covariance)
            save_alpha = np.copy(self.__alpha)
            save_gamma = np.copy(self.__gamma)
            save_mix_level = self.__mix_level
            ''''''

            j_merge_list = self.__J_merge()
            j_split_list = self.__J_split()
            candidates = []
            try:
                for index in range(len(j_merge_list)):
                    model_1, model_2 = j_merge_list[index][:-1]
                    for index_ in range(len(j_split_list)):
                        model_3 = j_split_list[index_][0]
                        if model_3 == model_1 or model_3 == model_2:
                            continue
                        else:
                            candidates.append([model_1, model_2, model_3])
                            if len(candidates) == c_max:
                                raise PassException
            except PassException:
                pass
            '''备份参数'''

            for index in range(len(candidates)):
                mean_list, covariance_list, alpha_list = [], [], []
                new_mean, new_covariance, new_alpha = self.__merge(candidates[index][0], candidates[index][1])
                new_sp = self.__split(candidates[index][2])

                if new_sp is False:
                    continue

                mean_list.append(new_mean)
                mean_list.extend(new_sp[0])
                covariance_list.append(new_covariance)
                covariance_list.extend(new_sp[1])
                alpha_list.append(new_alpha)
                alpha_list.extend(new_sp[2])

                gamma_sum = []
                for index_ in range(len(self.__data)):
                    column_sum = self.__gamma[candidates[index][0], index_] + self.__gamma[
                        candidates[index][1], index_] + self.__gamma[candidates[index][2], index_]
                    gamma_sum.append(column_sum)
                gamma = self.__reestimate(mean_list, covariance_list, alpha_list, gamma_sum)

                self.__mean = mean_list
                self.__covariance = covariance_list
                self.__alpha = alpha_list
                self.__gamma = gamma
                self.__mix_level = 3

                new_mean_list, new_covariance_list, new_alpha_list = self.maximization(c_covariance=c_covariance)
                self.__mean = new_mean_list
                self.__covariance = new_covariance_list
                self.__alpha = new_alpha_list

                q_1 = self.q_function()

                '''修正参数——删除候选行'''
                self.__mean = np.delete(save_mean, candidates[index], axis=0)
                self.__covariance = np.delete(save_covariance, candidates[index], axis=0)
                self.__alpha = np.delete(save_alpha, candidates[index])
                self.__gamma = np.delete(save_gamma, candidates[index], axis=0)
                self.__mix_level = save_mix_level - 3

                q_2 = self.q_function()
                new_q_value = q_1 + q_2
                if new_q_value > q_value:
                    self.__mean = np.append(self.__mean, new_mean_list, axis=0)
                    self.__covariance = np.append(self.__covariance, new_covariance_list, axis=0)
                    self.__alpha = np.append(self.__alpha, new_alpha_list, axis=0)
                    self.__gamma = np.append(self.__gamma, gamma, axis=0)
                    return new_q_value
                else:
                    self.__mean = save_mean
                    self.__covariance = save_covariance
                    self.__alpha = save_alpha
                    self.__gamma = save_gamma
                    self.__mix_level = save_mix_level
                    return False

        """     **********************  Expectation  ********************     """

        """计算gamma期望"""

        def expectation(self):
            """gamma代表该数据属于模型k的期望
               表示模型k对观测数据yj的响应程度"""
            """""""""""""""""""""""""""""""""""""""""""""
                             P(gammajk = 1|θ)
                gamma  =  ------------------------------
                       ∑K,k=1P(yj|gammajk=1,θ)P(gammajk=1|θ)
            """""""""""""""""""""""""""""""""""""""""""""
            for i in range(len(self.__data)):
                tmp_gamma_list = []
                for j in range(self.__mix_level):
                    self.__gamma[j][i] = gaussian_function(self.__data[i], self.__mean[j], self.__covariance[j],
                                                           self.__dimension, log=True) + np.log(
                        self.__alpha[j])
                    tmp_gamma_list.append(self.__gamma[j][i])
                """得到列之和，更新gamma表"""
                self.__gamma[:, i] = self.__gamma[:, i] - log_sum_exp(tmp_gamma_list)

        """
            EM算法核心：Q函数
            完全数据的对数似然函数关于给定观测数据和当前参数alpha_k、mean_k、variance_k下
            对为观测数据的条件概率分布P(z|y,θ)的期望
        """

        def q_function(self):
            gamma_exp = np.exp(self.__gamma)
            gamma_sum_1 = np.sum(gamma_exp, axis=1)
            value_1 = np.sum(gamma_sum_1 * np.log(self.__alpha))
            value_2 = 0.
            for i in range(self.__mix_level):
                for j in range(len(self.__data)):
                    value_2 += gamma_exp[i][j] * gaussian_function(self.__data[j], self.__mean[i], self.__covariance[i],
                                                                   self.__dimension, log=True)
            return value_1 + value_2

        """     ******************  Maximization    *****************   """

        """
            计算参数值，求偏导，使对数似然函数值更大
        """

        def maximization(self, c_covariance=1e-3):
            """计算gamma_sum"""
            gamma_sum_array = log_sum_exp(self.__gamma, vector=True)
            data_t = self.__data.T
            data_t_bias = np.log(data_t + self.__bias)
            """计算新mean值"""
            mean_array = []
            for k in range(self.__mix_level):
                mean_array.append(log_sum_exp(self.__gamma[k, :] + data_t_bias, vector=True))
            new_mean = np.exp(np.array(mean_array) - gamma_sum_array.reshape(-1, 1)) - self.__bias

            """计算新covariance值"""
            covariance_array = []
            for k in range(self.__mix_level):
                covariance = log_sum_exp(self.__gamma[k, :] + np.log((data_t - new_mean[k].reshape(-1, 1)) ** 2),
                                         vector=True)
                covariance = np.exp(covariance - gamma_sum_array[k])
                if (covariance < c_covariance).any():
                    '''纠正协方差过小问题'''
                    self.log.note('当前计算的协方差矩阵中某些值过小，纠正至%s' % c_covariance, cls='w')
                    covariance[np.where(covariance < c_covariance)] = c_covariance
                covariance_array.append(np.diag(covariance))
            new_covariance = covariance_array

            """计算新alpha值"""
            new_alpha = np.exp(gamma_sum_array) / len(self.__data)

            return new_mean, new_covariance, new_alpha

        def update_acc(self, l_value, b_value, o_value):
            """
            更新累加器
            :param l_value: 由HMM计算出的累加值
            :param b_value: 对观测序列的观测概率密度
            :param o_value: 观测序列的值
            :return:
            """
            record_array = np.array(self.__record).T
            record_array += l_value - b_value
            o_value_t = o_value.T
            log_o_value_t = np.log(o_value_t + self.__bias)  # 对观测值取对数
            ''''''
            self.__acc = log_sum_exp(np.append(record_array, self.__acc.reshape(-1, 1), axis=1), vector=True)
            '''alpha'''
            self.__alpha_acc = log_sum_exp(np.append(l_value, self.__alpha_acc))
            '''mean'''
            for index in range(self.__mix_level):
                self.__mean_acc[index] = log_sum_exp(
                    np.append(log_o_value_t + record_array[index], self.__mean_acc[index].reshape(-1, 1), axis=1),
                    vector=True)
            '''covariance'''
            for index in range(self.__mix_level):
                self.__covariance_acc[index] = log_sum_exp(
                    np.append(record_array[index] + np.log((o_value_t - self.__mean[index].reshape(-1, 1)) ** 2),
                              self.__covariance_acc[index].reshape(-1, 1), axis=1), vector=True)
            '''清除record缓存'''
            self.__record = []

        def update_param(self, show_q=False, c_covariance=1e-3):
            """更新参数"""
            self.log.note('正在通过BW算法训练GMM_%d，混合度为 %d' % (self.__gmm_id, self.__mix_level), cls='i', show_console=show_q)
            self.__alpha = np.exp(self.__acc - self.__alpha_acc)
            self.__mean = np.exp(self.__mean_acc - self.__acc.reshape(-1, 1)) - self.__bias
            for index in range(self.__mix_level):
                c_exp = np.exp(self.__covariance_acc[index] - self.__acc[index])
                if (c_exp < c_covariance).any():
                    '''纠正协方差过小问题'''
                    self.log.note('当前计算的协方差矩阵中某些值过小，纠正至%s' % c_covariance, cls='w')
                    c_exp[np.where(c_exp < c_covariance)] = c_covariance  # 纠正协方差过小
                self.__covariance[index] = np.diag(c_exp)

        def em(self, show_q=False, smem=False, c_covariance=1e-3):
            """用于自动执行迭代的方法"""
            self.log.note('正在通过EM算法训练GMM_%d，混合度为 %d' % (self.__gmm_id, self.__mix_level), cls='i', show_console=show_q)
            if len(self.__data) == 0:
                raise DataUnLoadError(self.log)
            q_value = -float('inf')
            while True:
                self.log.note('GMM 当前似然度：%f' % q_value, cls='i', show_console=show_q)
                self.expectation()
                self.__mean, self.__covariance, self.__alpha = self.maximization(c_covariance=c_covariance)
                _q = self.q_function()
                if _q - q_value > 1.28:
                    q_value = _q
                else:
                    if smem:
                        self.log.note('执行SMEM算法...', cls='i', show_console=show_q)
                        new_q_value = self.__SMEM(q_value, c_covariance=c_covariance)
                        if new_q_value is not False:
                            q_value = new_q_value
                            continue
                        else:
                            self.log.note('迭代结束', cls='i', show_console=show_q)
                            break
                    else:
                        break

        def theta(self):
            """获得已经收敛的参数"""
            theta = {}
            for i in range(self.__mix_level):
                theta['theta_%d' % i] = [self.__mean[i], self.__covariance[i].diagonal()]
            return theta

        def gmm(self, x):
            """返回属于的模型序号"""
            p = 0.
            index = 0
            for i in range(self.__mix_level):
                tmp_p = gaussian_function(x, self.__dimension, self.__mean[i], self.__covariance[i], self.__dimension)
                if tmp_p > p:
                    p = tmp_p
                    index = i
            '''返回该高斯模型产生该数据的概率和该高斯模型的序号'''
            return p, index

        def point(self, x, log=False, standard=False, record=False):
            """
            计算该高斯模型的得分(概率)
            :param x: 数据
            :param log: 对数形式
            :param standard: 以标准正态分布输出后验概率密度
            :param record: 记录各高斯分量的输出值，按时间的先后顺序保存在
            :return:
            """
            n = len(x)
            if n != self.dimension:
                raise DataDimensionError(self.dimension, n, self.log)
            p = 0.
            if log:
                p_list = []
                for i in range(self.__mix_level):
                    p_list.append(
                        np.log(self.__alpha[i]) + gaussian_function(x, self.__mean[i], self.__covariance[i],
                                                                    self.__dimension, log=log, standard=standard))
                if record:
                    self.__record.append(p_list)
                p = log_sum_exp(p_list)
            else:
                for i in range(self.__mix_level):
                    p += self.__alpha[i] * gaussian_function(x, self.dimension, self.__mean[i], self.__covariance[i],
                                                             self.__dimension)
            '''返回该高斯模型产生该数据的概率'''
            return p

    """
        聚类分析算法 用于初始化中心点
    """

    class ClusterInitialization(object):
        def __init__(self, data, k, dimension, log=None):
            """含有因变量的数据"""
            self.__data = data
            """数据维度"""
            self.__dimension = dimension
            """样本类别"""
            self.__k = k
            """层次聚类算法得到的参数树"""
            self.__k_tree = None
            """记录日志"""
            if log:
                self.log = log
            else:
                self.log = Clustering.LogInfoPrint()

        '''
            距离计算
            arg=1 曼哈顿距离（Manhattan Distance）
            arg=2 欧几里德距离（Euclidean Distance）
            arg>2 明可夫斯基距离（Minkowski Distance）
        '''

        @staticmethod
        def cal_distance(d1, d2, arg=2):
            _distance = 0.
            for index in range(len(d1)):
                _distance += abs(d1[index] - d2[index]) ** arg
                return _distance ** (1 / arg)

        '''
            计算方差(Variance)
        '''

        @staticmethod
        def cal_variance(cluster, algorithm=None):
            """
            计算标准差
            :param cluster: 数据簇
            :param algorithm: 算法
            :return: 均方差
            """
            center = cluster[0]
            points = cluster[1]
            if algorithm == 'kmeans':
                points = points.values()
            variance = []
            '''遍历维度'''
            for index_d in range(len(center)):
                tmp_variance = 0.
                '''遍历其他点'''
                for point in points:
                    tmp_variance += (center[index_d] - point[index_d]) ** 2
                '''求平均：除以类内点个数'''
                tmp_variance /= len(points)
                '''为防止奇异矩阵，检测全零矩阵'''
                if tmp_variance < 1e-4:
                    tmp_variance = 1e-4
                variance.append(tmp_variance ** 0.5)
            return variance

        """
            K-means 聚类算法
        """

        def kmeans(self, algorithm=0, cov_matrix=False):
            """
            K-means聚类
            :param algorithm: 算法类别：
                                algorithm=0: k-means
                                algorithm=1: k-means++
                                algorithm=2: k-means2
            :param cov_matrix: 将方差转化为协方差矩阵
            :return:
            """
            """初始化样本类别标记列表"""
            """二维数组，第二维储存该样本所在类别列表的位置"""
            pointmark = [[-1, -1] for _ in range(len(self.__data))]
            """
                K-means 算法:
                1、随机取K（这里K=2）个种子点。
                2、对图中的所有点求到这K个种子点的距离
                3、接下来，我们要移动种子点到属于他的“点群”的中心。（见图上的第三步）
                4、然后重复第2）和第3）步，直到，种子点没有移动
            """

            def kmeans(k_index=None):
                """"""
                '''记录各数据所属类别'''
                if k_index is None:
                    k_index = []
                    '''生成k个不同的中心点'''
                    for i in range(self.__k):
                        while True:
                            random_center = random.randint(0, len(self.__data) - 1)
                            if random_center not in k_index:
                                k_index.append(
                                    [copy.deepcopy(self.__data[random_center]), {-1: self.__data[random_center]}])
                                '''从点记录中将自己(中心点)剔除'''
                                pointmark[random_center][0] = i
                                break
                '''
                    计算新的中心点
                    参数为围绕旧中心点的簇(cluster)
                    cluster格式：[中心点,[其他点]]
                '''

                def cal_center(cluster):
                    new_center = np.copy(cluster[0])
                    points = cluster[1]
                    '''遍历维度'''
                    for index_d in range(len(new_center)):
                        new_center[index_d] = 0.
                        '''遍历其他点'''
                        for point in points.values():
                            new_center[index_d] += point[index_d]
                        '''求平均：除以其他点个数加一(原来的位置)'''
                        new_center[index_d] /= len(points)
                    return new_center

                '''当没有样本移动时停止'''
                movepoint = True
                while movepoint:
                    movepoint = False
                    '''寻找邻近点(Expectation)'''
                    for k in range(self.__k):
                        min_distance = sys.maxsize
                        min_index = -1
                        '''类别间转移点 True 表明此点是类间转移'''
                        move_class = False
                        for i in range(len(self.__data)):
                            '''如果此点已分类则跳过'''
                            if pointmark[i][0] == k:
                                continue
                            '''当此点已经分类，比较自己到这个点的距离和这个点原来类别中心点到这个点的距离哪个更近，如果自己到这个点
                            距离更近，则将其划为自己类别中'''
                            distance = Clustering.ClusterInitialization.cal_distance(k_index[k][0], self.__data[i])

                            if pointmark[i][0] != -1:
                                distance_ = Clustering.ClusterInitialization.cal_distance(k_index[pointmark[i][0]][0],
                                                                                          self.__data[i])
                                if distance_ <= distance:  # distance_ < distance: 18.01.09
                                    continue

                            if distance < min_distance:
                                min_distance = distance
                                min_index = i
                                if pointmark[i][0] != -1 and pointmark[i][1] != -1:
                                    move_class = True
                                else:
                                    move_class = False
                        '''如果没有满足要求的点，此中心点退出循环'''
                        if min_index == -1:
                            break
                        '''将距离最小点划到自己的簇内'''
                        movepoint = True
                        if move_class:
                            del k_index[pointmark[min_index][0]][1][pointmark[min_index][1]]
                        '''随机哈希key，同类别内必须不重复'''
                        random_number = int(random.random() * 1e15)
                        while k_index[k][1].get(random_number):
                            random_number = int(random.random() * 1e15)
                        pointmark[min_index][1] = random_number
                        pointmark[min_index][0] = k
                        k_index[k][1][random_number] = self.__data[min_index]
                    '''重新计算中心点(Maximization)'''
                    for k in range(self.__k):
                        k_index[k][0] = cal_center(k_index[k])
                meanlist = [k_index[_][0] for _ in range(self.__k)]
                '''计算方差'''
                variancelist = [Clustering.ClusterInitialization.cal_variance(k_index[_], algorithm='kmeans') for _ in
                                range(self.__k)]
                '''返回均值和方差'''
                '''计算每类样本数占总样本数的比例alpha'''
                alpha = [(len(k_index[_][1]) / len(self.__data)) for _ in range(len(k_index))]

                '''获取每簇数据'''
                clustered_data = []
                for i in range(self.__k):
                    clustered_data.append(list(k_index[i][1].values()))
                '''转换为矩阵'''
                meanlist, variancelist = np.array(meanlist), np.array(variancelist)
                if cov_matrix:
                    covariance = []
                    for index in range(len(variancelist)):
                        variance = variancelist[index]
                        covariance.append(np.diag(variance ** 2))
                    variancelist = np.array(covariance)
                return meanlist, variancelist, alpha, clustered_data

            """
                K-means++算法：
                1、先从数据中随机挑个随机点当“种子点”。
                2、对于每个点，我们都计算其和最近的一个“种子点”的距离D(x)并保存在一个数组里，
                然后把这些距离加起来得到Sum(D(x))。
                3、然后，再取一个随机值，用权重的方式来取计算下一个“种子点”。这个算法的实现是，
                先取一个能落在Sum(D(x))中的随机值Random，然后用Random -= D(x)，直到其<=0，
                此时的点就是下一个“种子点”。
                4、重复第（2）和第（3）步直到所有的K个种子点都被选出来。
                5、进行K-Means算法。
            """

            def kmeans_():
                """"""
                k_index = []
                # while True:
                '''randomint包括区间端点'''
                random_center = random.randint(0, len(self.__data) - 1)
                # if random_center not in k_index:
                k_index.append([copy.deepcopy(self.__data[random_center]), {-1: self.__data[random_center]}])
                '''设自己为第一个中心点0,并从样本类别列表中将自己(中心点)剔除'''
                pointmark[random_center][0] = 0
                # break
                '''选取剩余中心点'''
                sumdistance = 0.
                distancelist = []
                for d in self.__data:
                    mindistance = sys.maxsize
                    for center in k_index:
                        distance = Clustering.ClusterInitialization.cal_distance(center[0], d)
                        if distance < mindistance:
                            mindistance = distance
                    distancelist.append(mindistance)
                    sumdistance += mindistance
                '''
                异常数据，数据各分量均一致 
                sumdistance = 0.
                distancelist = [0.0, 0.0, ...
                '''
                if sumdistance == 0.:
                    index = random.sample(range(0, len(distancelist)), self.__k - 1)  # 中心点所在数据点的索引集合
                    for k in range(1, self.__k):
                        for i in index:
                            k_index.append([copy.deepcopy(self.__data[i]), {-1: self.__data[i]}])
                            pointmark[i][0] = k
                    assert len(k_index) == self.__k, '中心点缺少'
                    return kmeans(k_index=k_index)
                for k in range(1, self.__k):
                    '''取随机数'''
                    randomnumber = random.randint(0, int(sumdistance))
                    for index in range(len(distancelist)):
                        randomnumber -= distancelist[index]
                        if randomnumber < 0:
                            k_index.append([copy.deepcopy(self.__data[index]), {-1: self.__data[index]}])
                            pointmark[index][0] = k
                            break
                assert len(k_index) == self.__k, '中心点缺少'
                return kmeans(k_index=k_index)

            """
                            k-means∥算法原理分析:
                    k-means++ 最主要的缺点在于其内在的顺序执行特性，得到 k 个聚类中心必须遍历
                数据集 k 次，并且当前聚类中心的计算依赖于前面得到的所有聚类中心，这使得算法无
                法并行扩展，极大地限制了算法在大规模数据集上的应用。
                    k-means∥ 主要思路在于改变每次遍历时的取样策略，并非按照 k-means++ 那样
                每次遍历只取样一个样本，而是每次遍历取样 O(k) 个样本，重复该取样过程大约 O(logn)次，
                重复取样过后共得到 O(klogn) 个样本点组成的集合，该集合以常数因子近似于最优解，然后再
                聚类这 O(klogn) 个点成 k 个点，最后将这 k 个点作为初始聚类中心送入Lloyd迭代中，实际
                实验证明 O(logn) 次重复取样是不需要的，一般5次重复取样就可以得到一个较好的聚类初始中心
            """

            def kmeans2():
                pass

            if algorithm is 0:
                return kmeans()
            elif algorithm is 1:
                return kmeans_()
            elif algorithm is 2:
                kmeans2()
            else:
                raise ClassError(self.log)

        def ckmeans(self):
            """
            k-means算法的C++实现
            :return:
            """
            pass

        """
            随机中心算法
        """

        def randomcenter(self):
            k_index = []
            for i in range(self.__k):
                while True:
                    '''randomint包括区间端点'''
                    random_center = random.randint(0, len(self.__data) - 1)
                    if random_center not in k_index:
                        k_index.append([self.__data[random_center], [self.__data[random_center]]])
                        break
            for d in self.__data:
                mindistance = sys.maxsize
                min_class = -1
                for k in range(self.__k):
                    distance = Clustering.ClusterInitialization.cal_distance(k_index[k][0], d)
                    if distance < mindistance:
                        mindistance = distance
                        min_class = k
                k_index[min_class][1].append(d)
            '''计算均值'''
            meanlist = [k_index[_][0] for _ in range(self.__k)]
            '''计算方差'''
            variancelist = [Clustering.ClusterInitialization.cal_variance(k_index[_]) for _ in range(self.__k)]
            '''返回均值和方差'''
            '''计算每类样本数占总样本数的比例alpha'''
            alpha = [(len(k_index[_][1]) / len(self.__data)) for _ in range(len(k_index))]
            return meanlist, variancelist, alpha

        """
            层次聚类算法 (Hierarchical Clustering)
        """

        def layercluster(self):
            """"""
            '''格式:[[当前中心点,[同类样本点]] * 样本个数]'''
            k_index = [[self.__data[_], [self.__data[_]]] for _ in range(len(self.__data))]
            '''格式:[合并序号(用于参数获取),节点数，当前均值方差,[均值方差的左子树],[均值方差的右子树]]'''
            k_tree = [[0, 1, [self.__data[_], 0.], [], []] for _ in range(len(self.__data))]
            '''合并次数'''
            operation = 0
            while len(k_index) != self.__k:
                operation += 1
                print('合并次数:', operation)
                mindistance = sys.maxsize
                _min_index = -1
                min_index_ = -1
                for i in range(len(k_index)):
                    for j in range(len(k_index)):
                        if i == j:
                            continue
                        distance = Clustering.ClusterInitialization.cal_distance(k_index[i][0], k_index[j][0])
                        if distance < mindistance:
                            mindistance = distance
                            _min_index = i
                            min_index_ = j
                '''计算两点平均值'''
                k_index[_min_index][0] = k_index[_min_index][0] / 2 + k_index[min_index_][0] / 2
                '''将k_index[min_index][2]的样本集extend到k_index[i][2]'''
                k_index[_min_index][1].extend(k_index[min_index_][1])
                del k_index[min_index_]
                '''若构建可查找树，计算新方差，并合树'''
                variance_ = Clustering.ClusterInitialization.cal_variance(k_index[_min_index])
                k_tree[_min_index] = [operation, k_tree[_min_index][1] + k_tree[min_index_][1],
                                      [k_index[_min_index][0], variance_],
                                      k_tree[_min_index], k_tree[min_index_]]
                del k_tree[min_index_]
            '''对不可查找树计算最终的方差'''
            '''成员变量获取结果'''
            self.__k_tree = k_tree

        '''用于获得层次聚类算法所得参数'''

        def theta(self, k=None):

            if k is None:
                k = self.__k
            elif k < len(self.__k_tree):
                raise Exception('Error: 层次未达到指定类别，尝试减少类别数至<=k，并重新执行层次聚类算法')
            if self.__k_tree is None:
                raise Exception('Error: 在执行算法前尝试获得结果')

            _mean, _variance, alpha = [], [], []
            _tree = copy.deepcopy(self.__k_tree)

            '''拆解树，寻找对应层次结点'''
            sequence = len(self.__data) - len(self.__k_tree)
            while len(self.__data) - sequence < k:
                for node in _tree:
                    if node[0] == sequence:
                        _tree.append(node[3])
                        _tree.append(node[4])
                        _tree.remove(node)
                        sequence -= 1
                        break
            '''收集参数'''
            for _ in range(len(_tree)):
                _mean.append(_tree[_][2][0])
                _variance.append(_tree[_][2][1])
                alpha.append(_tree[_][1] / len(self.__data))
            '''矩阵化'''
            _mean = np.array(_mean)
            _variance = np.array(_variance)
            '''返回参数'''
            return _mean, _variance, alpha

        """
            Binning算法(装箱算法)
            参考文献：高斯混合模型聚类中EM算法及初始化的研究(2006)
        """

        def binning(self):
            pass

        """
            人工神经网络算法 SOM
            包括传统SOM和基于粒子群的SOM算法
            参数见ANN中的pso文档
            Example: ci.som((0.6, 20, 0.6, 20), scopev=[-1, 1], t_1=500, t_2=500, method=Distance.euclidean_metric)
        """

        def som(self, args, scopev=None, t_1=500, t_2=500, method=Distance.euclidean_metric, algorithm=0):
            if not scopev:
                scopev = [-1, 1]
            ann = ANN(data=self.__data, dimension=self.__dimension)
            if algorithm == 0:
                ann.som(self.__k, args, T=t_1, method=method)
            else:
                ann.p_som(self.__k, args, scopev, T=t_1, T_=t_2, method=method)

    class LogInfoPrint(object):
        def __init__(self):
            """"""
            pass

        def note(self, info, cls, show_console=True):
            """用于训练信息输出"""
            sys.stdout.write(info + '\n')
