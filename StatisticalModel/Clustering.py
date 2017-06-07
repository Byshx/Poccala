# -*-coding:utf-8-*-

"""
Clustering 聚类算法

应用：高斯混合模型(GMM,基于 Expectation Maximization Algorithm)、K-means聚类算法(EM算法特例)、层次聚类、自组织映射神经网络(SOM)

Author:Byshx

Date:2017.03.21

"""

import copy
import math
import random
import sys
import numpy as np
from StatisticalModel.ANN import ANN
from StatisticalModel.DataInitialization import DataInitialization
from StatisticalModel.Distance import Distance

datapath = '/home/luochong/PycharmProjects/MachineLearning/EMDataSet3.csv'


class Clustering(DataInitialization):
    def __init__(self):
        """"""
        super().__init__()

    """
        高斯混合模型
        初始化参数需要聚类分析算法
    """

    class GMM(object):
        def __init__(self, dimension, k, data=None, alpha=None, μ=None, σ=None, sigma=None, differentiation=True):
            """
                初始化参数
            :param data: 载入的数据
            :param dimension: 数据维度
            :param k: 高斯混合度
            :param alpha: 高斯加权系数
            :param μ: 均值
            :param σ: 特征向量各维度的方差
            :param sigma: 协方差矩阵
            :param differentiation: 初始值差异化
            """
            if data is None:
                self.__data = []
            else:
                self.__data = data
            """数据维度"""
            self.__dimension = dimension
            """模型数量"""
            self.__k = k
            """初始参数"""
            if μ is None:
                if differentiation:
                    self.__μ = 1e-2 * np.random.random((k, dimension))
                else:
                    self.__μ = np.zeros((k, dimension))
            else:
                self.__μ = μ
            '''生成对角矩阵'''
            eye_matrix = np.eye(dimension)
            if sigma is not None:
                self.__sigma = sigma
            elif σ is None:
                if differentiation:
                    self.__sigma = [np.random.random((dimension, dimension)) * eye_matrix for _ in range(k)]
                else:
                    self.__sigma = [1e-3 * np.eye(dimension) for _ in range(k)]
            else:
                self.__sigma = [eye_matrix * np.dot(σ[_:_ + 1, :].T, σ[_:_ + 1, :]) for _ in range(k)]
            if alpha is None:
                self.__alpha = 1. / k * np.ones((k,))
            else:
                self.__alpha = alpha
            if data:
                """γ的期望"""
                self.__gamma = np.ones((k, len(data)))

        def set_data(self, data):
            """
            载入新数据
            :param data:
            :return:
            """
            self.__data = data
            self.__gamma = np.ones((self.__k, len(data)))

        def add_data(self, data):
            """
            加入新数据
            :param data: 
            :return: 
            """
            self.__data.extend(data)
            self.__gamma = np.ones((self.__k, len(self.__data)))

        def clear_data(self):
            """清空所有数据"""
            self.__data = []

        def set_μ(self, μ):
            self.__μ = μ

        def set_sigma(self, σ=None, sigma=None):
            if sigma is not None:
                self.__sigma = sigma
            elif σ is not None:
                '''生成对角矩阵'''
                eye_matrix = np.eye(self.__dimension)
                self.__sigma = []
                for _ in range(self.__k):
                    sigma_ = eye_matrix * np.dot(σ[_:_ + 1, :].T, σ[_:_ + 1, :])
                    self.__sigma.append(sigma_)
            else:
                raise Exception('Error: 无参数传入(σ=None, sigma=None)')

        def set_alpha(self, alpha):
            self.__alpha = alpha

        def set_k(self, k):
            self.__k = k
            self.__gamma = np.ones((k, len(self.__data)))

        @property
        def μ(self):
            return self.__μ

        @property
        def sigma(self):
            return self.__sigma

        @property
        def alpha(self):
            return self.__alpha

        @property
        def dimension(self):
            """数据维度"""
            return self.__dimension

        @property
        def mixture(self):
            """高斯混合度"""
            return self.__k

        @property
        def data(self):
            """获得数据"""
            return self.__data

        @staticmethod
        def gaussian_function(y, dimension, μ, cov, log=False, standard=False):
            """根据高斯模型计算估计值 y为数据(行向量) μ为模型参数(μ为行向量) cov为协方差矩阵,log为是否计算对数值,standard为是否规范化"""
            x = y - μ
            if standard:
                x = np.dot(x, np.linalg.inv(cov) ** 0.5)
                cov_ = np.eye(dimension)
            else:
                cov_ = cov
            np.seterr(all='ignore')  # 取消错误提示
            if log:
                func = - (dimension / 2) * np.log(2 * math.pi) - 0.5 * np.log(np.linalg.det(cov_))
                exp = -0.5 * np.dot(np.dot(x, np.linalg.inv(cov_)), x.T)
                return func + exp
            else:
                sigma = (2 * math.pi) ** (dimension / 2) * np.linalg.det(cov_) ** 0.5
                func = 1. / sigma
                exp = np.exp(-0.5 * np.dot(np.dot(x, np.linalg.inv(cov_)), x.T))
                return func * exp

        """     ***************     Split and Merge     *****************     """

        def __J_merge(self):
            """
            计算合并得分并排序
            :return: 
            """
            j_merge_list = []
            for i in range(self.__k):
                for j in range(i + 1, self.__k):
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

                d_distance_list.sort(key=lambda x: x[1])
                for index in range(len(d_distance_list)):
                    p_1 += index / len(d_distance_list) * p_gamma[d_distance_list[index][0]]
                return p_1 / p_2

            '''将属于不同GMM的数据分类'''
            data = [[] for _ in range(self.__k)]
            for i in range(len(self.__data)):
                max_p = np.max(self.__gamma[:, i])
                max_index = int(np.where(self.__gamma[:, i] == max_p)[0])
                data[max_index].append(i)

            '''计算分裂分数'''
            j_split_list = []
            for i in range(self.__k):
                j_point = 0.
                for j in range(len(self.__data)):
                    p = cal_p(self.__data[j], data, i)
                    j_point += p * np.log(p / Clustering.GMM.gaussian_function(self.__data[j], self.__dimension,
                                                                               self.__μ[i], self.__sigma[i],
                                                                               standard=True))
                j_split_list.append([i, j_point])
            j_split_list.sort(key=lambda x: x[1], reverse=True)
            return j_split_list

        def __merge(self, x, y):
            """
            模型合并
            :return: 
            """
            new_alpha = self.__alpha[x] + self.__alpha[y]
            new_μ = (self.__μ[x] * self.__alpha[x] + self.__μ[y] * self.__alpha[y]) / new_alpha
            new_sigma = (self.__sigma[x] * self.__alpha[x] + self.__sigma[y] * self.__alpha[y]) / new_alpha
            return new_μ, new_sigma, new_alpha

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

            if len(data) < self.__k:
                return False

            cluster = Clustering.ClusterInitialization(data, 2, self.__dimension)
            μ_, σ_, alpha_, cluster_data = cluster.kmeans(algorithm=1)

            new_μ_1 = μ_[0] + np.random.rand(self.__dimension) * 1e-2
            new_μ_2 = μ_[1] + np.random.rand(self.__dimension) * 1e-2
            new_sigma_1 = np.eye(self.__dimension) * np.linalg.det(self.__sigma[x]) ** (1 / self.__dimension)
            new_sigma_2 = np.copy(new_sigma_1)
            new_alpha_1 = self.__alpha[x] * 0.5
            new_alpha_2 = self.__alpha[x] * 0.5
            new_sp = [[new_μ_1, new_μ_2], [new_sigma_1, new_sigma_2], [new_alpha_1, new_alpha_2]]
            return new_sp

        def __reestimate(self, new_μ, new_sigma, new_alpha, gamma_sum):
            """重估"""
            new_gamma = np.ones((len(new_μ), len(self.__data)))
            for i in range(len(self.__data)):
                tmp_gamma_list = []
                for j in range(len(new_μ)):
                    new_gamma[j][i] = Clustering.GMM.gaussian_function(self.__data[i], self.__dimension, new_μ[j],
                                                                       new_sigma[j], log=True) + np.log(new_alpha[j])
                    tmp_gamma_list.append(new_gamma[j][i])
                max_gamma = max(tmp_gamma_list)
                log_sum_exp = 0.  # 计算和的对数
                for index in range(len(tmp_gamma_list)):
                    log_sum_exp += np.exp(tmp_gamma_list[index] - max_gamma)
                log_sum_exp = np.log(log_sum_exp) + max_gamma
                """得到列之和，更新gamma表"""
                new_gamma[:, i] = np.exp(new_gamma[:, i] - log_sum_exp) * gamma_sum[i]
            return new_gamma

        def __SMEM(self, q_value, c_max=5):
            """
            执行SMEM算法
            :param q_value: 终止迭代时的Q值
            :param c_max: 候选模型数
            """

            class PassException(BaseException):
                """跳出多重循环"""
                pass

            if self.__k < 3:
                sys.stderr.write('聚类数目小于3，结束SMEM算法\n')
                return False
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
                μ_list, sigma_list, alpha_list = [], [], []
                new_μ, new_sigma, new_alpha = self.__merge(candidates[index][0], candidates[index][1])
                new_sp = self.__split(candidates[index][2])

                if new_sp is False:
                    continue

                μ_list.append(new_μ)
                μ_list.extend(new_sp[0])
                sigma_list.append(new_sigma)
                sigma_list.extend(new_sp[1])
                alpha_list.append(new_alpha)
                alpha_list.extend(new_sp[2])

                gamma_sum = []
                for index_ in range(len(self.__data)):
                    column_sum = self.__gamma[candidates[index][0], index_] + self.__gamma[
                        candidates[index][1], index_] + self.__gamma[candidates[index][2], index_]
                    gamma_sum.append(column_sum)
                gamma = self.__reestimate(μ_list, sigma_list, alpha_list, gamma_sum)
                new_μ_list, new_sigma_list, new_alpha_list = self.maximization(μ_list, sigma_list, alpha_list, gamma, 3)
                q_1 = self.q_function(new_μ_list, new_sigma_list, new_alpha_list, gamma, 3)

                '''修正参数——删除候选行'''
                modify_μ = np.delete(self.__μ, candidates[index], axis=0)
                modify_sigma = np.delete(self.__sigma, candidates[index], axis=0)
                modify_alpha = np.delete(self.__alpha, candidates[index])
                modify_gamma = np.delete(self.__gamma, candidates[index], axis=0)

                q_2 = self.q_function(modify_μ, modify_sigma, modify_alpha, modify_gamma, self.__k - 3)
                new_q_value = q_1 + q_2
                if new_q_value > q_value:
                    self.__μ = np.append(modify_μ, new_μ_list, axis=0)
                    self.__sigma = np.append(modify_sigma, new_sigma_list, axis=0)
                    self.__alpha = np.append(modify_alpha, new_alpha_list, axis=0)
                    self.__gamma = np.append(modify_gamma, gamma, axis=0)
                    return new_q_value
            return False

        """     **********************  Expectation  ********************     """

        """计算γ期望"""

        def cal_expectation(self):
            """gamma代表该数据属于模型k的期望
               表示模型k对观测数据yj的响应程度"""
            """""""""""""""""""""""""""""""""""""""""""""
                             P(γjk = 1|θ)
                γ  =  ------------------------------
                       ∑K,k=1P(yj|γjk=1,θ)P(γjk=1|θ)
            """""""""""""""""""""""""""""""""""""""""""""
            for i in range(len(self.__data)):
                tmp_gamma_list = []
                for j in range(self.__k):
                    self.__gamma[j][i] = Clustering.GMM.gaussian_function(self.__data[i], self.__dimension, self.__μ[j],
                                                                          self.__sigma[j], log=True) + np.log(
                        self.__alpha[j])
                    tmp_gamma_list.append(self.__gamma[j][i])
                max_gamma = max(tmp_gamma_list)
                log_sum_exp = 0.  # 计算和的对数
                for index in range(len(tmp_gamma_list)):
                    log_sum_exp += np.exp(tmp_gamma_list[index] - max_gamma)
                log_sum_exp = np.log(log_sum_exp) + max_gamma
                """得到列之和，更新gamma表"""
                self.__gamma[:, i] = np.exp(self.__gamma[:, i] - log_sum_exp)

        """
            EM算法核心：Q函数
            完全数据的对数似然函数关于给定观测数据和当前参数alpha_k、μ_k、σ_k下
            对为观测数据的条件概率分布P(z|y,theta)的期望
        """

        def q_function(self, μ, sigma, alpha, gamma, k):
            q = 0.
            for i in range(k):
                for j in range(len(self.__data)):
                    x = self.__data[j] - μ[i]
                    eye_matrix = np.diag(sigma[i])
                    det_sigma = np.sum(np.log(eye_matrix ** 0.5))
                    q += (np.log(alpha[i] / (2 * math.pi) ** (self.__dimension / 2)) - det_sigma
                          - 0.5 * np.dot(np.dot(x, np.linalg.inv(sigma[i])), x.T)) * gamma[i][j]
            return q

        """     ******************  Maximization    *****************   """

        """
            计算参数值，求偏导，使对数似然函数值更大
        """

        def maximization(self, μ, sigma, alpha, gamma, k):
            """计算γ_sum"""
            γ_sum = []
            for i in range(k):
                tmp_sum = 0.
                for j in range(len(self.__data)):
                    tmp_sum += gamma[i][j]
                γ_sum.append(tmp_sum)

            """计算新μ值"""
            for i in range(k):
                if γ_sum[i] == 0:
                    continue
                d_μ = []
                for d in range(self.__dimension):
                    μ_ = 0.
                    for j in range(len(self.__data)):
                        μ_ += gamma[i][j] * self.__data[j][d]
                    d_μ.append(μ_ / γ_sum[i])
                μ[i] = d_μ

            """计算新σ值"""
            eye_matrix = np.eye(self.__dimension)  # 用于生成对角矩阵
            for i in range(k):
                if γ_sum[i] == 0:
                    continue
                d_σ = []
                for d in range(self.__dimension):
                    σ_ = 0.
                    for j in range(len(self.__data)):
                        σ_ += gamma[i][j] * (self.__data[j][d] - μ[i][d]) ** 2
                    d_σ.append((σ_ / γ_sum[i]) ** 0.5)
                sigma_ = np.array([d_σ])
                if (sigma_ < 1e-3).any():
                    '''纠正协方差过小问题'''
                    sigma[i] = np.eye(self.__dimension) * 1e-3
                    continue
                sigma[i] = eye_matrix * np.dot(sigma_.T, sigma_)

            """计算新alpha值"""
            for i in range(k):
                alpha[i] = γ_sum[i] / len(self.__data)
            return μ, sigma, alpha

        """用于自动执行迭代的方法"""

        def baulm_welch(self, show_q=False, smem=False):
            if len(self.__data) == 0:
                raise Exception('Error: 未载入数据')
            q_value = -float('inf')
            while True:
                if show_q:
                    print('GMM 当前似然度：%f' % q_value)
                self.cal_expectation()
                self.maximization(self.__μ, self.__sigma, self.__alpha, self.__gamma, self.__k)
                _q = self.q_function(self.__μ, self.__sigma, self.__alpha, self.__gamma, self.__k)
                if _q - q_value > 1.28:
                    q_value = _q
                else:
                    if smem:
                        print('执行SMEM算法...')
                        new_q_value = self.__SMEM(q_value)
                        if new_q_value is not False:
                            q_value = new_q_value
                            continue
                        else:
                            print('迭代结束\n')
                            break
                    else:
                        break

        def theta(self):
            """获得已经收敛的参数"""
            theta = {}
            for i in range(self.__k):
                theta['theta_%d' % i] = [self.__μ[i], self.__sigma[i]]
            return theta

        def gmm(self, x):
            """返回属于的模型序号"""
            p = 0.
            index = 0
            for i in range(self.__k):
                tmp_p = self.gaussian_function(x, self.__dimension, self.__μ[i], self.__sigma[i])
                if tmp_p > p:
                    p = tmp_p
                    index = i
            '''返回该高斯模型产生该数据的概率和该高斯模型的序号'''
            return p, index

        def point(self, x, log=False, standard=False):
            """
            计算该高斯模型的得分(概率)
            :param x: 数据
            :param log: 对数形式
            :param standard: 以标准正态分布输出后验概率
            :return: 
            """
            n = len(x)
            if n != self.dimension:
                raise ValueError('Error: 数据维度错误。应为%d 实为%d' % (self.dimension, n))
            p = 0.
            if log:
                p_list = []
                for i in range(self.__k):
                    p_list.append(np.log(self.__alpha[i]) + self.gaussian_function(x, self.dimension, self.__μ[i],
                                                                                   self.__sigma[i], log=log,
                                                                                   standard=standard))
                max_p = max(p_list)
                log_sum_exp = 0.
                for i in range(self.__k):
                    log_sum_exp += np.exp(p_list[i] - max_p)
                p = max_p + np.log(log_sum_exp)
            else:
                for i in range(self.__k):
                    p += self.__alpha[i] * self.gaussian_function(x, self.dimension, self.__μ[i], self.__sigma[i])
            '''返回该高斯模型产生该数据的概率'''
            return p

        def update(self):
            """
            用于执行EM算法的统一方法
            解决各使用迭代优化的聚类方法名称不统一的问题
            :return: 
            """
            self.baulm_welch()

    """
        聚类分析算法 用于初始化中心点
    """

    class ClusterInitialization(object):
        def __init__(self, data, k, dimension):
            """含有因变量的数据"""
            self.__data = data
            """数据维度"""
            self.__dimension = dimension
            """样本类别"""
            self.__k = k
            """层次聚类算法得到的参数树"""
            self.__k_tree = None

        '''
            距离计算
            λ=1 曼哈顿距离（Manhattan Distance）
            λ=2 欧几里德距离（Euclidean Distance）
            λ>2 明可夫斯基距离（Minkowski Distance）
        '''

        @staticmethod
        def cal_distance(d1, d2, _λ=2):
            _distance = 0.
            for index in range(len(d1)):
                _distance += abs(d1[index] - d2[index]) ** _λ
                return _distance ** (1 / _λ)

        '''
            计算方差(Variance)
        '''

        @staticmethod
        def cal_variance(cluster, algorithm=None):
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

        def kmeans(self, algorithm=0):
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
                                if distance_ < distance:
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
                μlist = [k_index[_][0] for _ in range(self.__k)]
                '''计算方差'''
                σlist = [Clustering.ClusterInitialization.cal_variance(k_index[_], algorithm='kmeans') for _ in
                         range(self.__k)]
                '''返回均值和方差'''
                '''计算每类样本数占总样本数的比例alpha'''
                alpha = [(len(k_index[_][1]) / len(self.__data)) for _ in range(len(k_index))]

                '''获取每簇数据'''
                clustered_data = []
                for i in range(self.__k):
                    clustered_data.append(list(k_index[i][1].values()))
                '''转换为矩阵'''
                μlist, σlist = np.array(μlist), np.array(σlist)
                return μlist, σlist, alpha, clustered_data

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
                while True:
                    '''randomint包括区间端点'''
                    random_center = random.randint(0, len(self.__data) - 1)
                    if random_center not in k_index:
                        k_index.append([copy.deepcopy(self.__data[random_center]), {-1: self.__data[random_center]}])
                        '''设自己为第一个中心点0,并从样本类别列表中将自己(中心点)剔除'''
                        pointmark[random_center][0] = 0
                        break
                for k in range(1, self.__k):
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
                    '''取随机数'''
                    randomnumber = random.randint(0, int(sumdistance))
                    for index in range(len(distancelist)):
                        randomnumber -= distancelist[index]
                        if randomnumber < 0:
                            k_index.append([copy.deepcopy(self.__data[index]), {-1: self.__data[index]}])
                            pointmark[index][0] = k
                            break
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
                raise print('错误参数')

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
            μlist = [k_index[_][0] for _ in range(self.__k)]
            '''计算方差'''
            σlist = [Clustering.ClusterInitialization.cal_variance(k_index[_]) for _ in range(self.__k)]
            '''返回均值和方差'''
            '''计算每类样本数占总样本数的比例alpha'''
            alpha = [(len(k_index[_][1]) / len(self.__data)) for _ in range(len(k_index))]
            return μlist, σlist, alpha

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
                σ_ = Clustering.ClusterInitialization.cal_variance(k_index[_min_index])
                k_tree[_min_index] = [operation, k_tree[_min_index][1] + k_tree[min_index_][1],
                                      [k_index[_min_index][0], σ_],
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

            _μ, _σ, alpha = [], [], []
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
                _μ.append(_tree[_][2][0])
                _σ.append(_tree[_][2][1])
                alpha.append(_tree[_][1] / len(self.__data))
            '''矩阵化'''
            _μ = np.array(_μ)
            _σ = np.array(_σ)
            '''返回参数'''
            return _μ, _σ, alpha

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
            Example: ci.som((0.6, 20, 0.6, 20), [-1, 1], T=500, T_=500, method=Distance.euclidean_metric)
        """

        def som(self, args, scopev=[-1, 1], T=500, T_=500, method=Distance.euclidean_metric, algorithm=0):
            ann = ANN(data=self.__data, dimension=self.__dimension)
            if algorithm == 0:
                ann.som(self.__k, args, T=T, method=method)
            else:
                ann.p_som(self.__k, args, scopev, T=T, T_=T_, method=method)


if __name__ == '__main__':
    em = Clustering()
    em.init_data(hasmark=True, datapath=datapath)
    ci = em.ClusterInitialization(em.data, 2, em.dimension)
    μ, σ, alpha, c_data = ci.kmeans(algorithm=1)
    print(μ)
    print(σ)
    print(alpha)
    gmm = em.GMM(em.dimension, 2, em.data, alpha, μ, σ)
    gmm.baulm_welch(show_q=True)
    print(gmm.theta())
