# -*-coding:utf-8-*-

"""
    隐马尔可夫模型（Hidden Markov Model，HMM）

Author:Byshx

Date:2017.04.01  Happy Fool's Day →_→

简介：
    隐马尔可夫模型（Hidden Markov Model，HMM）是统计模型，它用来描述一个含有隐含未知参数的马尔可夫过程。其难点是从可观察的参数中确定
该过程的隐含参数。然后利用这些参数来作进一步的分析，例如模式识别。
                                                                                            ————《百度百科》

应用：
    语音识别、自然语言处理、OCR等

参考文献：
1、《统计学习方法》 李航
2、http://www.comp.leeds.ac.uk/roger/HiddenMarkovModels/html_dev/forward_algorithm/s1_pg11.html
3、<<A tutorial on Hidden Markov Models and Selected Applications in Speech Recognition>>
4、CSDN各个博客
5、http://digital.cs.usu.edu/~cyan/CS7960/hmm-tutorial.pdf ——解决HMM浮点数下溢(underflow)问题
"""

import math
import numpy as np
from StatisticalModel.DataInitialization import DataInitialization

datapath = '/home/luochong/PycharmProjects/ASR/StatisticalModel/HiddenMarkovModelDataSet.csv'


class HMM(DataInitialization):
    def __init__(self, states, observations, T=None, A=None, B=None, profunc=None, π=None):
        """
                参数说明：
                1、states ——状态集合(字典集:{序号:状态})
                2、observations ——观测集合(字典集:{观测:序号})
                3、T ——观测序列长度
                4、A ——状态转移矩阵(N*N，N为状态数)
                5、B ——混淆矩阵(Confusion Matrix)/观测矩阵  (N*M，N为状态数，M为观察数)
                6、profunc ——代替混淆矩阵的概率密度函数，如GMM
                7、π ——初始概率矩阵(1*M，M为观测数)

                *参数初值通常用聚类算法K-Means提前处理
            """
        super().__init__()
        '''状态集合'''
        self.__states = states
        self.__hmm_size = len(states)
        '''观测集合'''
        self.__observations = observations
        '''观测序列长度'''
        self.__T = T

        self.__q_value = -float('inf')

        '''声明参数'''
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.__A = None
        self.__B = None
        self.__profunction = None
        self.__π = None
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        '''用以判断B是否为矩阵'''
        self.__matrix = True

        '''状态转移矩阵'''
        if A is not None:
            self.__A = A
        else:
            self.__A = 1. / len(states) * np.ones((len(states), len(states)))

        if B is not None:
            '''混淆矩阵/观测矩阵'''
            self.__B = B
        elif profunc is not None:
            '''概率密度函数'''
            self.__matrix = False
            self.__profunction = profunc
        else:
            self.__B = 1. / len(observations) * np.ones((len(states), len(observations)))
        '''初始概率矩阵'''
        if π is not None:
            self.__π = π
        else:
            self.__π = 1. / len(states) * np.ones((len(states),))
        '''运算结果'''
        self.__result_f = None  # 钱箱算法结果保存
        self.__result_b = None  # 后向算法结果保存
        self.__result_p = None  # 概率密度函数计算结果
        '''前一次概率估计结果(向前算法计算)'''
        self.__p = None
        '''保存c_coefficient(缩放系数，用于scaling forward algorithm)'''
        self.__c_coefficient = 0.

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''getter方法'''

    @property
    def states(self):
        return self.__states

    @property
    def observations(self):
        return self.__observations

    @property
    def T(self):
        return self.__T

    @property
    def A(self):
        return self.__A

    @property
    def B(self):
        return self.__B

    @property
    def B_p(self):
        """
        profunction计算的概率矩阵
        :return: self.__result_p
        """
        return self.__result_p

    @property
    def π(self):
        return self.__π

    @property
    def profunction(self):
        return self.__profunction

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def cal_observation_pro(self, data, normalize=True):
        """
        计算观测概率
        :param data: 格式[n*数据向量长度]
        :param normalize: 规则化
        :return: 
        """
        data_p = []
        for d in range(len(data)):
            observations_pro = []
            for i in range(len(self.__states)):
                observations_pro.append([self.__profunction[i].point(data[d][_]) for _ in range(self.__T[d])])
            observations_pro = np.array(observations_pro)
            """normalize"""
            sum_row = np.sum(observations_pro, axis=1).reshape((len(self.__states), 1))
            if normalize:
                sum_row[np.where(sum_row == 0.)] = 1
                observations_pro /= sum_row
            """"""
            data_p.append(observations_pro)
        self.__result_p = data_p

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''清除方法'''

    def clear_result_cache(self):
        """
        清除临时存储结果的list，用于在使用不定长度训练数据前
        :return: None
        """
        self.__result_f = None  # 前向算法结果保存
        self.__result_b = None  # 后向算法结果保存
        self.__result_p = None  # 概率密度函数计算结果
        self.__p = None  # 上次前向算法运算结果保存
        self.__c_coefficient = 0.  # 缩放系数保存

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''改变参数'''

    def change_T(self, t):
        """
        改变序列长度，用于在使用不定长度训练数据前
        :param t: 新的序列长度
        :return: None
        """
        self.__T = t

    def change_π(self, π):
        self.__π = π

    def change_A(self, A):
        self.__A = A

    def change_B(self, B):
        self.__B = B

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def __scale_coefficient(self, result, result_index, t, sum_log=False):
        """
        防溢出处理
        :param result:结果列表
        :param result_index:结果列表序号
        :param t: 序列时间序号
        :param sum_log: 保存c_coefficient对数和
        :return: 
        """
        sum_column = np.sum(result[result_index][:, t], axis=0)
        if sum_column == 0.:
            result[result_index][:, t] = 1. / len(self.__states)
            sum_column = 1.
        result[result_index][:, t] /= sum_column
        if sum_log:
            self.__c_coefficient += math.log10(sum_column)

    def __forward_algorithm(self, result_f_index):
        """
             前向算法

        参数说明：
            A状态转移矩阵
            B混淆矩阵
            result ——结果集
            O ——观测序列(长度为T)

        """
        if self.__matrix:
            '''通过观测矩阵计算初值'''
            index = self.__observations[self.data[result_f_index][0]]
            self.__result_f[result_f_index][:, 0] = self.__π * self.__B[:, index]

            '''防下溢处理'''
            self.__scale_coefficient(self.__result_f, result_f_index, 0, sum_log=True)

            '''递推计算'''
            for i in range(1, self.__T[result_f_index]):
                p = np.dot(self.__result_f[result_f_index][:, i - 1], self.__A)
                index = self.__observations[self.data[result_f_index][i]]
                self.__result_f[result_f_index][:, i] = p * self.__B[:, index]
                '''防下溢处理'''
                self.__scale_coefficient(self.__result_f, result_f_index, i, sum_log=True)

        else:
            '''通过概率密度函数计算计算初值'''
            self.__result_f[result_f_index][:, 0] = self.__π * self.__result_p[result_f_index][:, 0]
            '''防下溢处理'''
            self.__scale_coefficient(self.__result_f, result_f_index, 0, sum_log=True)

            '''递推计算'''
            for i in range(1, self.__T[result_f_index]):
                '''对列向量分片，使之等于子字hmm的状态数'''
                p = np.dot(self.__result_f[result_f_index][:, i - 1], self.__A)
                self.__result_f[result_f_index][:, i] = p * self.__result_p[result_f_index][:, i]
                '''防下溢处理'''
                self.__scale_coefficient(self.__result_f, result_f_index, i, sum_log=True)

    def __backward_algorithm(self, result_b_index):
        """
             后向算法
    
        参数说明：
            A状态转移矩阵
            B混淆矩阵
            result ——结果集
            O ——观测序列(长度为T)
    
        """
        '''计算最后时刻β=1'''

        if self.__matrix:
            '''递推计算'''
            for i in range(self.__T[result_b_index] - 2, -1, -1):
                p = np.dot(self.__A, self.__result_b[result_b_index][:, i + 1])
                index = self.__observations[self.data[result_b_index][i + 1]]
                self.__result_b[result_b_index][:, i] = p * self.__B[:, index]
                '''防下溢处理'''
                self.__scale_coefficient(self.__result_b, result_b_index, i)

        else:
            '''如果观测矩阵为概率密度函数'''
            '''递推计算'''
            for i in range(self.__T[result_b_index] - 2, -1, -1):
                p = np.dot(self.__A, self.__result_b[result_b_index][:, i + 1])
                self.__result_b[result_b_index][:, i] = p * self.__result_p[result_b_index][:, i + 1]
                '''防下溢处理'''
                self.__scale_coefficient(self.__result_b, result_b_index, i)

    def __generate_result(self, p_normalize=True):
        """
        更新结果矩阵
        :param p_normalize: 观测概率规则化
        """
        '''缩放系数清零'''
        self.__c_coefficient = 0.

        if self.__T is None:
            '''若self.__T is None,则需根据每个数据的长度初始化self.__T'''
            self.__T = [len(self.data[index]) for index in range(len(self.data))]

        """一个result_f/result_b---------> 一个data"""

        if self.__result_f is None:
            self.__result_f = []
            self.__result_b = []
            for _ in range(len(self.data)):
                self.__result_f.append(np.zeros((len(self.__states), self.__T[_])))
                self.__result_b.append(np.ones((len(self.__states), self.__T[_])))
            if self.__profunction is not None:
                '''计算观测概率'''
                self.cal_observation_pro(self.data, p_normalize)

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        for data_index in range(len(self.data)):
            self.__forward_algorithm(data_index)
            self.__backward_algorithm(data_index)

    def __one_state_probability(self, index, t, q=None, index1=None):
        """
            已知单个状态概率计算
            (由矩阵运算代替)
        参数：
            1、t为状态时间
            2、q为状态
        """
        if index1 is None:
            if q is None:
                raise print('Error: 状态参数q缺失')
            index1 = self.__states[q]
        p = self.__result_f[index][index1][t] * self.__result_b[index][index1][t]
        return p

    def __two_states_probability(self, t, result_index, q1=None, q2=None, index1=None, index2=None):
        """
            已知连续两个状态概率计算
    
        参数：
            1、t为状态时间
            2、O ——观测序列
            3、q1、q2为状态
            4、index1、index2为状态所在位置
        """
        if index1 is None:
            if q1 is None:
                raise Exception('Error: 状态参数q1缺失')
            index1 = self.__states[q1]
        if index2 is None:
            if q2 is None:
                raise Exception('Error: 状态参数q2缺失')
            index2 = self.__states[q2]

        p1 = self.__result_f[result_index][index1][t]
        p2 = self.__result_b[result_index][index2][t + 1]

        if self.__matrix:
            _ = self.__observations[self.data[result_index][index2]]
            p = p1 * self.__A[index1][index2] * self.__B[index2][_] * p2
        else:
            p = p1 * self.__A[index1][index2] * self.__result_p[result_index][index2][t + 1] * p2
        return p

    """
        模型训练（Baum-Welch算法/EM算法）
    """
    """     Expectation     """

    def q_function(self, p_normalize=True):
        """
        Q函数计算
        :return:None
        """
        '''模型参数当前估计值所产生的概率'''
        p = self.__estimate(p_normalize)
        if self.__p is None:
            '''第一次迭代self.__p中的值等于此次向前算法计算出的值'''
            self.__p = p
        q = np.dot(np.log(p), self.__p.T)[0][0] + self.__c_coefficient
        ''''''
        self.__p = p
        return q

    def __estimate(self, p_normalize=True):
        """
        计算由向前算法计算出、并保存在结果矩阵的估计值
        :return: None
        """
        p = []
        if self.__result_p is None:
            self.__generate_result(p_normalize)
        for i in range(len(self.__result_f)):
            p.append(np.sum(self.__result_f[i][:, -1]))
        return np.array([p])

    """     Maximization    """

    def __maximization(self):
        """
        參數說明：
            1、A、B为待估计参数
            2、A_、B_为估计参数
        """

        def cal_ξ_1(t_, result_index):
            """计算分母"""
            sum_ξ = 0.
            for m in range(len(self.__states)):
                for n in range(len(self.__states)):
                    sum_ξ += self.__two_states_probability(t_, result_index, index1=m, index2=n)
            return sum_ξ

        def cal_ξ_2(i_, j_, t_, result_index, sum_ξ):
            """计算分子"""
            p = self.__two_states_probability(t_, result_index, index1=i_, index2=j_)
            return p / sum_ξ

        def cal_γ(t_, result_index):
            _γ = self.__result_f[result_index][:, t_] * self.__result_b[result_index][:, t_]
            _sum = np.sum(_γ)
            if _sum == 0.:
                _sum = 1.
            return _γ / _sum

        '''更新状态转移矩阵'''

        ξ = np.zeros((len(self.__states), len(self.__states)))
        γ = np.zeros((len(self.__states),))
        π = np.zeros_like(self.__π)

        for index in range(len(self.data)):
            for t in range(self.__T[index] - 1):
                sum_ = cal_ξ_1(t, index)
                if sum_ == 0.:
                    sum_ = 1.
                for i in range(len(self.__states)):
                    for j in range(len(self.__states)):
                        ξ[i][j] += cal_ξ_2(i, j, t, index, sum_)
                γ += cal_γ(t, index)
            π += cal_γ(0, index)

        '''重估初始概率矩阵'''
        π_sum = π.sum()
        if π_sum == 0.:
            self.__π = np.ones((len(self.__states),)) / len(self.__states)
        else:
            self.__π = π / π_sum

        '''除数不为零'''
        γ[np.where(γ == 0.)] = 1.

        self.__A = ξ / γ.reshape((len(self.__states), 1))

        '''规范化'''
        sum_A = self.__A.sum(axis=1)
        sum_A[np.where(sum_A == 0.)] = 1.
        self.__A /= sum_A.reshape((len(sum_A), 1))

        '''更新观测矩阵'''

        if self.__matrix:

            """补加时间位置为T的状态估计值"""

            for index in range(len(self.data)):
                γ += cal_γ(self.__T[index] - 1, index)

            γ_ = np.zeros((len(self.__states), len(self.__observations)))

            for k in range(len(self.__observations)):
                for index in range(len(self.data)):
                    for t in range(self.__T[index]):
                        if self.__observations[self.data[index][t]] == k:
                            γ_[:, k] += cal_γ(t, index)
            self.__B = γ_ / γ.reshape((len(self.__states), 1))

        else:
            '''GMM暂不更新'''
            pass

    def baulm_welch(self, show_q=False):
        """
            调用后自动完成迭代的方法
        :return: 收敛后的参数
        """
        '''计算前向后向结果'''
        self.__generate_result()
        # '''记录第一次Q函数值'''
        # _q = self.q_function()
        # if _q - self.__q_value > 1e1:
        #     self.__q_value = _q
        # else:
        #     return
        self.__q_value = -float('inf')
        while True:
            if show_q:
                print('HMM 当前似然度:', self.__q_value)
            self.__maximization()
            self.__generate_result()
            q_ = self.q_function()
            if q_ - self.__q_value > 12.8:
                self.__q_value = q_
            else:
                break

    @staticmethod
    def viterbi(states=None, observations=None, A=None, B=None, π=None, O=None, O_size=None, matrix=True,
                convert=False, end_state_back=False):
        """
    
              维特比算法
    
            基于动态规划，找出最优路径，用于解决隐马尔可夫模型中的预测问题
            
        :param states:状态集合
        :param observations:观测集合
        :param A:状态转移矩阵
        :param B:观测矩阵或概率密度函数
        :param π:初始概率矩阵
        :param O: 观测数据(若B为观测矩阵，则O为离散观测序列；若B为概率密度函数，则O为特征向量)
        :param matrix:是否为观测矩阵，True为矩阵，False为概率函数所计算结果集
        :param convert: convert=False 不将状态转换为标签
        :param end_state_back: 从最后一个状态开始回溯计算路径
        :return: 状态列表
        """
        if O_size is None:
            size = len(O)
        else:
            size = O_size
        '''最终路径'''
        mark_state = np.zeros((size,))
        '''标注路径'''
        before_state = [[0 for _ in range(size)] for _ in range(len(states))]

        '''对数运算消除警告'''
        np.seterr(divide='ignore')
        ''''''
        if matrix:
            '''B为观测矩阵'''
            index = observations[O[0]]
            p_list = np.log10(π) + np.log10(B[:, index])
            for i in range(1, size):
                p_ = np.zeros_like(p_list)
                index = observations[O[i]]
                for j in range(len(states)):
                    tmp = p_list + np.log10(A[:, j])
                    max_p = tmp.max()
                    p_[j] = max_p
                    '''最大值所在index'''
                    max_index = np.where(tmp == max_p)[0][0]
                    '''指向前面最优状态'''
                    before_state[max_index][i] = j
                p_list = p_ + np.log10(B[:, index])
        else:
            '''B为GMM评分函数'''
            profunction = B
            p_list = np.log10(π) + np.log10(profunction[:, 0])
            for i in range(1, size):
                p_ = np.zeros_like(p_list)
                for j in range(len(states)):
                    tmp = p_list + np.log10(A[:, j])
                    max_p = tmp.max()
                    p_[j] = max_p
                    '''最大值所在index'''
                    max_index = np.where(tmp == max_p)[0][0]
                    '''指向前面最优状态'''
                    before_state[j][i] = max_index
                p_list = p_ + np.log10(profunction[:, i])

        if end_state_back:
            end_index = len(p_list) - 4 + np.where(p_list[-4:] == p_list[-4:].max())[0][0]
            print(end_index, len(p_list))
            point = p_list[end_index]
            mark_state[size - 1] = end_index
        else:
            max_index = np.where(p_list == p_list.max())[0][0]
            point = p_list[max_index]
            mark_state[size - 1] = max_index

        '''回溯找出所有状态'''
        before_index = max_index
        for _ in range(size - 1, -1, -1):
            mark_state[_] = before_index
            before_index = before_state[before_index][_]

        if convert:
            c_mark_state = [None for _ in range(size)]
            for _ in range(len(mark_state)):
                c_mark_state[_] = states[mark_state[_]]
            c_mark_state = np.array(c_mark_state)
            print(c_mark_state)
            # a = input()
            # if a == '1':
            #     np.savetxt('/home/luochong/PycharmProjects/ASR/B_p.csv', B)
            return point, c_mark_state
        return point, mark_state


if __name__ == '__main__':
    states = {0: '盒子1', 1: '盒子2', 2: '盒子3'}
    observations = {'红': 0, '白': 1}
    # a = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # b = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    # pi = np.array([0.2, 0.4, 0.4])
    # t = 4
    hmm = HMM(states, observations)
    hmm.init_data(continuous=False, matrix=False, hasmark=False, datapath=datapath)
    hmm.baulm_welch()
    a = hmm.A
    b = hmm.B
    pi = hmm.π
    print(pi)
    hmm.viterbi(states, observations, a, b, pi, ['红', '红', '红', '白', '白'])
