# -*-coding:utf-8-*-

"""
    隐马尔可夫模型（Hidden Markov Model，HMM）对数计算形式

Author:Byshx

Date:2017.05.20

"""
import numpy as np
from StatisticalModel.DataInitialization import DataInitialization

datapath = '/home/luochong/PycharmProjects/ASR/StatisticalModel/HiddenMarkovModelDataSet.csv'


class LHMM(DataInitialization):
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
        if T is None:
            self.__T = []
        else:
            self.__T = T

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

    def cal_observation_pro(self, data, data_t, normalize=False, standard=False):
        """
        计算观测概率
        :param data: 格式[n*数据向量长度]
        :param data_t: 数据时间帧
        :param normalize: 规范化
        :param standard: 标准正态分布
        :return: 
        """
        data_p = []

        for d in range(len(data)):
            observations_pro = []
            normalize_array = np.zeros((len(self.__states), 1))  # 规范化除数(对数形式)
            for i in range(len(self.__states)):
                observations_pro.append(
                    [self.__profunction[i].point(data[d][_], log=True, standard=standard) for _ in range(data_t[d])])
            if normalize:
                for j in range(len(self.__states)):
                    normalize_array[j][0] = LHMM.__log_sum_exp(observations_pro[j])
            observations_pro = np.array(observations_pro) - normalize_array
            """"""
            data_p.append(observations_pro)
        self.__result_p = data_p

    @staticmethod
    def __matrix_dot(data1, data2, axis=0, log=True):
        """矩阵相乘"""
        p = []
        i, j = data2.shape
        if log:
            data_2 = np.log(data2)
        else:
            data_2 = data2
        if axis == 0:
            for index in range(i):
                tmp_p_list = data1 + data_2[index, :]
                p.append(LHMM.__log_sum_exp(tmp_p_list))
        elif axis == 1:
            for index in range(j):
                tmp_p_list = data1 + data_2[:, index]
                p.append(LHMM.__log_sum_exp(tmp_p_list))
        return np.array(p)

    @staticmethod
    def __log_sum_exp(p_list):
        """计算和的对数"""
        try:
            max_p = max(p_list)
        except ValueError:
            print(p_list)
        if abs(max_p) == float('inf'):
            return max_p
        log_sum_exp = 0.
        for index_ in range(len(p_list)):
            log_sum_exp += np.exp(p_list[index_] - max_p)
        log_sum_exp = max_p + np.log(log_sum_exp)
        return log_sum_exp

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

    def clear_data(self):
        """
        清除数据和序列长度
        :return: None
        """
        super().clear_data()
        self.__T = []

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''改变参数'''

    def change_T(self, t):
        """
        改变序列长度，用于在使用不定长度训练数据前
        :param t: 新的序列长度
        :return: None
        """
        self.__T = t

    def add_T(self, t):
        self.__T.extend(t)

    def change_π(self, π):
        self.__π = π

    def change_A(self, A):
        self.__A = A

    def change_B(self, B):
        self.__B = B

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
            self.__result_f[result_f_index][:, 0] = np.log(self.__π) + np.log(self.__B[:, index])

            '''递推计算'''
            for i in range(1, self.__T[result_f_index]):
                p = LHMM.__matrix_dot(self.__result_f[result_f_index][:, i - 1], self.__A, axis=1)
                index = self.__observations[self.data[result_f_index][i]]
                self.__result_f[result_f_index][:, i] = p + np.log(self.__B[:, index])

        else:
            '''通过概率密度函数计算计算初值'''
            self.__result_f[result_f_index][:, 0] = np.log(self.__π) + self.__result_p[result_f_index][:, 0]

            '''递推计算'''
            for i in range(1, self.__T[result_f_index]):
                '''对列向量分片，使之等于子字hmm的状态数'''
                p = LHMM.__matrix_dot(self.__result_f[result_f_index][:, i - 1], self.__A, axis=1)
                self.__result_f[result_f_index][:, i] = p + self.__result_p[result_f_index][:, i]

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
                p = LHMM.__matrix_dot(self.__result_b[result_b_index][:, i + 1], self.__A, axis=0)
                index = self.__observations[self.data[result_b_index][i + 1]]
                self.__result_b[result_b_index][:, i] = p + np.log(self.__B[:, index])

        else:
            '''如果观测矩阵为概率密度函数'''
            '''递推计算'''
            for i in range(self.__T[result_b_index] - 2, -1, -1):
                p = LHMM.__matrix_dot(self.__result_b[result_b_index][:, i + 1], self.__A, axis=0)
                self.__result_b[result_b_index][:, i] = p + self.__result_p[result_b_index][:, i + 1]

    def __generate_result(self):
        """
        更新结果矩阵
        """
        if len(self.__T) == 0:
            '''若self.__T is None,则需根据每个数据的长度初始化self.__T'''
            self.__T = [len(self.data[index]) for index in range(len(self.data))]

        """一个result_f/result_b---------> 一个data"""

        if self.__result_f is None:
            self.__result_f = []
            self.__result_b = []
            for _ in range(len(self.data)):
                self.__result_f.append(np.zeros((len(self.__states), self.__T[_])))
                self.__result_b.append(np.zeros((len(self.__states), self.__T[_])))
            if self.__profunction is not None:
                '''计算观测概率的对数值'''
                self.cal_observation_pro(self.data, self.__T, normalize=True)

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        for data_index in range(len(self.data)):
            self.__forward_algorithm(data_index)
            self.__backward_algorithm(data_index)

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
            p = p1 + np.log(self.__A[index1][index2]) + np.log(self.__B[index2][_]) + p2
        else:
            p = p1 + np.log(self.__A[index1][index2]) + self.__result_p[result_index][index2][t + 1] + p2
        return p

    """
        模型训练（Baum-Welch算法/EM算法）
    """
    """     Expectation     """

    def q_function(self):
        """
        Q函数计算
        :return:None
        """
        '''模型参数当前估计值所产生的概率'''
        p = self.__estimate()
        if self.__p is None:
            '''第一次迭代self.__p中的值等于此次向前算法计算出的值'''
            self.__p = p
        q = LHMM.__matrix_dot(p[0], self.__p.T, axis=1, log=False)
        ''''''
        self.__p = p
        return q[0]

    def __estimate(self):
        """
        计算由向前算法计算出、并保存在结果矩阵的估计值
        :return: None
        """
        p = []
        if self.__result_p is None:
            self.__generate_result()
        for i in range(len(self.__result_f)):
            p.append(LHMM.__log_sum_exp(self.__result_f[i][:, -1]))
        return np.array([p])

    """     Maximization    """

    def __maximization(self, correct=None):
        """
        最大似然估计
        :param correct: 参数类型为函数，用于修正状态转移矩阵，返回值须为状态转移矩阵
        """

        def cal_ξ_1(t_, result_index):
            """计算分母"""
            sum_ξ = []
            for m in range(len(self.__states)):
                for n in range(len(self.__states)):
                    sum_ξ.append(self.__two_states_probability(t_, result_index, index1=m, index2=n))
            return LHMM.__log_sum_exp(sum_ξ)

        def cal_ξ_2(i_, j_, t_, result_index, sum_ξ):
            """计算分子"""
            p = self.__two_states_probability(t_, result_index, index1=i_, index2=j_)
            return p - sum_ξ

        def cal_γ(t_, result_index):
            _γ = self.__result_f[result_index][:, t_] + self.__result_b[result_index][:, t_]
            _sum = LHMM.__log_sum_exp(_γ)
            return _γ - _sum

        '''更新状态转移矩阵'''
        size = len(self.__states)
        ξ_list = [[[] for _ in range(size)] for _ in range(size)]
        ξ = np.zeros((size, size))
        γ_list = []
        γ = np.zeros((size,))
        π_list = []
        π = np.zeros_like(self.__π)

        for index in range(len(self.data)):
            for t in range(self.__T[index] - 1):
                sum_ = cal_ξ_1(t, index)
                for i in range(len(self.__states)):
                    for j in range(len(self.__states)):
                        ξ_list[i][j].append(cal_ξ_2(i, j, t, index, sum_))
                γ_list.append(cal_γ(t, index))
            π_list.append(cal_γ(0, index))

        '''计算ξ'''
        for i in range(size):
            for j in range(size):
                ξ[i][j] = LHMM.__log_sum_exp(ξ_list[i][j])

        '''计算γ'''
        γ_list = np.array(γ_list)
        for i in range(size):
            γ[i] = LHMM.__log_sum_exp(γ_list[:, i])

        '''计算π'''
        π_list = np.array(π_list)
        for i in range(size):
            π[i] = LHMM.__log_sum_exp(π_list[:, i])
        self.__π = np.exp(π)

        γ[np.where(γ == -float('inf'))] = 0.

        self.__A = ξ - γ.reshape((len(self.__states), 1))

        '''规范化'''
        for i in range(len(self.__states)):
            normalize_num = LHMM.__log_sum_exp(self.__A[i])
            if np.isinf(normalize_num):
                '''inf相减为nan，故略去'''
                continue
            self.__A[i] -= normalize_num

        self.__A = np.exp(self.__A)
        print(self.__A)

        if correct is not None:
            '''修正状态转移矩阵'''
            self.__A = correct(self.__A)

        '''更新观测矩阵'''

        if self.__matrix:

            """补加时间位置为T的状态估计值"""

            γ_add_list = []
            for index in range(len(self.data)):
                γ_add_list.append(cal_γ(self.__T[index] - 1, index))
            γ_add_list = np.array(γ_add_list)
            tmp_γ = γ.reshape(1, size)
            tmp_γ = np.append(tmp_γ, γ_add_list, axis=0)

            for i in range(size):
                γ[i] = LHMM.__log_sum_exp(tmp_γ[:, i])

            tmp_γ_ = [[] for _ in range(len(self.__observations))]
            γ_ = np.zeros((size, len(self.__observations)))

            for k in range(len(self.__observations)):
                for index in range(len(self.data)):
                    for t in range(self.__T[index]):
                        if self.__observations[self.data[index][t]] == k:
                            tmp_γ_[k].append(cal_γ(t, index))

            for i in range(len(self.__observations)):
                tmp_p_list = np.array(tmp_γ_[i])
                for j in range(size):
                    γ_[j][i] = LHMM.__log_sum_exp(tmp_p_list[:, j])
            self.__B = γ_ - γ.reshape((size, 1))
            self.__B = np.exp(self.__B)

        else:
            '''GMM暂不更新'''
            pass

    def baulm_welch(self, correct=None, show_q=False):
        """
            调用后自动完成迭代的方法
        :return: 收敛后的参数
        """
        '''计算前向后向结果'''
        self.__generate_result()
        q_value = -float('inf')
        while True:
            if show_q:
                print('HMM 当前似然度:', q_value)
            self.__maximization(correct)
            self.__generate_result()
            q_ = self.q_function()
            if q_ - q_value > 1.28:
                q_value = q_
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
            p_list = np.log10(π) + profunction[:, 0]
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
                p_list = p_ + profunction[:, i]

        if end_state_back:
            end_index = len(p_list) - 4 + np.where(p_list[-4:] == p_list[-4:].max())[0][0]
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
            return point, c_mark_state
        return point, mark_state


if __name__ == '__main__':
    states = {0: '盒子1', 1: '盒子2', 2: '盒子3'}
    observations = {'红': 0, '白': 1}
    # a = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # b = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    # pi = np.array([0.2, 0.4, 0.4])
    # t = 4
    hmm = LHMM(states, observations)
    hmm.init_data(continuous=False, matrix=False, hasmark=False, datapath=datapath)
    hmm.baulm_welch(show_q=True)
    a = hmm.A
    b = hmm.B
    pi = hmm.π
    print('初始概率矩阵：\n', pi)
    print('状态转移矩阵：\n', a)
    print('观测矩阵：\n', b)
    hmm.viterbi(states, observations, a, b, pi, ['红', '红', '红', '白', '白'])
