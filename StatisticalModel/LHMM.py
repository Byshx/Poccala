# -*-coding:utf-8-*-

"""
    隐马尔可夫模型（Hidden Markov Model，HMM）对数计算形式

Author:Byshx

Date:2017.05.20

"""
import numpy as np
from StatisticalModel.DataInitialization import DataInitialization


class LHMM(DataInitialization):
    def __init__(self, states, log, t=None, transmat=None, profunc=None, probmat=None, pi=None):
        """
        对数计算的隐马尔可夫模型
        --------------------
        *模型参数初值通常用聚类算法K-Means提前处理
        :param states: 状态集合(字典集:{序号:状态})
        :param log: 记录日志
        :param t: 观测序列长度
        :param transmat: 状态转移矩阵(N*N，N为状态数)
        :param profunc: 代替混淆矩阵的概率密度函数，如GMM
        :param probmat: 混淆矩阵/观测矩阵(当profunc不为None时，使用profunc计算观测矩阵，并覆盖probmat)
        :param pi: 初始概率矩阵(1*M，M为观测数)
        """
        super().__init__()
        '''状态集合'''
        self.__states = states
        self.__hmm_size = len(states)

        '''记录日志'''
        self.log = log
        '''观测序列长度'''
        if t is None:
            self.__t = []
        else:
            self.__t = t

        '''声明参数'''
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.__transmat = None
        self.__profunction = profunc
        self.__pi = None
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        '''状态转移矩阵'''
        if transmat is not None:
            self.__transmat = transmat
        else:
            self.__transmat = 1. / len(states) * np.ones((len(states), len(states)))

        '''初始概率矩阵'''
        if pi is not None:
            self.__pi = pi
        else:
            self.__pi = 1. / len(states) * np.ones((len(states),))

        assert profunc is not None or probmat is not None, '必须为 profunc 和 probmat 之一指定参数值'

        '''运算结果'''
        self.__result_f = None  # 前向算法结果保存
        self.__result_b = None  # 后向算法结果保存
        ''''''
        self.__result_p = probmat  # 概率密度函数计算结果[probmat_1,probmat_2,...]  理论上profunc和probmat只能一个不为None
        # (list中有N个probmat，N为数据量，probmat为numpy.array形式，原先为批量训练设计，现做保留)
        ''''''
        '''前一次概率估计结果(向前算法计算)'''
        self.__p = None

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''getter方法'''

    @property
    def states(self):
        return self.__states

    @property
    def t(self):
        return self.__t

    @property
    def transmat(self):
        return self.__transmat

    @property
    def B_p(self):
        """
        profunction计算的概率矩阵
        :return: self.__result_p
        """
        return self.__result_p

    @property
    def pi(self):
        return self.__pi

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
        max_p = max(p_list)
        if abs(max_p) == float('inf'):
            return max_p
        log_sum_exp = max_p + np.log(np.sum(np.exp(p_list - max_p)))
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
        self.__t = []

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''改变参数'''

    def change_t(self, t):
        """
        改变序列长度，用于在使用不定长度训练数据前
        :param t: 新的序列长度
        :return: None
        """
        self.__t = t

    def add_T(self, t):
        self.__t.extend(t)

    def change_pi(self, pi):
        self.__pi = pi

    def change_A(self, transmat):
        self.__transmat = transmat

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def __forward_algorithm(self, result_f_index):
        """
             前向算法
        :param result_f_index: 数据序号
        """
        '''通过概率密度函数计算计算初值'''
        self.__result_f[result_f_index][:, 0] = np.log(self.__pi) + self.__result_p[result_f_index][:, 0]

        '''递推计算'''
        for i in range(1, self.__t[result_f_index]):
            '''对列向量分片，使之等于子字hmm的状态数'''
            p = LHMM.__matrix_dot(self.__result_f[result_f_index][:, i - 1], self.__transmat, axis=1)
            self.__result_f[result_f_index][:, i] = p + self.__result_p[result_f_index][:, i]

    def __backward_algorithm(self, result_b_index):
        """
             后向算法
        :param result_b_index: 数据序号
        """
        '''如果观测矩阵为概率密度函数'''
        '''递推计算'''
        for i in range(self.__t[result_b_index] - 2, -1, -1):
            p = LHMM.__matrix_dot(self.__result_b[result_b_index][:, i + 1], self.__transmat, axis=0)
            self.__result_b[result_b_index][:, i] = p + self.__result_p[result_b_index][:, i + 1]

    def __generate_result(self):
        """
            更新概率累积矩阵
        """
        if len(self.__t) == 0:
            '''若self.__t is None,则需根据每个数据的长度初始化self.__t'''
            self.__t = [len(self.data[index]) for index in range(len(self.data))]

        """一个result_f/result_b---------> 一个data"""

        if self.__result_f is None:
            self.__result_f = []
            self.__result_b = []
            for _ in range(len(self.data)):
                self.__result_f.append(np.zeros((len(self.__states), self.__t[_])))
                self.__result_b.append(np.zeros((len(self.__states), self.__t[_])))
            if self.__profunction is not None:
                '''计算观测概率的对数值'''
                self.cal_observation_pro(self.data, self.__t, normalize=False)

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        for data_index in range(len(self.data)):
            self.__forward_algorithm(data_index)
            self.__backward_algorithm(data_index)

    def __two_states_probability(self, t, result_index, index):
        """
            已知连续两个状态概率计算
        :param t: 状态时间
        :param result_index: 概率累积矩阵(前向&后向)位置
        :param index: state状态所在位置
        :return:
        """
        p1 = self.__result_f[result_index][index][t] * np.ones((len(self.__states),))
        p2 = self.__result_b[result_index][:, t + 1]
        p_arr = p1 + np.log(self.__transmat[index]) + self.__result_p[result_index][:, t + 1] + p2
        return p_arr

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

    def __maximization(self, show_a=False):
        """
        最大似然估计
        :param show_a: 显示重估后的状态转移矩阵
        """
        def cal_ksai_1(t_, result_index):
            """计算分母"""
            sum_ksai = self.__two_states_probability(t_, result_index, 0)
            for m in range(1, len(self.__states)):
                sum_ksai = np.append(sum_ksai, self.__two_states_probability(t_, result_index, m), axis=0)
            return LHMM.__log_sum_exp(sum_ksai)

        def cal_ksai_2(i_, t_, result_index, sum_ksai):
            """计算分子"""
            p = self.__two_states_probability(t_, result_index, i_)
            return p - sum_ksai

        def cal_gamma(t_, result_index):
            _gamma = self.__result_f[result_index][:, t_] + self.__result_b[result_index][:, t_]
            _sum = LHMM.__log_sum_exp(_gamma)
            return _gamma - _sum

        '''更新状态转移矩阵'''
        size = len(self.__states)
        ksai_list = [[[] for _ in range(size)] for _ in range(size)]
        ksai = np.zeros((size, size))
        gamma_list = []
        gamma = np.zeros((size,))
        pi_list = []
        pi = np.zeros_like(self.__pi)

        for index in range(len(self.data)):
            for t in range(self.__t[index] - 1):
                sum_ = cal_ksai_1(t, index)
                for i in range(size):
                    p_arr = cal_ksai_2(i, t, index, sum_)
                    for j in range(size):
                        ksai_list[i][j].append(p_arr[j])
                gamma_list.append(cal_gamma(t, index))
            pi_list.append(cal_gamma(0, index))

        '''计算ksai'''
        for i in range(size):
            for j in range(size):
                ksai[i][j] = LHMM.__log_sum_exp(ksai_list[i][j])

        '''计算gamma'''
        gamma_list = np.array(gamma_list)
        for i in range(size):
            gamma[i] = LHMM.__log_sum_exp(gamma_list[:, i])

        '''计算pi'''
        pi_list = np.array(pi_list)
        for i in range(size):
            pi[i] = LHMM.__log_sum_exp(pi_list[:, i])
        self.__pi = np.exp(pi)

        gamma[np.where(gamma == -float('inf'))] = 0.

        self.__transmat = ksai - gamma.reshape((len(self.__states), 1))

        '''规范化'''
        for i in range(len(self.__states)):
            normalize_num = LHMM.__log_sum_exp(self.__transmat[i])
            if np.isinf(normalize_num):
                '''inf相减为nan，故略去'''
                continue
            self.__transmat[i] -= normalize_num

        self.__transmat = np.exp(self.__transmat)
        if show_a:
            self.log.note('\n' + str(self.__transmat), cls='i')

    def baulm_welch(self, show_q=False, show_a=False):
        """
            调用后自动完成迭代的方法
        :param show_q: 显示当前似然度
        :param show_a: 显示重估后的状态转移矩阵
        :return: 收敛后的参数
        """
        '''计算前向后向结果'''
        self.__generate_result()
        q_value = -float('inf')
        while True:
            if show_q:
                self.log.note('HMM 当前似然度:%f' % q_value, cls='i')
            self.__maximization(show_a=show_a)
            self.__generate_result()
            q_ = self.q_function()
            if q_ > q_value:
                q_value = q_
            else:
                break

    @staticmethod
    def viterbi(log, states, transmat, prob, pi, convert=False, end_state_back=False, show_mark_state=False):
        """
            维特比算法
                --基于动态规划，找出最优路径，用于解决隐马尔可夫模型中的预测问题
        :param log: 记录日志
        :param states:状态集合
        :param transmat:状态转移矩阵
        :param prob: 概率(评分)矩阵
        :param pi:初始概率矩阵
        :param convert: convert=False 不将状态转换为标签
        :param end_state_back: 从最后一个状态开始回溯计算路径
        :param show_mark_state: 显示viterbi路径
        :return: 状态列表
        """
        '''数据序列长度'''
        s_len, t = prob.shape
        assert s_len == len(states), '状态数不对应概率输出'
        '''最终路径'''
        mark_state = np.zeros((t,))
        '''标注路径'''
        before_state = [[0 for _ in range(t)] for _ in range(s_len)]

        '''对数运算消除警告'''
        np.seterr(divide='ignore')
        p_list = np.log(pi) + prob[:, 0]
        ''''''
        max_index = 0.
        for i in range(1, t):
            p_ = np.zeros_like(p_list)
            for j in range(s_len):
                tmp = p_list + np.log(transmat[:, j])
                max_p = tmp.max()
                p_[j] = max_p
                '''最大值所在index'''
                max_index = np.where(tmp == max_p)[0][0]
                '''指向前面最优状态'''
                before_state[j][i] = max_index
            p_list = p_ + prob[:, i]

        if end_state_back:
            end_index = len(p_list) - 4 + np.where(p_list[-4:] == p_list[-4:].max())[0][0]
            point = p_list[end_index]
            mark_state[t - 1] = end_index
        else:
            max_index = np.where(p_list == p_list.max())[0][0]
            point = p_list[max_index]
            mark_state[t - 1] = max_index

        '''回溯找出所有状态'''
        before_index = max_index
        for _ in range(t - 1, -1, -1):
            mark_state[_] = before_index
            before_index = before_state[before_index][_]

        if convert:
            c_mark_state = [None for _ in range(t)]
            for _ in range(len(mark_state)):
                c_mark_state[_] = states[mark_state[_]]
            c_mark_state = np.array(c_mark_state)
            if show_mark_state:
                log.note(c_mark_state, cls='i')
            return point, c_mark_state
        if show_mark_state:
            log.note(mark_state, cls='i')
        return point, mark_state
