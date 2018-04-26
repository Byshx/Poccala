# -*-coding:utf-8-*-

"""
    隐马尔可夫模型（Hidden Markov Model，HMM）对数计算形式

Author:Byshx

Date:2017.05.20

"""
import os
import time
import configparser
from StatisticalModel.util import *
from StatisticalModel.DataInitialization import DataInitialization


class LHMM(DataInitialization):
    def __init__(self, states, statesnum, log, t=None, transmat=None, profunc=None, probmat=None, pi=None,
                 hmm_list=None, fix_code=0):
        """
        对数计算的隐马尔可夫模型
        --------------------
        *模型参数初值通常用聚类算法K-Means提前处理
        :param states: 状态集合(字典集:{序号:状态})
        :param statesnum: 状态的数量，取决于HMM的结构，一般情况下statesnum == len(states)。在嵌入式训练时statesnum == 单基元HMM结构
                          定义的数量，而len(states) == 嵌入式HMM的所有状态数(前后两个非发射状态 + 句子中基元数 * (statesnum - 2))
        :param log: 记录日志
        :param t: 观测序列长度
        :param transmat: 状态转移矩阵(N*N，N为状态数)
        :param profunc: 代替混淆矩阵的概率密度函数，如GMM
        :param probmat: 混淆矩阵/观测矩阵(当probmat不为None时，profunc仅用来更新参数，probmat参与运算)
        :param pi: 初始概率矩阵(1*M，M为观测数)
        :param hmm_list: 在嵌入式训练中，需要嵌入式HMM中各基元HMM实例
        :param fix_code: 从左到右三位分别代表三个参数状态转移矩阵transmat、概率密度函数profunc和初始概率矩阵pi，对应位置为0时不锁定，
                        为1时锁值不更新。当观测矩阵由probmat产生而不是profunc计算产生时，第二个参数不更新
        """
        super().__init__()
        '''状态集合'''
        self.__states = states
        self.__statesnum = statesnum
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
        '''用于重估的变量'''
        self.__ksai = np.zeros((self.__hmm_size, self.__hmm_size))
        self.__gamma = np.zeros((self.__hmm_size,))
        '''Accumulator'''
        self.__ksai_acc = np.log(np.zeros((self.__statesnum - 2, self.__statesnum)))
        self.__gamma_acc = np.log(np.zeros((self.__statesnum - 2,)))
        self.__acc_file = True
        '''子HMM集合，一般为自己本身，但在Embedded Training中，状态转移矩阵包含多个子HMM'''
        if hmm_list is None:
            self.__hmm_list = [self]
        else:
            self.__hmm_list = hmm_list
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''参数锁定列表'''
        self.__fix_code = None  # 声明fix_code
        self.__fix_list = None  # 声明fix_list
        self.fix_code = fix_code

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

    @property
    def ksai_acc(self):
        return self.__ksai_acc

    @property
    def gamma_acc(self):
        return self.__gamma_acc

    @property
    def fix_code(self):
        return self.__fix_code

    @fix_code.setter
    def fix_code(self, fix_code):
        self.__fix_code = fix_code
        self.__fix_list = [bool(fix_code & 2 ** e) for e in range(2, -1, -1)]
        if self.__profunction is None and self in self.__hmm_list:
            self.__fix_list[1] = True

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def add_acc(self, ksai_value, gamma_value):
        """
        累加
        :param ksai_value: ksai累加值
        :param gamma_value: gamma累加值
        :return:
        """
        '''累加ksai'''
        ksai_array = [self.__ksai_acc, ksai_value]
        self.__ksai_acc = matrix_log_sum_exp(ksai_array, axis_x=self.__statesnum - 2)
        '''累加gamma'''
        gamma_array = [self.__gamma_acc.reshape(1, -1), gamma_value.reshape(1, -1)]
        self.__gamma_acc = matrix_log_sum_exp(gamma_array, axis_x=1).reshape(-1, )

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
                    [self.__profunction[i].point(data[d][_], log=True, standard=standard, record=True) for _ in
                     range(data_t[d])])
            if normalize:
                for j in range(len(self.__states)):
                    normalize_array[j][0] = log_sum_exp(observations_pro[j])
            observations_pro = np.array(observations_pro) - normalize_array
            """"""
            data_p.append(observations_pro)
        self.__result_p = data_p

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """参数操作"""

    def save_parameter(self, path):
        """
        参数保存
        :param path: 参数保存地址
        :type path: str
        :return:
        """
        path.strip('/')
        path = path + '/HMM'
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path + '/transmat.npy', self.__transmat)  # 保存状态转移矩阵
        np.save(path + '/pi.npy', self.__pi)  # 保存初始概率矩阵
        with open(path + '/HMM_config.ini', 'w+') as hmm_config_file:
            hmm_config = configparser.ConfigParser()
            hmm_config.add_section('Configuration')
            hmm_config.set('Configuration', 'FIX_CODE', value=str(self.__fix_code))
            hmm_config.write(hmm_config_file)

    def save_acc(self, path):
        """
        累加器保存
        :param path: 累加器保存地址
        :type path: str
        :return:
        """
        path.strip('/')
        path = path + '/HMM'
        if not os.path.exists(path):
            os.mkdir(path)
        path_ksai_acc = path + '/ksai-acc'
        path_gamma_acc = path + '/gamma-acc'
        try:
            os.mkdir(path_ksai_acc)
            os.mkdir(path_gamma_acc)
        except FileExistsError:
            pass
        localtime = int(time.time())  # 时间戳
        np.save(path_ksai_acc + '/ksai_acc_%d.npy' % localtime, self.__ksai_acc)  # 保存ksai累加器
        np.save(path_gamma_acc + '/gamma_acc_%d.npy' % localtime, self.__gamma_acc)  # 保存gamma累加器

    def init_parameter(self, path):
        """
        参数读取
        :param path: 参数读取地址
        :type path: str
        :return:
        """
        path.strip('/')
        path = path + '/HMM'
        '''读取均值、协方差、权重矩阵'''
        transmat = np.load(path + '/transmat.npy')
        pi = np.load(path + '/pi.npy')
        '''将数据初始化到GMM'''
        self.__transmat = transmat
        self.__pi = pi
        with open(path + '/HMM_config.ini', 'r+') as hmm_config_file:
            hmm_config = configparser.ConfigParser()
            hmm_config.read(hmm_config_file)
            sections = hmm_config.sections()
            for section in sections:
                items = dict(hmm_config.items(section))
                self.fix_code = int(items['fix_code'])

    def init_acc(self, path):
        """
        收集基元参数目录下的所有acc文件，并初始化到HMM中
        :param path: 累加器读取地址
        :type path: str
        :return:
        """
        path.strip('/')
        path = path + '/HMM'
        path_ksai_acc = path + '/ksai-acc'
        path_gamma_acc = path + '/gamma-acc'
        if os.path.exists(path_ksai_acc) and os.path.exists(path_gamma_acc):
            self.__acc_file = True
        else:
            self.__acc_file = False
            return
        '''读取ksai_acc文件'''
        ksai_acc = []
        for dir in os.walk(path_ksai_acc):
            for filename in dir[2]:
                file = open(path_ksai_acc + '/' + filename, 'rb')
                data = np.load(file)
                ksai_acc.append(data)
                file.close()
        self.__ksai_acc = matrix_log_sum_exp(ksai_acc, axis_x=self.__statesnum - 2)
        gamma_acc = []
        '''读取gamma_acc文件'''
        for dir in os.walk(path_gamma_acc):
            for filename in dir[2]:
                file = open(path_gamma_acc + '/' + filename, 'rb')
                data = np.load(file)
                gamma_acc.append(data)
                file.close()
        gamma_acc = np.array(gamma_acc).T
        self.__gamma_acc = log_sum_exp(gamma_acc, vector=True)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''清除方法'''

    def clear_result_buffer(self):
        """
        清除临时存储结果的list，用于在使用不定长度训练数据前
        :return: None
        """
        self.__result_f = None  # 前向算法结果保存
        self.__result_b = None  # 后向算法结果保存
        # self.__result_p = None  # 概率密度函数计算结果
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
        :param t: 新的序列长度[数据1长度，数据2长度，...]
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
        log_transmat = np.log(self.__transmat)
        '''通过概率密度函数计算计算初值'''
        self.__result_f[result_f_index][:, 0] = np.log(self.__pi) + self.__result_p[result_f_index][:, 0]

        '''递推计算'''
        for i in range(1, self.__t[result_f_index]):
            '''对列向量分片，使之等于子字hmm的状态数'''
            p = []
            for j in range(self.__hmm_size):
                tmp_p_list = self.__result_f[result_f_index][:, i - 1] + log_transmat[:, j]
                p.append(log_sum_exp(tmp_p_list))
            self.__result_f[result_f_index][:, i] = np.array(p) + self.__result_p[result_f_index][:, i]

    def __backward_algorithm(self, result_b_index):
        """
             后向算法
        :param result_b_index: 数据序号
        """
        log_transmat = np.log(self.__transmat)
        '''递推计算'''
        for i in range(self.__t[result_b_index] - 2, -1, -1):
            back_array = []
            for j in range(self.__hmm_size):
                beta_array = log_transmat[j, :] + self.__result_p[result_b_index][:, i + 1] \
                             + self.__result_b[result_b_index][:, i + 1]
                back_array.append(beta_array)
            self.__result_b[result_b_index][:, i] = log_sum_exp(back_array, vector=True)

    def __generate_result(self):
        """
            更新概率累积矩阵
        """
        if len(self.__t) == 0:
            '''若self.__t is None,则需根据每个数据的长度初始化self.__t'''
            self.__t = [len(self.data[index]) for index in range(self.datasize)]

        """一个result_f/result_b---------> 一个data"""

        if self.__result_f is None:
            self.__result_f = []
            self.__result_b = []
            for _ in range(self.datasize):
                self.__result_f.append(np.zeros((len(self.__states), self.__t[_])))
                self.__result_b.append(np.zeros((len(self.__states), self.__t[_])))
            if self.__profunction is not None:
                '''计算观测概率的对数值'''
                self.cal_observation_pro(self.data, self.__t, normalize=False)

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''计算前向/后向累计对数概率'''
        for data_index in range(self.datasize):
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
        p1 = self.__result_f[result_index][index][t]
        p2 = self.__result_b[result_index][:, t + 1]
        p_arr = p1 + np.log(self.__transmat[index]) + self.__result_p[result_index][:, t + 1] + p2
        return p_arr

    """
        模型训练（Baum-Welch算法）
    """
    """     Expectation     """

    def __expectation(self):
        """
        计算由向前算法计算出、并保存在结果矩阵的估计值P(O|λ)
        :return: None
        """
        p = []
        if self.__result_p is None:
            self.__generate_result()
        for i in range(self.datasize):
            p = np.append(p, self.__result_f[i][:, -1])
        return log_sum_exp(p)

    """     Maximization    """

    def __maximization(self):
        """
        最大似然估计
        """

        def cal_ksai(data_index):
            """计算ksai值"""
            ksai_array = []
            for t in range(self.__t[data_index] - 1):
                ksai_array_t = []
                for m in range(0, len(self.__states)):
                    tmp_array = self.__two_states_probability(t, data_index, m)
                    ksai_array_t.append(tmp_array)
                ksai_array.append(np.array(ksai_array_t))
            return matrix_log_sum_exp(ksai_array, axis_x=self.__hmm_size)

        def cal_gamma(data_index):
            """计算gamma值"""
            gamma_array = self.__result_f[data_index][:, :-1] + self.__result_b[data_index][:, :-1]
            return log_sum_exp(gamma_array, vector=True)

        def cal_pi(data_index):
            """计算pi值"""
            pi_array = self.__result_f[data_index][:, 0] + self.__result_b[data_index][:, 0]
            pi_sum_value = log_sum_exp(pi_array)
            pi_array -= pi_sum_value
            return pi_array

        if self.datasize > 1:
            ksai_list = []
            gamma_list = []
            pi_list = []
            for index in range(self.datasize):
                ksai_list.append(cal_ksai(index))
                gamma_list.append(cal_gamma(index).reshape(1, -1))
                pi_list.append(cal_pi(index).reshape(1, -1))
            self.__ksai = matrix_log_sum_exp(ksai_list, axis_x=self.__hmm_size)
            self.__gamma = matrix_log_sum_exp(gamma_list, axis_x=1).reshape((-1,))

            if self.__fix_list[2] is False:
                self.__pi = np.exp(matrix_log_sum_exp(pi_list, axis_x=1))
        else:
            self.__ksai = cal_ksai(0)
            self.__gamma = cal_gamma(0)
            if self.__fix_list[2] is False:
                self.__pi = np.exp(cal_pi(0))

    def update_acc(self):
        """
        更新各基元HMM中的累加器
        :return:
        """
        ksai_view = self.__ksai[1:-1, :]
        gamma_view = self.__gamma[1:-1]
        emit_state_num = self.__statesnum - 2  # 发射状态数
        l_value = None
        b_value = None
        sum_value = None
        for index in range(self.datasize):
            if self.__fix_list[1] is False:
                l_value = self.__result_f[index] + self.__result_b[index]
                b_value = self.__result_p[index][1:-1, :]
                sum_value = log_sum_exp(l_value.T, vector=True)
                l_value = l_value[1:-1]
            index_x, index_y = 0, 0
            for hmm in self.__hmm_list:
                if self.__fix_list[0] is False:
                    '''更新HMM累加器'''
                    ksai_value = ksai_view[index_y:index_y + emit_state_num, index_x:index_x + self.__statesnum]
                    gamma_value = gamma_view[index_y:index_y + emit_state_num]
                    hmm.add_acc(ksai_value, gamma_value)
                if self.__fix_list[1] is False:
                    # '''更新概率密度函数累加器'''
                    l_value_states = l_value[index_y:index_y + emit_state_num, :]
                    np.subtract(l_value_states, sum_value, out=l_value_states)
                    b_value_states = b_value[index_y:index_y + emit_state_num, :]
                    gmms = hmm.profunction[1:-1]
                    for i in range(emit_state_num):
                        gmm = gmms[i]
                        gmm.update_acc(l_value_states[i, :], b_value_states[i, :], self.data[index])
                index_y += emit_state_num
                index_x += self.__statesnum - 2

    def update_param(self, show_q=False, show_a=False, c_covariance=1e-3):
        """
        更新参数
        :param show_q: 显示GMM重估信息
        :param show_a: 显示重估后的状态转移矩阵
        :param c_covariance: 协方差纠正值
        :return:
        """
        if not self.__acc_file:
            return
        if self.__fix_list[0] is False:
            self.__transmat[1:-1, :] = np.exp(self.__ksai_acc - self.__gamma_acc.reshape((self.__statesnum - 2, 1)))
        if self.__fix_list[1] is False:
            for index in range(1, len(self.__profunction) - 1):
                self.__profunction[index].update_param(show_q=show_q, c_covariance=c_covariance)
        self.log.note('HMM 状态转移矩阵：\n' + str(self.__transmat), cls='i', show_console=show_a)

    def baulm_welch(self, show_q=False):
        """
            调用后自动完成迭代的方法
        :param show_q: 显示当前似然度
        :return: 收敛后的参数
        """
        '''计算前向后向结果'''
        q_value = -float('inf')
        while True:
            self.log.note('HMM 当前似然度:%f' % q_value, cls='i', show_console=show_q)
            self.__generate_result()
            self.__maximization()
            self.__maximization_2()
            q_value_new = self.__expectation()
            if q_value_new - q_value > 0.64:
                q_value = q_value_new
            else:
                self.update_acc()
                break
            self.clear_result_buffer()

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
            log.note('Viterbi Sequence:\n' + str(c_mark_state), cls='i', show_console=show_mark_state)
            return point, c_mark_state
        log.note('Viterbi Sequence:\n' + str(mark_state), cls='i', show_console=show_mark_state)
        return point, mark_state
