# -*-coding:utf-8-*-
"""
    语音识别——声学模型

Author:Byshx

Date: 2017.04.15

"""

import sys
import os
import numpy as np
from StatisticalModel.LHMM import LHMM
from StatisticalModel.Clustering import Clustering
from StatisticalModel.AudioProcessing import AudioProcessing
from StatisticalModel.DataInitialization import DataInitialization

"""基元文件路径"""
unit = sys.path[1] + '/AcousticModel/Unit/'

path_training = sys.path[1] + '/AcousticModel/Audio/lib/'

path_recognize = sys.path[1] + '/AcousticModel/Audio/record'

path_l = sys.path[1] + '/AcousticModel/Audio/label'

path_parameter = unit + '/Parameters/'


class AcousticModel(DataInitialization):
    """声学模型"""

    def __init__(self, state_num=5, mix_level=1, max_mix_level=None, savepath=path_parameter, dct_num=13, delta_1=True,
                 delta_2=True):
        """
            初始化
        :param state_num: 每个基元的状态数，一般为5个及以上
        :param mix_level: 高斯混合度
        :param max_mix_level: 高斯最大混合度
        :param dct_num: 标准MFCC维数(如13)
        :param delta_1: True——计算一阶差分系数，MFCC维度 + vector_size(如13 + 13 = 26)
        :param delta_2: True——计算二阶差分系数，MFCC维度 + vector_size(如13 + 13 + 13  = 39)
        """
        super().__init__()
        '''复合HMM模型'''
        self.__unit = {}
        self.__unit_type = None
        '''基元状态数'''
        self.__state_num = state_num

        '''基元参数保存路径'''
        self.__address = savepath

        '''高斯混合度'''
        self.__mix_level = mix_level
        if max_mix_level is None:
            self.__max_mix_level = mix_level
        else:
            if max_mix_level < mix_level:
                raise ValueError('Error: 高斯最大混合度小于初始混合度')
            self.__max_mix_level = max_mix_level

        '''音频处理'''
        self.__audio = AudioProcessing()
        '''音频特征向量参数'''
        self.__dct_num = dct_num
        self.__delta_1 = delta_1
        self.__delta_2 = delta_2
        '''特征向量总维度'''
        self.__vector_size = self.__dct_num
        if self.__delta_2 is True:
            self.__vector_size *= 3
        elif self.__delta_1 is True:
            self.__vector_size *= 2

    @property
    def unit(self):
        return self.__unit

    @property
    def statenum(self):
        return self.__state_num

    def initialize_unit(self, unit_type='XIF'):
        """
        初始化基元
        :param unit_type: 基元类型
        :return:
        """
        unit_type_path = unit + unit_type
        self.__unit_type = unit_type
        if os.path.exists(unit_type_path) is False:
            raise FileExistsError('Error: 基元文件%s不存在' % unit_type)
        else:
            if os.path.exists(path_parameter + unit_type) is False:
                os.mkdir(path_parameter + unit_type)
        with open(unit_type_path) as f:
            u = f.readline()
            print('使用基元:', u)
            print('载入基元中...')
            while u:
                u = f.readline()
                if len(u) == 0:
                    break
                u = u.strip('\n').split(',')
                for i in range(len(u)):
                    '''状态集合'''
                    states = {_: u[i] for _ in range(self.__state_num)}
                    '''观测概率表示(GMM)'''
                    observations = ['GMM_probability']
                    '''状态转移矩阵'''
                    A = np.zeros((self.__state_num, self.__state_num))
                    '''开始状态，为虚状态，只允许向下一个状态转移'''
                    A[0][1] = 1.
                    for j in range(1, self.__state_num - 1):
                        for k in range(j, j + 2):
                            A[j][k] = 0.5

                    '''初始化GMM'''
                    gmm = [Clustering.GMM(self.__vector_size, self.__mix_level) for _ in
                           range(self.__state_num - 2)]

                    '''初始化虚状态评分类'''
                    virtual_gmm_1 = AcousticModel.VirtualState(0.)
                    virtual_gmm_2 = AcousticModel.VirtualState(0.)

                    gmm.insert(0, virtual_gmm_1)
                    gmm.append(virtual_gmm_2)

                    '''生成hmm实例'''
                    lhmm = LHMM(states, observations, None, A=A, profunc=gmm)

                    '''数据结构：{基元：[训练次数,HMM],}'''
                    self.__unit[u[i]] = [0, lhmm]
        print('基元载入完成 √')

    def init_parameter(self, *args):
        """
        加载参数
        :param args: 为从本地读取参数传递的参数文件位置信息或数据库配置信息
                     args为(None,)时从默认地址读取参数，为(path,)时从path
                     读取参宿；args为(数据库参数信息)时从数据库读取
                     数据库参数信息：
                        1、数据库类型(mysql/sqlserver/..)
                        2、dict{host, usr, passwd, database}
                        3、表(Table)名
        :param path: 参数保存路径
        :return: 
        """
        print('载入参数中...')
        if len(args) == 0:
            address = self.__address + self.__unit_type
            units = list(self.__unit.keys())
            counter_file = open(address + '/counter.csv')
            '''读取训练次数'''
            counter_file.readline()  # 读取文件描述信息
            units_ = counter_file.readline().strip('\n')
            while units_:
                units_ = units_.split(' ')
                self.__unit[units_[0]][0] = int(units_[1])
                units_ = counter_file.readline().strip('\n')
            counter_file.close()
            '''读取参数'''
            for index in range(len(units)):
                unit_path = address + '/%s' % units[index]
                detail_file = open(unit_path + '/detail.csv')
                A = np.load(unit_path + '/A.npy')
                π = np.load(unit_path + '/π.npy')
                hmm = self.__unit[units[index]][1]
                hmm.change_A(A)
                hmm.change_π(π)
                for index_ in range(1, self.__state_num - 1):
                    '''读取均值、协方差、权重矩阵'''
                    means = np.load(unit_path + '/GMM_%d_means.npy' % index_)
                    covariance = np.load(unit_path + '/GMM_%d_covariance.npy' % index_)
                    alpha = np.load(unit_path + '/GMM_%d_weight.npy' % index_)
                    mix_level = int(detail_file.readline().split()[-1])  # 获取高斯混合度
                    '''将数据初始化到GMM'''
                    gmm = hmm.profunction[index_]
                    gmm.set_μ(means)
                    gmm.set_sigma(sigma=covariance)
                    gmm.set_alpha(alpha)
                    gmm.set_k(mix_level)
                detail_file.close()
        elif len(args) == 3:
            '''从数据库读取参数，参数包括: 数据库类型(mysql/sqlserver/..), dict{host, usr, passwd, database}, 表(Table)名'''
            units = list(self.__unit.keys())
            '''初始化数据库'''
            self.init_data_from_database(**args[0], database=args[1])
            for index in range(len(units)):
                sql = "SELECT * FROM %s WHERE unit == '%s'" % (args[2], units[index])
                data = self.init_data_from_database(sql=args[2])
                """#############待续###########"""
        else:
            raise ValueError('Error: 参数数量错误')
        print('参数载入完成 √')

    def __save_parameter(self):
        """
        保存训练后的结果
        :return: 
        """
        path_parameter_ = self.__address + self.__unit_type
        '''保存基元训练次数'''
        counter_file = open(path_parameter_ + '/counter.csv', 'w+')
        counter_file.writelines('###各基元训练次数###\n')
        units = list(self.__unit.keys())
        for index in range(len(units)):
            '''写入训练次数'''
            counter_file.writelines('%s %d\n' % (units[index], self.__unit[units[index]][0]))
            unit_parameter = path_parameter_ + '/%s' % units[index]
            if os.path.exists(unit_parameter) is False:
                '''基元文件夹不存在时创建'''
                os.mkdir(unit_parameter)
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            hmm = self.__unit[units[index]][1]
            np.save(unit_parameter + '/A.npy', hmm.A)  # 保存状态转移矩阵
            np.save(unit_parameter + '/π.npy', hmm.π)  # 保存初始概率矩阵
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            detail_file = open(unit_parameter + '/detail.csv', 'w+')
            '''保存GMM参数'''
            for index_ in range(1, self.__state_num - 1):
                '''除去前后两个虚函数'''
                gmm = hmm.profunction[index_]
                np.save(unit_parameter + '/GMM_%d_means.npy' % index_, gmm.μ)  # 保存均值
                np.save(unit_parameter + '/GMM_%d_covariance.npy' % index_, gmm.sigma)  # 保存协方差矩阵
                np.save(unit_parameter + '/GMM_%d_weight.npy' % index_, gmm.alpha)  # 保存权重矩阵
                detail_file.writelines('GMM_%d %d\n' % (index_, gmm.mixture))  # 保存高斯混合度
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            detail_file.close()
        counter_file.close()

    def __init_audio(self, audiopath, labelpath):
        """
        加载音频
        :param audiopath: 音频路径
        :param labelpath: 标注路径
        :return: label_data(标注和数据的映射，每个标注对应数据包括 [MFCC * n个音频, 帧长度 * n个音频])
        """
        label_data = {}
        audio_data = {}
        framesize = {}
        label_list = []

        with open(labelpath) as f:
            '''载入音频标注'''
            f.readline()
            label = f.readline()
            while label:
                label = label.strip('\n').split(' ')
                label_list.append(label[0])
                if label_data.get(label[0]) is None:
                    label_data[label[0]] = [label[1]]
                else:
                    label_data[label[0]].append(label[1])
                label = f.readline()

        for dir in os.walk(audiopath):
            for file in dir[2]:
                '''文件名即标注'''
                number = file.split('.')[0]
                mfcc = self.__load_audio(audiopath=os.path.join(dir[0], file))
                audio_data[number] = mfcc
                framesize[number] = len(mfcc)

        for key in list(label_data.keys()):
            index_list = label_data[key]
            data = []
            f_size = []
            for index in index_list:
                data.append(audio_data[index])
                f_size.append(framesize[index])
            label_data[key] = [data, f_size]

        label_list = list(set(label_list))
        return label_list, label_data

    def __flat_start(self, label_data):
        """
        均一起步
        计算全局均值和方差
        :param label_data: 所有数据
        :return: 
        """
        '''data为所有数据的合集'''
        data = list(label_data.values())
        _data = None
        size = 0
        for index in range(len(data)):
            size += sum(data[index][1])
            tmp_data = data[index][0][0]
            for d in range(1, len(data[index][0])):
                tmp_data = np.append(tmp_data, data[index][0][d], axis=0)
            if _data is None:
                _data = tmp_data
            else:
                _data = np.append(_data, tmp_data, axis=0)
        label = list(label_data.keys())
        _label = []
        for l in label:
            _label.extend(l.split(','))
        '''取不重复基元'''
        label = list(set(_label))
        cluster = Clustering.ClusterInitialization(_data, self.__mix_level, self.__vector_size)
        _μ, _σ, _alpha, _clustered_data = cluster.kmeans(algorithm=1)

        '''训练GMM'''
        tmp_gmm = Clustering.GMM(None, self.__vector_size, self.__mix_level)
        tmp_gmm.set_data(_data)
        tmp_gmm.set_μ(_μ)
        tmp_gmm.set_sigma(σ=_σ)
        tmp_gmm.set_alpha(_alpha)
        '''GMM Baulm-Welch迭代'''
        tmp_gmm.baulm_welch()

        '''获取均值、协方差和权重值'''
        mean = tmp_gmm.μ
        covariance = tmp_gmm.sigma
        alpha = tmp_gmm.alpha

        for i in range(len(label)):
            hmm = self.__unit[label[i]][1]
            for j in range(1, len(hmm.profunction) - 1):
                '''除去前后两个虚方法'''
                gmm = hmm.profunction[j]
                gmm.set_μ(mean)
                gmm.set_sigma(sigma=covariance)
                gmm.set_alpha(alpha)

    def __initialize_data(self, data, label):
        """
        初始化基元参数
        :param data: 音频数据
        :param label: 句中基元标记
        :return:
        """
        set_label = list(set(label))
        set_label.sort(key=label.index)
        '''数据等分'''
        union_data_list, union_data_t_list = AcousticModel.__eq_segment(data, label)

        virtual_label = [_ for _ in range(self.__state_num - 2)]  # 数据再细分
        for index in range(len(set_label)):
            union_data_list_2, union_data_t_list_2 = AcousticModel.__eq_segment(
                [[union_data_list[index]], [union_data_t_list[index]]], virtual_label)

            hmm = self.__unit[set_label[index]][1]
            hmm.add_data(union_data_list)
            hmm.add_T(union_data_t_list)

            for index_ in range(self.__state_num - 2):
                hmm.profunction[index_ + 1].add_data(union_data_list_2[index_])

    @staticmethod
    def __eq_segment(data, label, supplement=False):
        """数据均分"""
        set_label = list(set(label))
        set_label.sort(key=label.index)
        label = np.array(label)
        label_num = len(label)
        """"""
        union_data_list = []
        union_data_t_list = []

        for index in range(len(set_label)):
            loc = np.where(label == set_label[index])[0]
            '''数据拼接'''
            union_data = None
            union_data_t = 0
            for index_ in range(len(data[0])):
                data_t = data[1][index_]  # 对应标注中的一个语音数据数据
                if supplement:
                    mod = data_t % label_num
                else:
                    mod = 0
                chunk = (data_t + mod) // label_num
                if mod > 0:
                    data[0][index_] = np.append(data[0][index_], data[0][index_][-mod:], axis=0)
                    data[1][index_] += mod
                union_data_t += chunk
                union_loc_data = [rv for r in
                                  zip(*[data[0][index_][loc[_] * chunk:(loc[_] + 1) * chunk] for _ in range(len(loc))])
                                  for rv in r]
                if union_data is None:
                    union_data = union_loc_data
                else:
                    union_data = np.append(union_data, union_loc_data, axis=0)
            union_data_list.append(union_data)
            union_data_t_list.append(union_data_t)
        return union_data_list, union_data_t_list

    def __cal_hmm(self, label, correct=None):
        """计算HMM"""
        '''开始训练'''
        hmm = self.__unit[label][1]
        if len(hmm.data) == 0:
            return
        hmm.baulm_welch(correct=correct, show_q=True)

    def __cal_gmm(self, label, c=True):
        """初始化GMM"""
        hmm = self.__unit[label][1]
        for i in range(1, len(hmm.profunction) - 1):
            '''除去前后两个虚方法'''
            gmm = hmm.profunction[i]
            data = gmm.data
            if len(data) < self.__mix_level:
                continue
            if c:
                '''重新聚类'''
                cluster = Clustering.ClusterInitialization(data, self.__mix_level, self.__vector_size)
                μ, σ, alpha, clustered_data = cluster.kmeans(algorithm=1)
                gmm.set_k(self.__mix_level)
                gmm.set_μ(μ)
                gmm.set_sigma(σ=σ)
                gmm.set_alpha(alpha)
                '''GMM Baulm-Welch迭代'''
                gmm.baulm_welch(show_q=True, smem=True)
            else:
                gmm.baulm_welch(show_q=True)
            gmm.clear_data()  # 清空数据内存

    def training(self, audiopath, labelpath, flat_start=True, t=4, batch_train=False):
        """
        训练参数
        采用嵌入式训练，串联标注所对应的HMM
        :param audiopath: 音频路径
        :param labelpath: 标注路径
        :param flat_start: 采用均一起步初始化
        :param t: 训练迭代次数
        :param batch_train: 批量训练，默认False（逐条数据训练）
        :return: 
        """

        def correct(A):
            """状态转移矩阵修正函数"""
            A[np.where((A > 0) & (A < 1e-4))] = 1e-4
            '''规范化'''
            sum_A = A.sum(axis=1)
            sum_A[np.where(sum_A == 0.)] = 1.
            A /= sum_A.reshape((len(sum_A), 1))
            return A

        label_list, label_data = self.__init_audio(audiopath=audiopath, labelpath=labelpath)
        if flat_start:
            '''均一起步'''
            self.__flat_start(label_data)
        else:
            '''分段K均值'''
            print('分割数据中...')
            for label in label_list:
                _label = label.split(',')
                self.__initialize_data(label_data[label], _label)

            print('初始化参数...')
            units = list(self.__unit.keys())
            for key in units:
                print('初始化基元:', key)
                self.__cal_gmm(key, c=True)
                self.__cal_hmm(key, correct=None)

        point = -float('inf')  # 总评分
        current_t = 0
        flag = False  # 高斯混合度增加的标志

        while current_t < t:
            point_sum = 0.  # 累计评分
            print('当前评分：', point)
            for label in label_list:
                '''分解标记为基元'''
                _label = label.split(',')
                label_ = list(set(_label))
                '''保持原来顺序'''
                label_.sort(key=_label.index)

                data = []
                data_t = []
                if batch_train:
                    data.append(label_data[label][0])
                    data_t.append(label_data[label][1])
                else:
                    data = [[_] for _ in label_data[label][0]]
                    data_t = [[_] for _ in label_data[label][1]]

                for data_index in range(len(data)):
                    for j in range(len(label_)):
                        '''清空结果集'''
                        hmm = self.__unit[label_[j]][1]
                        hmm.clear_result_cache()
                        hmm.cal_observation_pro(data[data_index], data_t[data_index])
                        hmm.clear_data()

                    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    sequence_label = []
                    for i in range(len(data_t[data_index])):
                        point_, sequence = self.viterbi(_label, data_t[data_index][i], i)
                        point_sum += point_
                        '''对每个数据进行维特比切分'''
                        sequence_label.append(sequence)

                    '''viterbi切分重估'''
                    self.__re_estimate(_label, data, data_index, sequence_label)

            units = list(self.__unit.keys())
            for key in units:
                print('重估基元:', key)
                if flag:
                    self.__cal_gmm(key, c=True)
                else:
                    '''直接使用BW算法'''
                    self.__cal_gmm(key, c=False)
                self.__cal_hmm(key)
            '''训练次数+1'''
            current_t += 1
            if self.__mix_level < self.__max_mix_level:
                self.__mix_level += 1
                flag = True
            else:
                flag = False

        '''保存参数'''
        self.__save_parameter()

    def __re_estimate(self, label, data, data_index, sequence_label):
        """
        对GMM进行参数重估
        :param label: 数据标注的重复基元
        :param data: 训练数据
        :param data_index: 数据索引
        :param sequence_label: 维特比切分后的标注序列
        :return: 
        """
        label_ = list(set(label))
        label_.sort(key=label.index)

        for label_index in range(len(label_)):
            hmm = self.__unit[label_[label_index]][1]
            label_data = [None for _ in range(self.__state_num - 2)]
            for seq_index in range(len(sequence_label)):
                loc = np.where(sequence_label[seq_index] == label_[label_index])[0]
                '''区分化相同标注的不同位置'''
                seq_num = np.arange(len(loc))
                sub = loc - seq_num
                set_sub = list(set(sub))

                '''为HMM添加数据'''
                hmm_data = data[data_index][seq_index][np.where(sequence_label[seq_index] == label_[label_index])]
                if len(hmm_data) == 0:
                    continue
                hmm.add_data([hmm_data])
                hmm.add_T([len(hmm_data)])

                for index_ in range(len(set_sub)):
                    data_ = data[data_index][seq_index][loc[np.where(sub == set_sub[index_])]]
                    chunk = len(data_) // (self.__state_num - 2)
                    tmp_data = [data_[:chunk], data_[chunk:chunk * 2], data_[chunk * 2:]]
                    for index in range(self.__state_num - 2):
                        if label_data[index] is None:
                            label_data[index] = tmp_data[index]
                        else:
                            label_data[index] = np.append(label_data[index], tmp_data[index], axis=0)

            for i in range(1, len(hmm.profunction) - 1):
                gmm = hmm.profunction[i]
                gmm.add_data(label_data[i - 1])

    def embedded(self, label, data_index, alter=31):
        """
        嵌入式模型
        :param label: 标注
        :param data_index: 数据索引
        :param alter: 选择序号，11111(二进制) = 31为全部生成
        :return: 
        """
        state_size = (self.__state_num - 2) * len(label) + 2

        def embedded_states():
            """复合状态集合"""
            state_list = [label[0]]
            for i in range(len(label)):
                state_list.extend([label[i] for _ in range(self.__state_num - 2)])
            state_list.append(label[len(label) - 1])
            map_list = list(map(lambda x, y: [x, y], [_ for _ in range(state_size)], state_list))
            complex_states = dict(map_list)
            return complex_states

        def embedded_observation():
            """复合观测集合——None"""
            complex_observation = ['GMM_probability']
            return complex_observation

        def embedded_A():
            """复合状态转移矩阵"""
            complex_A = np.zeros((state_size, state_size))
            complex_A[:self.__state_num - 1, :self.__state_num] = self.__unit[label[0]][1].A[:-1]
            for i in range(1, len(label)):
                A = self.__unit[label[i]][1].A
                '''拼接'''
                a = i * (self.__state_num - 2) + 1
                b = (i + 1) * (self.__state_num - 2) + 1
                complex_A[a:b, a - 1: a - 1 + self.__state_num] = A[1:-1]
            return complex_A

        def embedded_B():
            """复合观测矩阵"""
            i = 0
            complex_B = self.__unit[label[0]][1].B_p[data_index][0:-1]
            for i in range(1, len(label)):
                B = self.__unit[label[i]][1].B_p[data_index]
                '''拼接'''
                complex_B = np.append(complex_B, B[1:-1, :], axis=0)
            '''后续处理'''
            B = self.__unit[label[i]][1].B_p[data_index][-1:, :]
            complex_B = np.append(complex_B, B, axis=0)  # 添加最后一个虚函数
            return complex_B

        def embedded_π():
            """复合初始概率矩阵"""
            complex_π = np.ones((state_size,)) / state_size
            return complex_π

        ''''''
        func_list = [embedded_states, embedded_observation, embedded_A, embedded_B, embedded_π]
        embedded_list = []
        for index in range(5):
            if 2 ** (5 - index - 1) & alter != 0:
                embedded_list.append(func_list[index]())
        return embedded_list

    def viterbi(self, label, data_size, data_index):
        """
        维特比切分
        :param label: 标注
        :param data_size: 数据长度 
        :param data_index: 数据索引
        :return: 
        """
        complex_states, complex_observation, complex_A, complex_B, complex_π = self.embedded(label, data_index, 31)
        '''维特比强制对齐'''
        return LHMM.viterbi(complex_states, complex_observation, complex_A, complex_B, complex_π, O_size=data_size,
                            matrix=False, convert=True, end_state_back=False)

    def __load_audio(self, audiopath):
        """
        读取音频
        :param audiopath: 音频地址，为None时录音生成数据
        :return:
        """
        '''获取音频特征向量'''
        mfcc = self.__audio.MFCC(self.__dct_num)
        mfcc.init_audio(path=audiopath)
        '''计算一阶和二阶差分系数'''
        m = mfcc.mfcc(nfft=1024, d1=self.__delta_1, d2=self.__delta_2)
        vad = self.__audio.VAD()
        vad.init_mfcc(m)
        filtered_mfcc = vad.mfcc()
        return m

    class VirtualState(object):
        """
        为虚状态设立的评分类
        """

        def __init__(self, p=0.):
            self.__p = p

        def point(self, x, log=False, standard=False):
            """返回评分为p,x接受参数，但不处理"""
            return self.__p


if __name__ == '__main__':
    am = AcousticModel(mix_level=2, max_mix_level=4)
    am.initialize_unit(unit_type='XIF_tone')
    # am.init_parameter()
    am.training(audiopath=path_training, labelpath=path_l, flat_start=False, t=4, batch_train=True)
