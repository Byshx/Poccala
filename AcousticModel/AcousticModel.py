# -*-coding:utf-8-*-
"""
    语音识别——声学模型

Author:Byshx

Date: 2017.04.15

"""

import os
import time
import pickle
import shutil
import numpy as np
from LogPrint import Log
from Exceptions import *
from multiprocessing import Pool
from StatisticalModel.LHMM import LHMM
from StatisticalModel.Clustering import Clustering
from StatisticalModel.AudioProcessing import AudioProcessing
from StatisticalModel.DataInitialization import DataInitialization

"""基元、数据文件路径"""
UNIT_FILE_PATH = os.path.abspath(os.environ['unit_file_path'])
PARAMETERS_FILE_PATH = os.path.abspath(os.environ['parameters_file_path'])


class AcousticModel(DataInitialization):
    """声学模型"""

    def __init__(self,
                 log,
                 unit_type,
                 processes=None,
                 job_id=0,
                 console=True,
                 state_num=5,
                 mix_level=1,
                 dct_num=13,
                 delta_1=True,
                 delta_2=True):
        """
            初始化
        :param log: 记录日志
        :param unit_type: 基元类型
        :param processes: 进程数
        :param job_id: 作业id
        :param console: 控制台是否显示基元训练信息
        :param state_num: 每个基元的状态数，一般为5个及以上
        :param mix_level: 高斯混合度
        :param dct_num: 标准MFCC维数(如13)
        :param delta_1: True——计算一阶差分系数，MFCC维度 + vector_size(如13 + 13 = 26)
        :param delta_2: True——计算二阶差分系数，MFCC维度 + vector_size(如13 + 13 + 13  = 39)
        """
        super().__init__()
        '''作业id'''
        self.__job_id = job_id
        self.__console = console
        '''保存日志'''
        self.log = log
        '''存储所有基元{基元:音频数据文件数目,...}'''
        self.__loaded_units = []
        '''复合HMM模型'''
        self.__unit = {}
        '''基元类型'''
        self.__unit_type = unit_type
        '''基元状态数'''
        self.__state_num = state_num

        '''基元参数保存路径'''
        self.__address = PARAMETERS_FILE_PATH

        '''高斯混合度'''
        self.__mix_level = mix_level

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

        '''获取cpu核心数'''
        cpu_count = os.cpu_count()
        '''进程池'''
        if processes is None:
            self.processes = cpu_count
            self.log.note('训练进程数：%s' % cpu_count, cls='i')
        elif processes < 1 or type(processes) is not int:
            raise CpuCountError
        elif processes > cpu_count:
            self.processes = processes
            self.log.note('训练进程数：%s (>cpu核心数%s，可能影响训练性能)' % (int(processes), cpu_count), cls='w')
        else:
            self.processes = processes
            self.log.note('训练进程数：%s' % processes, cls='i')

    @property
    def unit(self):
        return self.__unit

    @property
    def loaded_units(self):
        return self.__loaded_units

    @property
    def statenum(self):
        return self.__state_num

    """解决pickle序列化问题"""

    def __getstate__(self):
        state = self.__dict__.copy()
        state['unit_type'] = state['log'].unit_type
        state['console'] = state['log'].console
        del state['log']  # log instance is unpickable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__['log'] = Log(state['unit_type'], console=state['console'])
        self.__dict__['log'].append()
        del state['unit_type']
        del state['console']

    """"""""""""""""""""""""

    def load_unit(self, unit_type=None):
        """
        加载基元文件
        :param unit_type: 根据基元类型的文件批量初始化基元 如 路径..AcousticModel/Unit/下有XIF_tone.csv文件，里面含有所有预设基元
        :return:
        """
        if unit_type:
            self.__unit_type = unit_type
        unit_type_path = UNIT_FILE_PATH + '/' + self.__unit_type
        if not os.path.exists(unit_type_path):
            raise UnitFileExistsError(self.__unit_type, self.log)
        else:
            '''判断文件夹是否存在，不存在则创建'''
            if not os.path.exists(PARAMETERS_FILE_PATH):
                os.mkdir(PARAMETERS_FILE_PATH)
            if not os.path.exists(PARAMETERS_FILE_PATH + '/' + self.__unit_type):
                os.mkdir(PARAMETERS_FILE_PATH + '/' + self.__unit_type)
        with open(unit_type_path) as f:
            u = f.readline().strip('\n')
            self.log.note('使用基元: %s' % u, cls='i')
            self.log.note('加载基元中...', cls='i')
            while u:
                u = f.readline()
                if len(u) == 0:
                    break
                u = u.strip('\n').split(',')
                for i in range(len(u)):
                    self.__loaded_units.append(u[i])
        self.log.note('基元加载完成   √', cls='i')  # 基元载入完成

    def init_unit(self, unit=None, new_log=True):
        """
        初始化基元，生成基元的复合数据结构
        :param unit: 初始化指定基元，为None时初始化所有loaded_units里的基元
        :param new_log: 是否删除先前日志
        :return:
        """

        def generate(_unit):
            """"""
            '''状态集合'''
            states = {_: _unit for _ in range(self.__state_num)}
            '''观测概率表示(GMM)'''
            observations = ['GMM_probability']
            '''状态转移矩阵'''
            A = np.zeros((self.__state_num, self.__state_num))
            '''开始状态，为虚状态，只允许向下一个状态转移'''
            A[0][1] = 1.
            for j in range(1, self.__state_num - 1):
                for k in range(j, j + 2):
                    A[j][k] = 0.5
            '''创建基元文件夹'''
            unit_path = PARAMETERS_FILE_PATH + '/%s/%s' % (self.__unit_type, _unit)
            if not os.path.exists(unit_path):
                os.mkdir(unit_path)
            ''''''''''''''''''
            log = Log(self.__unit_type, _unit, console=self.__console)
            if new_log:
                log.generate()
            else:
                log.append()
            '''初始化GMM'''
            gmm = [Clustering.GMM(self.__vector_size, self.__mix_level, log) for _ in
                   range(self.__state_num - 2)]

            '''初始化虚状态评分类'''
            virtual_gmm_1 = AcousticModel.VirtualState(0.)
            virtual_gmm_2 = AcousticModel.VirtualState(0.)

            gmm.insert(0, virtual_gmm_1)
            gmm.append(virtual_gmm_2)

            '''生成hmm实例'''
            lhmm = LHMM(states, observations, log, T=None, A=A, profunc=gmm, pi=None)

            '''数据结构：{基元：HMM,...}'''
            self.__unit[_unit] = lhmm

        """"""""""""""""""""""""""""""""""""""""""
        if unit:
            generate(unit)
            return
        else:
            for unit in self.__loaded_units:
                generate(unit)

    def init_parameter(self, unit=None, *args):
        """
        加载参数
        :param unit: 基元unit为None时，加载所有基元参数
        :param args: 为从本地读取参数传递的参数文件位置信息或数据库配置信息
                     args为(None,)时从默认地址读取参数，为(path,)时从path
                     读取参宿；args为(数据库参数信息)时从数据库读取
                     数据库参数信息：
                        1、数据库类型(mysql/sqlserver/..)
                        2、dict{host, usr, passwd, database}
                        3、表(Table)名
        :return: 
        """

        def load(_unit):
            address = self.__address + '/' + self.__unit_type
            unit_path = address + '/%s' % _unit
            detail_file = open(unit_path + '/detail.csv')
            A = np.load(unit_path + '/A.npy')
            pi = np.load(unit_path + '/pi.npy')
            hmm = self.__unit[_unit]
            hmm.change_A(A)
            hmm.change_pi(pi)
            for index_ in range(1, self.__state_num - 1):
                '''读取均值、协方差、权重矩阵'''
                means = np.load(unit_path + '/GMM_%d_means.npy' % index_)
                covariance = np.load(unit_path + '/GMM_%d_covariance.npy' % index_)
                alpha = np.load(unit_path + '/GMM_%d_weight.npy' % index_)
                mix_level = int(detail_file.readline().split()[-1])  # 获取高斯混合度
                '''将数据初始化到GMM'''
                gmm = hmm.profunction[index_]
                gmm.mean = means
                gmm.covariance = covariance
                gmm.alpha = alpha
                gmm.mixture = mix_level
            detail_file.close()

        if unit:
            load(unit)
            return
        if len(args) == 0:
            units = self.__loaded_units
            '''读取参数'''
            for index in range(len(units)):
                load(units[index])

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
            raise ArgumentNumberError(self.log)

    def __save_parameter(self, unit):
        """
        保存基元unit训练后的参数
        :param unit:基元
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_parameter = path_parameter_ + '/%s' % unit
        if not os.path.exists(unit_parameter):
            '''基元文件夹不存在时创建'''
            os.mkdir(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        hmm = self.__unit[unit]
        np.save(unit_parameter + '/A.npy', hmm.A)  # 保存状态转移矩阵
        np.save(unit_parameter + '/pi.npy', hmm.pi)  # 保存初始概率矩阵
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        detail_file = open(unit_parameter + '/detail.csv', 'w+')
        '''保存GMM参数'''
        for index_ in range(1, self.__state_num - 1):
            '''除去前后两个虚函数'''
            gmm = hmm.profunction[index_]
            np.save(unit_parameter + '/GMM_%d_means.npy' % index_, gmm.mean)  # 保存均值
            np.save(unit_parameter + '/GMM_%d_covariance.npy' % index_, gmm.covariance)  # 保存协方差矩阵
            np.save(unit_parameter + '/GMM_%d_weight.npy' % index_, gmm.alpha)  # 保存权重矩阵
            detail_file.writelines('GMM_%d %d\n' % (index_, gmm.mixture))  # 保存高斯混合度
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        with open(path_parameter_ + '/trainInfo_%s.csv' % self.__job_id, 'a+') as trainInfo:
            trainInfo.writelines('%s\n' % unit)
        detail_file.close()

    def __load_trainInfo(self):
        """
        读取上一次训练结果
        :return: 未训练基元集合
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        trainInfoLocation = path_parameter_ + '/trainInfo_%s.csv' % self.__job_id
        trained_units = []
        units = self.__loaded_units
        if os.path.exists(trainInfoLocation):
            with open(trainInfoLocation) as trainInfo:
                u = trainInfo.readline()
                while u:
                    trained_units.append(u.strip('\n'))
                    u = trainInfo.readline()
        else:
            return units
        wwt_unit = list(set(units).difference(set(trained_units)))  # 待训练基元
        return wwt_unit

    def __save_data(self, unit, unit_data):
        """
        保存每次切分的数据
        :param unit: 基元unit
        :param unit_data: 基元对应数据
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_path = path_parameter_ + '/%s' % unit
        if not os.path.exists(unit_path):
            '''基元数据文件夹不存在时创建'''
            os.mkdir(unit_path)
        unit_data_path = path_parameter_ + '/%s/data' % unit
        if not os.path.exists(unit_data_path):
            '''基元数据文件夹不存在时创建'''
            os.mkdir(unit_data_path)
        pid = os.getpid()
        localtime = int(time.time())
        file = open(unit_data_path + '/%s_data_%s_%s.pkl' % (unit, pid, localtime), 'wb')
        pickle.dump(unit_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def __load_data(self, unit):
        """
        读取切分的数据
        :param unit: 基元unit
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_data_path = path_parameter_ + '/%s/data' % unit
        if not os.path.exists(unit_data_path):
            hmm = self.__unit[unit]
            log = hmm.log  # 获取对应基元的日志文件
            '''基元数据文件夹不存在'''
            log.note('基元 %s 不存在数据，结束训练' % unit, cls='w')
            return None, None
        unit_data = []
        unit_data_t = []
        for dir in os.walk(unit_data_path):
            for filename in dir[2]:
                file = open(unit_data_path + '/' + filename, 'rb')
                data = pickle.load(file)
                unit_data.append(data)
                unit_data_t.append(len(data))
                file.close()
        return unit_data, unit_data_t

    def delete_data(self, unit, show_err=False):
        """
        用于清空data文件夹，释放空间
        :param unit: 基元
        :param show_err: 文件夹不存在时，输出错误信息
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_path = path_parameter_ + '/%s' % unit
        if os.path.exists(unit_path) is False:
            '''基元数据文件夹不存在'''
            if show_err:
                self.log.note('不存在基元 %s' % unit, cls='w')
            return
        unit_data_path = path_parameter_ + '/%s/data' % unit
        if os.path.exists(unit_data_path) is False:
            '''基元数据文件夹不存在'''
            if show_err:
                self.log.note('基元 %s 不存在数据' % unit, cls='w')
            return
        shutil.rmtree(unit_data_path)

    def delete_trainInfo(self):
        """
        清除基元训练信息
        :return:
        """
        self.log.note('正在清除基元训练信息......', cls='i')
        trainInfoLocation = self.__address + '/' + self.__unit_type + '/trainInfo_%s.csv' % self.__job_id
        if os.path.exists(trainInfoLocation):
            os.remove(trainInfoLocation)
        self.log.note('清除基元训练信息完成   √', cls='i')

    @staticmethod
    def init_audio(audiopath, labelpath):
        """
        加载音频地址
        :param audiopath: 音频路径
        :param labelpath: 标注路径
        :return: 音频及其标注地址的迭代器
        """
        count = 0  # 音频文件数目
        for dir in os.walk(audiopath):
            for _ in dir[2]:
                count += 1
        yield count
        for dir in os.walk(audiopath):
            for file in dir[2]:
                name = file.split('.')[0]
                wav_name = audiopath + '/%s.wav\n' % name
                label_name = labelpath + '/%s.wav.trn\n' % name
                yield wav_name, label_name

    def __flat_start(self, label_data):
        """
        均一起步
        (停止使用)
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
        cluster = Clustering.ClusterInitialization(_data, self.__mix_level, self.__vector_size, self.log)
        mean, covariance, alpha, clustered_data = cluster.kmeans(algorithm=1)

        '''训练GMM'''
        tmp_gmm = Clustering.GMM(None, self.__vector_size, self.__mix_level)
        tmp_gmm.data = data
        tmp_gmm.mean = mean
        tmp_gmm.covariance = covariance
        tmp_gmm.alpha = alpha
        '''GMM Baulm-Welch迭代'''
        tmp_gmm.baulm_welch()

        '''获取均值、协方差和权重值'''
        mean = tmp_gmm.mean
        covariance = tmp_gmm.covariance
        alpha = tmp_gmm.alpha

        for i in range(len(label)):
            hmm = self.__unit[label[i]][1]
            for j in range(1, len(hmm.profunction) - 1):
                '''除去前后两个虚方法'''
                gmm = hmm.profunction[j]
                gmm.mean = mean
                gmm.covariance = covariance
                gmm.alpha = alpha

    def __cal_hmm(self, unit, unit_data, unit_data_t, correct=None, show_q=False, show_a=False):
        """
        计算HMM
        :param unit: 基元
        :param unit_data: 该基元对应的数据
        :param unit_data_t: 该基元对应的数据长度
        :param correct: 纠正函数
        :param show_q: 显示当前似然度
        :param show_a: 显示重估后状态转移矩阵
        :return:
        """
        '''开始训练'''
        unit_hmm = self.__unit[unit]
        '''获取数据'''
        unit_hmm.add_data(unit_data)
        unit_hmm.add_T(unit_data_t)
        if len(unit_hmm.data) == 0:
            return
        unit_hmm.baulm_welch(correct=correct, show_q=show_q, show_a=show_a)
        '''清除数据，释放内存'''
        unit_hmm.clear_result_cache()
        unit_hmm.clear_data()

    def __cal_gmm(self, unit, unit_data, init=False, smem=False, show_q=False, c_covariance=1e-3):
        """
        计算GMM
        :param unit: 当前基元
        :param unit_data: 基元数据
        :param init: 是否初始化
        :param smem: 是否进行SMEM算法
        :param show_q: 显示当前似然度
        :param c_covariance: 修正数值，纠正GMM中的Singular Matrix
        :return:
        """
        hmm = self.__unit[unit]
        gmms_num = len(hmm.profunction) - 2  # 高斯混合模型数

        for i in range(1, len(hmm.profunction) - 1):
            '''除去前后两个虚方法'''

            gmm = hmm.profunction[i]
            gmm.log.note('正在训练GMM%d，共 %d GMM，混合度为 %d' % (i, gmms_num, self.__mix_level), cls='i')
            data = unit_data[i - 1]  # 对应高斯模型的数据
            gmm.add_data(data)

            if len(data) < self.__mix_level:
                gmm.log.note('数据过少，忽略该组数据', cls='w')
                continue
            if init or gmm.mixture != self.__mix_level:  # 当初始化模型或高斯混合度变化时，重新聚类
                cluster = Clustering.ClusterInitialization(data, self.__mix_level, self.__vector_size, gmm.log)
                mean, covariance, alpha, clustered_data = cluster.kmeans(algorithm=1, cov_matrix=True)
                gmm.mixture = self.__mix_level
                gmm.mean = mean
                gmm.covariance = covariance
                gmm.alpha = alpha
            '''GMM Baulm-Welch迭代'''
            gmm.baulm_welch(show_q=show_q, smem=smem, c_covariance=c_covariance)
            gmm.clear_data()  # 清空数据内存

    @staticmethod
    def __eq_distribution(data, state_num):
        """数据均分"""
        '''
            for z_tuple in zip(a,b):
                for frame in z_tuple:
                    ...
            各段数据进行整合，返回的元组长度为压缩前各列表长度的最小值
        '''
        _data = [frame for z_tuple in zip(*[child_data for child_data in data]) for frame in z_tuple]
        chunk = round(len(_data) / state_num)
        union_data = [None for _ in range(state_num)]
        start = 0
        end = chunk
        '''处理前N-1个数据集'''
        for _ in range(state_num - 1):
            union_data[_] = _data[start:end]
            start += chunk
            end += chunk
        '''处理最后一个数据集'''
        last_index = state_num - 1
        union_data[last_index] = _data[start:len(_data)]
        return union_data

    def __eq_segment(self, data, arg=None, mode='e'):
        """
        训练等分数据
        :param data: 数据序列
        :param arg: 可变参数(根据mode不同，意义不同)
        :param mode: 均分模式：
                        模式1:(mode='e')
                            :type arg :list // [unit1,unit2,unit3...] 数据标注列表
                            用于首词训练等分数据，并且保存
                        模式2:(mode='g')
                            :type arg :int //HMM状态数
                            重估训练中，将viterbi切分出的数据按HMM状态数等分，将等分的数据返回，用于GMM的训练
        :return: mode='e':
                    None
                 mode='g':
                    list[[各段数据] * 状态数]
        """

        if mode is 'e':  # equal模式
            chunk = len(data) // len(arg)
            start = 0
            for u in arg:
                end = start + chunk
                unit_data = data[start:end]
                start += chunk
                self.__save_data(u, unit_data)  # 保存切分好的数据分片
        elif mode is 'g':  # gmm模式
            chunk = len(data) // arg
            g_slice = []  # 处理单个语音数据后的均分数据
            start = 0
            '''添加1 ~ (arg-1)的数据'''
            for _ in range(arg - 1):
                end = start + chunk
                data_slice = data[start:end]
                g_slice.append(data_slice)
                start += chunk
            '''添加最后一个数据'''
            g_slice.append(data[start:])
            return g_slice
        else:
            raise ClassError(self.log)

    def get_gmmdata(self, data):
        """
        获取重估后gmm数据，用于gmm重估
        :param data: 对应基元的所有语音数据
        :return: gmm的重估数据
        """
        data_size = len(data)
        gmm_num = self.__state_num - 2
        g_data = [_ for _ in self.__eq_segment(data[0], gmm_num, mode='g')]  # 所有gmm语音数据的集合

        '''合并各段GMM数据'''
        for index in range(1, data_size):
            g_slice = self.__eq_segment(data[index], gmm_num, mode='g')
            for gmm_index in range(gmm_num):
                g_data[gmm_index] = np.append(g_data[gmm_index], g_slice[gmm_index], axis=0)
        return g_data

    def split_data(self, load_line=0, init=True):
        """
        处理音频数据
        :param load_line: 读取标注文件中，标注文字所在的行，默认第0行(即第一行)
                            如     你好
                                n i3 h ao3
                            标注在第2行，因此load_line=1
        :param init: 是否初始化参数(均分数据)
        :return:
        """
        self.log.note('正在获取数据路径列表...', cls='i')
        load_num = 0  # 已处理音频文件数目
        path_file = PARAMETERS_FILE_PATH + '/%s/pathInfo_%s.csv' % (self.__unit_type, self.__job_id)
        if not os.path.exists(path_file):
            raise PathInfoExistError(self.log)
        path_list = []  # [[音频路径],[标注路径],数据数目]
        file_count = 0
        with open(path_file) as pathInfo:
            line_audio = pathInfo.readline().strip('\n')
            line_label = pathInfo.readline().strip('\n')
            while line_audio:
                path_list.append([line_audio, line_label])
                line_audio = pathInfo.readline().strip('\n')
                line_label = pathInfo.readline().strip('\n')
                file_count += 1
        pool = Pool(processes=self.processes)
        self.log.note('处理音频数据中...', cls='i')
        for path in path_list:  # 遍历迭代器 path[0]为音频路径，path[1]为标注路径,path[2]为文件总数
            load_num += 1
            f_label = open(path[1])
            line = load_line
            label = ''
            while line >= 0:  # 获取对应行
                label = f_label.readline()
                line -= 1
            f_label.close()
            label = label.strip('\n').split(" ")  # 标注去回车并按空格形成列表
            _args = [load_num, file_count]
            pool.apply_async(self.multi_split_data, args=(label, path[0], init, *_args))
        pool.close()
        pool.join()
        self.log.note('音频处理完成   √', cls='i')

    def multi_split_data(self, label, path, init, *args):
        """
        多进程处理音频
        :param label: 音频标注
        :param path: 音频数据路径
        :param init: 是否均分音频
        :param args: 其他参数(当前处理音频数、总音频数)
        :return:
        """
        if init:
            data = self.__load_audio(path)  # 加载音频数据
            self.__eq_segment(data, label, mode='e')
            self.log.note('当前已处理音频：%d / %d' % (args[0], args[1]), cls='i')
        else:
            data = self.__load_audio(path)  # 加载音频数据
            label_set = list(set(label))  # 标注基元集合
            for label_u in label_set:
                '''初始化B_p'''
                self.init_unit(label_u, new_log=init)
                self.init_parameter(unit=label_u)
                unit_hmm = self.__unit[label_u]
                unit_hmm.cal_observation_pro([data], [len(data)])
                unit_hmm.clear_data()
            '''viterbi切分数据'''
            point, sequence = self.viterbi(label, len(data), 0)
            sequence_num = len(set(sequence))  # viterbi切分所得序列中，基元个数
            label_num = len(label_set)  # 数据标注中应出现的基元个数
            """序列基元数少于应出现基元数，视为viterbi切分失败，抛弃该数据"""
            if sequence_num < label_num:
                self.log.note('viterbi切分失败', cls='w')
                self.log.note('当前已处理音频：%d / %d' % (args[0], args[1]), cls='i')
                return
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""
            for label_u in label_set:  # 将同类基元数据合并保存
                loc_list = AcousticModel.discriminate(label_u, sequence)
                for loc in loc_list:
                    '''获取同一基元、不同部分的数据'''
                    data_u = data[loc]
                    self.__save_data(label_u, data_u)
            self.log.note('当前已处理音频：%d / %d' % (args[0], args[1]), cls='i')
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def training(self, init=True, show_q=False, show_a=False, c_covariance=1e-3):
        """
        训练参数
        采用嵌入式训练，串联标注所对应的HMM
        :param init: 初始化参数
        :param show_q: 显示HMM/GMM当前似然度
        :param show_a: 显示HMM重估后状态转移矩阵
        :param c_covariance: 修正数值，纠正GMM中的Singular Matrix
        :return:
        """
        wwt_units = self.__load_trainInfo()  # 未训练基元集合
        unit_num = len(self.__loaded_units)  # 基元全集数目
        wwt_unit_num = len(wwt_units)  # 未初始化基元集合
        init_unit_num = 0  # 已初始化基元数
        pool = Pool(processes=self.processes)
        '''基元训练(初始化/重估)'''
        for unit in wwt_units:
            init_unit_num += 1
            _args = (show_q, show_a, c_covariance, init_unit_num, wwt_unit_num, unit_num)
            pool.apply_async(self.multi_training, args=(unit, init, *_args))
        pool.close()
        pool.join()
        """"""
        '''清空data文件夹释放空间'''
        self.log.note('正在释放存储空间......', cls='i')
        for unit in self.__loaded_units:
            self.delete_data(unit)
        self.log.note('释放空间完成   √', cls='i')
        '''清除基元训练信息'''
        self.delete_trainInfo()
        ''''''

    def multi_training(self, unit, init=True, *args):
        """
        多进程训练
        :param unit: 基元
        :param init: 是否初始化
        :param args: 承接training方法的各种参数和变量(按在training的出现顺序排列)
        :return:
        """
        self.init_unit(unit=unit, new_log=init)
        if init:  # 首次训练，需要初始化
            self.log.note('正在初始化基元%s,已初始化 %d / %d , 共 %d' % (unit, args[3], args[4], args[5]), cls='i')
            unit_data, unit_data_t = self.__load_data(unit)
            if unit_data is None:
                self.__save_parameter(unit)
                return
            '''为GMM划分数据'''
            unit_data_2 = self.get_gmmdata(unit_data)
            self.__cal_gmm(unit, unit_data_2, init=init, smem=False, show_q=args[0], c_covariance=args[2])
            self.__cal_hmm(unit, unit_data, unit_data_t, show_q=args[0], show_a=args[1])
            '''保存该基元参数'''
            self.__save_parameter(unit)
        else:  # 进行重估
            self.init_parameter(unit=unit)
            self.log.note('正在重估基元%s,已重估 %d / %d , 共 %d' % (unit, args[3], args[4], args[5]), cls='i')
            unit_data, unit_data_t = self.__load_data(unit)
            if unit_data is None:
                self.__save_parameter(unit)
                return
            '''为GMM划分数据'''
            unit_data_2 = self.get_gmmdata(unit_data)
            self.__cal_gmm(unit, unit_data_2, smem=True, show_q=args[0], c_covariance=args[2])
            self.__cal_hmm(unit, unit_data, unit_data_t, show_q=args[0], show_a=args[1])
            '''保存该基元参数'''
            self.__save_parameter(unit)
        '''关闭日志'''
        self.log.close()
        ''''''

    @staticmethod
    def discriminate(unit, sequence):
        """
        区分数据中多次出现的基元，并进行分割处理
        :param unit: 基元
        :param sequence: vitebi切分后的数据标注
        :return:
        """
        loc = np.where(sequence == unit)[0]
        '''区分化相同标注的不同位置'''
        seq_num = np.arange(len(loc))
        sub = loc - seq_num
        set_sub = list(set(sub))
        '''同一标注、不同位置的数据索引集合'''
        loc_list = []

        for index in range(len(set_sub)):
            loc_list.append(loc[np.where(sub == set_sub[index])])
        return loc_list

    def embedded(self, label, data_index, alter=31):
        """
        嵌入式模型(构建句子级HMM)
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
            complex_A[:self.__state_num - 1, :self.__state_num] = self.__unit[label[0]].A[:-1]
            for i in range(1, len(label)):
                A = self.__unit[label[i]].A
                '''拼接'''
                a = i * (self.__state_num - 2) + 1
                b = (i + 1) * (self.__state_num - 2) + 1
                complex_A[a:b, a - 1: a - 1 + self.__state_num] = A[1:-1]
            return complex_A

        def embedded_B():
            """复合观测矩阵"""
            i = 0
            complex_B = self.__unit[label[0]].B_p[data_index][0:-1]
            for i in range(1, len(label)):
                B = self.__unit[label[i]].B_p[data_index]
                '''拼接'''
                complex_B = np.append(complex_B, B[1:-1, :], axis=0)
            '''后续处理'''
            B = self.__unit[label[i]].B_p[data_index][-1:, :]
            complex_B = np.append(complex_B, B, axis=0)  # 添加最后一个虚函数
            return complex_B

        def embedded_pi():
            """复合初始概率矩阵"""
            complex_pi = np.ones((state_size,)) / state_size
            return complex_pi

        ''''''
        func_list = [embedded_states, embedded_observation, embedded_A, embedded_B, embedded_pi]
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
        complex_states, complex_observation, complex_A, complex_B, complex_pi = self.embedded(label, data_index, 31)
        '''维特比强制对齐'''
        return LHMM.viterbi(self.log, complex_states, complex_observation, complex_A, complex_B, complex_pi,
                            O_size=data_size, matrix=False, convert=True, end_state_back=False)

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
        m = mfcc.mfcc(nfft=512, d1=self.__delta_1, d2=self.__delta_2)
        vad = self.__audio.VAD()
        vad.init_mfcc(m)
        filtered_mfcc = vad.mfcc()
        return filtered_mfcc

    class VirtualState(object):
        """
        为虚状态设立的评分类
        """

        def __init__(self, p=0.):
            self.__p = p

        def point(self, x, log=False, standard=False):
            """返回评分为p,x接受参数，但不处理"""
            return self.__p
