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
                 mode=0,
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
        :param mode: 训练模式
        :param unit_type: 基元类型
        :param processes: 进程数
        :param job_id: 作业id
        :param console: 控制台是否显示基元训练信息
        :param state_num: 每个基元的状态数，一般为5个及以上(首尾两个状态为非发射状态)
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
        '''存储所有基元'''
        self.__loaded_units = []
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

        '''声学模型训练模式'''
        self.__mode = mode

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

    def init_unit(self, unit, new_log=True, fix_code=0):
        """
        初始化基元，生成基元的复合数据结构
        :param unit: 初始化指定基元
        :param new_log: 是否删除先前日志
        :param fix_code: 关闭参数更新，000=0 001=1 010=2 100=4...
        :return:
        """
        """"""
        '''状态集合'''
        states = {_: unit for _ in range(self.__state_num)}
        '''状态转移矩阵'''
        transmat = np.zeros((self.__state_num, self.__state_num))
        '''开始状态，为虚状态，只允许向下一个状态转移'''
        transmat[0][1] = 1.
        for j in range(1, self.__state_num - 1):
            transmat[j][j] = 0.5  # 第一个转移概率
            transmat[j][j + 1] = 0.5  # 第二个转移概率
        '''创建基元文件夹'''
        unit_path = PARAMETERS_FILE_PATH + '/%s/%s' % (self.__unit_type, unit)
        log_hmm_path = unit_path + '/HMM'
        log_gmm_path = [unit_path + '/GMM_%d' % gmm_id for gmm_id in range(self.__state_num - 2)]
        try:
            os.mkdir(unit_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(log_hmm_path)
        except FileExistsError:
            pass
        try:
            for gmm_id in range(self.__state_num - 2):
                os.mkdir(log_gmm_path[gmm_id])
        except FileExistsError:
            pass
        ''''''''''''''''''
        log_hmm = Log(self.__unit_type, log_hmm_path, console=self.__console)
        log_gmm = [Log(self.__unit_type, path=log_gmm_path[gmm_id], console=self.__console) for gmm_id in
                   range(self.__state_num - 2)]
        if new_log:
            log_hmm.generate()
            for gmm_id in range(self.__state_num - 2):
                log_gmm[gmm_id].generate()
        else:
            log_hmm.append()
            for gmm_id in range(self.__state_num - 2):
                log_gmm[gmm_id].append()
        '''初始化GMM'''
        gmm = []
        for gmm_id in range(self.__state_num - 2):
            gmm.append(Clustering.GMM(log_gmm[gmm_id], dimension=self.__vector_size, mix_level=self.__mix_level,
                                      gmm_id=gmm_id))

        '''初始化虚状态评分类'''
        virtual_gmm_1 = AcousticModel.VirtualState(1.)
        virtual_gmm_2 = AcousticModel.VirtualState(0.)

        gmm.insert(0, virtual_gmm_1)
        gmm.append(virtual_gmm_2)

        '''生成hmm实例'''
        lhmm = LHMM(states, self.__state_num, log_hmm, transmat=transmat, profunc=gmm, fix_code=fix_code)
        return lhmm

    def init_parameter(self, unit, hmm):
        """
        加载参数
        :param unit: 基元
        :param hmm: 外部传入的HMM实例
        :return: 
        """
        address = self.__address + '/' + self.__unit_type
        unit_path = address + '/%s' % unit
        hmm.init_parameter(unit_path)
        for index in range(1, self.__state_num - 1):
            gmm = hmm.profunction[index]
            gmm.init_parameter(unit_path)

    def __save_parameter(self, unit, hmm):
        """
        保存基元unit训练后的参数
        :param unit:基元
        :param hmm: 外部传入HMM实例
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_parameter = path_parameter_ + '/%s' % unit
        if not os.path.exists(unit_parameter):
            '''基元文件夹不存在时创建'''
            os.mkdir(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''保存HMM参数'''
        hmm.save_parameter(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''保存GMM参数'''
        for index_ in range(1, self.__state_num - 1):
            '''除去前后两个虚函数'''
            profunc = hmm.profunction[index_]
            profunc.save_parameter(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        with open(path_parameter_ + '/trainInfo_%s.csv' % self.__job_id, 'a+') as trainInfo:
            trainInfo.writelines('%s\n' % unit)

    def __save_acc(self, unit, hmm):
        """
        保存基元的累加器
        :param unit: 基元
        :param hmm: 外部传入的HMM实例
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_parameter = path_parameter_ + '/%s' % unit
        if not os.path.exists(unit_parameter):
            '''基元文件夹不存在时创建'''
            os.mkdir(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''保存HMM参数'''
        hmm.save_acc(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''保存GMM参数'''
        for index_ in range(1, self.__state_num - 1):
            '''除去前后两个虚函数'''
            profunc = hmm.profunction[index_]
            profunc.save_acc(unit_parameter)

    def __init_acc(self, unit, hmm):
        """
        读取基元的累加器
        :param unit: 基元
        :param hmm: 外部传入的HMM实例
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_parameter = path_parameter_ + '/%s' % unit
        if not os.path.exists(unit_parameter):
            '''基元文件夹不存在时创建'''
            os.mkdir(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''保存HMM参数'''
        hmm.init_acc(unit_parameter)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''保存GMM参数'''
        for index_ in range(1, self.__state_num - 1):
            '''除去前后两个虚函数'''
            profunc = hmm.profunction[index_]
            profunc.init_acc(unit_parameter)

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

    def __load_data(self, unit, hmm):
        """
        读取切分的数据
        :param unit: 基元unit
        :param hmm: 外部传入HMM实例
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_data_path = path_parameter_ + '/%s/data' % unit
        if not os.path.exists(unit_data_path):
            log = hmm.log  # 获取对应基元的日志文件
            '''基元数据文件夹不存在'''
            log.note('基元 %s 不存在数据，结束训练' % unit, cls='w')
            return None
        unit_data = []
        for dir in os.walk(unit_data_path):
            for filename in dir[2]:
                file = open(unit_data_path + '/' + filename, 'rb')
                data = pickle.load(file)
                unit_data.append(data)
                file.close()
        return unit_data

    def delete_buffer_file(self, unit, show_info=False):
        """
        用于清空缓存文件夹(data/acc)，释放空间
        :param unit: 基元
        :param show_info: 文件夹不存在时，输出错误信息
        :return:
        """
        path_parameter_ = self.__address + '/' + self.__unit_type
        unit_path = path_parameter_ + '/%s' % unit
        if os.path.exists(unit_path) is False:
            '''基元数据文件夹不存在'''
            self.log.note('不存在基元 %s' % unit, cls='w', show_console=show_info)
            return

        """删除data文件"""

        unit_data_path = path_parameter_ + '/%s/data' % unit
        try:
            shutil.rmtree(unit_data_path)
            info_detail_data = '\033[0;30;1mSucceed\033[0m'
        except FileNotFoundError:
            info_detail_data = '\033[0;31;1mFailed\033[0m'

        info = 'Unit: \033[0;30;1m%-5s\033[0m\t**Data File**\tGMM--\t%s\t' % (unit, info_detail_data)

        """删除acc文件"""
        hmm_file = unit_path + '/HMM'
        hmm_ksai_acc_file = hmm_file + '/ksai-acc'
        hmm_gamma_acc_file = hmm_file + '/gamma-acc'

        try:
            shutil.rmtree(hmm_ksai_acc_file)
            shutil.rmtree(hmm_gamma_acc_file)
            info_detail_acc_h = '\033[0;30;1mSucceed\033[0m'
        except FileNotFoundError:
            info_detail_acc_h = '\033[0;31;1mFailed\033[0m'

        info_detail_acc_g = None
        for gmm_id in range(self.__state_num - 2):
            gmm_file = unit_path + '/GMM_%d' % gmm_id
            gmm_acc_file = gmm_file + '/acc'
            gmm_alpha_acc_file = gmm_file + '/alpha-acc'
            gmm_mean_acc_file = gmm_file + '/mean-acc'
            gmm_covariance_acc_file = gmm_file + '/covariance-acc'
            try:
                shutil.rmtree(gmm_acc_file)
                shutil.rmtree(gmm_alpha_acc_file)
                shutil.rmtree(gmm_mean_acc_file)
                shutil.rmtree(gmm_covariance_acc_file)
                info_detail_acc_g = '\033[0;30;1mSucceed\033[0m'
            except FileNotFoundError:
                info_detail_acc_g = '\033[0;31;1mFailed\033[0m'

        info += '**Acc File**\tHMM--\t%s\tGMM--\t%s' % (info_detail_acc_h, info_detail_acc_g)
        self.log.note(info, cls='i', show_console=show_info)

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

    def __flat_start(self, path_list, file_count, proportion=0.25, step=1, differentiation=True, coefficient=1.):
        """
            均一起步
        计算全局均值和方差
        :param path_list: 数据路径列表
        :param file_count: 数据总量
        :param proportion: 训练数据中，用于计算全局均值和协方差的数据占比
        :param step: 在帧中跳跃选取的跳跃步长
        :param differentiation: GMM中各分模型参数差异化处理
        :param coefficient: 差异化程度，区间[0,1]
        :return:
        """
        self.log.note('flat starting...', cls='i')
        p_file_count = int(file_count * proportion)
        p_data = self.__load_audio(path_list[0][0])
        p_data = p_data[::step]
        for index in range(1, p_file_count):
            data = self.__load_audio(path_list[index][0])  # 加载音频数据
            data = data[::step]
            p_data = np.append(p_data, data, axis=0)
        cluster = Clustering.ClusterInitialization(p_data, 1, self.__vector_size, self.log)
        mean, covariance, alpha, clustered_data = cluster.kmeans(algorithm=1, cov_matrix=True)
        covariance_diagonal = covariance[0].diagonal()
        units = self.__loaded_units
        ''''''
        diff_coefficient = np.zeros((self.__mix_level, 1))
        if differentiation:
            '''差异化处理'''
            assert 0 <= coefficient <= 1, '差异化系数不满足区间[0,1]'
            diff_coefficient = (np.random.random((self.__mix_level, 1)) - np.random.random(
                (self.__mix_level, 1))) * coefficient
        for unit in units:
            hmm = self.init_unit(unit, new_log=True)
            gmms = hmm.profunction[1: -1]
            for g in gmms:
                g.mean = mean.repeat(self.__mix_level, axis=0) + diff_coefficient * covariance_diagonal
                g.covariance = covariance.repeat(self.__mix_level, axis=0)
            self.__save_parameter(unit, hmm)
        self.delete_trainInfo()

    def __cal_hmm(self, unit, hmm, show_q=False, show_a=False, c_covariance=1e-3):
        """
        计算HMM
        :param unit: 基元
        :param hmm: 外部传入HMM实例
        :param show_q: 显示GMM重估信息
        :param show_a: 显示重估后状态转移矩阵
        :return:
        """
        '''获取数据'''
        self.__init_acc(unit, hmm)
        hmm.update_param(show_q=show_q, show_a=show_a, c_covariance=c_covariance)

    def __cal_gmm(self, hmm, unit_data, init=False, smem=False, show_q=False, c_covariance=1e-3):
        """
        计算GMM
        :param hmm: 外部传入HMM实例
        :param unit_data: 基元数据
        :param init: 是否初始化
        :param smem: 是否进行SMEM算法
        :param show_q: 显示当前似然度
        :param c_covariance: 修正数值，纠正GMM中协方差值过小问题
        :return:
        """
        for i in range(1, self.__state_num - 1):
            '''除去前后两个虚方法'''
            gmm = hmm.profunction[i]
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
            gmm.em(show_q=show_q, smem=smem, c_covariance=c_covariance)
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

    def __get_gmmdata(self, data):
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

    def __get_data_list(self):
        """读取音频、标注文件路径列表，统计音频数"""
        self.log.note('正在获取数据路径列表...', cls='i')
        path_file = PARAMETERS_FILE_PATH + '/%s/pathInfo_%s.csv' % (self.__unit_type, self.__job_id)
        if not os.path.exists(path_file):
            raise PathInfoExistError(self.log)
        path_list = []  # [[音频路径,标注路径],...]
        file_count = 0
        with open(path_file) as pathInfo:
            line_audio = pathInfo.readline().strip('\n')
            line_label = pathInfo.readline().strip('\n')
            while line_audio:
                path_list.append([line_audio, line_label])
                line_audio = pathInfo.readline().strip('\n')
                line_label = pathInfo.readline().strip('\n')
                file_count += 1
        return path_list, file_count

    def __generator(self, path_list, load_line):
        """
        根据音频路径列表处理音频，并返回标注和处理后的音频特征(MFCC)
        :param path_list: 音频及其标注的路径列表
        :param load_line: 标注文字所在行
        :return:
        """
        for path in path_list:  # 遍历迭代器 path[0]为音频路径，path[1]为标注路径,path[2]为文件总数
            f_label = open(path[1])
            line = load_line
            label = ''
            while line >= 0:  # 获取对应行
                label = f_label.readline()
                line -= 1
            f_label.close()
            label = label.strip('\n').split(" ")  # 标注数据，去回车并按空格形成列表
            data = self.__load_audio(path[0])  # 音频数据
            yield label, data

    def process_data(self, mode=1, load_line=0, init=True, proportion=0.25, step=1, differentiation=True,
                     coefficient=1):
        """
        处理音频数据
        :param mode: 训练方案
        :param load_line: 读取标注文件中，标注文字所在的行，默认第0行(即第一行)
                            如     你好
                                n i3 h ao3
                            标注在第2行，因此load_line=1
        :param init: 是否初始化参数(均分数据)
        :param proportion: (Flat-starting 参数) 训练数据中，用于计算全局均值和协方差的数据占比
        :param step: (Flat-starting 参数) 在帧中跳跃选取的跳跃步长
        :param differentiation: (Flat-starting 参数) GMM中各分模型参数差异化处理
        :param coefficient: (Flat-starting 参数) 差异化程度，区间[0,1]
        :return:
        """
        self.log.note('处理音频数据中...', cls='i')
        load_num = 0  # 已处理音频文件数目

        path_list, file_count = self.__get_data_list()
        label_data_generator = self.__generator(path_list, load_line)
        if mode == 1:
            fix_code = 2  # 锁定发射概率密度函数的重估(b'010')
            if not init:
                self.log.note('Viterbi Alignment...', cls='i')
            pool = Pool(processes=self.processes)
            for l, d in label_data_generator:
                load_num += 1
                args = [load_num, file_count, fix_code]
                pool.apply_async(self.multi_process_data, args=(l, d, init, *args))
            pool.close()
            pool.join()
        elif mode == 2:
            if init:
                self.__flat_start(path_list, file_count, proportion=proportion, step=step,
                                  differentiation=differentiation, coefficient=coefficient)
        else:
            raise ModeError(self.log)
        self.log.note('音频处理完成   √', cls='i')

    def multi_process_data(self, label, data, init, *args):
        """
        多进程处理数据(方案1)
        :param label: 音频标注(基元为单位的列表[a,b,c,...])
        :param data: 音频数据
        :param init: 是否均分音频
        :param args: 其他参数(当前处理音频数、总音频数、fix_code)
        :return:
        """
        data_list = [data]
        data_t_list = [len(data)]
        if init:
            self.__eq_segment(data, label, mode='e')
        else:
            label_set = list(set(label))  # 标注基元集合
            hmm_list = []
            for label_u in label:
                '''初始化B_p'''
                unit_hmm = self.init_unit(unit=label_u, new_log=init)
                self.init_parameter(unit=label_u, hmm=unit_hmm)
                hmm_list.append(unit_hmm)
                unit_hmm.cal_observation_pro(data_list, data_t_list)
                unit_hmm.clear_data()
            '''生成嵌入式HMM'''
            complex_states, complex_transmat, complex_prob, complex_pi = self.embedded(label, hmm_list, 0, 15)
            np.savetxt('prob.csv', complex_prob)
            '''viterbi切分数据'''
            point, sequence = self.viterbi(complex_states, complex_transmat, complex_prob, complex_pi)
            sequence_num = len(set(sequence))  # viterbi切分所得序列中，基元个数
            label_num = len(label_set)  # 数据标注中应出现的基元个数
            """序列基元数少于应出现基元数，视为viterbi切分失败，抛弃该数据"""
            if sequence_num < label_num:
                self.log.note('viterbi切分失败', cls='w')
                self.log.note('当前已切分音频：%d / %d' % (args[0], args[1]), cls='i')
                return
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""
            for label_u in label_set:  # 将同类基元数据合并保存
                loc_list = AcousticModel.discriminate(label_u, sequence)
                for loc in loc_list:
                    '''获取同一基元、不同部分的数据'''
                    data_u = data[loc]
                    self.__save_data(label_u, data_u)
        ''''''
        self.log.note('当前已切分音频：%d / %d' % (args[0], args[1]), cls='i')
        ''''''
        self.log.close()
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def training(self, mode=1, init=True, show_q=False, show_a=False, load_line=0, c_covariance=1e-3):
        """
        训练参数
        采用嵌入式训练，串联标注所对应的HMM
        :param mode: 训练方案
        :param init: 初始化参数
        :param show_q: 显示HMM/GMM当前似然度
        :param show_a: 显示HMM重估后状态转移矩阵
        :param load_line: 读取标注文件中，标注文字所在的行，默认第0行(即第一行)
        :param c_covariance: 修正数值，纠正GMM中的Singular Matrix
        :return:
        """

        wwt_units = self.__load_trainInfo()  # 未训练基元集合
        unit_num = len(self.__loaded_units)  # 基元全集数目
        wwt_unit_num = len(wwt_units)  # 未初始化/训练基元集合
        trained_unit_num = 0  # 已初始化/训练基元数
        if mode == 1:
            fix_code = 2
            pool = Pool(processes=self.processes)
            '''基元训练(初始化/重估)'''
            for unit in wwt_units:
                trained_unit_num += 1
                args = (show_q, show_a, c_covariance, trained_unit_num, wwt_unit_num, unit_num)
                pool.apply_async(self.multi_training, args=(unit, init, *args))
            pool.close()
            pool.join()
        elif mode == 2:
            fix_code = 0
        else:
            raise ModeError(self.log)
        '''嵌入式训练HMM'''
        self.embedded_training(wwt_units, init=init, load_line=load_line, fix_code=fix_code, show_q=show_q,
                               show_a=show_a, c_covariance=c_covariance)

        '''清空data文件夹释放空间'''
        self.log.note('正在释放存储空间......', cls='i')
        for unit in self.__loaded_units:
            self.delete_buffer_file(unit, show_info=True)
        self.log.note('释放空间完成   √', cls='i')
        '''清除基元训练信息'''
        self.delete_trainInfo()
        ''''''

    def multi_training(self, unit, init, *args):
        """
        多进程训练(方案1)
        :param unit: 基元
        :param init: 是否初始化
        :param args: 承接training方法的各种参数和变量(按在training的出现顺序排列)
        :return:
        """
        hmm = self.init_unit(unit, new_log=init, fix_code=2)
        if init:  # 首次训练，需要初始化
            self.log.note('正在初始化基元%s,已初始化 %d / %d , 共 %d' % (unit, args[3], args[4], args[5]), cls='i')
        else:  # 进行重估
            self.log.note('正在重估基元%s,已重估 %d / %d , 共 %d' % (unit, args[3], args[4], args[5]), cls='i')

        unit_data = self.__load_data(unit, hmm)
        if unit_data is None:
            self.__save_parameter(unit, hmm)
            return
        '''为GMM划分数据'''
        unit_data_2 = self.__get_gmmdata(unit_data)
        self.__cal_gmm(hmm, unit_data_2, init=init, smem=init, show_q=args[0], c_covariance=args[2])
        '''保存该基元参数'''
        self.__save_parameter(unit, hmm)
        '''关闭日志'''
        self.log.close()
        ''''''

    def embedded_training(self, wwt_units, init=True, load_line=0, fix_code=0, show_q=False, show_a=False,
                          c_covariance=1e-3):
        """
        嵌入式训练HMM
        :param wwt_units: 未训练基元集合
        :param init: 是否初始化
        :param load_line: 读取标注文件中，标注文字所在的行，默认第0行(即第一行)
                            如     你好
                                n i3 h ao3
                            标注在第2行，因此load_line=1
        :param fix_code: 参数锁定
        :param show_q: 显示HMM/GMM当前似然度
        :param show_a: 显示HMM重估后状态转移矩阵
        :param c_covariance: 修正数值，纠正GMM中的Singular Matrix
        :return:
        """
        self.log.note('Embedded Training...', cls='i')
        """累计ACC"""
        load_num = 0  # 已处理音频文件数目
        pool = Pool(processes=self.processes)
        path_list, file_count = self.__get_data_list()
        label_data_generator = self.__generator(path_list, load_line)

        for l, d in label_data_generator:
            load_num += 1
            args = [show_q, load_num, file_count, fix_code]
            pool.apply_async(self.multi_embedded_training_1, args=(l, d, init, *args))
        pool.close()
        pool.join()
        """读取ACC 训练HMM"""
        pool = Pool(processes=self.processes)
        unit_num = len(self.__loaded_units)  # 基元全集数目
        wwt_unit_num = len(wwt_units)  # 未初始化/训练基元集合
        trained_unit_num = 0  # 已初始化/训练基元数

        for unit in wwt_units:
            trained_unit_num += 1
            args = (show_q, show_a, c_covariance, trained_unit_num, wwt_unit_num, unit_num, fix_code)
            pool.apply_async(self.multi_embedded_training_2, args=(unit, init, *args))
        pool.close()
        pool.join()

    def multi_embedded_training_1(self, label, data, init, *args):
        """
        多进程嵌入式训练HMM
        :param label: 音频标注(基元为单位的列表[a,b,c,...])
        :param data: 音频数据
        :param init: 是否初始化
        :param args: 其他参数(show_q、当前处理音频数、总音频数、fix_code)
        :return:
        """
        hmm_list = []
        data_list = [data]
        data_t_list = [len(data)]
        '''为每个HMM计算观测概率密度'''
        for unit in label:
            hmm = self.init_unit(unit=unit, new_log=init)
            self.init_parameter(unit, hmm=hmm)
            hmm_list.append(hmm)
            hmm.cal_observation_pro(data_list, data_t_list)
            hmm.clear_data()
        '''生成嵌入式HMM'''
        complex_states, complex_transmat, complex_prob, complex_pi = self.embedded(label, hmm_list, 0, 15)

        embed_hmm = LHMM(complex_states, self.__state_num, self.log, transmat=complex_transmat,
                         probmat=[complex_prob], pi=complex_pi, hmm_list=hmm_list, fix_code=args[3])
        embed_hmm.add_data(data_list)
        embed_hmm.add_T(data_t_list)
        embed_hmm.baulm_welch(show_q=args[0])

        for index in range(len(label)):
            self.__save_acc(label[index], hmm_list[index])
        self.log.note('当前已处理音频：%d / %d' % (args[1], args[2]), cls='i')
        '''关闭日志'''
        self.log.close()

    def multi_embedded_training_2(self, unit, init, *args):
        """
        多进程训练(方案2)
        :param unit: 基元
        :param init: 是否初始化
        :param args: 承接training方法的各种参数和变量(show_q,show_a, c_covariance, trained_unit_num, wwt_unit_num, unit_num,
                    fix_code)
        :return:
        """
        self.log.note('正在训练基元%s,已完成 %d / %d , 共 %d' % (unit, args[3], args[4], args[5]), cls='i')
        hmm = self.init_unit(unit, new_log=init, fix_code=args[6])
        self.init_parameter(unit, hmm)
        self.__cal_hmm(unit, hmm, show_q=args[0], show_a=args[1], c_covariance=args[2])
        '''保存参数'''
        self.__save_parameter(unit, hmm)
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

    def embedded(self, label, hmm_list, data_index, alter=15):
        """
        嵌入式模型(构建句子级HMM)
        :param label: 标注
        :param hmm_list: 标注所对应的HMM列表[hmm_a,hmm_b,hmm_c,...]
        :param data_index: 数据索引
        :param alter: 选择性生成，1111(二进制) = 15为全部生成
        :return:
        """
        state_size = (self.__state_num - 2) * len(hmm_list) + 2

        def embedded_states():
            """复合状态集合"""
            state_list = [label[0]]
            for i in range(len(label)):
                state_list.extend([label[i] for _ in range(self.__state_num - 2)])
            state_list.append(label[len(label) - 1])
            map_list = list(map(lambda x, y: [x, y], [_ for _ in range(state_size)], state_list))
            complex_states = dict(map_list)
            return complex_states

        def embedded_transmat():
            """复合状态转移矩阵"""
            complex_transmat = np.zeros((state_size, state_size))
            complex_transmat[:self.__state_num - 1, :self.__state_num] = hmm_list[0].transmat[:-1]
            for i in range(0, len(label)):
                transmat = hmm_list[i].transmat
                '''拼接'''
                a = i * (self.__state_num - 2) + 1
                b = (i + 1) * (self.__state_num - 2) + 1
                complex_transmat[a:b, a - 1: a - 1 + self.__state_num] = transmat[1:-1]
            return complex_transmat

        def embedded_prob():
            """复合观测矩阵"""
            i = 0
            complex_prob = hmm_list[0].B_p[data_index][0:-1]
            for i in range(1, len(label)):
                prob = hmm_list[i].B_p[data_index]
                '''拼接'''
                complex_prob = np.append(complex_prob, prob[1:-1, :], axis=0)
            '''后续处理'''
            prob = hmm_list[i].B_p[data_index][-1:, :]
            complex_prob = np.append(complex_prob, prob, axis=0)  # 添加最后一个虚函数
            return complex_prob

        def embedded_pi():
            """复合初始概率矩阵"""
            complex_pi = np.ones((state_size,)) / state_size
            return complex_pi

        ''''''
        func_list = [embedded_states, embedded_transmat, embedded_prob, embedded_pi]
        embedded_list = []
        for index in range(4):
            if 2 ** (4 - index - 1) & alter != 0:
                embedded_list.append(func_list[index]())
        return embedded_list

    def viterbi(self, complex_states, complex_transmat, complex_prob, complex_pi):
        """
        维特比切分
        :param complex_states: 复合状态矩阵
        :param complex_transmat: 复合状态转移矩阵
        :param complex_prob: 复合观测矩阵
        :param complex_pi: 复合初始概率矩阵
        :return:
        """
        '''维特比强制对齐'''
        return LHMM.viterbi(self.log, complex_states, complex_transmat, complex_prob, complex_pi, convert=True,
                            show_mark_state=True)

    class VirtualState(object):
        """
        为虚状态设立的评分类
        """

        def __init__(self, p=0.):
            """"""
            np.seterr(divide='ignore')  # 忽略np.log(0.)错误
            self.__p = p

        def point(self, x, log=False, standard=False, record=False):
            """返回评分为p,x接受参数，但不处理"""
            if log:
                return np.log(self.__p)
            return self.__p
