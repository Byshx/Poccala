# -*-coding:utf-8-*-
"""
模型训练控制器
------------
用于任务划分、并行训练

Author:Byshx

Date:2018.01.17

"""
from init import *  # 加载配置config.ini
from LogPrint import Log
from Exceptions import *
from AcousticModel.AcousticModel import AcousticModel

AUDIO_FILE_PATH = os.path.abspath(os.environ['audio_file_path'])
LABEL_FILE_PATH = os.path.abspath(os.environ['label_file_path'])
PARAMETERS_FILE_PATH = os.path.abspath(os.environ['parameters_file_path'])


class Task(object):
    def __init__(self, num=1, console=False, **kwargs):
        """
        :param num: 参与训练的机器数量(单机情况下，num=1)
        :param console: 是否显示基元训练信息
        :param kwargs: 声学模型参数
        """
        if 'task_num' in kwargs.keys():
            self.num = kwargs['task_num']
        else:
            self.num = num
        '''检测必要的目录是否存在，不存在则创建'''
        path = PARAMETERS_FILE_PATH + '/' + kwargs['unit_type']
        if not os.path.exists(path):
            os.makedirs(path)
        ''''''''''''''''''''''''''''''''''''
        self.console = console  # 当console=True时，每个基元的训练信息将输出到控制台上。在这种情况下，训练进程的数量最好为1。
        self.kwargs = kwargs
        self.log = Log(kwargs['unit_type'], console=self.console)
        self.log.generate()
        '''检查最大高斯混合度约束'''
        if self.kwargs['mix_level'] > self.kwargs['max_mix_level']:
            raise MixtureNumberError(self.kwargs['mix_level'], self.kwargs['max_mix_level'], self.log)
        self.c_covariance = kwargs['c_covariance']

    def split_unit(self):
        """
        基元分割
        :return: 各机器的训练清单
        """
        amodel = AcousticModel(self.log, self.kwargs['unit_type'], processes=self.kwargs['processes'],
                               console=self.console, state_num=self.kwargs['state_num'],
                               mix_level=self.kwargs['mix_level'], delta_1=self.kwargs['delta_1'],
                               delta_2=self.kwargs['delta_2'])
        amodel.load_unit()
        units = amodel.loaded_units
        chunk = len(units) // self.num
        traininfo_path = PARAMETERS_FILE_PATH + '/%s' % self.kwargs['unit_type']
        for job_id in range(self.num - 1):
            trainInfo_file = traininfo_path + '/trainInfo_%d.csv' % job_id
            '''删除现存trainInfo文件'''
            if os.path.exists(trainInfo_file):
                os.remove(trainInfo_file)
            units_slice = set(units).difference(set(units[job_id * chunk:(job_id + 1) * chunk]))
            for unit in units_slice:
                trainInfo = open(trainInfo_file, 'a+')
                trainInfo.writelines('%s\n' % unit)
        '''最后一个机器的训练清单'''
        units_slice = set(units).difference(set(units[(self.num - 1) * chunk:]))
        trainInfo_file = traininfo_path + '/trainInfo_%d.csv' % (self.num - 1)
        '''删除现存trainInfo文件'''
        if os.path.exists(trainInfo_file):
            os.remove(trainInfo_file)
        for unit in units_slice:
            trainInfo = open(traininfo_path + '/trainInfo_%d.csv' % (self.num - 1), 'a+')
            trainInfo.writelines('%s\n' % unit)

    def split_data(self, audiopath, labelpath):
        """
        音频数据、标注地址分割
        :param audiopath: 音频路径
        :param labelpath: 标注路径
        :return:
        """
        generator = AcousticModel.init_audio(audiopath, labelpath)  # 音频和标注的迭代器
        file_count = generator.__next__()  # 文件总数
        chunk = file_count // self.num
        for job_id in range(self.num - 1):
            path_file = PARAMETERS_FILE_PATH + '/%s/pathInfo_%d.csv' % (self.kwargs['unit_type'], job_id)
            '''文件已存在'''
            if os.path.exists(path_file):
                continue
            with open(path_file, 'w') as f:
                count = 0
                while count < chunk:
                    count += 1
                    path = generator.__next__()
                    f.write(path[0])  # 写入音频路径
                    f.write(path[1])  # 写入标注路径
        '''写入最后一个路径列表'''
        path_file = PARAMETERS_FILE_PATH + '/%s/pathInfo_%d.csv' % (self.kwargs['unit_type'], (self.num - 1))
        with open(path_file, 'w') as f:
            for path in generator:
                f.write(path[0])  # 写入音频路径
                f.write(path[1])  # 写入标注路径

    def parallel_data(self, env, mode=1, init=True):
        """
        并行处理数据
        :param env: 用于标识机器的环境变量名
        :param mode: 训练方案
        :param init: 是否初始化
        :return:
        """
        '''获取作业ID'''
        try:
            job_id = os.environ[env]
        except JobIDExistError:
            raise JobIDExistError(self.log)
        amodel = AcousticModel(self.log, self.kwargs['unit_type'], processes=self.kwargs['processes'], job_id=job_id,
                               console=self.console, state_num=self.kwargs['state_num'],
                               mix_level=self.kwargs['mix_level'], delta_1=self.kwargs['delta_1'],
                               delta_2=self.kwargs['delta_2'])
        amodel.load_unit()
        amodel.process_data(mode=mode, load_line=self.kwargs['load_line'], init=init,
                            proportion=self.kwargs['proportion'], step=self.kwargs['step'],
                            differentiation=self.kwargs['differentiation'], coefficient=self.kwargs['coefficient'])

    def parallel_train(self, env, mode=1, init=True, show_q=False, show_a=False):
        """
        并行训练
        :param env: 用于标识机器的环境变量名
        :param mode: 训练方案
        :param init: 是否初始化
        :param show_q: 显示HMM/GMM当前似然度
        :param show_a: 显示HMM重估后状态转移矩阵
        :return:
        """
        '''获取作业ID'''
        try:
            job_id = os.environ[env]
        except JobIDExistError:
            raise JobIDExistError(self.log)
        amodel = AcousticModel(self.log, self.kwargs['unit_type'], processes=self.kwargs['processes'], job_id=job_id,
                               console=self.console, state_num=self.kwargs['state_num'],
                               mix_level=self.kwargs['mix_level'], delta_1=self.kwargs['delta_1'],
                               delta_2=self.kwargs['delta_2'])
        amodel.load_unit()
        amodel.training(mode=mode, init=init, show_q=show_q, show_a=show_a, load_line=self.kwargs['load_line'],
                        c_covariance=self.c_covariance)

    def add_mix_level(self):
        """
        增加高斯混合度
        :return:
        """
        if self.kwargs['mix_level'] < self.kwargs['max_mix_level']:
            self.kwargs['mix_level'] += 1

    def auto(self, init=True, t=1, mode=1, add_mix=False, show_a=False, show_q=False):
        """
        声学模型单机自动训练
        :param init: 是否初始化
        :param t: 训练次数
        :param mode: 训练方案(training scheme):
                    --方案1(mode=1，适合孤立词识别系统或语料库语速平缓的语音识别系统声学模型的训练):
                                    1、初始化靠均分数据(uniformly segmentation)，每个HMM(state)的数据收集起来，最后集中训练GMM&HMM。
                                    (其中HMM是由Embedded Training训练而得，这里的Embedded Training与方案2不同之处在于，它并不使用
                                    前向后向的累计概率来训练GMM，仅用于HMM中状态概率矩阵和初始概率矩阵的训练。)
                                    2、重估(Re-estimation)依靠声学模型，经过维特比对齐(Viterbi Alignment)后，得出最佳序列，将序列
                                    中不同的数据收集到不同的状态文件夹中，等待最后集中训练；重估训练中，GMM会使用SMEM算法调整空间。
                                    3、每一步训练后，高斯混合度将会增加，在下一轮训练时将使用K-Means算法按照新的混合度重新聚类
                    --方案2(mode=2，适合一般语料库下的连续语音识别系统声学模型的训练):
                                    1、所有子字HMM使用全局均值和协方差(flat-start)
                                    2、对每一个训练语料建立句子级的HMM(Embedded HMM)，进行嵌入式训练(Embedded Training)，得到每个句子
                                    的参数累加值，最后用各模型的累加器(Accumulator)对模型的各参数进行训练。
                                    3、嵌入式训练中，运行Baum-Welch算法，进行迭代。
        :param add_mix: 每轮训练后是否增加高斯混合度
        :param show_q: 显示HMM/GMM当前似然度
        :param show_a: 显示HMM重估后状态转移矩阵
        :return:
        """
        current_t = 1
        env_key = 'env_id'
        self.log.note('训练方案：方案%d' % mode, cls='i', show_console=self.console)
        self.split_data(AUDIO_FILE_PATH, LABEL_FILE_PATH)
        if mode == 1:
            self.split_unit()
            while current_t <= t:
                current_t += 1
                self.parallel_data(env_key, init=init)
                self.parallel_train(env_key, mode=mode, init=init, show_q=show_q, show_a=show_a)
                if add_mix:
                    self.add_mix_level()
                init = False
        elif mode == 2:
            while current_t <= t:
                current_t += 1
                self.parallel_data(env_key, mode=mode, init=init)
                self.parallel_train(env_key, mode=mode, init=init, show_q=show_q, show_a=show_a)
                init = False

    def end(self):
        self.log.close()


if __name__ == '__main__':
    task = Task(num=1, console=True, **args)
    task.auto(init=True, t=2, mode=2, show_a=True, show_q=True)
    # task.auto(init=True, t=2, mode=2)
    task.auto(init=False, t=2, mode=1, show_a=True, show_q=True)
    task.end()
