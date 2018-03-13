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
        :param num: 并行任务数
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
        self.console = console
        self.kwargs = kwargs
        self.log = Log(kwargs['unit_type'])
        self.log.generate()
        '''检查最大高斯混合度约束'''
        if self.kwargs['mix_level'] > self.kwargs['max_mix_level']:
            raise MixtureNumberError(self.kwargs['mix_level'], self.kwargs['max_mix_level'], self.log)

    def split_unit(self):
        """
        基元分割
        :return: 各机器的训练清单
        """
        amodel = AcousticModel(self.log, self.kwargs['unit_type'], console=self.console,
                               state_num=self.kwargs['state_num'],
                               mix_level=self.kwargs['mix_level'],
                               delta_1=self.kwargs['delta_1'], delta_2=self.kwargs['delta_2'])
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

    def parallel_data(self, env, init=True):
        """
        并行处理数据
        :param env: 用于标识机器的环境变量名
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
        amodel.split_data(load_line=self.kwargs['load_line'], init=init)

    def parallel_train(self, env, init=True, show_q=False, show_a=False, c_covariance=1e-3):
        """
        并行训练
        :param env: 用于标识机器的环境变量名
        :param init: 是否初始化
        :param show_q: 显示HMM/GMM当前似然度
        :param show_a: 显示HMM重估后状态转移矩阵
        :param c_covariance: 修正数值，纠正GMM中的Singular Matrix
        :return:
        """
        '''获取作业ID'''
        try:
            job_id = os.environ[env]
        except JobIDExistError:
            raise JobIDExistError(self.log)
        '''从配置项中获取协方差修正数值，获取失败则在参数中获取，默认为1e-3'''
        try:
            c_covariance_ = os.environ['c_covariance']
        except KeyError:
            c_covariance_ = c_covariance
        amodel = AcousticModel(self.log, self.kwargs['unit_type'], processes=self.kwargs['processes'], job_id=job_id,
                               console=self.console, state_num=self.kwargs['state_num'],
                               mix_level=self.kwargs['mix_level'], delta_1=self.kwargs['delta_1'],
                               delta_2=self.kwargs['delta_2'])
        amodel.load_unit()
        amodel.training(init=init, show_q=show_q, show_a=show_a, c_covariance=c_covariance_)

    def add_mix_level(self):
        """
        增加高斯混合度
        :return:
        """
        if self.kwargs['mix_level'] < self.kwargs['max_mix_level']:
            self.kwargs['mix_level'] += 1

    def auto(self, init=True, t=1):
        """
        单机自动训练
        :param init: 是否初始化
        :param t: 训练次数
        :return:
        """
        current_t = 1
        env_key = 'env_id'
        self.split_unit()
        self.split_data(AUDIO_FILE_PATH, LABEL_FILE_PATH)
        while current_t <= t:
            current_t += 1
            self.parallel_data(env_key, init=init)
            self.parallel_train(env_key, init=init, show_q=True, show_a=True)
            self.add_mix_level()
            init = False

    def end(self):
        self.log.close()


if __name__ == '__main__':
    task = Task(1, **args)
    task.auto(init=False, t=1)
    task.end()
