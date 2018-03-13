# -*-coding:utf-8-*-
"""
读取配置文件
----------
Author: Byshx

Date: 2018.01.23
"""

import os
import configparser
from Exceptions import *

CONFIG_FILE_PATH = os.path.abspath('./config.ini')

""""""""""""""""""""
# 默认环境变量(当config.ini中路径缺省时，使用以下路径)
env = {'unit_file_path': './AcousticModel/Unit',
       'parameters_file_path': './AcousticModel/Unit/Parameters',
       'log_file_path': './AcousticModel/Unit/Parameters',  # 主日志文件位置
       'audio_file_path': './data/data_80/record',
       'label_file_path': './data/data_80/label'
       }

""""""""""""""""""""
# 配置项
args = {
    'task_num': 1,  # 分配任务数
    'unit_type': 'XIF_tone',  # 基元类型
    'c_covariance': 1e-3,  # 协方差修正值
    'processes': None,  # 多进程训练的进程数，要求为正整数,None为默认cpu核心数(若开启超线程，以超线程后核心数为准)
    'load_line': 2,  # 标注文件中标注行所在行数(从0开始)
    'state_num': 5,  # HMM状态数
    'mix_level': 8,  # GMM初始混合度
    'max_mix_level': 13,  # GMM最大混合度
    'dct_num': 13,  # 梅尔倒谱系数维度
    'delta_1': True,  # 计算一阶差分系数
    'delta_2': True  # 计算二阶差分系数
}
""""""""""""""""""""


def initconfig():
    """
    加载配置项到环境变量
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        raise ConfigExitsError
    cf = configparser.ConfigParser()
    cf.read(CONFIG_FILE_PATH)
    sections = cf.sections()
    for section in sections:
        items = cf.items(section)
        for item in items:
            if item[1] is '' and item[0] in env.keys():
                os.environ[item[0]] = env[item[0]]
            else:
                os.environ[item[0]] = item[1]


"""加载环境配置"""
initconfig()
