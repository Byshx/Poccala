# -*-coding:utf-8-*-
"""
日志信息重定向
------------
Author: Byshx

Date: 2018.01.21

用于控制训练信息的输出

"""
import os
import sys
import time
from Exceptions import *

PARAMETERS_FILE_PATH = os.path.abspath(os.environ['parameters_file_path'])  # 获取训练参数保存位置，用以保存各参数训练日志
LOG_FILE_PATH = os.path.abspath(os.environ['log_file_path'])  # 获取主日志文件位置


class Log(object):
    def __init__(self, unit_type, unit=None, job_id=0, console=True):
        """
        :param unit_type: 基元类型
        :param unit: 基元，若基元为None，则将主日志文件作为输出；若基元不为None，则将基元训练日志文件作为输出
        :param job_id: 作业id，用于标注并行训练的机器
        :param console: 输出到控制台
        """
        if unit:
            path = PARAMETERS_FILE_PATH + '/%s/%s' % (unit_type, unit)
            self.filename = path + '/log.csv'
        else:
            path = LOG_FILE_PATH
            if not os.path.exists(path):
                raise LogDirExistError
            path = path + '/%s' % unit_type
            self.filename = path + '/log_%d.csv' % job_id
        self.unit_type = unit_type
        self.log = None
        self.console = console

    def generate(self):
        """
        创建日志文件
        :return:
        """
        '''清除先前日志文件'''
        self.clear()
        '''新建日志文件'''
        self.log = open(self.filename, 'w')

    def append(self):
        """
        获取先前日志，以将新信息添加到已有日志末尾
        :return:
        """
        self.log = open(self.filename, 'a')

    def clear(self):
        """清除先前日志"""
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def note(self, content, cls):
        """
        记录日志
        :param content: 训练信息内容
        :param cls: 信息类别
        :return:
        """
        localtime = time.localtime(int(time.time()))
        format_time = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
        self.output_log(format_time, content, cls)
        if self.console:
            Log.print_console(format_time, content, cls)

    def output_log(self, format_time, content, cls='i'):
        """
        将训练信息输出到控制台
        :param format_time: 当前时间
        :param content: 训练信息内容
        :param cls: 信息类别（3种）：
                        'i' --- [INFO]正常信息
                        'w' --- [WARN]警告信息
                        'e' --- [ERROR]错误信息
        """
        if cls is 'i':
            output = '%s [INFO] %s\n' % (format_time, content)
            self.log.write(output)
        elif cls is 'w':
            output = '%s [WARN] %s\n' % (format_time, content)
            self.log.write(output)
        elif cls is 'e':
            output = '%s [ERROR] %s\n' % (format_time, content)
            self.log.write(output)
        else:
            raise ClassError('错误类别')

    @staticmethod
    def print_console(format_time, content, cls='i'):
        """
        将训练信息输出到控制台
        :param format_time: 当前时间
        :param content: 训练信息内容
        :param cls: 信息类别
        :return:
        """
        if cls is 'i':
            info = '%s [INFO] %s\n' % (format_time, content)
            sys.stdout.write(info)
        elif cls is 'w':
            info = '%s [WARN] %s\n' % (format_time, content)
            sys.stderr.write(info)
        elif cls is 'e':
            info = '%s [ERROR] %s\n' % (format_time, content)
            sys.stderr.write(info)
        else:
            raise ClassError('错误类别')

    def close(self):
        """
        关闭日志文件输出流
        :return:
        """
        self.log.close()
