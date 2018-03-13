# -*-coding:utf-8-*-

"""
异常类
-----
Author: Byshx

Date: 2018.01.22
"""


class MixtureNumberError(Exception):
    def __init__(self, min_mixture, max_mixture, log):
        self.min = min_mixture
        self.max = max_mixture
        self.log = log

    def __str__(self):
        info = '高斯初始混合度大于最大混合度(%d > %d)' % (self.min, self.max)
        self.log.note(info, cls='e')
        return info


class UnitFileExistsError(FileExistsError):
    def __init__(self, unit_type, log):
        self.unit_type = unit_type
        self.log = log

    def __str__(self):
        info = '基元文件%s不存在' % self.unit_type
        self.log.note(info, cls='e')
        return info


class ArgumentNumberError(ValueError):
    def __init__(self, log, *args):
        self.log = log
        self.args = args

    def __str__(self):
        info = '参数数量错误,参数为：%s' % str(self.args)
        self.log.note(info, cls='e')
        return info


class ClassError(Exception):
    def __init__(self, log):
        self.log = log

    def __str__(self):
        info = '未知类别/模式'
        self.log.note(info, cls='e')
        return info


class DataUnLoadError(Exception):
    def __init__(self, log):
        self.log = log

    def __str__(self):
        info = '数据未载入'
        self.log.note(info, cls='e')
        return info


class DataDimensionError(Exception):
    def __init__(self, expect, real, log):
        self.expect = expect
        self.real = real
        self.log = log

    def __str__(self):
        info = '数据维度错误，应为%d 实为%d' % (self.expect, self.real)
        self.log.note(info, cls='e')
        return info


class PassException(Exception):
    """跳出多重循环"""
    pass


class ConfigExitsError(FileExistsError):
    def __init__(self):
        pass

    def __str__(self):
        info = '配置文件不存在'
        return info


class JobIDExistError(KeyError):
    def __init__(self, log):
        self.log = log

    def __str__(self):
        info = '不存在标识该机器的环境变量'
        self.log.note(info, cls='e')
        return info


class PathInfoExistError(FileExistsError):
    def __init__(self, log):
        self.log = log

    def __str__(self):
        info = '数据路径信息文件不存在,先尝试执行init_audio方法'
        self.log.note(info, cls='e')
        return info


class LogDirExistError(FileExistsError):
    def __init__(self):
        pass

    def __str__(self):
        info = '指定日志文件目录不存在'
        return info


class CpuCountError(ValueError):
    def __init__(self, log):
        self.log = log

    def __str__(self):
        info = 'cpu数目应为正整数'
        self.log.note(info, cls='e')
        return info
