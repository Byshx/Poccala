# -*-coding:utf-8-*-

"""
    汉语转拼音
    
Author: Byshx

Date: 2017.05.05

"""

import sys
import copy

'''汉语——拼音映射表'''
dict_path = sys.path[0] + '/Mandarin.dat'


class PinYin(object):
    def __init__(self, path=dict_path):
        self.__dict = {}
        self.__init_dict(path)
        """"""""""""""""""""""""
        self.__syllable = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'z',
                           'c', 's', 'r', 'y', 'w']  # 声母集合（其中y和w不是声母）
        self.__extend_dict = \
            {'ai': '#_a',
             'ao': '#_a',
             'an': '#_a',
             'ang': '#_a',
             'o': '#_o',
             'ou': '#_o',
             'e': '#_e',
             'ei': '#_e',
             'er': '#_e',
             'en': '#_e',
             '?': '#_v'}  # 扩展的零声母 由于#_I 和 #_u 的特殊性，在for循环内判断

    def __init_dict(self, path, lower=True):
        """
        初始化字典
        :param path: 映射文件路径
        :param lower: 拼音转为小写
        :return: 
        """
        with open(path, 'r') as f:
            s = f.readline()
            while s:
                s = s.strip('\n').split('\t')
                key = s[0].lower()
                pinyin = s[1].split(' ')
                if lower:
                    pinyin = [pinyin[_].lower() for _ in range(len(pinyin))]
                self.__dict[key] = pinyin
                s = f.readline()
        f.close()

    def word2pinyin(self, string, separate=True, check_tone=True, extend=True, show_tone_mark=True):
        """
        汉字转拼音
        :param string: 字符串 
        :param separate: 分隔声韵母
        :param check_tone: 韵母
        :param extend: 扩展零声母
        :param show_tone_mark: 显示音调 
        :return: 
        """
        try:
            str_list = list(string)
            if show_tone_mark:
                return [self.__check_tone(copy.deepcopy(self.__dict[u'%x' % ord(_)]), separate, check_tone, extend, show_tone_mark) for
                        _
                        in str_list]
            else:
                result = [self.__check_tone(copy.deepcopy(self.__dict[u'%x' % ord(_)]), separate, check_tone, extend,
                                            show_tone_mark) for _ in str_list]
                '''去除音标'''
                return list(map(lambda x: list(set([x[_][:-1] for _ in range(len(x))])), result))
        except KeyError:
            return None

    def __check_tone(self, tone, separate, check_tone, extend, show_tone_mark):
        """
        纠正tone
        如 juan ——> jvuan  que ——> qve
        :param tone: 拼音
        :param separate: 分隔声韵母
        :param check_tone: 韵母
        :param extend: 扩展零声母
        :param show_tone_mark: 显示音调 
        :return: 
        """
        mark = ['j', 'q', 'x']  # 替换声母
        if separate:
            for i in range(len(tone)):
                if tone[i][0] in self.__syllable:
                    if len(tone[i]) >= 3 and tone[i][:2] in self.__syllable:
                        tone[i] = tone[i][:2] + ',' + tone[i][2:]  # 插入逗号分隔
                    else:
                        tone[i] = tone[i][0] + ',' + tone[i][1:]  # 插入逗号分隔
        if check_tone:
            for i in range(len(tone)):
                if tone[i][0] in mark:
                    if 'u' in tone[i] and 'iu' not in tone[i]:
                        tone[i] = tone[i].replace('u', 'v')
                if 'ue' in tone[i]:
                    tone[i] = tone[i].replace('ue', 've')

        if extend:
            for i in range(len(tone)):
                if 'y' in tone[i]:
                    tone[i] = tone[i].replace('y', '#_I')
                elif 'w' in tone[i]:
                    tone[i] = tone[i].replace('w', '#_u')
                else:
                    if show_tone_mark:
                        if int(tone[i][-1]) == 5:  # 将零声修正为0
                            tone[i] = tone[i][:-1] + '0'
                        tone_tmp = tone[i]

                    else:
                        tone_tmp = tone[i][:-1]
                    if self.__extend_dict.get(tone_tmp) is not None:
                        if separate:
                            tone[i] = self.__extend_dict.get(tone_tmp) + ',' + tone[i]
                        else:
                            tone[i] = self.__extend_dict.get(tone_tmp) + tone[i]
        else:
            for i in range(len(tone)):
                if 'y' in tone[i] or 'w' in tone[i]:
                    tone[i] = tone[i][1:]  # 除去y和w
        return tone


if __name__ == '__main__':
    p = PinYin(dict_path)
    a = p.word2pinyin('家', separate=True, check_tone=True, extend=True, show_tone_mark=True)
    print(a)
