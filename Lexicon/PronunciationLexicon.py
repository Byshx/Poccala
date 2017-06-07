# -*-coding:utf-8-*-
"""
语音识别——发音词典(Lexicon)

用于将音频标记(Label)转换为对应基元(音素、音节或声韵母)

Author: Byshx

Date: 2017.04.17

"""

import os
import sys
import pickle
from Lexicon.PinYin import PinYin

'''词典实例位置'''
lexicon_path = sys.path[1] + '/Lexicon/lexicon_ch_small.pkl'
'''词典数据'''
lexicon_data_path = sys.path[1] + '/Lexicon/LexiconData/'


class PronunciationLexicon(object):
    def __init__(self):
        self.__lexicon = {}
        self.__lexicon_size = 0

    def init_lexicon(self, loadpath=lexicon_path):
        """
        从序列化文件中初始化发音词典
        :param loadpath: 字典路径
        :return:
        """
        print('载入发音词典中...')
        file = open(loadpath, 'rb')
        self.__lexicon = pickle.load(file)
        file.close()
        print('发音词典载入完成 √')

    @property
    def lexicon(self):
        return self.__lexicon

    def generate_lexicon(self, path=lexicon_data_path, savepath=lexicon_path):
        """
        通过发音词典映射文件生成词典
        :param path: 语料文件路径
        :param savepath: 词典pickle保存路径
        :return: 
        """
        pinyin = PinYin()
        for dir in os.walk(path):
            for file in dir[2]:
                with open(path + file, 'r') as f:
                    s = f.readline().strip('\n')
                    while s:
                        self.__lexicon_size += 1
                        p = pinyin.word2pinyin(s, separate=True, check_tone=True, extend=True, show_tone_mark=True)
                        '''处理首个字'''
                        for index in range(len(p[0])):
                            unit_pinyin = p[0][index]  # 单字拼音
                            unit_pinyin_ = unit_pinyin.split(',')  # 分割声韵母
                            if self.__lexicon.get(unit_pinyin_[0]) is None:
                                self.__lexicon[unit_pinyin_[0]] = {
                                    unit_pinyin: {}}  # {unit_pinyin: {'state': [None, False]}}
                            node_dict = self.__lexicon[unit_pinyin_[0]]
                            if node_dict.get(unit_pinyin) is None:
                                node_dict[unit_pinyin] = {}  # {'state': [None, False]}
                            self.__create_tree(node_dict[unit_pinyin], p[1:], 0, s)
                        s = f.readline().strip('\n')
        print('存储词数：%d' % self.__lexicon_size)
        file = open(savepath, 'wb')
        picker = pickle.Pickler(file=file)
        picker.dump(self.__lexicon)
        file.close()
        print('词典已保存')

    def __create_tree(self, node, p, row, word):
        """
        构建发音树
        :param p: 拼音 
        :return: 
        """
        if row == len(p):
            if node.get('word') is None:
                node['word'] = []
            if word not in node['word']:
                node['word'].append(word)
            return
        for i in range(len(p[row])):
            if node.get(p[row][i]) is None:
                node[p[row][i]] = {}  # {'state': [None, False]}  # state [上下文context,active是否活跃]
            self.__create_tree(node[p[row][i]], p, row + 1, word)


if __name__ == '__main__':
    l = PronunciationLexicon()
    l.generate_lexicon()
    # l.init_lexicon()
    # a = l.lexicon
    # print(a['g'])
    # print(a['g']['g,ou3']['l,i4']['g,uo2'])
