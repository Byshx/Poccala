# -*-coding:utf-8-*-
"""
    语音识别——语言模型
    
Author: Byshx

Date: 2017.05.01 假期结束了...

    语言模型是一个单纯的、统一的、抽象的形式系统，语言客观事实经过语言模型的描述，比较适合于电子计算机进行自动处理，
因而语言模型对于自然语言的信息处理具有重大的意义。
                                                                                    ————百度百科
                                                                                    
模型：N-gram模型

应用： 自然语言处理、语音识别

"""
import os
import sys
import re
import time
import pickle
from LanguageModel import StringProgress

path_gram = sys.path[1] + '/LanguageModel/'

path_data = path_gram + '/data/'

path_dict = path_gram


class Ngram(object):
    def __init__(self, n=2):
        self.__gram = {}
        self.__n = n

    def init_gram(self, loadpath=path_gram):
        """读取"""
        print('载入语言模型中...')
        file = open(loadpath + '%sgram_ch.pkl' % self.__n, 'rb')
        self.__gram = pickle.load(file)
        file.close()
        print('Ngram 模型载入完成 √')

    def n_gram(self, word):
        """获取下一个词"""
        word = ''.join(word)
        return self.__search(self.__gram, word, 0)

    def __search(self, gram, w, index):
        if type(gram) is not dict or index == len(w):
            return gram
        return self.__search(gram[w[index]], w, index + 1)

    def generate_gram(self, datapath=path_data, savepath=path_dict):
        """
        生成N-Gram模型
        :param datapath: 语料数据
        :param savepath: 存储路径
        :return: 
        """
        print('开始时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        ch_punctuation = """～`！@#￥%……&×（）——、“”‘’？％：，。.；《》【】"""
        other_punctuation = """"""
        en_punctuation = """~`!@#$%^&*()-=+/*-<>?,._{}[]\\"""
        punctuation = ch_punctuation + other_punctuation + en_punctuation

        for dir in os.walk(datapath):
            for file in dir[2]:
                with open(datapath + file) as f:
                    input_data = f.readline()
                    while input_data:
                        input_data = input_data.strip('\n')
                        '''全/半角转化'''
                        input_data = StringProgress.strQ2B(input_data)
                        input_data = re.sub('[a-zA-Z0-9%s]*' % punctuation, ' ', input_data)
                        input_data = ','.join(input_data.split())
                        if len(input_data) == 0:
                            input_data = f.readline()
                            continue
                        data = input_data.split(',')
                        self.__deal_data(data)
                        input_data = f.readline()
                f.close()
        """持久化"""
        file = open(savepath + '%sgram_ch.pkl' % self.__n, 'wb')
        pickle.dump(self.__gram, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()
        print('结束时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    def __deal_data(self, data):
        """
        处理数据
        :param data: 分割的数据
        :return: 
        """
        for index in range(len(data)):
            if len(data[index]) >= self.__n:
                _index = 0
                index_ = _index + self.__n
                while index_ <= len(data[index]):
                    word = ' '.join(data[index][_index:index_])
                    word = word.split()
                    self.__build_tree(self.__gram, word, 0)
                    _index += 1
                    index_ += 1

    def __build_tree(self, node, word, index):
        """
        构建Ngram树
        :param node: 树节点
        :param word: 切割后的词语
        :return: 
        """
        if node.get(word[index]) is None:
            if index < self.__n - 1:
                node[word[index]] = {}
            else:
                if node.get(word[index]) is None:
                    node[word[index]] = 0
        if index == self.__n - 1:
            node[word[index]] += 1
            return
        return self.__build_tree(node[word[index]], word, index + 1)


if __name__ == '__main__':
    lm = Ngram(n=3)
    lm.generate_gram()
    # lm.init_gram()
    # a = lm.n_gram('尼潦')
    # print(a)
