# -*-coding:utf-8-*-
"""
    语音识别——解码部分
    
Author: Byshx

Date: 2017.05.07

算法： 帧同步解码算法（维特比束算法、树状令牌传递算法）
"""
import os
import sys
import time
import math
import numpy as np
from multiprocessing import cpu_count
from LanguageModel.Ngram import Ngram
from AcousticModel.AcousticModel import AcousticModel
from StatisticalModel.AudioProcessing import AudioProcessing
from StatisticalModel.LHMM import LHMM
from Lexicon.PronunciationLexicon import PronunciationLexicon

"""获取系统核心数"""
cores = cpu_count()
'''语音暂储路径'''
path_record = sys.path[1] + '/AcousticModel/Audio/record/'

###########################################################
unit = None  # 基元，从声学模型中获取
state_num = -1  # HMM状态数,从声学模型中获取
lexicon = None  # 发音字典
mfcc = None  # MFCC数据
t = 0  # 时间点。
beam = 0.85  # 束(beam)大小
tokens = {}  # 令牌字典
decode_tree = {}  # 解码树
"""Ngram模型"""
ngram = []  # Ngram模型
unit_time = None  # 保存基元计算的时间点，避免重复计算
viterbi = LHMM.viterbi  # viterbi算法


###########################################################


def record_audio(record_seconds, savepath=path_record, del_file=False):
    """接受语音"""
    present_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    file_name = present_time + '.wav'  # 文件名
    ap = AudioProcessing()
    ap.record(record_seconds=record_seconds, output_path=savepath + file_name)
    m = ap.MFCC()
    m.init_audio(path=savepath + '20.wav')
    mfcc_data = m.mfcc(nfft=1024, cal_energy=True, d1=True, d2=True)
    vad = ap.VAD()
    vad.init_mfcc(mfcc_data)
    mfcc_vad_data = vad.mfcc()
    if del_file:
        os.remove(savepath + '%s.wav' % file_name)
    return mfcc_vad_data


def generate_first_word(am, candidate=10):
    """根据第一帧生成首词模型"""
    syllable_units = list(lexicon.keys())
    syllable_list = list(map(lambda x: [x, 0], syllable_units))
    score = dict(syllable_list)
    first_mfcc = [mfcc[0:20]]

    for index in range(len(syllable_units)):
        hmm = unit[syllable_units[index]][1]
        '''传入第一帧数据'''
        hmm.change_T([20])
        hmm.init_data(first_mfcc)
        score[syllable_units[index]] = hmm.q_function()
    score_sorted = list(score.items())
    score_sorted.sort(key=lambda x: x[1], reverse=True)
    '''找到最接近的基元'''
    candidate_syllable = score_sorted[:candidate]
    '''建立首词模型'''
    for syllable in candidate_syllable:
        score_ = syllable[1]  # 该基元得分
        syllable_lexicon = lexicon[syllable[0]]
        first_letter_list = list(syllable_lexicon.keys())  # 首词第一个字
        print(syllable, len(first_letter_list))
        for index in range(len(first_letter_list)):
            tokens[first_letter_list[index]] = Token(score_, first_letter_list[index], [], [], am, syllable_lexicon)
            tokens[first_letter_list[index]].viterbi(first_mfcc, 0, unit_time)


def token_passing():
    """令牌传递"""
    global t
    while t < len(mfcc) - 1:
        t += 1
        data = [mfcc[t:t + 1]]
        key_list = list(tokens.keys())
        score_list = []
        for index in range(len(key_list)):
            token = tokens[key_list[index]]
            result = token.viterbi(data, t, unit_time)
            if result is True:
                '''评分累计下降次数达到上限，进行词内转移'''
                passing_in_word(token, data)
                tokens.__delitem__(token.state)
                continue
            score_list.append([key_list[index], tokens[key_list[index]].score])
        score_list.sort(key=lambda x: x[1])  # 从小到大排序
        '''剪枝'''
        pruning(score_list)
        print(t, len(score_list), score_list)


def passing_in_word(token, data):
    """词内转移"""
    '''获取下一组基元'''
    lexicon_ = token.lexicon
    key_list = list(lexicon_[token.state].keys())
    print(token.state, key_list, token.score)
    # key_list.remove('state')  # 去掉state键
    flag = False  # 该节点是否有词
    if 'word' in key_list:
        flag = True
        key_list.remove('word')  # 去掉word键

    for index in range(len(key_list)):
        if tokens.get(key_list[index]) is not None:
            '''如果哈希表中存在该Token，取出'''
            new_token = tokens[key_list[index]]
            score_sum = token.score
            if score_sum > new_token.score:
                new_token.score = score_sum
                new_token.context = token.context
                new_token.stack = token.stack
                new_token.lexicon = lexicon_[token.state]
                new_token.descend_time = 0
        else:
            '''不存在时新建'''
            new_token = Token(token.score, key_list[index], token.context, token.stack, token.am,
                              lexicon_[token.state])
            new_token.viterbi(data, t, unit_time)
            tokens[key_list[index]] = new_token
    return flag


def passing_between_word(token):
    """词间转移"""
    lexicon = token.lexicon
    word_list = lexicon[token.state]['word']
    for index in range(len(word_list)):
        candidate_word_dict = ngram.ngram(word_list[index])
        candidate_word = list(candidate_word_dict.items())
        sum_time = sum(candidate_word_dict.values())
        for index_ in range(len(candidate_word)):
            l_score = math.log10(candidate_word[1] / sum_time)
            new_token = Token()


def pruning(score_list):
    """剪枝"""
    global tokens
    if len(set([score_list[_][1] for _ in range(len(score_list))])) < 8:
        return
    width = len(score_list)
    pruning_num = int(width * (1 - beam))
    for node in score_list[:pruning_num]:
        tokens.__delitem__(node[0])


def build_tree(word):
    """将结果写入解码树"""
    pass


def transfer(candidate=5):
    """词间转移写入栈和上下文"""
    '''将最后的结果加在stack中'''
    score_list = list(tokens.items())
    score_list.sort(key=lambda x: x[1].score, reverse=True)
    score_list = score_list[:candidate]
    result = []
    for index in range(len(score_list)):
        token = tokens[score_list[index][0]]
        print(token.mark, token.state)
        if token.lexicon[token.state].get('word') is not None:
            result.append(token.lexicon[token.state]['word'])
    print(result)


def main(record_seconds, n=2, savepath=path_record):
    """执行整个过程"""
    global state_num, mfcc, unit_time, lexicon, unit, ngram
    '''初始化声学模型'''
    am = AcousticModel(mix_level=4)
    am.initialize_unit(unit_type='XIF_tone')
    am.init_parameter()
    unit = am.unit
    unit_time = dict([[key, -1] for key in list(am.unit.keys())])
    state_num = am.statenum  # 设置HMM状态数
    '''初始化语言模型'''
    for i in range(n):
        ngram_ = Ngram(n=i + 1)
        ngram_.init_gram()
        ngram.append(ngram_)
    '''初始化发音词典'''
    p_lexicon = PronunciationLexicon()
    p_lexicon.init_lexicon()
    lexicon = p_lexicon.lexicon
    '''获取MFCC帧'''
    mfcc = record_audio(record_seconds, savepath)
    '''生成首词模型'''
    generate_first_word(am)
    a = time.time()
    '''执行令牌传递算法'''
    token_passing()
    b = time.time()
    transfer()
    print('耗时:', b - a)


class Token(object):
    """令牌"""

    def __init__(self, score, state, context, stack, am, p_lexicon):
        self.score = score  # 当前得分
        self.state = state  # 词内状态(词内的字所对应基元)
        self.__label = state.split(',')  # 切割后的词基元
        self.context = context  # 上下文(词)
        self.stack = stack  # 解码栈
        self.lexicon = p_lexicon  # 发音词典
        self.am = am  # 声学模型实例
        self.descend_time = 0  # 下降次数
        self.mark = -1
        self.mark_max = -1
        self.__embedded_list = am.embedded(self.__label, 0, 29)  # 计算除观测矩阵外的复合模型
        self.__embedded_list.insert(3, None)  # 为观测概率B占位(place holder)
        self.__p_list = None  # 存储viterbi切分的概率得分

    def __cal_data(self, data, data_time, unit_train_time):
        """更新观测矩阵，生成复合观测矩阵"""
        for index in range(len(self.__label)):
            if unit_train_time[self.__label[index]] == data_time:
                continue
            else:
                unit_train_time[self.__label[index]] = data_time
            hmm = self.am.unit[self.__label[index]][1]
            hmm.cal_observation_pro(data, [1], normalize=False, standard=False)
        self.__embedded_list[3] = self.am.embedded(self.__label, 0, 2).__getitem__(0)  # 仅计算观测矩阵

    def viterbi(self, data, data_time, unit_train_time=unit_time):
        """
        viterbi束算法
        :param data: 传入数据(一帧)
        :param data_time: 数据帧所在时间点
        :param unit_train_time: 各基元计算时间表
        :return: 
        """

        def info(p):
            """寻找最大分值点"""
            point_ = p.max()
            mark_ = np.where(p == point_)
            mark_state = mark_[0][0]
            return point_, mark_state

        self.__cal_data(data, data_time, unit_train_time)
        complex_states, complex_observation, complex_A, complex_B, complex_π = self.__embedded_list
        '''消除警告'''
        np.seterr(divide='ignore')
        if self.__p_list is None:
            self.__p_list = np.log(complex_π) + complex_B[:, 0]
            point, mark = info(self.__p_list)
            self.score += point
            self.mark = mark
            if mark == len(complex_states) - 1:
                return True
        else:
            p_ = np.zeros_like(self.__p_list)
            for j in range(len(complex_states)):
                tmp = self.__p_list + np.log(complex_A[:, j])
                max_p = tmp.max()
                p_[j] = max_p
            self.__p_list = p_ + complex_B[:, 0]
            point, mark = info(self.__p_list)
            self.score += point
            self.mark = mark
            if len(complex_states) - mark <= 1:
                return True


if __name__ == '__main__':
    main(1)
