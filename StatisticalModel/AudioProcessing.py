# -*-coding:utf-8-*-

"""
    音频处理

包括录音、音频播放和音频特征提取(梅尔倒谱系数，MFFC)

Author:Byshx

Date:2017.04.04

"""
import os
import sys
import wave
import math
import pylab
import pyaudio
import numpy as np
from contextlib import contextmanager


@contextmanager
def ignore_stderr():
    """捕捉pyaudio错误信息"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


class AudioProcessing(object):
    def __init__(self):
        """"""
        with ignore_stderr():
            self.__pyaudio = pyaudio.PyAudio()

    """音频播放"""

    def play(self, input_path, chunk=1024):
        """"""
        '''读取音频文件'''
        wav = wave.open(input_path, 'rb')
        stream = self.__pyaudio.open(format=self.__pyaudio.get_format_from_width(wav.getsampwidth()),
                                     channels=wav.getnchannels(), rate=wav.getframerate(), output=True)
        data = wav.readframes(chunk)
        while data != '':
            stream.write(data)
            data = wav.readframes(chunk)

        stream.stop_stream()
        stream.close()
        wav.close()
        self.__pyaudio.terminate()

    """音频录制"""

    def record(self,
               record_seconds,
               chunk=1024,
               format=pyaudio.paInt16,
               channels=2,
               rate=16000,
               output_path=None):
        stream = self.__pyaudio.open(format=format, channels=channels,
                                     rate=rate, input=True, frames_per_buffer=chunk)
        print('Recording ...')
        frames = []

        for i in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
        '''加入剩余的帧'''
        frames.append(stream.read(rate * record_seconds - int(rate / chunk * record_seconds) * chunk))

        print('Done.')
        stream.stop_stream()
        stream.close()
        self.__pyaudio.terminate()

        '''保存文件'''
        print('Saving ...')

        wav = wave.open(output_path, 'wb')
        wav.setnchannels(channels)
        wav.setsampwidth(self.__pyaudio.get_sample_size(format))
        wav.setframerate(rate)
        wav.writeframes(b''.join(frames))
        wav.close()

        print('Done.')

    class MFCC(object):

        """
                梅尔倒谱系数（Mel-scale Frequency Cepstral Coefficients，简称MFCC）

            简介：
                MFCC：Mel频率倒谱系数的缩写。Mel频率是基于人耳听觉特性提出来的，它与Hz频率成非线性对应关系。Mel频率倒谱系数(MFCC)
            则是利用它们之间的这种关系，计算得到的Hz频谱特征，MFCC已经广泛地应用在语音识别领域。
                                                                                                    ————《百度百科》
            相关算法：
            1、快速傅里叶变换（Fast Fourier Transform,简称FFT）
            2、离散余弦变换(Discrete Cosine Transformation,简称DCT)

            参考文献：
            1、https://my.oschina.net/jamesju/blog/193343#comment-list
            2、http://blog.csdn.net/xiaoding133/article/details/8106672
            3、http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn1

        """

        def __init__(self, vec_num=13):
            self.__pyaudio = pyaudio.PyAudio()
            '''已读取的音频文件对象'''
            self.__wav = None
            '''信号样本点'''
            self.__wdata = None
            '''标准MFCC阶数'''
            self.__vec_num = vec_num

        @property
        def data(self):
            return self.__wdata

        @property
        def wav(self):
            return self.__wav

        @property
        def params(self):
            """
            返回wave文件参数

                return _wave_params(self._nchannels, self._sampwidth, self._framerate,
            self._nframes, self._comptype, self._compname)
            :return: 参数依次是：声道、样本位数(byte)、采样率、帧数、压缩类型和压缩名称
            """
            return self.__wav.getparams()

        def init_audio(self, wav=None, path=None, show_pic=False):
            """
            加载音频文件
            :param wav: 已读取的音频文件wave对象
            :param path: 音频文件地址
            :param show_pic: 显示波形图
            :return: None
            """
            if wav is None:
                if path is None:
                    print('Error: 未给出音频文件路径')
                self.__wav = wave.open(path, 'rb')
            else:
                self.__wav = wav
            nframes = self.__wav.getnframes()
            '''byte表示的十六进制信号流'''
            str_data = self.__wav.readframes(nframes)
            '''转换为十进制矩阵'''
            self.__wdata = np.fromstring(str_data, dtype=np.short)
            '''双声道拆分矩阵'''
            if self.__wav.getnchannels() == 2:
                '''将矩阵设置为2列，行数自动匹配'''
                self.__wdata.shape = -1, 2
                self.__wdata = self.__wdata.T
                '''合并矩阵，两值取频率高的，放在wdata[0]中'''
                for _ in range(len(self.__wdata[0])):
                    if self.__wdata[0][_] < self.__wdata[1][_]:
                        self.__wdata[0][_] = self.__wdata[1][_]
                self.__wdata = self.__wdata[0]
            self.__wdata = np.delete(self.__wdata, np.where(self.__wdata == 0))

            if show_pic:
                x = [_ for _ in range(len(self.__wdata))]
                pylab.plot(x, self.__wdata, 'b')
                pylab.show()

        @staticmethod
        def pre_emphasis(signal, alpha=0.98):
            """
            对信号进行预加重

                预加重的目的是提升高频部分，使信号的频谱变得平坦，保持在低频到高频的整个频带中，
            能用同样的信噪比求频谱。同时，也是为了消除发生过程中声带和嘴唇的效应，来补偿语音
            信号受到发音系统所抑制的高频部分，也为了突出高频的共振峰。
            :param signal:信号采样点
            :param alpha:预加重系数 取值 0.9 ~ 1.0 ，一般为0.98
            :return: None
            """
            y = signal[1:] - alpha * signal[:-1]
            '''最后少一位，用0填充'''
            y = np.append(y, 0.)
            return y

        @staticmethod
        def frame_blocking(signal, framerate, sampletime=0.025, overlap=0.5):
            """
            分帧

                将N个采样点集合成一个观测单位，称为帧。通常情况下N的值为256或512，涵盖的时间约
            为20~30ms左右
            :param framerate:采样率
            :param signal:采样信号
            :param sampletime:取帧所占时间
            :param overlap:重叠区所占采样集合的比例
            :return: 分帧后的样本集合
            """
            samplenum = len(signal)  # 采集的样本数
            framesize = int(framerate * sampletime)  # 一帧的大小
            step = int(framesize * overlap)  # 帧移动步长
            framenum = 1 + math.ceil((samplenum - framesize) / step)  # 分帧的总帧数
            padnum = (framenum - 1) * step + framesize
            zeros = np.zeros((int(padnum - samplenum),))
            padsignal = np.concatenate((signal, zeros))

            indices = np.tile(np.arange(0, framesize), (framenum, 1)) + np.tile(np.arange(0, framenum * step, step),
                                                                                (framesize, 1)).T
            indices = np.array(indices, dtype=np.int32)
            frames = padsignal[indices]
            return frames

        @staticmethod
        def hamming_window(frames, alpha=0.46):
            """
            汉明加窗

                将每一帧乘以汉明窗，以增加帧左端和右端的连续性。

            假设分帧后的信号为S(n),n=0,1,...N-1. N为帧数，乘上汉明窗后：
                    S'(n) = S(n) * W(n)
            其中 W(n,alpha) = (1 - alpha) - alpha * cos[2πn/(N-1)], 0<=n<=N-1

            :param frames:分帧数据
            :param alpha:不同的a值会产生不同的汉明窗，一般情况下a取0.46
            :return: 加窗后的样本集合
            """
            frames.dtype = np.float64
            length = len(frames)
            for i in range(length):
                frames[i] *= (1 - alpha) - alpha * math.cos(2 * math.pi * i / (length - 1))
            return frames

        @staticmethod
        def fft(frames, nfft=512):
            """
            快速傅里叶变换

                由于信号在时域上的变换通常很难看出信号的特性，所以通常将它转换为频域上的
            能量分布来观察，不同的能量分布，就能代表不同语音的特性。所以在乘上汉明窗后，
            每帧还必须再经过快速傅里叶变换以得到在频谱上的能量分布。对分帧加窗后的各帧信
            号进行快速傅里叶变换得到各帧的频谱。并对语音信号的频谱取模平方得到语音信号的
            功率谱。

            :param frames:加窗后的分帧样本
            :param nfft:fft长度，不足补零
            :return:长度为nfft//2+1长度的帧矩阵，结果取绝对值
            """
            complex_spec = np.fft.rfft(frames, nfft)
            return np.absolute(complex_spec)

        @staticmethod
        def power(fpiece):
            """
            计算每帧的功率谱
            :param fpiece:经过FFT的帧集合
            :return:功率谱集合
            """
            p = []
            for i in range(len(fpiece)):
                p.append(abs(fpiece[i]) ** 2 / len(fpiece[i]))
            return p

        @staticmethod
        def mel_filter_bank(complex_spec, samplerate, nfft=512, low_hz=0., high_hz=None,
                            filterbanks=26):
            """
            梅尔滤波器组/梅尔三角带通滤波器(Mel triangular band-pass filter)

                将能量谱通过一组Mel尺度(是Mel标度，要事先进行Mel频率转换)的三角形滤波器组，定义一个有M个滤波器的滤波器组
            （滤波器的个数和临界带的个数相近），采用的滤波器为三角滤波器，中心频率为f(m),
            m=1,2,...,M。M通常取22-26。各f(m)之间的间隔随着m值的减小而缩小，随着m值的
            增大而增宽。

                对频谱进行平滑化，并消除谐波的作用，突显原先语音的共振峰。（因此一段语音
            的音调或音高，是不会呈现在 MFCC 参数内，换句话说，以 MFCC 为特征的语音辨识系
            统，并不会受到输入语音的音调不同而有所影响）此外，还可以降低运算量。

            Mel标度频率域与频率的关系：

                    Mel(f) = 2595 * log(1 + f/700)

            :param complex_spec:经过实序列的快速傅里叶变换后的频域分帧集合
            :param samplerate:采样率
            :param nfft:FFT长度，默认为512
            :param low_hz:音频最低频率，默认为0
            :param high_hz:音频最高频率，默认为采样率的一半
            :param filterbanks:滤波器数目

            :return:
            """

            def cal_mel_frequency(frequency):
                return 2595 * math.log(1 + frequency / 700, math.e)

            def cal_frequency(mel_frequency):
                return 700 * (np.exp(mel_frequency / 2595) - 1)

            def cal_bin_index(hz_):
                return np.floor((nfft + 1) / samplerate * hz_)

            def cal_frequency_response(bin_):
                """
                计算三角滤波器的频率响应
                :param bin_:FFT组
                """
                response = np.zeros((filterbanks, nfft // 2 + 1))
                for i in range(filterbanks):
                    for j in range(int(bin_[i]), int(bin_[i + 1])):
                        response[i][j] = (j - int(bin_[i])) / (bin_[i + 1] - bin_[i])
                    for j in range(int(bin_[i + 1]), int(bin_[i + 2])):
                        response[i][j] = (j - int(bin_[i + 1])) / (bin_[i + 2] - bin_[i + 1])
                return response

            high_hz = high_hz or samplerate / 2
            '''梅尔标度最小值'''
            mel_min = cal_mel_frequency(low_hz)
            '''梅尔标度最大值'''
            mel_max = cal_mel_frequency(high_hz)
            mel = np.linspace(mel_min, mel_max, filterbanks + 2)
            '''Mel标度频域转换为频率(frequency)'''
            hz = cal_frequency(mel)
            '''计算帧能量'''
            energy = np.sum(complex_spec, 1)
            '''计算各中心频率在FFT的bin组下标'''
            bin_index = cal_bin_index(hz)
            '''三角滤波器频率响应'''
            fbank = cal_frequency_response(bin_index)
            fbank = np.dot(complex_spec, fbank.T)
            return fbank, energy

        @staticmethod
        def dct(s, rank=13):
            """
            离散余弦变换(Discrete Cosine Transform,DCT)
            :param s: 滤波器输出的能量
            :param rank: MFCC的阶数，通常取12~16
            :return:
            """
            '''滤波器的对数能量'''
            log_energy = np.log(s)
            '''三角滤波器个数'''
            filters_num = len(s[0])
            '''帧数'''
            framenum = len(s)
            '''前系数'''
            coefficient = 2 / filters_num ** 0.5
            dct_frame = np.zeros((framenum, rank))
            '''MFCC系数集合'''
            for i in range(framenum):
                for j in range(rank):
                    sum_ = 0.
                    for k in range(filters_num):
                        sum_ += coefficient * log_energy[i][k] * math.cos(math.pi * (2 * k - 1) * j / (2 * filters_num))
                    dct_frame[i][j] = sum_
            return dct_frame

        @staticmethod
        def lifter(cepstra, lifter=22):
            """
            由于系数低位的值过小，需要向前“抬升”。源于线性预测分析。

            原文：
                he principal advantage of cepstral coefficients is that they are generally decorrelated and this allows
            diagonal covariances to be used in the HMMs. However, one minor problem with them is that the higher order
            cepstra are numerically quite small and this results in a very wide range of variances when going from the
            low to high cepstral coefficients . HTK does not have a problem with this but for pragmatic reasons such as
            displaying model parameters, flooring variances, etc, it is convenient to re-scale the cepstral coefficients
            to have similar magnitudes. This is done by setting the configuration parameter CEPLIFTER  to some value L
            to lifter the cepstra according to the following formula.

            来源:
                http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node53.html#eceplifter

            :param cepstra:梅尔倒谱系数
            :param lifter: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
            """
            if lifter > 0:
                nframes, ncoeff = np.shape(cepstra)
                n = np.arange(ncoeff)
                lift = 1 + (lifter / 2.) * np.sin(np.pi * n / lifter)
                return lift * cepstra
            else:
                return cepstra

        @staticmethod
        def cal_delta(feat, n=2):
            """
            计算差分系数
            :param feat: 标准梅尔倒谱系数或一阶差分系数
            :param n: 间隔帧数，一般为1或2
            :returns: 差分系数矩阵
            """
            framenum = len(feat)
            denominator = 2 * sum([i ** 2 for i in range(1, n + 1)])
            delta_feat = np.empty_like(feat)
            padded = np.pad(feat, ((n, n), (0, 0)), mode='edge')
            for t in range(framenum):
                delta_feat[t] = np.dot(np.arange(-n, n + 1), padded[t: t + 2 * n + 1]) / denominator
            return delta_feat

        def mfcc(self, sampletime=0.025, overlap=0.5, nfft=512, cal_energy=True, d1=False, d2=False):
            """
            用于自动执行获取mfcc的方法
            :return: [N维MFCC参数（N/3 MFCC系数+ N/3 一阶差分参数+ N/3 二阶差分参数）+帧能量（此项可根据需求替换）] * 数据帧数
                    即： matrix.shape = (帧数，mfcc维度)
            """
            '''获取音频参数'''
            params = self.params
            '''预加重'''
            pe = self.pre_emphasis(self.data)
            '''分帧'''
            fb = self.frame_blocking(pe, params[2], sampletime=sampletime, overlap=overlap)
            '''Hamming加窗'''
            win = self.hamming_window(fb)
            '''快速傅里叶变换'''
            fft = self.fft(win, nfft)
            '''三角带通滤波器'''
            fbank, energy = self.mel_filter_bank(fft, params[2], nfft=nfft)
            '''离散余弦变换获得标准MFCC参数'''
            standard_coefficient = self.dct(fbank, rank=self.__vec_num)
            '''将第一位替换为帧能量'''
            if cal_energy:
                standard_coefficient[:, 0] = np.log(energy)
            '''计算一阶差分系数'''
            if d1 is True:
                delta = self.cal_delta(standard_coefficient)
                coefficient = np.concatenate((standard_coefficient, delta), 1)
                '''计算二阶差分系数'''
                if d2 is True:
                    delta_ = self.cal_delta(delta)
                    coefficient = np.concatenate((coefficient, delta_), 1)
                return coefficient
            return standard_coefficient

    class VAD(object):
        def __init__(self, simple_size=16):
            """
            初始化梅尔倒谱系数特征向量
            :param simple_size: 取样长度，默认前16帧
            """
            self.__mfcc = None
            self.__simple_size = simple_size

        def init_mfcc(self, mfcc):
            self.__mfcc = mfcc

        def mel_distance(self, alpha=0.5):
            """
            计算梅尔倒谱距离
            :return: 每个帧与噪声的梅尔倒谱距离
            """
            simple = self.__mfcc[:self.__simple_size]
            '''噪声特征向量初值，取所有噪声特征向量的平均值'''
            noise = 1. / self.__simple_size * simple.sum(axis=0)
            '''迭代更新噪声近似值'''
            for i in range(self.__simple_size):
                noise = alpha * noise + (1 - alpha) * self.__mfcc[i]
            '''计算所有帧与噪声梅尔特征向量距离'''
            mel_distance = []
            for i in range(len(self.__mfcc)):
                distance = noise - self.__mfcc[i]
                mel_distance.append(np.dot(distance, distance) ** 0.5)
            return np.array(mel_distance)

        def osf(self, mel_distance, beta=0.93):
            """
            顺序统计滤波(order statistics filter,OSF)
            
            由于噪声的影响，倒谱距离的波动性较大从而影响检测的准确性，因此计算当前帧与背景噪声的倒谱距离时，使用一组
            顺序统计滤波器对倒谱距离进行平滑处理
            
            :param mel_distance: 每帧与噪声的梅尔倒谱向量距离
            :param beta: 采样分位数
            :return: 
            """

            def sort(array, h_):
                """
                升序排序
                """
                array.sort()
                return array[h_], array[h_ + 1]

            '''平滑处理后的梅尔倒谱距离'''
            mel_distance_ = np.copy(mel_distance)
            '''采样位置h = beta * (2 * self.__simple_size + 1)'''
            h = int(beta * (2 * self.__simple_size + 1))
            '''滤波器采样长度(滤波器阶数) = 2 * self.__simple_size + 1'''
            for i in range(self.__simple_size, len(self.__mfcc) - self.__simple_size):
                d, d_ = sort(np.copy(mel_distance[i - self.__simple_size:i + self.__simple_size]), h)
                mel_distance_[i] = (1 - beta) * d + beta * d_
            return mel_distance_

        def detect(self, mel_distance, show_pic=False):
            """
            语音/非语音分类判别
            :param mel_distance :平滑后的梅尔倒谱距离
            :param show_pic : 展示曲线图
            """
            '''计算中值'''
            simple = np.copy(mel_distance[:self.__simple_size])
            simple.sort()
            d_mid = mel_distance[int(self.__simple_size / 2)]

            '''最大、最小值'''
            max_distance = mel_distance.max()
            min_distance = mel_distance.min()

            '''阈值'''
            beta = d_mid * (max_distance - min_distance) / max_distance

            check = mel_distance - np.ones_like(mel_distance) * beta
            if show_pic:
                x1 = [_ for _ in range(len(self.__mfcc))]
                y1 = mel_distance.tolist()
                x2 = list(np.where(check > 0.)[0])
                y2 = mel_distance[np.where(check > 0.)]
                pylab.plot(x1, y1, 'b')
                pylab.plot(x2, y2, 'r')
                pylab.show()
            return self.__mfcc[np.where(check > 0.)]

        def mfcc(self, show_pic=False):
            """获取端点检测后的MFCC帧"""
            mel_distance = self.mel_distance()
            mel_distance_osf = self.osf(mel_distance)
            mfcc = self.detect(mel_distance_osf, show_pic=show_pic)
            return mfcc
