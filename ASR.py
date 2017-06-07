"""
    基于GMM-HMM和梅尔倒谱系数提取技术的自动语音识别系统(Automatic Speech Recognition,ASR)

Author:Byshx

Date:2017.04.08

"""

import os
import time
import numpy as np
from StatisticalModel.AudioProcessing import AudioProcessing
from StatisticalModel.Clustering import Clustering
import scipy

print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))

path1 = '/home/luochong/PycharmProjects/ASR/AcousticModel/Audio/wavedata/male/'
path2 = '/home/luochong/PycharmProjects/ASR/AcousticModel/Audio/wavedata/female'

ap = AudioProcessing.MFCC()

male = []
"""读取音频数据"""
for dir in os.walk(path1):
    for file in dir[2]:
        name = file.split('.')[0]
        sign = name[:-1]
        ap.init_audio(path=os.path.join(dir[0], file))
        mfcc = ap.mfcc(d1=True, d2=True)
        if name == 'Qa':
            for index in range(len(mfcc)):
                male.append(mfcc[index])
male = np.array(male)

ci = Clustering.ClusterInitialization(male, 36, 39)
u, o, alpha = ci.kmeans(algorithm=1)
# ci.som((0.6, 20, 0.6, 20), [-1, 1], T=500, T_=500, method=Distance.euclidean_metric)
print(u)
print(o)
print(alpha)

gmm = Clustering.GMM(male, 39, 3, alpha, u, o)
gmm.auto_update()
print(gmm.theta())
# for i in range(4):
#     print('混合度：%d' % gmm.mixture)
#     gmm.gaussian_division(2 ** i)
#     gmm.auto_update()
#     print(gmm.theta())

print('############################################################')
print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
