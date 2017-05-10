# -*- coding: utf-8 -*-

import cPickle
import numpy as np
import warnings
import tensorflow as tf
from transwarpnlp.multi_class_classify.cnn_config import CnnConfig
from matplotlib import pylab
pylab.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
pylab.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.manifold import TSNE
import os

warnings.filterwarnings("ignore")

config = CnnConfig()

"""
使用tensorflow构建CNN模型进行多标签文本分类
"""

# 一些数据预处理的方法======================================
def get_idx_from_sent(sent, word_idx_map, max_l):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:  # 长度不够的，补充0
        x.append(0)
    # 一个训练的一个输入 形式为[0,0,0,0,x11,x12,,,,0,0,0] 向量长度为max_l+2*filter_h-2
    return x


def generate_batch(revs, word_idx_map, minibatch_index):
    batch_size = config.batch_size
    sentence_length = config.sentence_length
    minibatch_data = revs[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
    batchs = np.ndarray(shape=(batch_size, sentence_length), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 2), dtype=np.int32)

    for i in range(batch_size):
        sentece = minibatch_data[i]["text"]
        lable = minibatch_data[i]["y"]
        if lable == 1:
            labels[i] = [0, 1]  #
        else:
            labels[i] = [1, 0]  #
        batch = get_idx_from_sent(sentece, word_idx_map, sentence_length)
        batchs[i] = batch
    return batchs, labels

