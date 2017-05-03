# -*- coding: utf-8 -*-

import cPickle
import numpy as np
import warnings
import tensorflow as tf
from transwarpnlp.textclassify.cnn_config import CnnConfig
from matplotlib import pylab
pylab.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
pylab.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.manifold import TSNE
import os

warnings.filterwarnings("ignore")

config = CnnConfig()

"""
使用tensorflow构建CNN模型进行文本分类
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
    while len(x) < max_l:
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


def get_test_batch(revs, word_idx_map, cv=1):
    sentence_length = config.sentence_length
    test = []
    for rev in revs:
        if rev["split"] == cv:
            test.append(rev)
    minibatch_data = test
    test_szie = len(minibatch_data)
    batchs = np.ndarray(shape=(test_szie, sentence_length), dtype=np.int32)
    labels = np.ndarray(shape=(test_szie, 2), dtype=np.int32)
    for i in range(test_szie):
        sentece = minibatch_data[i]["text"]
        lable = minibatch_data[i]["y"]
        if lable == 1:
            labels[i] = [0, 1]
        else:
            labels[i] = [1, 0]
        batch = get_idx_from_sent(sentece, word_idx_map, sentence_length)
        batchs[i] = batch

    return batchs, labels


# 卷积图层 第一个卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

# 定义pooling图层
def max_pool(x, filter_h):
    return tf.nn.max_pool(x, ksize=[1, config.img_h - filter_h + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')


def build_model(x_in, y_in, keep_prob):
    # Embedding layer===============================
    # 要学习的词向量矩阵
    embeddings = tf.Variable(tf.random_uniform([config.word_idx_map_szie, config.vector_size], -1.0, 1.0))
    # 输入reshape
    x_image_tmp = tf.nn.embedding_lookup(embeddings, x_in)
    # 输入size: sentence_length*vector_size
    # x_image = tf.reshape(x_image_tmp, [-1,sentence_length,vector_size,1])======>>>>>
    # 将[None, sequence_length, embedding_size]转为[None, sequence_length, embedding_size, 1]
    x_image = tf.expand_dims(x_image_tmp, -1)  # 单通道

    # 定义卷积层，进行卷积操作===================================
    h_conv = []
    for filter_h in config.filter_hs:
        # 卷积的patch大小：vector_size*filter_h, 通道数量：1, 卷积数量：hidden_layer_input_size
        filter_shape = [filter_h, config.vector_size, 1, config.num_filters]
        W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name="b")
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出szie: (sentence_length-filter_h+1,1)
        h_conv.append(h_conv1)


    # pool层========================================
    h_pool_output = []
    for h_conv1, filter_h in zip(h_conv, config.filter_hs):
        h_pool1 = max_pool(h_conv1, filter_h)  # 输出szie:1
        h_pool_output.append(h_pool1)

    # 全连接层=========================================
    l2_reg_lambda = 0.001
    # 输入reshape
    num_filters_total = config.num_filters * len(config.filter_hs)
    h_pool = tf.concat(h_pool_output, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    h_drop = tf.nn.dropout(h_pool_flat, keep_prob)


    W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
    l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")  # wx+b
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_in)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y_in, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy, embeddings

def word2vec(embeddings, train_path, sess):
    with sess.as_default():
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        final_embeddings = normalized_embeddings.eval()
        filename = os.path.join(train_path, "ckpt/CNN_result_embeddings")
        cPickle.dump(final_embeddings, open(filename, "wb"))
        return final_embeddings

def display_word2vec(final_embeddings, idx_word_map):
    num_points = 200
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])

    def plot(embeddings, labels):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(15, 15))
        for i, label in enumerate(labels):
            x, y = embeddings[i, :]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()

    words = [idx_word_map[i].decode("utf-8") for i in range(1, num_points + 1)]
    plot(two_d_embeddings, words)
