# -*- coding: utf-8 -*-

import os, time
from transwarpnlp.joint_seg_tagger.config import Config
from transwarpnlp.dataprocess import joint_data_transform
import tensorflow as tf

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = Config()

def train_joint(data_path):
    train_file = "/data/train.txt"
    dev_file = "/data/dev.txt"
    test_file = "/data/test.txt"

    if config.ngram > 1 and not os.path.isfile(data_path + '/model/' + str(config.ngram) + 'gram.txt') \
        or (not os.path.isfile(data_path + '/model/' + 'chars.txt')):
        joint_data_transform.get_vocab_tag(data_path, [train_file, dev_file], ngram=config.ngram)

    # 字集，标签集，n元词集
    chars, tags, ngram = joint_data_transform.read_vocab_tag(data_path, config.ngram)

    emb = None
    if config.pre_embeddings:
        short_emb = "glove.txt"
        if not os.path.isfile(data_path + '/model/' + short_emb + '_sub.txt'):
            joint_data_transform.get_sample_embedding(data_path, short_emb, chars)
            emb_dim, emb = joint_data_transform.read_sample_embedding(data_path, short_emb)
            assert config.embeddings_dimension == emb_dim
    else:
        print('Using random embeddings...')

    char2idx, idx2char, tag2idx, idx2tag = joint_data_transform.get_dic(chars, tags)

    # 训练样本id，训练标签id，句子的最大字符数，句子的最大词数，句子的最大词长
    train_x, train_y, train_max_slen_c, train_max_slen_w, train_max_wlen =\
        joint_data_transform.get_input_vec(data_path, train_file, char2idx, tag2idx, config.tag_scheme)

    dev_x, dev_y, dev_max_slen_c, dev_max_slen_w, dev_max_wlen = \
        joint_data_transform.get_input_vec(data_path, dev_file, char2idx, tag2idx, config.tag_scheme)

    # 将多元词加入训练和验证语料
    if config.ngram > 1:
        gram2idx = joint_data_transform.get_ngram_dic(ngram)
        train_gram = joint_data_transform.get_gram_vec(data_path, train_file, gram2idx)
        dev_gram = joint_data_transform.get_gram_vec(data_path, dev_file, gram2idx)
        train_x += train_gram
        dev_x += dev_gram
        nums_grams = []
        for dic in gram2idx:
            nums_grams.append(len(dic.keys()))

    tag_map = {'seg': 0, 'BI': 1, 'BIE': 2, 'BIES': 3}

    max_step_c = max(train_max_slen_c, dev_max_slen_c)
    max_step_w = max(train_max_slen_w, dev_max_slen_w)
    max_w_len = max(train_max_wlen, dev_max_wlen)
    print('Longest sentence by character is %d. ' % max_step_c)
    print('Longest sentence by word is %d. ' % max_step_w)
    print('Longest word is %d. ' % max_w_len)

    b_train_x, b_train_y = joint_data_transform.buckets(train_x, train_y, size=config.bucket_size)
    b_dev_x, b_dev_y = joint_data_transform.buckets(dev_x, dev_y, size=config.bucket_size)

    b_train_x, b_train_y, b_buckets, b_counts = joint_data_transform.pad_bucket(b_train_x, b_train_y)
    b_dev_x, b_dev_y, b_buckets, _ = joint_data_transform.pad_bucket(b_dev_x, b_dev_y, bucket_len_c=b_buckets)

    print('Training set: %d instances; Dev set: %d instances.' % (len(train_x[0]), len(dev_x[0])))

    nums_tags = joint_data_transform.get_nums_tags(tag2idx, config.tag_scheme)

    # allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
    configProto = tf.ConfigProto(allow_soft_placement=True)
    print('Initialization....')
    t = time()
    initializer = tf.contrib.layers.xavier_initializer()
    main_graph = tf.Graph()

if __name__ == "__main__":
    data_path = os.path.join(pkg_path, "data/joint")
    train_joint(data_path)