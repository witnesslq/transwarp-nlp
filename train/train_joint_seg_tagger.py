# -*- coding: utf-8 -*-

import os, time
from transwarpnlp.joint_seg_tagger.config import Config
from transwarpnlp.dataprocess import joint_data_transform
from transwarpnlp.joint_seg_tagger.model import Model
import tensorflow as tf

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = Config()

def train_joint(data_path):
    train_file = "/data/train1.txt"
    dev_file = "/data/dev1.txt"

    if config.ngram > 1 and not os.path.isfile(data_path + '/model/' + str(config.ngram) + 'gram.txt') \
        or (not os.path.isfile(data_path + '/model/' + 'chars.txt')):
        joint_data_transform.get_vocab_tag(data_path, [train_file, dev_file], ngram=config.ngram)

    # 字集，标签集，n元词集
    chars, tags, ngram = joint_data_transform.read_vocab_tag(data_path, config.ngram)

    emb = None
    emb_dim = config.embeddings_dimension
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
    nums_grams = []
    ng_embs = None
    if config.ngram > 1:
        gram2idx = joint_data_transform.get_ngram_dic(ngram)
        train_gram = joint_data_transform.get_gram_vec(data_path, train_file, gram2idx)
        dev_gram = joint_data_transform.get_gram_vec(data_path, dev_file, gram2idx)
        train_x += train_gram
        dev_x += dev_gram

        for dic in gram2idx:
            nums_grams.append(len(dic.keys()))

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
    t = time.time()
    initializer = tf.contrib.layers.xavier_initializer()
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger", reuse=None, initializer=initializer) as scope:
            model = Model(nums_chars=len(chars) + 2, nums_tags=nums_tags, buckets_char=b_buckets,
                          counts=b_counts, tag_scheme=config.tag_scheme, word_vec=config.word_vector,
                          crf=config.crf, ngram=nums_grams, batch_size=config.train_batch)
            model.model_graph(trained_model=data_path + '/model/trained_model', scope=scope, emb_dim=emb_dim,
                              gru=config.gru, rnn_dim=config.rnn_cell_dimension, rnn_num=config.rnn_layer_number,
                              emb=emb, ng_embs=ng_embs, drop_out=config.dropout_rate,con_width=config.filter_size,
                              filters=config.filters_number, pooling_size=config.max_pooling)
            t = time.time()
            model.config(scope=scope, optimizer=config.optimizer, decay=config.decay_rate, lr_v=config.learning_rate,
                         momentum=config.momentum, clipping=config.clipping)
            init = tf.global_variables_initializer()

    main_graph.finalize()

    main_sess = tf.Session(config=configProto, graph=main_graph)

    if config.crf:
        decode_graph = tf.Graph()
        with decode_graph.as_default():
            model.decode_graph()
        decode_graph.finalize()
        decode_sess = tf.Session(config=configProto, graph=decode_graph)
        sess = [main_sess, decode_sess]
    else:
        sess = [main_sess]

    with tf.device("/cpu:0"):
        main_sess.run(init)
        print('Done. Time consumed: %d seconds' % int(time.time() - t))
        t = time.time()
        model.train(t_x=b_train_x, t_y=b_train_y, v_x=b_dev_x, v_y=b_dev_y, idx2tag=idx2tag, idx2char=idx2char,
                    sess=sess, epochs=config.epochs, trained_model=data_path + '/model/trained_model_weights',
                    lr=config.learning_rate, decay=config.decay_rate)
        print('Done. Time consumed: %d seconds' % int(time.time() - t))

if __name__ == "__main__":
    data_path = os.path.join(pkg_path, "data/joint")
    train_joint(data_path)