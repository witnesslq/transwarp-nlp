# -*- coding: utf-8 -*-

import os, time
import codecs
import cPickle as pickle
from transwarpnlp.joint_seg_tagger.config import Config
from transwarpnlp.dataprocess import joint_data_transform
from transwarpnlp.joint_seg_tagger.model import Model
import tensorflow as tf

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = Config()

def test_joint(data_path, method="test"):
    test_file = "/data/test.txt"
    raw_file = "/data/raw.txt"
    model_file = "model/trained"
    output_file = "model/output"

    emb_path = None
    ng_emb_path = None

    if not os.path.isfile(data_path + '/' + model_file + '_model')\
        or not os.path.isfile(data_path + '/' + model_file + '_weights'):
        raise Exception('No model file or weights file under the name of ' + model_file + '.')

    fin = open(data_path + '/' + model_file + '_model', 'rb')
    weight_path = data_path + '/' + model_file
    param_dic = pickle.load(fin)
    fin.close()

    nums_chars = param_dic['nums_chars']
    nums_tags = param_dic['nums_tags']
    tag_scheme = param_dic['tag_scheme']
    word_vector = param_dic['word_vec']
    crf = param_dic['crf']
    emb_dim = param_dic['emb_dim']
    gru = param_dic['gru']
    rnn_dim = param_dic['rnn_dim']
    rnn_num = param_dic['rnn_num']
    drop_out = param_dic['drop_out']
    con_width = param_dic['filter_size']
    cv_kernels = param_dic['filters']
    pooling_size = param_dic['pooling_size']
    num_ngram = param_dic['ngram']

    ngram = 1

    if num_ngram is not None:
        ngram = len(num_ngram) + 1

    chars, tags, grams = joint_data_transform.read_vocab_tag(data_path, ngram)
    char2idx, idx2char, tag2idx, idx2tag = joint_data_transform.get_dic(chars, tags)
    new_chars, new_grams, new_gram_emb, gram2idx = None, None, None, None
    test_x, test_y, raw_x = None, None, None
    max_step = None

    if method == "test":
        new_chars = joint_data_transform.get_new_chars(data_path + '/data/' + test_file, char2idx)
        char2idx, idx2char = joint_data_transform.update_char_dict(char2idx, new_chars)
        test_x, test_y, test_max_slen_c, test_max_slen_w, test_max_wlen, _ =\
            joint_data_transform.get_input_vec(data_path, test_file,char2idx, tag2idx,tag_scheme=tag_scheme)
        print('Test set: %d instances.' % len(test_x[0]))

        max_step = test_max_slen_c

        print('Longest sentence by character is %d. ' % test_max_slen_c)
        print('Longest sentence by word is %d. ' % test_max_slen_w)
        print('Longest word is %d. ' % test_max_wlen)

        if ngram > 1:
            gram2idx = joint_data_transform.get_ngram_dic(grams)
            new_grams = joint_data_transform.get_new_grams(data_path + '/' + test_file, gram2idx)
            test_gram = joint_data_transform.get_gram_vec(data_path, test_file, gram2idx)
            test_x += test_gram
        for k in range(len(test_x)):
            test_x[k] = joint_data_transform.pad_zeros(test_x[k], max_step)
        for k in range(len(test_y)):
            test_y[k] = joint_data_transform.pad_zeros(test_y[k], max_step)
    elif method == "tag":
        new_chars = joint_data_transform.get_new_chars(data_path + '/' + raw_file, char2idx, type='raw')
        char2idx, idx2char = joint_data_transform.update_char_dict(char2idx, new_chars)

        if not config.tag_large:
            raw_x, raw_len = joint_data_transform.get_input_vec_raw(data_path, raw_file, char2idx)
            print('Numbers of sentences: %d.' % len(raw_x[0]))
            max_step = raw_len
        else:
            max_step = joint_data_transform.get_maxstep(data_path, raw_file)

        print('Longest sentence is %d. ' % max_step)
        if ngram > 1:
            gram2idx = joint_data_transform.get_ngram_dic(grams)
            if not config.tag_large:
                raw_gram = joint_data_transform.get_gram_vec(data_path, raw_file, gram2idx, is_raw=True)
                raw_x += raw_gram
        if not config.tag_large:
            for k in range(len(raw_x)):
                raw_x[k] = joint_data_transform.pad_zeros(raw_x[k], max_step)

    configProto = tf.ConfigProto(allow_soft_placement=True)
    print('Initialization....')
    t = time.time()
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger") as scope:
            model = Model(nums_chars=nums_chars, nums_tags=nums_tags, buckets_char=[max_step], counts=[200], tag_scheme=tag_scheme, word_vec=word_vector,
                          crf=crf, ngram=num_ngram, batch_size=config.tag_batch)
            model.model_graph(trained_model=None, scope=scope, emb_dim=emb_dim, gru=gru, rnn_dim=rnn_dim,
                             rnn_num=rnn_num, drop_out=drop_out,  con_width=con_width, filters=cv_kernels,
                             pooling_size=pooling_size)

            model.define_updates(new_chars=new_chars, emb_path=emb_path, char2idx=char2idx, new_grams=new_grams,
                                 ng_emb_path=ng_emb_path, gram2idx=gram2idx)
            init = tf.global_variables_initializer()

            print('Done. Time consumed: %d seconds' % int(time() - t))

        main_graph.finalize()
        main_sess = tf.Session(config=configProto, graph=main_graph)

        if crf:
            decode_graph = tf.Graph()
            with decode_graph.as_default():
                model.decode_graph()
            decode_graph.finalize()
            decode_sess = tf.Session(config=configProto, graph=decode_graph)
            sess = [main_sess, decode_sess]
        else:
            sess = [main_sess]

        with tf.device("/cpu:0"):
            ens_model = None
            print('Loading weights....')
            main_sess.run(init)
            model.run_updates(main_sess, weight_path + '_weights')

            if method == 'test':
                model.test(sess=sess, t_x=test_x, t_y=test_y, idx2tag=idx2tag, idx2char=idx2char,
                           outpath=output_file, ensemble=config.ensemble, batch_size=config.test_batch)
            elif method == 'tag':
                if not config.tag_large:
                    model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char, outpath=output_file,
                              ensemble=config.ensemble, batch_size=config.tag_batch, large_file=config.tag_large)
                else:
                    l_writer = codecs.open(output_file, 'w', encoding='utf-8')
                    out = []
                    with codecs.open(data_path + '/' + raw_file, 'r', encoding='utf-8') as l_file:
                        lines = []
                        for line in l_file:
                            lines.append(line.strip())
                            if len(lines) >= config.large_size:
                                raw_x, _ = joint_data_transform.get_input_vec_line(lines, char2idx)

                                if ngram > 1:
                                    raw_gram = joint_data_transform.get_gram_vec_raw(lines, gram2idx)
                                    raw_x += raw_gram

                                for k in range(len(raw_x)):
                                    raw_x[k] = joint_data_transform.pad_zeros(raw_x[k], max_step)

                                out = model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char,
                                                outpath=output_file, ensemble=config.ensemble,
                                                batch_size=config.tag_batch, large_file=config.tag_large)

                                for l_out in out:
                                    l_writer.write(l_out + '\n')
                                lines = []
                        if len(lines) > 0:
                            raw_x, _ = joint_data_transform.get_input_vec_line(lines, char2idx)

                            if ngram > 1:
                                raw_gram = joint_data_transform.get_gram_vec_raw(lines, gram2idx)
                                raw_x += raw_gram

                            for k in range(len(raw_x)):
                                raw_x[k] = joint_data_transform.pad_zeros(raw_x[k], max_step)

                            out = model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char,
                                            outpath=output_file, ensemble=config.ensemble, batch_size=config.tag_batch,
                                            large_file=config.tag_large)

                            for l_out in out:
                                l_writer.write(l_out + '\n')
                    l_writer.close()