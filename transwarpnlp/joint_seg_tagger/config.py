# -*- coding: utf-8 -*-

class Config(object):
    ngram = 3
    word_vector = True
    pre_embeddings = True
    embeddings_dimension = 64
    bucket_size = 10
    gru = True
    rnn_cell_dimension = 200
    rnn_layer_number = 1
    dropout_rate = 0.5
    epochs = 30
    optimizer = "adagrad"
    learning_rate = 0.1
    decay_rate = 0.05
    momentum = None
    clipping = True
    train_batch = 20
    test_batch = 200
    tag_batch = 200
    ensemble = False
    tag_scheme = 'BIES'
    crf = 1
    tag_large = False
    large_size = 200000