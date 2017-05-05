# -*- coding: utf-8 -*-

class Config(object):
    ngram = 3
    word_vector = False
    pre_embeddings = True
    embeddings_dimension = 64
    radical = False
    radical_dimension = 30
    bucket_size = 10
    gru = False
    rnn_cell_dimension = 200
    rnn_layer_number = 1
    dropout_rate = 0.5
    filter_size = 5
    filters_number = 32
    max_pooling = 2
    epochs = 30
    optimizer = "adagrad"
    learning_rate = 0.1
    decay_rate = 0.05
    momentum = None
    clipping = False
    train_batch = 20
    test_batch = 200
    tag_batch = 200
    ensemble = False
    tag_scheme = 'BIES'