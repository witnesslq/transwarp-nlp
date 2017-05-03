# -*- coding: utf-8 -*-

class CnnConfig(object):
    vector_size = 100
    sentence_length = 120
    batch_size = 100
    hidden_layer_input_size = 100
    filter_hs = [3, 4, 5]
    num_filters = 128
    img_h = sentence_length
    img_w = vector_size
    filter_w = img_w
    word_idx_map_szie = 75924  # 18766#75924