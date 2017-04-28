#!/usr/bin/python
# -*- coding:utf-8 -*-


class LargeConfig(object):
    learning_rate = 1.0
    init_scale = 0.04
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    num_samples = 1024 # Sampled Softmax
    batch_size = 8
    size = 128 # Number of Node of each layer
    num_layers = 2
    vocab_size = 10000