#!/usr/bin/python
# -*- coding:utf-8 -*-

class LargeConfig(object):
    learning_rate = 1.0
    init_scale = 0.04
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    num_samples = 128
    batch_size = 8
    size = 64
    num_layers = 2
    vocab_size = 1000
    buckets = [(120, 30), (200, 35), (300, 40), (400, 40), (500, 40)]

