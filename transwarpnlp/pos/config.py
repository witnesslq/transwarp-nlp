#!/usr/bin/python
# -*- coding:utf-8 -*-

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.5
  max_grad_norm = 10
  num_layers = 2
  num_steps = 30
  hidden_size = 128
  max_epoch = 5
  max_max_epoch = 10
  keep_prob = 1.0
  lr_decay = 1 / 1.15
  batch_size = 10        # single sample batch
  vocab_size = 50000
  target_num = 48       # POS tagging tag number for Chinese