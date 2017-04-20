#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
NER tagger for building a LSTM based NER tagging model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
import os

from ner import reader

def data_type():
  return tf.float32

class NERTagger(object):
  """The NER Tagger Model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num # target output number
    
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    
    # Check if Model is Training
    self.is_training = is_training
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
    
    self._initial_state = cell.zero_state(batch_size, data_type())
    
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    
    outputs = []
    state = self._initial_state
    with tf.variable_scope("ner_lstm"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    
    output = tf.reshape(tf.concat(outputs, 1), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, target_num], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        logits = [logits],
        targets = [tf.reshape(self._targets, [-1])],
        weights = [tf.ones([batch_size * num_steps], dtype=data_type())])
    
    # Fetch Reults in session.run()
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    self._logits = logits
    
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    self._new_lr = tf.placeholder(
        data_type(), shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self.saver = tf.train.Saver(tf.global_variables())
  
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    
  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state
  
  @property
  def logits(self):
    return self._logits

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def run_epoch(session, model, word_data, tag_data, eval_op, ner_train_dir, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                    model.num_steps)):
    fetches = [model.cost, model.final_state, eval_op]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    cost, state, _ = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps
    
    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
    
    # Save Model to CheckPoint when is_training is True
    if model.is_training:
      if step % (epoch_size // 10) == 10:
        checkpoint_path = os.path.join(ner_train_dir, "lstm", "lstm.ckpt")
        model.saver.save(session, checkpoint_path)
        print("Model Saved... at time step " + str(step))

  return np.exp(costs / iters)