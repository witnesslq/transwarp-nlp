#!/usr/bin/python
# -*- coding:utf-8 -*-
""" 
NER tagger for building a BI-LSTM based NER tagging model.
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
    self.is_training = is_training
    size = config.hidden_size
    vocab_size = config.vocab_size
    
    # Define input and target tensors
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)
    
    self._cost, self._logits = _bilstm_model(inputs, self._targets, config)
    
    # Gradients and SGD update operation for training the model.
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    
    self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
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
  def cost(self):
    return self._cost
  
  @property
  def logits(self):
    return self._logits

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
  
  @property
  def accuracy(self):
    return self._accuracy


def _bilstm_model(inputs, targets, config):
    '''
    @Use BasicLSTMCell, MultiRNNCell method to build LSTM model, 
    @return logits, cost and others
    '''
    batch_size = config.batch_size
    num_steps = config.num_steps
    num_layers = config.num_layers
    size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num # target output number    
    
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        
    cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell] * num_layers, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell] * num_layers, state_is_tuple=True)
    
    initial_state_fw = cell_fw.zero_state(batch_size, data_type())
    initial_state_bw = cell_bw.zero_state(batch_size, data_type())
    
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    inputs_list = [tf.squeeze(s, axis = 1) for s in tf.split(value = inputs, num_or_size_splits = num_steps, axis = 1)]
    
    with tf.variable_scope("ner_bilstm"):
        outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
            cell_fw, cell_bw, inputs_list, initial_state_fw = initial_state_fw, 
            initial_state_bw = initial_state_bw)
    
    # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
    # [time][batch][cell_fw.output_size + cell_bw.output_size]
    
    output = tf.reshape(tf.concat(outputs, 1), [-1, size * 2])
    # output has size: [T, size * 2]
    
    softmax_w = tf.get_variable("softmax_w", [size * 2, target_num], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(targets, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(targets, [-1]), logits = logits)
    cost = tf.reduce_sum(loss)/batch_size # loss [time_step]
    return cost, logits

def run_epoch(session, model, word_data, tag_data, eval_op, ner_train_dir, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps

  start_time = time.time()
  costs = 0.0
  iters = 0
  
  for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                    model.num_steps)):
    fetches = [model.cost, model.logits, eval_op]  # eval_op define the m.train_op or m.eval_op
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    cost, logits, _ = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
    
    # Save Model to CheckPoint when is_training is True
    if model.is_training:
      if step % (epoch_size // 10) == 10:
        checkpoint_path = os.path.join(ner_train_dir, "bilstm", "bilstm.ckpt")
        model.saver.save(session, checkpoint_path)
        print("Model Saved... at time step " + str(step))
  
  return np.exp(costs / iters)
