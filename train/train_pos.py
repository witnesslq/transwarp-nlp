#coding:utf-8
from __future__ import unicode_literals
import sys,os

import tensorflow as tf

from transwarpnlp.pos import reader
from transwarpnlp.pos import pos_model
from transwarpnlp.pos import pos_model_bilstm

from transwarpnlp.pos.config import LargeConfig

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)
train_dir = os.path.join(pkg_path, "data", "pos", "ckpt")

flags = tf.flags
flags.DEFINE_string("pos_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("pos_scope_name", "pos_var_scope", "Variable scope of pos Model")

FLAGS = flags.FLAGS

def train_lstm(data_path):
    raw_data = reader.load_data(data_path)
    # train_data, valid_data, test_data, _ = raw_data
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocab_size = raw_data
    config = LargeConfig()
    eval_config = LargeConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=None, initializer=initializer):
            m = pos_model.POSTagger(is_training=True, config=config)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=True, initializer=initializer):
            valid_m = pos_model.POSTagger(is_training=False, config=config)
            test_m = pos_model.POSTagger(is_training=False, config=eval_config)

        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.pos_train_dir, "lstm"))

        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pos_train_dir, "lstm")))
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(float(i - config.max_epoch), 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))

            train_perplexity = pos_model.run_epoch(sess, m, train_word, train_tag, m.train_op, pos_train_dir=FLAGS.pos_train_dir, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            valid_perplexity = pos_model.run_epoch(sess, valid_m, dev_word, dev_tag, tf.no_op(), pos_train_dir=FLAGS.pos_train_dir)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = pos_model.run_epoch(sess, test_m, test_word, test_tag, tf.no_op(), pos_train_dir=FLAGS.pos_train_dir)
        print("Test Perplexity: %.3f" % test_perplexity)

def train_bilstm(data_path):
    raw_data = reader.load_data(data_path)
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocabulary = raw_data
    config = LargeConfig
    eval_config = LargeConfig
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=None, initializer=initializer):
            m = pos_model_bilstm.POSTagger(is_training=True, config=config)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=True, initializer=initializer):
            valid_m = pos_model_bilstm.POSTagger(is_training=False, config=config)
            test_m = pos_model_bilstm.POSTagger(is_training=False, config=eval_config)

        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.pos_train_dir, "bilstm"))

        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pos_train_dir, "bilstm")))
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(float(i - config.max_epoch), 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
            train_perplexity = pos_model_bilstm.run_epoch(sess, m, train_word, train_tag, m.train_op,
                                         verbose=True, pos_train_dir=FLAGS.pos_train_dir)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = pos_model_bilstm.run_epoch(sess, valid_m, dev_word, dev_tag, tf.no_op(),
                                                          pos_train_dir=FLAGS.pos_train_dir)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = pos_model_bilstm.run_epoch(sess, test_m, test_word, test_tag, tf.no_op(),
                                                     pos_train_dir=FLAGS.pos_train_dir)
        print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    data_path = os.path.join(pkg_path, "data", "pos", "data")
    train_lstm(data_path)
    #train_bilstm(data_path)