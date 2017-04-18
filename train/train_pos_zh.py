#coding:utf-8
from __future__ import unicode_literals
import sys,os

import tensorflow as tf

from transwarpnlp.pos import reader
from transwarpnlp.pos.pos_model import LargeConfigChinese, POSTagger, run_epoch

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)
# file_path = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(pkg_path, "data", "pos", "ckpt")

flags = tf.flags
flags.DEFINE_string("pos_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("pos_scope_name", "pos_var_scope", "Variable scope of pos Model")

FLAGS = flags.FLAGS

def train_lstm(data_path):
    raw_data = reader.load_data(data_path)
    # train_data, valid_data, test_data, _ = raw_data
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocab_size = raw_data
    config = LargeConfigChinese
    eval_config = LargeConfigChinese
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(- config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=None, initializer=initializer):
            m = POSTagger(is_training=True, config=config)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=True, initializer=initializer):
            valid_m = POSTagger(is_training=False, config=config)
            test_m = POSTagger(is_training=False, config=eval_config)

        cktp = tf.train.get_checkpoint_state(FLAGS.pos_train_dir)

        if cktp:
            print("Loading model parameters from %s" % cktp.model_checkpoint_path)
            m.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pos_train_dir))
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))

            train_perplexity = run_epoch(sess, m, train_word, train_tag, m.train_op, pos_train_dir=FLAGS.pos_train_dir, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            valid_perplexity = run_epoch(sess, valid_m, dev_word, dev_tag, tf.no_op(), pos_train_dir=FLAGS.pos_train_dir)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(sess, test_m, test_word, test_tag, tf.no_op(), pos_train_dir=FLAGS.pos_train_dir)
        print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    data_path = os.path.join(pkg_path, "data", "pos", "data")
    train_lstm(data_path)