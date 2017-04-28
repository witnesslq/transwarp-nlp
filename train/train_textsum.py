#coding:utf-8
from __future__ import unicode_literals
import sys,os

import tensorflow as tf
import numpy as np
import time
import math

from transwarpnlp.textsum.config import LargeConfig
from transwarpnlp.textsum import data_utils
from transwarpnlp.textsum import seq2seq_model

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)
train_dir = os.path.join(pkg_path, "data", "textsum", "ckpt")

flags = tf.flags
flags.DEFINE_string("text_sum_train_dir", train_dir, "Training directory.")

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# article length padded to 120 and summary padded to 30
config = LargeConfig()

tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("headline_scope_name", "headline_var_scope", "Variable scope of Headline textsum model")
FLAGS = tf.app.flags.FLAGS

def create_model(sess, forward_only):
    initializer = tf.random_normal_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope(FLAGS.headline_scope_name, reuse=None, initializer= initializer):
        model = seq2seq_model.Seq2SeqModel(config.vocab_size,
                                           config.vocab_size,
                                           buckets=config.buckets,
                                           size=config.size,
                                           num_layers=config.num_layers,
                                           max_gradient_norm=config.max_gradient_norm,
                                           batch_size=config.batch_size,
                                           learning_rate=config.learning_rate,
                                           learning_rate_decay_factor=config.learning_rate_decay_factor,
                                           use_lstm=True,
                                           num_samples=config.num_samples,
                                           forward_only=forward_only)

        ckpt = tf.train.get_checkpoint_state(FLAGS.text_sum_train_dir)
        if ckpt:
            model_checkpoint_path = ckpt.model_checkpoint_path
            print("Reading model parameters from %s" % model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.text_sum_train_dir))
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        return model

def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

      Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
          it must be aligned with the source file: n-th line contains the desired
          output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
          if 0 or None, data files will be read completely (no limit).

      Returns:
        data_set: a list of length len(buckets); data_set[n] contains a list of
          (source, target) pairs read from the provided data files that fit
          into the n-th bucket, i.e., such that len(source) < buckets[n][0] and
          len(target) < buckets[n][1]; source and target are lists of token-ids.
      """
    data_set = [[] for _ in config.buckets]
    with tf.gfile.GFile(source_path, mode='r') as source_file,\
         tf.gfile.GFile(target_path, mode='r') as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(config.buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def trainSeq2Seq(data_path):
    # Prepare Headline data.
    print("Preparing Headline data in %s" % data_path)
    src_train, dest_train, src_dev, dest_dev, _, _ = data_utils.prepare_headline_data(data_path, config.vocab_size)

    dev_config = tf.ConfigProto(device_count={"CPU": 6},
                   inter_op_parallelism_threads=1,
                   intra_op_parallelism_threads=2)

    with tf.Session(config=dev_config) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (config.num_layers, config.size))
        model = create_model(sess, False)
        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d). 0 is not limit"
               % FLAGS.max_train_data_size)

        dev_set = read_data(src_dev, dest_dev)
        train_set = read_data(src_train, dest_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                              for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while current_step < 100:
            print(current_step)
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                if len(previous_losses) > 2 and max(previous_losses[-3:]) > loss:
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.text_sum_train_dir, "headline.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(config.buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

if __name__ == "__main__":
    data_path = os.path.join(pkg_path, "data/textsum/data/")
    trainSeq2Seq(data_path)