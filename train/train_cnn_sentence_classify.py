# -*- coding: utf-8 -*-

import cPickle
import numpy as np
import tensorflow as tf
import os, time
from transwarpnlp.textclassify.cnn_config import CnnConfig
from transwarpnlp.textclassify import cnn_classfier

config = CnnConfig()
pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_classfier(train_path):
    print("loading data...")
    x = cPickle.load(open(os.path.join(train_path, "data/mr.txt"), "rb"))
    # 读取出预处理后的数据 revs {"y":label,"text":"word1 word2 ..."}
    #                          word_idx_map["word"]==>index
    #                        vocab["word"]==>frequency
    revs, _, _, word_idx_map, idx_word_map, vocab = x[0], x[1], x[2], x[3], x[4], x[5]
    print("data loaded!")
    revs = np.random.permutation(revs)  # 原始的sample正负样本是分别聚在一起的，这里随机打散
    n_batches = len(revs) / config.batch_size  #
    n_train_batches = int(np.round(n_batches * 0.9))

    # 开始定义模型============================================
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        # 占位符 真实的输入输出
        x_in = tf.placeholder(tf.int32, shape=[None, config.sentence_length], name="input_x")
        y_in = tf.placeholder(tf.float32, [None, 2], name="input_y")  # 2分类问题
        keep_prob = tf.placeholder(tf.float32)

        # 构建模型
        loss, accuracy, embeddings = cnn_classfier.build_model(x_in, y_in, keep_prob)

        # 训练模型========================================

        num_steps = 1

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(1e-4, global_step, num_steps, 0.99, staircase=True)  # 学习率递减
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # summaries,====================
        timestamp = str(int(time.time()))
        out_dir = os.path.join(train_path, "summary", timestamp)
        print("Writing to {}\n".format(out_dir))
        loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        checkpoint_dir = os.path.join(train_path, "ckpt")
        checkpoint_prefix = os.path.join(checkpoint_dir, "classify")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        current_step = tf.train.global_step(sess, global_step)
        print("current_step:", current_step)
        if num_steps > int(current_step / 135):
            num_steps = num_steps - int(current_step / 135)
            print("continute step:", num_steps)
        else:
            num_steps = 0

        batch_x_test, batch_y_test = cnn_classfier.get_test_batch(revs, word_idx_map)

        for i in range(num_steps):
            for minibatch_index in np.random.permutation(range(n_train_batches)):  # 随机打散 每次输入的样本的顺序都不一样
                batch_x, batch_y = cnn_classfier.generate_batch(revs, word_idx_map, minibatch_index)
                # train_step.run(feed_dict={x_in: batch_x, y_in: batch_y, keep_prob: 0.5})
                feed_dict = {x_in: batch_x, y_in: batch_y, keep_prob: 0.5}
                _, step, summaries = sess.run([train_step, global_step, train_summary_op], feed_dict)
                train_summary_writer.add_summary(summaries, step)
            train_accuracy = accuracy.eval(feed_dict={x_in: batch_x_test, y_in: batch_y_test, keep_prob: 1.0})
            current_step = tf.train.global_step(sess, global_step)
            print("step %d, training accuracy %g" % (current_step, train_accuracy))
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

        return embeddings, sess, idx_word_map


if __name__ == "__main__":
    train_path = os.path.join(pkg_path, "data/textclassify")
    embeddings, sess, idx_word_map = train_classfier(train_path)
    final_embeddings = cnn_classfier.word2vec(embeddings, train_path, sess)
    # cnn_classfier.display_word2vec(final_embeddings, idx_word_map)