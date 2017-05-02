import pprint
import tensorflow as tf
import numpy as np
import os, sys

from transwarpnlp.sentiment.reader import read_data
from transwarpnlp.sentiment.model import MemN2N

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)
train_dir = os.path.join(pkg_path, "data", "sentiment", "ckpt")

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 7, "number of hops [7]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")

FLAGS = flags.FLAGS

def init_word_embeddings(word2idx, data_path):
  wt = np.random.normal(0, FLAGS.init_std, [len(word2idx), FLAGS.edim])

  pretrain_file = os.path.join(data_path, "glove.6B.300d.txt")
  with open(pretrain_file, 'r') as f:
    for line in f:
      content = line.strip().split()
      if content[0] in word2idx:
        wt[word2idx[content[0]]] = np.array(map(float, content[1:]))
  return wt


def train_sentiment(data_path):
    source_count, target_count = [], []
    source_word2idx, target_word2idx = {}, {}

    train_file = os.path.join(data_path, "Laptop_Train_v2.xml")
    test_file = os.path.join(data_path, "Laptops_Test_Gold.xml")

    train_data = read_data(train_file, source_count, source_word2idx, target_count, target_word2idx)
    test_data = read_data(test_file, source_count, source_word2idx, target_count, target_word2idx)

    FLAGS.pad_idx = source_word2idx['<pad>']
    FLAGS.nwords = len(source_word2idx)
    FLAGS.mem_size = train_data[4] if train_data[4] > test_data[4] else test_data[4]

    pp.pprint(flags.FLAGS.__flags)

    print('loading pre-trained word vectors...')
    FLAGS.pre_trained_context_wt = init_word_embeddings(source_word2idx, data_path)
    FLAGS.pre_trained_target_wt = init_word_embeddings(target_word2idx, data_path)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()
        model.run(train_data, test_data)



if __name__ == "__main__":
    data_file = os.path.join(pkg_path, "data/sentiment/data")
    train_sentiment(data_file)