"""Trains a seq2seq model.
WORK IN PROGRESS.
Implement "Abstractive Text Summarization using Sequence-to-sequence RNNS and
Beyond."
"""
import sys
import os

import tensorflow as tf
from transwarpnlp.textsum import seq2seq_attention_model
from transwarpnlp.textsum.textsum_config import Config
from transwarpnlp.textsum import data, batch_reader

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)

textsum_config = Config()

def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  print('running_avg_loss: %f\n' % running_avg_loss)
  return running_avg_loss

def _Train(model, data_batcher, train_dir, log_root):
  """Runs model training."""
  with tf.device('/cpu:0'):

    print("start build graph...")
    model.build_graph()
    print("end build graph")

    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(train_dir)
    sv = tf.train.Supervisor(logdir=log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=textsum_config.checkpoint_secs,
                             global_step=model.global_step)

    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))

    running_avg_loss = 0
    step = 0
    while not sv.should_stop() and step < textsum_config.max_run_steps:
      print("step %d" % step)

      article_batch, abstract_batch, targets, article_lens, abstract_lens,loss_weights, _, _=\
          data_batcher.getNextBatch()

      _, summaries, loss, train_step =\
          model.run_train_step(sess, article_batch, abstract_batch, targets, article_lens, abstract_lens, loss_weights)

      summary_writer.add_summary(summaries, train_step)
      running_avg_loss = _RunningAvgLoss(running_avg_loss, loss, summary_writer, train_step)

      step += 1
      if step % 100 == 0:
        summary_writer.flush()
    sv.stop()
    return running_avg_loss

def train_textsum(vocab_path, data_path, train_dir, log_root):
    vocab = data.Vocab(vocab_path, 10000)
    # Check for presence of required special tokens.
    assert vocab.CheckVocab(data.PAD_TOKEN) > 0
    assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
    assert vocab.CheckVocab(data.SENTENCE_START) > 0
    assert vocab.CheckVocab(data.SENTENCE_END) > 0

    batch_size = 2

    hps = seq2seq_attention_model.HParams(
        mode='train',  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=batch_size,
        enc_layers=2,
        enc_timesteps=100,
        dec_timesteps=20,
        min_input_len=2,  # discard articles/summaries < than this
        num_hidden=256,  # for rnn cell
        emb_dim=128,  # If 0, don't use embedding
        max_grad_norm=2,
        num_softmax_samples=0)  # If 0, no sampled softmax.

    batcher = batch_reader.Batcher(data_path, vocab, hps, textsum_config.article_key,
        textsum_config.abstract_key, textsum_config.max_article_sentences,
        textsum_config.max_abstract_sentences, bucketing=textsum_config.use_bucketing,
        truncate_input=textsum_config.truncate_input)
    tf.set_random_seed(textsum_config.random_seed)

    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab)
    _Train(model, batcher, train_dir, log_root)


if __name__ == "__main__":
    vocab_path = os.path.join(pkg_path, "data/textsum/data/vocab.txt")
    data_path = os.path.join(pkg_path, "data/textsum/data/train.txt")
    train_dir = os.path.join(pkg_path, "data/textsum/model/")
    log_root = os.path.join(pkg_path, "data/textsum/ckpt/")
    train_textsum(vocab_path, data_path, train_dir, log_root)