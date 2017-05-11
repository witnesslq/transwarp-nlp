"""Batch reader to seq2seq attention model, with bucketing support."""

from collections import namedtuple
from random import shuffle
from threading import Thread
import time

import numpy as np
import random
import tensorflow as tf

import data

ModelInput = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_abstract')

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100


class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self, data_path, vocab, hps,
               article_key, abstract_key, max_article_sentences,
               max_abstract_sentences, bucketing=True, truncate_input=False):
    """Batcher constructor.
    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary.
      hps: Seq2SeqAttention model hyperparameters.
      article_key: article feature key in tf.Example.
      abstract_key: abstract feature key in tf.Example.
      max_article_sentences: Max number of sentences used from article.
      max_abstract_sentences: Max number of sentences used from abstract.
      bucketing: Whether bucket articles of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._article_key = article_key
    self._abstract_key = abstract_key
    self._max_article_sentences = max_article_sentences
    self._max_abstract_sentences = max_abstract_sentences
    self._bucketing = bucketing
    self._truncate_input = truncate_input
    self._input_data = []
    self.fillInput()
    # self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
    # self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)
    #
    # self._input_threads = []
    # for _ in range(1):
    #   self._input_threads.append(Thread(target=self._FillInputQueue))
    #   self._input_threads[-1].daemon = True
    #   self._input_threads[-1].start()
    # self._bucketing_threads = []
    # for _ in range(1):
    #   self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
    #   self._bucketing_threads[-1].daemon = True
    #   self._bucketing_threads[-1].start()
    #
    # self._watch_thread = Thread(target=self._WatchThreads)
    # self._watch_thread.daemon = True
    # self._watch_thread.start()

  def getNextBatch(self):
    """Returns a batch of inputs for seq2seq attention model.
    Returns:
      enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
      dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].
      target_batch: A batch of targets [batch_size, hps.dec_timestamps].
      enc_input_len: encoder input lengths of the batch.
      dec_input_len: decoder input lengths of the batch.
      loss_weights: weights for loss function, 1 if not padded, 0 if padded.
      origin_articles: original article words.
      origin_abstracts: original abstract words.
    """
    enc_batch = np.zeros(
        (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
    enc_input_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    dec_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    dec_output_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    target_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    loss_weights = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.float32)
    origin_articles = ['None'] * self._hps.batch_size
    origin_abstracts = ['None'] * self._hps.batch_size

    input_len = len(self._input_data)
    for i in range(self._hps.batch_size):
      (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len,
       article, abstract) = random.choice(self._input_data)

      origin_articles[i] = article
      origin_abstracts[i] = abstract
      enc_input_lens[i] = enc_input_len
      dec_output_lens[i] = dec_output_len
      enc_batch[i, :] = enc_inputs[:]
      dec_batch[i, :] = dec_inputs[:]
      target_batch[i, :] = targets[:]
      for j in range(dec_output_len):
        loss_weights[i][j] = 1
    return (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
            loss_weights, origin_articles, origin_abstracts)

  def fillInput(self):
    """Fill input queue with ModelInput."""
    start_id = self._vocab.WordToId(data.SENTENCE_START)
    end_id = self._vocab.WordToId(data.SENTENCE_END)
    pad_id = self._vocab.WordToId(data.PAD_TOKEN)

    articles_abstracts = data.getArticlesAndAbstracts(self._data_path)
    for article, abstract in articles_abstracts:
        enc_inputs = []
        # Use the <s> as the <GO> symbol for decoder inputs.
        dec_inputs = [start_id]

        enc_inputs += data.GetWordIds(article, self._vocab)
        dec_inputs += data.GetWordIds(abstract, self._vocab)

        # Filter out too-short input
        if (len(enc_inputs) < self._hps.min_input_len or
                len(dec_inputs) < self._hps.min_input_len):
          tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d',
                             len(enc_inputs), len(dec_inputs))
          continue

        # If we're not truncating input, throw out too-long input
        if not self._truncate_input:
          if (len(enc_inputs) > self._hps.enc_timesteps or
                  len(dec_inputs) > self._hps.dec_timesteps):
            tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d',
                                 len(enc_inputs), len(dec_inputs))
            continue
        # If we are truncating input, do so if necessary
        else:
          if len(enc_inputs) > self._hps.enc_timesteps:
            enc_inputs = enc_inputs[:self._hps.enc_timesteps]
          if len(dec_inputs) > self._hps.dec_timesteps:
            dec_inputs = dec_inputs[:self._hps.dec_timesteps]

        # targets is dec_inputs without <s> at beginning, plus </s> at end
        targets = dec_inputs[1:]
        targets.append(end_id)

        # Now len(enc_inputs) should be <= enc_timesteps, and
        # len(targets) = len(dec_inputs) should be <= dec_timesteps

        enc_input_len = len(enc_inputs)
        dec_output_len = len(targets)

        # Pad if necessary
        while len(enc_inputs) < self._hps.enc_timesteps:
          enc_inputs.append(pad_id)
        while len(dec_inputs) < self._hps.dec_timesteps:
          dec_inputs.append(end_id)
        while len(targets) < self._hps.dec_timesteps:
          targets.append(end_id)

        element = ModelInput(enc_inputs, dec_inputs, targets, enc_input_len,
                            dec_output_len, article, abstract)

        self._input_data.append(element)

  # def _FillBucketInputQueue(self):
  #   """Fill bucketed batches into the bucket_input_queue."""
  #   while True:
  #     inputs = []
  #     for _ in range(self._hps.batch_size * BUCKET_CACHE_BATCH):
  #       inputs.append(self._input_queue.get())
  #     if self._bucketing:
  #       inputs = sorted(inputs, key=lambda inp: inp.enc_len)
  #
  #     batches = []
  #     for i in range(0, len(inputs), self._hps.batch_size):
  #       batches.append(inputs[i:i+self._hps.batch_size])
  #     shuffle(batches)
  #     for b in batches:
  #       self._bucket_input_queue.put(b)
  #
  # def _WatchThreads(self):
  #   """Watch the daemon input threads and restart if dead."""
  #   while True:
  #     time.sleep(60)
  #     input_threads = []
  #     for t in self._input_threads:
  #       if t.is_alive():
  #         input_threads.append(t)
  #       else:
  #         tf.logging.error('Found input thread dead.')
  #         new_t = Thread(target=self._FillInputQueue)
  #         input_threads.append(new_t)
  #         input_threads[-1].daemon = True
  #         input_threads[-1].start()
  #     self._input_threads = input_threads
  #
  #     bucketing_threads = []
  #     for t in self._bucketing_threads:
  #       if t.is_alive():
  #         bucketing_threads.append(t)
  #       else:
  #         tf.logging.error('Found bucketing thread dead.')
  #         new_t = Thread(target=self._FillBucketInputQueue)
  #         bucketing_threads.append(new_t)
  #         bucketing_threads[-1].daemon = True
  #         bucketing_threads[-1].start()
  #     self._bucketing_threads = bucketing_threads