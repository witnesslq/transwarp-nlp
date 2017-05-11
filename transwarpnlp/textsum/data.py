# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data batchers for data described in ..//data_prep/README.md."""

import sys, os
import re
from tensorflow.python.platform import gfile

# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'

_START_VOCAB = [PARAGRAPH_START, PARAGRAPH_END, SENTENCE_START,
                SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, DOCUMENT_START, DOCUMENT_END]

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

pkg_path = os.path.dirname(os.path.dirname(os.getcwd()))

class Vocab(object):
    """Vocabulary class for mapping words and ids."""

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                self._word_to_id[pieces[0]] = self._count
                self._id_to_word[self._count] = pieces[0]
                self._count += 1
                if self._count > max_size:
                    raise ValueError('Too many words: >%d.' % max_size)

    def CheckVocab(self, word):
        if word not in self._word_to_id:
            return None
        return self._word_to_id[word]

    def WordToId(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        return self._count

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f,\
         gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      for line in f:
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

      for w in vocab_list:
        vocab_file.write(w + b"\n")

      vocab_file.flush()
      vocab_file.close()
      f.close()

def getArticlesAndAbstracts(data_path):
    with open(data_path, 'r') as line_f:
        for line in line_f:
            if line != '\n':
                contents = line.split("|")
                yield contents[0], contents[1]

def Pad(ids, pad_id, length):
    """Pad or trim list to len length.
    Args:
      ids: list of ints to pad
      pad_id: what to pad with
      length: length to pad or trim to
    Returns:
      ids trimmed or padded with pad_id
    """
    assert pad_id is not None
    assert length is not None

    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return ids + a
    else:
        return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
    """Get ids corresponding to words in text.
    Assumes tokens separated by space.
    Args:
      text: a string
      vocab: TextVocabularyFile object
      pad_len: int, length to pad to
      pad_id: int, word id for pad symbol
    Returns:
      A list of ints representing word ids.
    """
    ids = []
    for w in text.split():
        i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(UNKNOWN_TOKEN))
    if pad_len is not None:
        return Pad(ids, pad_id, pad_len)
    return ids

if __name__ == "__main__":
    vocab_file = os.path.join(pkg_path, "data/textsum/data/vocab.txt")
    train_file = os.path.join(pkg_path, "data/textsum/data/train.txt")
    create_vocabulary(vocab_file, train_file, 10000)