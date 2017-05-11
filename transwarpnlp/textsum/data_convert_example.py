import struct
import sys

from tensorflow.core.example import example_pb2

def _binary_to_text(in_file, out_file):
  reader = open(in_file, 'rb')
  writer = open(out_file, 'w')
  while True:
    len_bytes = reader.read(8)
    if not len_bytes:
      sys.stderr.write('Done reading\n')
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    examples = []
    for key in tf_example.features.feature:
      examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
    writer.write('%s\n' % '\t'.join(examples))
  reader.close()
  writer.close()


def _text_to_binary(in_file, out_file):
  inputs = open(in_file, 'r').readlines()
  writer = open(out_file, 'wb')
  for inp in inputs:
    tf_example = example_pb2.Example()
    for feature in inp.strip().split('\t'):
      (k, v) = feature.split('=')
      tf_example.features.feature[k].bytes_list.value.extend([v])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
  writer.close()


if __name__ == "__main__":
    in_file = '/Users/endy/nlp/transwarp-nlp/data/textsum/data/data'
    out_file = '/Users/endy/nlp/transwarp-nlp/data/textsum/data/data-text'
    _binary_to_text(in_file, out_file)
