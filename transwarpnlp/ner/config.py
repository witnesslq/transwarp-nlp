# NER Model Configuration, Set Target Num, and input vocab_Size
class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.1
  max_grad_norm = 10
  num_layers = 2
  num_steps = 30
  hidden_size = 128
  max_epoch = 10
  max_max_epoch = 15
  keep_prob = 1.00    # remember to set to 1.00 when making new prediction
  lr_decay = 1 / 1.15
  batch_size = 10 # single sample batch
  vocab_size = 60000
  nerTags = ["nr", "nrf", "nz", "ns", "nsf", "nt", "t", "nto", "ntc", "o"]
  target_num = len(nerTags) + 1


