# -*- coding:utf-8 -*-

class Config(object):
    article_key = 'article'
    abstract_key = 'headline'
    max_run_steps = 1000
    max_article_sentences = 2
    max_abstract_sentences = 100
    beam_size = 4
    eval_interval_secs = 60
    checkpoint_secs = 60
    use_bucketing = False
    truncate_input = False
    num_gpus = 0
    random_seed = 111