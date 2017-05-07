# -*- coding: utf-8 -*-

import os, time
from transwarpnlp.joint_seg_tagger.config import Config
from transwarpnlp.dataprocess import joint_data_transform
from transwarpnlp.joint_seg_tagger.model import Model
import tensorflow as tf

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = Config()

def test_joint(data_path):
    test_file = "/data/test.txt"
    