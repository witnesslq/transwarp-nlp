# coding:utf-8

import pandas as pd
import os, sys
import numpy as np

pkg_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pkg_path)

def getSmallDataset(source_file, target_dir):
    data = pd.read_csv(source_file, sep="||").sample(20000)
    train, validate, test = np.split(data.sample(frac=1), [int(.8 * len(data)), int(.9 * len(data))])

    train.to_csv(target_dir + "/train.txt", header=False, index=False, quotechar=" ")
    validate.to_csv(target_dir + "/dev.txt", header=False, index=False, quotechar=" ")
    test.to_csv(target_dir + "/test.txt", header=False, index=False, quotechar=" ")

def getFullDataset(source_file, target_dir):
    data = pd.read_csv(source_file)
    train, validate, test = np.split(data.sample(frac=1), [int(.8 * len(data)), int(.9 * len(data))])
    train.to_csv(target_dir + "/train.txt", header=False, index=False, quotechar=" ")
    validate.to_csv(target_dir + "/dev.txt", header=False, index=False, quotechar=" ")
    test.to_csv(target_dir + "/test.txt", header=False, index=False, quotechar=" ")

if __name__ == "__main__":
    ner_source_file = os.path.join(pkg_path, "data/source/ner_data.txt")
    ner_target_dir = os.path.join(pkg_path, "data/ner/data")

    getSmallDataset(ner_source_file, ner_target_dir)

    pos_source_file = os.path.join(pkg_path, "data/source/pos_data.txt")
    pos_target_dir = os.path.join(pkg_path, "data/pos/data")
    getSmallDataset(pos_source_file, pos_target_dir)


