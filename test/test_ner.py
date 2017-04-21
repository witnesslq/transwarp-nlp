#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode

from transwarpnlp import segmenter
from transwarpnlp import ner_tagger
import os, sys

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)

tagger = ner_tagger.load_model(pkg_path, "lstm")

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text)
print (" ".join(words).encode('utf-8'))

#NER tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

#Results
#我/nt
#爱/nt
#吃/nt
#北京/p
#烤鸭/nt
