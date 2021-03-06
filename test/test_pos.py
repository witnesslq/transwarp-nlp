#coding:utf-8
from __future__ import unicode_literals
import sys,os

from transwarpnlp import segmenter
from transwarpnlp import pos_tagger

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)

tagger = pos_tagger.load_model(pkg_path, 'lstm')

#tagger = pos_tagger.load_model(pkg_path, 'bilstm')


#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text)
print(" ".join(words).encode('utf-8'))

#POS Tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print(str.encode('utf-8'))

#Results
#我/r
#爱/v
#吃/v
#北京/ns
#烤鸭/n
