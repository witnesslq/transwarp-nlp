# coding=utf-8
from __future__ import unicode_literals

import os
from transwarpnlp import segmenter

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def segment(all_the_text):
    re = ""
    relist = ""
    words = segmenter.seg(all_the_text)
    count = 0
    for w in words:

        if len(w) > 1 and w >= u'/u4e00' and w <= u'\u9fa5':
            re = re + " " + w
            count = count + 1
        if count % 100 == 0:
            print(re)
            re = re.replace("\n", " ")
            relist = relist + "\n" + re
            re = ""
            count = count + 1
    re = re.replace("\n", " ").replace("\r", " ")
    if len(relist) > 1 and len(re) > 40:
        relist = relist + "\n" + re
    elif len(re) > 40:
        relist = re
    relist = relist + "\n"
    relist = relist.replace("\r\n", "\n").replace("\n\n", "\n")

    return relist


def handleTrainData(input_path, output_file):
    fw = open(output_file, "a")
    for filename in os.listdir(input_path):
        print(filename)
        file_object = open(input_path + "/" + filename)
        try:
            all_the_text = file_object.read()
            all_the_text = all_the_text.decode("utf-8")
            pre_text = segment(all_the_text)
            if len(pre_text) > 30:
                fw.write(pre_text.encode("utf-8"))
        except Exception:
            print(Exception.message)
        finally:
            file_object.close()

if __name__ == "__main__":
    # input_path = os.path.join(os.path.dirname(pkg_path), "data/source/sogo/C000008")
    # output_file = os.path.join(os.path.dirname(pkg_path), "data/source/sogo", "C000008.txt")
    # handleTrainData(input_path, output_file)

    input_path = os.path.join(os.path.dirname(pkg_path), "data/source/sogo/C000010")
    output_file = os.path.join(os.path.dirname(pkg_path), "data/source/sogo", "C000010.txt")
    handleTrainData(input_path, output_file)

