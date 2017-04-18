NER (Named Entity Recognition)
==============================
命名实体识别

Train your model
--------------------
自己训练模型

###NER model
#### Prepare corpus the same way as POS
#### Put data files in folder ../transwarpnlp/ner/data/'your_language_code'
#### Running script
```python
python ner_model.py zh # training Chinese model
```
#### The trained model can be found under folder ../transwarpnlp/ner/ckpt/'your_language_code'

