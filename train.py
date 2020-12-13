# -*- coding: utf-8 -*- 
import sys
import numpy as np

#import bilstm
import bilstm_crf
import dataload
import param

FOLD_ID = int(sys.argv[1])

#載入訓練與測試資料集
train_sentences, test_sentences, words, tags = dataload.load(FOLD_ID)

n_words = len(words) #詞彙總數

# 這裡的 tags 清單改用手動指定的，比較能夠確認順序
tags = param.TAGS
n_tags = len(tags) #標籤類型總數

#將詞彙與標籤按順序產生index對應
word2idx = { w: i for i, w in enumerate(words) }
tag2idx = { t: i for i, t in enumerate(tags) }

# 把詞彙對應的順序寫入檔案，之後 predict.py 時會需要用到
import pickle
with open("model/word2idx-{0}.pkl".format(FOLD_ID), 'wb') as wordf:
    pickle.dump(word2idx, wordf)

# 產生Deep Learning用的訓練與測試資料矩陣
# 目前採用固定長度序列，所以每一句若小於200個字，都要填充到200長度，採用Keras提供的 pad_sequences 函式進行填充
from keras.preprocessing.sequence import pad_sequences
X_train = [[word2idx[w[0]] for w in s] for s in train_sentences]
X_train = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=X_train, padding="post",value=n_words - 1)

y_train = [[tag2idx[w[1]] for w in s] for s in train_sentences]
y_train = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=y_train, padding="post", value=tag2idx["s"])

# 將輸出標籤從數字轉換為一個 one-hot 向量，例：
# 0 --> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 --> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 以此類推，採用Keras提供的 to_categorical 函式進行轉換
from keras.utils import to_categorical
y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]

X_test = [[word2idx[w[0]] for w in s] for s in test_sentences]
X_test = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=X_test, padding="post",value=n_words - 1)

y_test = [[tag2idx[w[1]] for w in s] for s in test_sentences]
y_test = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=y_test, padding="post", value=tag2idx["s"])
y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

embedding_matrix = None
# 如果要使用預訓練的詞向量，載入pre-trained word2vec
if param.USE_PRETRAINED_EMBEDDING:
    from gensim.models import word2vec
    from gensim import models

    w2v_model = models.Word2Vec.load("data/word2vec.taisho.dim100.model")
    embedding_matrix = np.zeros((n_words, param.EMBEDDING_DIMENSION))
    for w, i in word2idx.items():
        if w in w2v_model:
            embedding_vector = w2v_model[w]
            embedding_matrix[i] = embedding_vector

# 建立雙向 LSTM 模型
#model = bilstm.createModel(n_words, n_tags, param.SENTENCE_MAX_LEN, embedding_matrix) 
# 建立雙向 LSTM-CRF 模型
model = bilstm_crf.createModel(n_words, n_tags, param.SENTENCE_MAX_LEN, embedding_matrix) 

model.summary()

# 開始訓練
history = model.fit(X_train, np.array(y_train), batch_size=param.BATCH_SIZE, epochs=param.EPOCHS, validation_split=0.2, verbose=1)

# 將訓練好的模型存檔
model.save("model/bilstm_model-{0}.h5".format(FOLD_ID))

# 用測試資料集進行測試，估測accuracy
p = model.evaluate(X_test, np.array(y_test), verbose=1)
for i in range(len(model.metrics_names)):
    print("{0}:{1}".format(model.metrics_names[i], p[i]))

