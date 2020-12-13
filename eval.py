import sys
import pickle
import numpy as np
#import bilstm
import bilstm_crf
import param
from keras.preprocessing.sequence import pad_sequences

FOLD_ID = int(sys.argv[1])

TEST_FILE_DIR = 'data/'
OUT_FILE_DIR = 'data/'

# 標籤種類，需與訓練時的一致
tags = param.TAGS
n_tags = len(tags)

# 載入之前訓練時的 word2idx
wordf = open("model/word2idx-{0}.pkl".format(FOLD_ID), 'rb')
word2idx = pickle.load(wordf)
n_words = len(word2idx)

embedding_matrix = None

#載入pre-trained word2vec
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
model = bilstm_crf.createModel(n_words, n_tags, param.SENTENCE_MAX_LEN, embedding_matrix) 

# 載入已經訓練好的網路參數權重
model.load_weights("model/bilstm_model-{0}.h5".format(FOLD_ID))

TEST_FILE = TEST_FILE_DIR + "test-{0}.txt".format(FOLD_ID)
OUT_FILE = OUT_FILE_DIR + "out-{0}.txt".format(FOLD_ID)

outf = open(OUT_FILE, 'w', encoding='utf-8')

test_sentences = []
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    temp_segs = []
    for line in f:
        ss = line.strip().split(' ')
        temp_tags = []
        for annot in ss:
            tks = annot.split('/')
            word = tks[0]
            tag = tks[1]
            temp_tags.append( (word, tag) )
        test_sentences.append(temp_tags)
#        if line.strip() == '':
#            test_sentences.append(temp_segs)
#            temp_segs = []
#        else:
#            ss = line.strip().split(' ')
#            if len(ss) < 3:
#                print("ERROR line:{0}".format(line))
#                continue
#            word = ss[0]
#            tag = ss[-1].strip()
#            temp_segs.append( (word, tag) )


for seg_sentence in test_sentences:
    # 將使用者輸入的句子進行填充到固定的長度
    X_test = [[word2idx[w] for w, tag in seg_sentence]]
    X_test = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=X_test, padding="post",value=n_words - 1)

    y_tags = [ tag for w, tag in seg_sentence ]

    # 利用模型進行預測
    p = model.predict(np.array([X_test[0]]), verbose=0)
    p = np.argmax(p, axis=-1) # 取得預測出來機率最大的類別
    count = 0
    for w, pred in zip(X_test[0], p[0]):
        if w == n_words - 1: # 如果已經到達用於填充的 ENDPAD 字元，迴圈跳出
            break
        outf.write("{}\tZH\t{}\t{}\n".format(seg_sentence[count][0],tags[pred], y_tags[count]))
        count += 1
    outf.write("\n")

outf.close()
