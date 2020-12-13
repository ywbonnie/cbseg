import param

TRAIN_FILE_DIR = "data/"
TEST_FILE_DIR = "data/"

def load(fold_index):
    train_sentences = []
    test_sentences = []
    words = []
    tags = []

    TRAIN_FILE = TRAIN_FILE_DIR + "train-{0}.txt".format(fold_index)
    TEST_FILE = TEST_FILE_DIR + "test-{0}.txt".format(fold_index)
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            temp_segs = []
            ss = line.strip().split(' ')
            for tok in ss:
                segs = tok.split('/')
                word = segs[0].strip()
                tag = segs[1].strip()
                if word not in words:
                    words.append(word)
                if tag not in tags:
                    tags.append(tag)
                temp_segs.append( (word, tag) )
            train_sentences.append(temp_segs)
    with open(TEST_FILE, 'r') as f:
        for line in f:
            temp_segs = []
            ss = line.strip().split(' ')
            for tok in ss:
                segs = tok.split('/')
                word = segs[0].strip()
                tag = segs[1].strip()
                if word not in words:
                    words.append(word)
                if tag not in tags:
                    tags.append(tag)
                temp_segs.append( (word, tag) )
            test_sentences.append(temp_segs)
#    with open(TRAIN_FILE, 'rb') as f:
#        temp_segs = []
#        for line in f:
#            line = line.decode('utf-8', 'ignore')
#            line = line.strip()
#            if line == '':
#                train_sentences.append(temp_segs)
#                temp_segs = []
#            else:
#                ss = line.split(' ')
#                word = ss[0].strip()
#                tag = ss[1].strip()
#                if word not in words:
#                    words.append(word)
#                if tag not in tags:
#                    tags.append(tag)
#                temp_segs.append( (word, tag) )
#    with open(TEST_FILE, 'rb') as f:
#        temp_segs = []
#        for line in f:
#            line = line.decode('utf-8', 'ignore')
#            line = line.strip()
#            if line == '':
#                test_sentences.append(temp_segs)
#                temp_segs = []
#            else:
#                ss = line.split(' ')
#                word = ss[0].strip()
#                tag = ss[1].strip()
#                if word not in words:
#                    words.append(word)
#                temp_segs.append( (word, tag) )

    # 過濾掉長度大於200個字的句子
    train_sentences = list(filter(lambda x: len(x) < param.SENTENCE_MAX_LEN, train_sentences))
    test_sentences = list(filter(lambda x: len(x) < param.SENTENCE_MAX_LEN, test_sentences))
    #將vocabulary裡面加上用來填充的單字 ENDPAD
    words.append('ENDPAD')

    return train_sentences, test_sentences, words, tags

    
