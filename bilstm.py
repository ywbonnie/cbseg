from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import param

def createModel(vocab_size, tag_size, max_len, emb_matrix=None):
    input = Input(shape=(max_len,))
    if emb_matrix is None:
        model = Embedding(input_dim=vocab_size, output_dim=param.EMBEDDING_DIMENSION, input_length=max_len)(input)
    else:
        model = Embedding(input_dim=vocab_size, output_dim=param.EMBEDDING_DIMENSION, weights=[emb_matrix], input_length=max_len, trainable=False)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=param.LSTM_UNITS, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(tag_size, activation="softmax"))(model)  # softmax output layer

    model = Model(input, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
