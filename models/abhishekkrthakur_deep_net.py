# -*- coding: utf-8 -*-

from keras.engine import Input, Model
from keras.layers import Embedding, Dense, Lambda, Convolution1D, Dropout, \
    GlobalMaxPooling1D, BatchNormalization, LSTM, Merge, PReLU
from keras import backend as k

EMBEDDING_DIM = 300
filter_length = 5
nb_filter = 64
pool_length = 4
max_len = 40


class DeepNet(object):
    def __init__(self, embedding_matrix):
        self.model = self.creat_deepnet(embedding_matrix)

    def creat_deepnet(self, embedding_matrix):
        input_query_1 = Input(shape=(None,), name='input_query_1', dtype='int32')
        input_query_2 = Input(shape=(None,), name='input_query_2', dtype='int32')

        nb_words = len(embedding_matrix)
        shared_embedding = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],
                                     input_length=40, trainable=False)

        q1_embedding = shared_embedding(input_query_1)
        q1_dense = (Dense(EMBEDDING_DIM, activation='relu'))(q1_embedding)
        q1_lambda = Lambda(lambda x: k.sum(x, axis=1), output_shape=(EMBEDDING_DIM,))(q1_dense)

        q2_embedding = shared_embedding(input_query_2)
        q2_dense = (Dense(EMBEDDING_DIM, activation='relu'))(q2_embedding)
        q2_lambda = Lambda(lambda x: k.sum(x, axis=1), output_shape=(EMBEDDING_DIM,))(q2_dense)

        q3_embedding = shared_embedding(input_query_1)
        q3_conv_1d_1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid',
                                     activation='relu', subsample_length=1)(q3_embedding)
        q3_dropout_1 = Dropout(0.2)(q3_conv_1d_1)
        q3_conv_1d_2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid',
                                     activation='relu', subsample_length=1)(q3_dropout_1)
        q3_global_pooling_1d_1 = GlobalMaxPooling1D()(q3_conv_1d_2)
        q3_dropout_2 = Dropout(0.2)(q3_global_pooling_1d_1)
        q3_dense = Dense(EMBEDDING_DIM)(q3_dropout_2)
        q3_dropout_3 = Dropout(0.2)(q3_dense)
        q3_bn = BatchNormalization()(q3_dropout_3)

        q4_embedding = shared_embedding(input_query_2)
        q4_conv_1d_1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid',
                                     activation='relu', subsample_length=1)(q4_embedding)
        q4_dropout_1 = Dropout(0.2)(q4_conv_1d_1)
        q4_conv_1d_2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid',
                                     activation='relu', subsample_length=1)(q4_dropout_1)
        q4_global_pooling_1d_1 = GlobalMaxPooling1D()(q4_conv_1d_2)
        q4_dropout_2 = Dropout(0.2)(q4_global_pooling_1d_1)
        q4_dense = Dense(EMBEDDING_DIM)(q4_dropout_2)
        q4_dropout_3 = Dropout(0.2)(q4_dense)
        q4_bn = BatchNormalization()(q4_dropout_3)

        q5_embedding = shared_embedding(input_query_1)
        q5_lstm = LSTM(EMBEDDING_DIM, dropout_W=0.2, dropout_U=0.2)(q5_embedding)

        q6_embedding = shared_embedding(input_query_2)
        q6_lstm = LSTM(EMBEDDING_DIM, dropout_W=0.2, dropout_U=0.2)(q6_embedding)

        merge_layer = Merge(mode='concat')([q1_lambda, q2_lambda, q3_bn, q4_bn, q5_lstm, q6_lstm])
        bn_layer_1 = BatchNormalization()(merge_layer)
        dense_layer_1 = Dense(EMBEDDING_DIM)(bn_layer_1)
        prelu_layer_1 = PReLU()(dense_layer_1)
        dropout_layer_1 = Dropout(0.2)(prelu_layer_1)

        bn_layer_2 = BatchNormalization()(dropout_layer_1)
        dense_layer_2 = Dense(EMBEDDING_DIM)(bn_layer_2)
        prelu_layer_2 = PReLU()(dense_layer_2)
        dropout_layer_2 = Dropout(0.2)(prelu_layer_2)

        bn_layer_3 = BatchNormalization()(dropout_layer_2)
        dense_layer_3 = Dense(EMBEDDING_DIM)(bn_layer_3)
        prelu_layer_3 = PReLU()(dense_layer_3)
        dropout_layer_3 = Dropout(0.2)(prelu_layer_3)

        bn_layer_4 = BatchNormalization()(dropout_layer_3)
        dense_layer_4 = Dense(EMBEDDING_DIM)(bn_layer_4)
        prelu_layer_4 = PReLU()(dense_layer_4)
        dropout_layer_4 = Dropout(0.2)(prelu_layer_4)

        bn_layer_5 = BatchNormalization()(dropout_layer_4)
        dense_layer_5 = Dense(EMBEDDING_DIM)(bn_layer_5)
        prelu_layer_5 = PReLU()(dense_layer_5)
        dropout_layer_5 = Dropout(0.2)(prelu_layer_5)

        output = Dense(1, activation='sigmoid')(dropout_layer_5)

        model = Model(input=[input_query_1, input_query_2], output=output)
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        return model
