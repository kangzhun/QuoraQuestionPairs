# -*- coding: utf-8 -*-
from keras.engine import Input, Model
from keras.layers import Embedding, TimeDistributed, Dense, Lambda, Merge, BatchNormalization
from keras import backend as K

EMBEDDING_DIM = 300


class QuoraModel(object):
    def __init__(self, embedding_matrix):
        self.model = self.create_model(embedding_matrix)

    def create_model(self, embedding_matrix):
        input_query_1 = Input(shape=(None,), name='input_query_1', dtype='int32')
        input_query_2 = Input(shape=(None,), name='input_query_2', dtype='int32')

        nb_words = len(embedding_matrix)
        shared_embedding = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],
                                     input_length=40, trainable=False)
        q1_embedding_layer = shared_embedding(input_query_1)
        q1_time_distributed_layer = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1_embedding_layer)
        q1_lambda_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1_time_distributed_layer)

        q2_embedding_layer = shared_embedding(input_query_2)
        q2_time_distributed_layer = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2_embedding_layer)
        q2_lambda_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,))(q2_time_distributed_layer)

        merge_layer = Merge(mode='concat')([q1_lambda_layer, q2_lambda_layer])
        bn_layer_1 = BatchNormalization()(merge_layer)
        dense_layer_1 = Dense(200, activation='relu')(bn_layer_1)
        bn_layer_2 = BatchNormalization()(dense_layer_1)
        dense_layer_2 = Dense(200, activation='relu')(bn_layer_2)
        bn_layer_3 = BatchNormalization()(dense_layer_2)
        dense_layer_3 = Dense(200, activation='relu')(bn_layer_3)
        bn_layer_4 = BatchNormalization()(dense_layer_3)
        dense_layer_4 = Dense(200, activation='relu')(bn_layer_4)
        output = Dense(1, activation='sigmoid')(dense_layer_4)

        model = Model(inputs=[input_query_1, input_query_2], outputs=output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

