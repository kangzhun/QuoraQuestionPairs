# -*- coding: utf-8 -*-
import os
from time import strftime, localtime

import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from keras.preprocessing import sequence, text
from keras.utils import np_utils

from config import HERE
from models.bradleypallen_model import QuoraModel

data_path = os.path.join(HERE, "data/quora_duplicate_questions.tsv")
data = pd.read_csv(data_path, sep='\t')
y = data.is_duplicate.values

tk = text.Tokenizer(num_words=200000)
max_len = 40

tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)

embeddings_index = {}
embedding_path = os.path.join(HERE, 'data/glove_word_embedding/glove.840B.300d.txt')
with open(embedding_path, 'r') as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = QuoraModel(embedding_matrix).model
model_name = "bradleypallen_model"
start_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
model_path = os.path.join(HERE, "saved_model", '%s_%s_weights.h5' % (model_name, start_time))
checkpoint = ModelCheckpoint(model_path, monitor='val_acc',
                             save_best_only=True, verbose=2)

model.fit(x=[x1, x2],
          y=y,
          batch_size=384,
          epochs=200,
          verbose=1,
          validation_split=0.1,
          shuffle=True,
          callbacks=[checkpoint])
