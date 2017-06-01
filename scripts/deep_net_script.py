# -*- coding: utf-8 -*-
import os
from time import strftime, localtime

import logging
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from keras.preprocessing import sequence, text
from keras.utils import np_utils

from config import HERE
from models.abhishekkrthakur_deep_net import DeepNet
from scripts.custom_callback import ConfusionMatrixCallback
from utils.setup_logger import setup_logging

# 配置日志文件
from wordembedding.embedding import H5EmbeddingManager

logger_config_path = os.path.join(HERE, 'scripts/configs/logging.yaml')
setup_logging(default_path=logger_config_path, default_level=logging.DEBUG, add_time_stamp=False)
logger = logging.getLogger(__name__)

source_path = os.path.join(HERE, 'data/quora_duplicate_questions.tsv')
data = pd.read_csv(source_path, sep='\t')
y = data.is_duplicate.values

tk = text.Tokenizer(num_words=200000)
max_len = 40

tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tk.word_index

y_train_enc = np_utils.to_categorical(y)

# 载入词向量
word_embedding_h5_path = os.path.join(HERE, 'data/glove_word_embedding/glove.840B.300d.txt.h5')
word_embedding_path = os.path.join(HERE, 'data/glove_word_embedding/glove.840B.300d.txt')
if os.path.exists(word_embedding_h5_path):
    logger.debug('load from h5 file, path=%s', word_embedding_h5_path)
    embeddings_index = H5EmbeddingManager(word_embedding_h5_path, mode='in-memory')
else:
    logger.debug('load from txt file, path=%s', word_embedding_path)
    embeddings_index = {}
    with open(word_embedding_path, 'r') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    logger.debug('Found %s word vectors.', len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    try:
        embedding_vector = embeddings_index[word]
    except Exception, e:
        embedding_vector = None
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

deep_net = DeepNet(embedding_matrix)
model = deep_net.create_deepnet()

model_name = "deep_net"
start_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
model_path = os.path.join(HERE, "saved_model", '%s_%s_weights.h5' % (model_name, start_time))
checkpoint = ModelCheckpoint(model_path, monitor='val_acc',
                             save_best_only=True, verbose=2)
# confusion_matrix = ConfusionMatrixCallback(x=[x1, x2], y=y)

model.fit(x=[x1, x2], y=y, batch_size=384, epochs=200, verbose=1,
          validation_split=0.1, shuffle=True, callbacks=[checkpoint])
