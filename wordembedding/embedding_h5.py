# encoding: utf-8
import zipfile
from itertools import izip

import numpy as np
import h5py
from gensim.models import word2vec
import pandas as pd
import csv

from config import ACTIVE_WORD2VEC_PATH
from wordembedding.embedding import H5EmbeddingManager

FUNCTION_WORDS = ['PADDING', 'OOV_WORD']


def export_data_h5(vocabulary, embedding_matrix, output='embedding.h5'):
    f = h5py.File(output, "w")

    compress_option = dict(compression="gzip", compression_opts=9, shuffle=True)
    words_flatten = '\n'.join(vocabulary)
    f.attrs['vocab_len'] = len(vocabulary)
    print len(vocabulary)
    dt = h5py.special_dtype(vlen=str)
    _dset_vocab = f.create_dataset('words_flatten', (1, ), dtype=dt, **compress_option)
    _dset_vocab[...] = [words_flatten]
    _dset = f.create_dataset('embedding', embedding_matrix.shape, dtype=embedding_matrix.dtype, **compress_option)
    _dset[...] = embedding_matrix
    f.flush()
    f.close()


def glove_export(embedding_file):
    with zipfile.ZipFile(embedding_file) as zf:
        for name in zf.namelist():
            vocabulary = []
            embeddings = []
            with zf.open(name) as f:
                for line in f:
                    vals = line.split(' ')
                    vocabulary.append(vals[0])
                    embeddings.append([float(x) for x in vals[1:]])
            print(set(map(len, embeddings)))
            export_data_h5(vocabulary, np.array(embeddings, dtype=np.float32), output=name + ".h5")


def w2v_export(embedding_file):
    model = word2vec.Word2Vec.load_word2vec_format(embedding_file)
    vocabulary = model.vocab
    embeddings = []
    for word in vocabulary:
        embeddings.append(model[word])
    export_data_h5(vocabulary, np.array(embeddings, dtype=np.float32), output=embedding_file + ".h5")

# 保存新的词向量文件为h5格式
def tsv_export(embedding_file):
    df = pd.read_csv(embedding_file,sep='\t',quoting=csv.QUOTE_NONE,encoding='utf-8',header=None)
    vocabulary = df.loc[:,0].values
    embeddings = df.loc[:,1:].values
    export_data_h5(vocabulary, np.array(embeddings, dtype=np.float32), output=embedding_file + ".h5")



def normalize_word(embedding_file):
    # 将词向量进行规范化
    model = word2vec.Word2Vec.load_word2vec_format(embedding_file)
    vocabulary = model.vocab
    embeddings = []
    distances = np.zeros(100)
    import math
    for word in vocabulary:
        for i in range(100):
            distances[i] += math.pow(model[word][i], 2)

    for i in range(100):
        distances[i] = math.sqrt(distances[i])

    for word in vocabulary:
        vec = model[word]
        for i in range(100):
            vec[i] = vec[i] / distances[i]
        embeddings.append(vec)
    export_data_h5(vocabulary, np.array(embeddings, dtype=np.float32), output=embedding_file + "_normalize.h5")



#if __name__ == '__main__':

    #tsvfile = "../data/word_embedding/w2v.tsv"
    #tsvfile = "../data/word_embedding/swivel_row.tsv"
    #tsvfile = "../data/word_embedding/swivel_col.tsv"
    #tsv_export(tsvfile)
