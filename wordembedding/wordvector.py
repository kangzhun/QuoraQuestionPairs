# -*- coding: utf-8 -*-
import os
from time import strftime, localtime

from gensim.models import word2vec

from config import HERE


def load_embedding(embedding_file_path):
    try:
        model = word2vec.Word2Vec.load(embedding_file_path)
    except Exception, e:
        model = word2vec.Word2Vec.load_word2vec_format(embedding_file_path)
    return model


def train_embedding(model_path, corpus_path, dimension, window, min_count):
    """
    训练词向量
    :param model_path:
    :param corpus_path:
    :param dimension:
    :param window:
    :param min_count:
    :return:
    """
    sentences = word2vec.Text8Corpus(corpus_path)
    model = word2vec.Word2Vec(sentences, size=dimension, window=window, min_count=min_count)
    model.save(model_path)


def get_similarity(model, word):
    """

    :param model:
    :param word: 编码为unicode
    :return:
    """
    similar_list = model.similar_by_word(word, topn=50)
    for (word, simi) in similar_list:
        print word
        print simi

if __name__ == '__main__':
    # 训练词向量
    start_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    model_path = os.path.join(HERE, 'data/chinese_word_embedding', 'word_embedding_%s.vec' % start_time)
    corpus_path = os.path.join(HERE, 'data/corpus/quora')
    dimension = 100
    window = 8
    min_count = 0
    train_embedding(model_path, corpus_path, dimension, window, min_count)

    # 加载词向量并测试
    model = load_embedding(model_path)
    get_similarity(model, u'you')

