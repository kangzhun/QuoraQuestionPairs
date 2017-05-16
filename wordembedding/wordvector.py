# -*- coding: utf-8 -*-

from gensim.models import word2vec


def load_embedding_txt(embedding_file_path):
    model = word2vec.Word2Vec.load_word2vec_format(embedding_file_path)
    # model.save(u"/home/jcc/frame_switch_data/wordembedding/total_vector_retrofit_100d.vec")
    # print model[u'泰国铢']
    similar_list = model.similar_by_word(u"人民币", topn=50)
    for (word, simi) in similar_list:
        print word
        print simi

    similar_list = model.similar_by_word(u"王力宏", topn=50)
    for (word, simi) in similar_list:
        print word
        print simi


def trainEmbedding(model_path, corpus_path, dimension, window, min_count):
    sentences = word2vec.Text8Corpus(corpus_path)
    model = word2vec.Word2Vec(sentences, size=dimension, window=window, min_count=min_count)
    model.save(model_path)


def getSimilarity(model_path):
    model = word2vec.Word2Vec.load(model_path)
    # print model.similarity(u"北京", u"武汉")
    similar_list = model.similar_by_word(u"北京", topn= 50)
    for (word, simi) in similar_list:
        print word
        print simi

if __name__ == '__main__':
    # 训练词向量
    model_path = u"../../data/wordembedding/w2v_100.vec"
    corpus_path = u"../../data/wordembedding/data/wordssets"
    dimension = 100
    window = 8
    min_count = 0
    trainEmbedding(model_path, corpus_path, dimension, window, min_count)

    # 加载词向量并测试
    model_path = u"../../data/wordembedding/w2v_100.vec"
    getSimilarity(model_path)

    load_embedding_txt()
