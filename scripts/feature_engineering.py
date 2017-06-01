# -*- coding: utf-8 -*-
"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
本函数用于抽取问句对的特征，博客中说明特征只用于[Logistic Regression， Xgboost]，未用于深度学习，最好结果为0.81
博客链接：https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
"""

import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize

stop_words = stopwords.words('english')


def wmd(s1, s2):
    # gensim自带计算words move distance的方法，文档分词-->去停词-->计算文档距离
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]  # 是否只由字母组成
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())  # 词向量归一化


data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
data = data.drop(['id', 'qid1', 'qid2'], axis=1)

# 抽取quora queries特征
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))  # 问句1的长度
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))  # 问句2的长度
data['diff_len'] = data.len_q1 - data.len_q2  # 问句1与问句2的长度差距
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))  # 问句1的字符长度（去除空格）
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))  # 问句2的字符串长度（去除空格）
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))  # 问句1的单词个数
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))  # 问句2的单词个数
data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)  # 问句1与问句2的相同单词数（通过集合实现）

# 调用fuzz抽取问句对特征
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((data.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]  # 余弦距离

data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]  # 曼哈顿距离

data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]  # 欧式距离

data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

# 统计的是每一个问句的偏度和峰度
data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]  # 偏度（skewness），是统计数据分布偏斜方向和程度的度量
data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]  # 偏度（skewness），是统计数据分布偏斜方向和程度的度量
data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]  # 峰度（kurtosis），表征概率密度分布曲线在平均值处峰值高低的特征数
data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]  # 峰度（kurtosis），表征概率密度分布曲线在平均值处峰值高低的特征数

cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)

data.to_csv('data/quora_features.csv', index=False)
