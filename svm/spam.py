# coding: utf8
# svm/spam.py

"""垃圾邮件分类器
"""

import numpy as np
from stemming.porter2 import stem
from pydash import py_

# 获得词汇表
vocabList = []
with open('vocab.txt') as f:
    for line in f:
        idx, w = line.split()
        vocabList.append(w)


def processEmail(email):
    """预处理邮件

    Args:
        email 邮件内容
    Returns:
        indices 单词在词表中的位置
    """
    # 转换为小写 --> 标准化 URL --> 标准化 邮箱地址
    # --> 去除 HTML 标签 --> 标准化数字
    # --> 标准化美元 --> 删除非空格字符
    return py_(email) \
        .strip_tags() \
        .reg_exp_replace(r'(http|https)://[^\s]*', 'httpaddr') \
        .reg_exp_replace(r'[^\s]+@[^\s]+', 'emailaddr') \
        .reg_exp_replace(r'\d+', 'number') \
        .reg_exp_replace(r'[$]+', 'dollar') \
        .lower_case() \
        .trim() \
        .words() \
        .map(stem) \
        .map(lambda word : py_.index_of(vocabList, word) + 1) \
        .value()

def extractFeatures(indices):
    """提取特征

    Args:
        indices 单词索引
    Returns:
        feature 邮件特征
    """
    feature = py_.map(range(1, len(vocabList) + 1),
                      lambda index: py_.index_of(indices, index) > -1)
    return np.array(feature, dtype=np.uint)

def getTopPredictors(weights, count):
    """获得最佳标识词汇

    Args:
        weights 权值
        count top count
    Returns:
        predictors predicators
    """
    return py_(vocabList) \
        .map(lambda word, idx: (word, weights[idx])) \
        .sort_by(lambda item: item[1], reverse = True) \
        .take(count) \
        .value()
