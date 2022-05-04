"""
author:yqtong@buaa.edu.cn
date:2022-05-04
利用random_forest对菜品名进行分类预测
"""
import json

import sklearn.pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def load_stop_words(file_path):
    """
    :param file_path:
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as fa:
        line_list = fa.readlines()
    cleaned_line_list = []
    for line in line_list:
        cleaned_line_list.append(line.strip())
    return cleaned_line_list


def stringify_features(feature_list):
    """
    :param feature_list:
    :return:
    """
    return ', '.join(feature_list)


def write_pred(y_list, id_list):
    """
    :param y_list:
    :param id_list:
    :return:
    """
    with open('submission.txt', 'w', encoding='utf-8') as fa:
        for idx in range(len(y_list)):
            fa.write(id_list[idx] + ' ' + y_list[idx] + '\n')


def train_and_eval(train_file_path, test_file_path):
    """
    :param train_file_path:
    :param test_file_path:
    :return:
    """
    with open(train_file_path, 'r', encoding='utf-8') as fa:
        with open(test_file_path, 'r', encoding='utf-8') as fb:
            train_dict = json.load(fa)
            test_dict = json.load(fb)
    fa.close()
    fb.close()
    # 1. 将特征向量化, 有哪些方法? 孰优孰劣?
    train_label_list = []
    train_feature_list, test_feature_list = [], []
    test_id_list = []
    for idx, (key, value) in enumerate(train_dict.items()):
        train_label_list.append(value['cuisine'])
        train_feature_list.append(value['ingredients'])
    for idx, (key, value) in enumerate(test_dict.items()):
        test_feature_list.append(value['ingredients'])
        test_id_list.append(key)
    print(train_feature_list[:5])
    print(train_label_list[:5])
    print(test_id_list[:5])
    vectorizer = TfidfVectorizer(
        preprocessor=stringify_features,
        stop_words=load_stop_words('english_stop_word.txt'),
    )
    vectorizer.fit(np.concatenate([train_feature_list, test_feature_list], dtype=object))
    print('Num of features:', len(vectorizer.get_feature_names()))
    # 2. 分类器训练, 不同的参数会有什么样的效果?如果进行对比实验?
    classifier = MultinomialNB(
        alpha=0.03
    )
    model = sklearn.pipeline.Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])
    model.fit(train_feature_list, train_label_list)
    # 3. 预测 submission type id + ' ' + label
    y_pred = model.predict(test_feature_list)
    print(len(y_pred))
    write_pred(y_pred, test_id_list)


if __name__ == '__main__':
    train_and_eval('trainraw.json', 'testraw.json')