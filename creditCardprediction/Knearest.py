"""
author:yqtong@buaa.edu.cn
date:2022-05-05
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def create_dataset(train_file_name, test_file_name):
    train_pd = pd.read_csv(train_file_name)
    test_pd = pd.read_csv(test_file_name)
    # 直接用数据集原始特征就可以吗？是否需要采样，再次预处理？
    # 所有的特征都一定对分类有用吗？会不会有负面影响？
    X_train = train_pd.iloc[:, train_pd.columns != 'Class']
    X_test = test_pd.iloc[:, test_pd.columns != 'Class']
    y_train = train_pd.iloc[:, train_pd.columns == 'Class']
    y_test = test_pd.iloc[:, test_pd.columns == 'Class']
    return np.array(X_train), np.array(y_train).reshape(-1), np.array(X_test), np.array(y_test).reshape(-1)


def train():
    train_file_name = 'train_creditcard.csv'
    test_file_name = 'test_creditcard.csv'
    X_train, y_train, X_test, y_test = create_dataset(train_file_name, test_file_name)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # 用准确率是否会有问题？
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print('accuracy: {}%'.format(accuracy*100))


if __name__ == '__main__':
    train()




