"""
author:yqtong@buaa.edu.cn
data:2022-05-04
构建模型前首先需要对数据整体情况有个了解
"""
import json
import matplotlib.pyplot as plt


def plot_train_label_bar(train_label_dict):
    """
    :param train_label_dict:
    :return:
    """
    label_list, num_list = [], []
    for label, num in train_label_dict:
        label_list.append(label)
        num_list.append(num)
    plt.figure(figsize=(30, 15))
    plt.xticks(rotation=20, fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(label_list, num_list, width=0.5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def plot_feature_bar(feature_dict):
    """
    feature 按0~10, 10~20, ..., >100来作为x轴
    :param feature_dict:
    :return:
    """
    x_list = ['0~9', '10~19', '20~29', '30~39', '40~49', '50~59', '60~69', '70~79', '80~89', '90~99', '>100']
    y_list = [0 for x in range(11)]
    for feature, num in feature_dict:
        if 0 <= num <= 9:
            y_list[0] += 1
        elif 10 <= num <= 19:
            y_list[1] += 1
        elif 20 <= num <= 29:
            y_list[2] += 1
        elif 30 <= num <= 39:
            y_list[3] += 1
        elif 40 <= num <= 49:
            y_list[4] += 1
        elif 50 <= num <= 59:
            y_list[5] += 1
        elif 60 <= num <= 69:
            y_list[6] += 1
        elif 70 <= num <= 79:
            y_list[7] += 1
        elif 80 <= num <= 89:
            y_list[8] += 1
        elif 90 <= num <= 99:
            y_list[9] += 1
        else:
            y_list[10] += 1
    plt.figure(figsize=(30, 15))
    plt.xticks(rotation=20, fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(x_list, y_list, width=0.5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def eval_json(train_file_path, test_file_path):
    """
    :param train_file_path:
    :param test_file_path:
    :return:
    """
    # 1. 样本数量有多少?
    with open(train_file_path, 'r', encoding='utf-8') as fa:
        train_file = json.load(fa)
    fa.close()
    print('num of train examples:', len(train_file))
    with open(test_file_path, 'r', encoding='utf-8') as fa:
        test_file = json.load(fa)
    fa.close()
    print('num of test examples:', len(test_file))
    # 2. 有哪些标签, 特征, 数量各是多少, 平均数量又是多少
    train_label_set, train_feature_set = set(), set()
    total_num_of_train_features, total_num_of_test_features = 0, 0
    test_feature_set = set()
    for idx, (key, value) in enumerate(train_file.items()):
        train_label_set.add(value['cuisine'])
        for f in value['ingredients']:
            train_feature_set.add(f)
            total_num_of_train_features += 1
    print('num of uniq train labels:', len(train_label_set))
    print('num of uniq train feature:', len(train_feature_set))
    print('train label details:', train_label_set)
    for idx, (key, value) in enumerate(test_file.items()):
        for f in value['ingredients']:
            test_feature_set.add(f)
            total_num_of_test_features += 1
    print('num of test feature:', len(test_feature_set))
    print('avg num of train features for each example:', total_num_of_train_features / len(train_file))
    print('avg num of test features for each example:', total_num_of_test_features / len(test_file))
    # 3. 训练集和测试集特征的交集、并集以及训练集特有的特征和测试集特有的特征
    print('features that appear only in the train set:', len(train_feature_set - test_feature_set))
    print('features that appear only in the test set:', len(test_feature_set - train_feature_set))
    print('total features in training and test sets:', len(train_feature_set | test_feature_set))
    print('features that appear both in both training and test sets:', len(train_feature_set & test_feature_set))
    # 3. 训练集特征和标签分布情况
    train_feature_dict, train_label_dict = dict(), dict()
    for idx, (key, value) in enumerate(train_file.items()):
        if train_label_dict.__contains__(value['cuisine']):
            train_label_dict[value['cuisine']] += 1
        else:
            train_label_dict[value['cuisine']] = 1
        for f in value['ingredients']:
            if train_feature_dict.__contains__(f):
                train_feature_dict[f] += 1
            else:
                train_feature_dict[f] = 1
    train_label_dict = sorted(train_label_dict.items(), key=lambda x: x[1], reverse=True)
    train_feature_dict = sorted(train_feature_dict.items(), key=lambda x: x[1], reverse=True)
    print('train label_dict:', train_label_dict)
    print('top 5 train features:', train_feature_dict[:5])
    print('last 5 train features:', train_feature_dict[-5:])
    # 4. 测试集特征分布情况
    test_feature_dict = dict()
    for idx, (key, value) in enumerate(test_file.items()):
        for f in value['ingredients']:
            if test_feature_dict.__contains__(f):
                test_feature_dict[f] += 1
            else:
                test_feature_dict[f] = 1
    test_feature_dict = sorted(test_feature_dict.items(), key=lambda x: x[1], reverse=True)
    print('top 5 test features:', test_feature_dict[:5])
    print('last 5 test features:', test_feature_dict[-5:])
    # 5. 可视化label分布
    plot_train_label_bar(train_label_dict)
    # 6. 可视化feature分布
    plot_feature_bar(train_feature_dict)
    plot_feature_bar(test_feature_dict)


if __name__ == '__main__':
    print('***Details of Food examples***')
    eval_json('trainraw.json', 'testraw.json')