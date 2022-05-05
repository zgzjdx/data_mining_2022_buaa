"""
author:yqtong@buaa.edu.cn
date:2022-05-05
"""
import pandas as pd
import numpy as np


def csv_reader(file_name):
    data = pd.read_csv(file_name)
    print('Total num of data:', len(data))
    keys = [x for x in data]
    print('Total type of features:', keys)
    # calculate feature distribution
    print('*****************************')
    for idx, key in enumerate(keys[1:-1]):
        key_list = [row[key] for idy, row in data.iterrows()]
        print('Min of {}: {}'.format(key, min(key_list)))
        print('Max of {}: {}'.format(key, max(key_list)))
        print('Mean of {}: {}'.format(key, sum(key_list) / len(key_list)))
        print('Median of {}: {}'.format(key, np.median(key_list)))
        print('Var of {}: {}'.format(key, np.var(key_list)))
        print('Std of {}: {}'.format(key, np.std(key_list)))
        print('*****************************')
    # 可以发现什么结论？如何从几十万的数据中发现异常值（噪声）？
    # calculate label distribution
    label_list = [row['Class'] for idx, row in data.iterrows()]
    print('Total num of label:', len(label_list))
    label_set = set(label_list)
    print('Label set:', label_set)
    print('Total type of label:', len(label_set))
    for label in label_set:
        print('Total num of {}: {}'.format(label, label_list.count(label)))
    # 可以发现什么结论？根据这种标签分布该如何取metrics？
    # 可视化部分略


if __name__ == '__main__':
    train_file_name = 'train_creditcard.csv'
    print('**************Training data statistics**************')
    csv_reader(train_file_name)
    test_file_name = 'test_creditcard.csv'
    print('**************Test data statistics**************')
    csv_reader(train_file_name)