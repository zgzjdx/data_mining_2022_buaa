"""
author:yqtong@buaa.edu.cn
date:2022-05-03
"""
import os

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import metrics
import cv2


def get_training_dirs(training_dir_path):
    """
    [root+dir_name1, root+dir_name2, ...]
    :param training_dir_path:
    :return:
    """
    return[x[0] for x in os.walk(training_dir_path)][1:]


def get_training_labels(training_dir_path):
    return[x[1] for x in os.walk(training_dir_path)][0]


def get_each_labels_files(training_dir_path):
    return [x[2] for x in os.walk(training_dir_path)][1:]


def get_label_encoder(labels):
    return LabelEncoder().fit(labels)


def image_normalization(image: np.ndarray) -> np.ndarray:
    mean = np.mean(image)
    var = np.var(image)
    return (image - mean) / np.sqrt(var)


def create_recognition_dataset(training_image_path, labelEncoder):
    X, Y = [], []
    dirs = get_training_dirs(training_image_path)
    images = get_each_labels_files(training_image_path)
    labels = get_training_labels(training_image_path)
    for idx, item in enumerate(zip(dirs, images, labels)):
        label = labelEncoder.transform([item[-1]])[0]
        for file_name in item[1]:
            file_path = os.path.join(item[0], file_name)
            # height * width * channel
            image = cv2.imread(file_path)
            # height * width
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 413 * 295 标准一寸照片
            image = cv2.resize(image, (295, 413))
            # normalization, 也可以试试归一化 or not
            image = image_normalization(image)
            # cv2.imshow("input image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            X.append(image.flatten())
            Y.append(label)
    return X, Y


def load_recognition_test_image(test_image_path: str) -> list:
    X = []
    test_dir = os.path.join(os.getcwd(), test_image_path)
    image_list = os.listdir(test_dir)
    for idx, file_name in enumerate(image_list):
        file_path = os.path.join(test_image_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (295, 413))
        image = image_normalization(image)
        X.append(image.flatten())

    return X


def load_detection_test_image(test_image_path: str) -> list:
    X = []
    test_dir = os.path.join(os.getcwd(), test_image_path)
    image_list = os.listdir(test_dir)
    for idx, file_name in enumerate(image_list):
        file_path = os.path.join(test_image_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50, 50))
        image = image_normalization(image)
        X.append(image.flatten())

    return X


def recognition_train():
    # preprocessing
    training_image_path = 'training_images'
    labels = get_training_labels(training_image_path)
    print('recognition labels:', labels)
    labelEncoder = get_label_encoder(labels)
    X, Y = create_recognition_dataset(training_image_path, labelEncoder)
    # model training, n_components must be between 0 and min(n_samples, n_features)
    pca = PCA(n_components=2).fit(X)
    X_train = pca.transform(X)
    SVM_classifier = SVC(C=0.5, probability=True)
    SVM_classifier.fit(X=X_train, y=Y)
    # 除了accuracy外, 还有哪些指标? 只用accuracy作为评判依据是否足够了?
    y_pred = SVM_classifier.predict(X_train).tolist()
    # 这里的测试有没有问题？测试集是否出现在训练集中了？
    training_accuracy = metrics.accuracy_score(y_true=Y, y_pred=y_pred)
    print("recognition accuracy: {}%".format(training_accuracy * 100))
    # prediction
    test_image_path = 'test_images'
    X_test = load_recognition_test_image(test_image_path)
    X_test = pca.transform(X_test)
    prediction_results = SVM_classifier.predict(X_test)
    prediction_results_transform = labelEncoder.inverse_transform(prediction_results)
    # 可以把每次实验的结果写到文件中进行记录, or some other ways
    print(prediction_results_transform)
    # 可以考虑把模型保存下来
    return labelEncoder, pca, SVM_classifier


def create_detection_datasets(face_images_dir):
    """
    :param face_images_dir:
    :return:
    """
    X = []
    path_list = os.listdir(face_images_dir)
    for path in path_list:
        current_path = os.path.join(face_images_dir, path)
        image_list = os.listdir(current_path)
        for file_name in image_list:
            file_path = os.path.join(current_path, file_name)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("input image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # 这里如何resize?
            image = cv2.resize(image, (50, 50))
            image = image_normalization(image)
            X.append(image.flatten())
    return X


def detection_train():
    face_images_dir = "face_images"
    noface_image_dir = "noface_images"
    labels = ['face', 'noface']
    print('detection labels:', labels)
    labelEncoder = get_label_encoder(labels)
    face_X = create_detection_datasets(face_images_dir)
    face_y = [0 for x in range(len(face_X))]
    noface_X = create_detection_datasets(noface_image_dir)
    noface_y = [1 for x in range(len(noface_X))]
    # 如何解决正负样本不均衡的问题？
    X = face_X + noface_X
    y = face_y + noface_y
    # 为什么要做shuffle?
    X_shuff, y_shuff = sklearn.utils.shuffle(X, y)
    pca = PCA(n_components=100).fit(X_shuff)
    X_train_shuff = pca.transform(X_shuff)
    SVM_classifier = SVC(C=0.5, probability=True)
    SVM_classifier.fit(X=X_train_shuff, y=y_shuff)
    # prediction
    test_image_path = 'test_images'
    X_test = load_detection_test_image(test_image_path)
    X_test = pca.transform(X_test)
    prediction_results = SVM_classifier.predict(X_test)
    prediction_results_transform = labelEncoder.inverse_transform(prediction_results)
    print(prediction_results_transform)
    # 可以考虑把模型保存下来
    return labelEncoder, pca, SVM_classifier


def sliding(image):
    # 滑动窗口大小该如何设置？
    win_width = 200
    win_height = 200
    # 步长
    step_size = 100
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield image[y:y + win_height, x:x + win_width]


def is_recognition(recognition_result):
    for idx, score in enumerate(recognition_result):
        if score >= 0.6:
            return idx, True
    return 'Others', False


def check_in(recognition_label_encoder, recognition_encoder, recognition_classifier,
             detection_label_encoder, detection_encoder, detection_classifier):
    """
    :param recognition_label_encoder:
    :param recognition_encoder:
    :param recognition_classifier:
    :param detection_label_encoder:
    :param detection_encoder:
    :param detection_classifier:
    :return:
    """
    attendance_images_dir = 'attendance_images'
    file_list = os.listdir(attendance_images_dir)

    for file_name in file_list:
        results = []
        file_path = os.path.join(attendance_images_dir, file_name)
        image = cv2.imread(file_path)
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images = sliding(image)
        for sub_image in images:
            sub_image_detection = cv2.resize(sub_image, (50, 50)).flatten()
            sub_image_detection = sub_image_detection.reshape(1, -1)
            sub_image_detection_features = detection_encoder.transform(sub_image_detection)
            detection_result = detection_classifier.predict_proba(sub_image_detection_features)[0]
            _, detection_flag = is_recognition(detection_result)
            if detection_flag:
                sub_image_recognition = cv2.resize(sub_image, (295, 413)).flatten()
                sub_image_recognition = sub_image_recognition.reshape(1, -1)
                sub_image_recognition_features = recognition_encoder.transform(sub_image_recognition)
                recognition_result = recognition_classifier.predict_proba(sub_image_recognition_features)[0]
                label, recognition_flag = is_recognition(recognition_result)
                if recognition_flag:
                    results.extend(recognition_label_encoder.inverse_transform([label]).tolist())
            else:
                continue
        print('Recognition results of current image {}: {}'.format(file_name, set(results)))


if __name__ == '__main__':
    # face recognition model
    recognition_label_encoder, recognition_encoder, recognition_classifier = recognition_train()
    # face detection model
    detectoin_label_encoder, detectoin_encoder, detection_classifier = detection_train()
    check_in(recognition_label_encoder, recognition_encoder, recognition_classifier, detectoin_label_encoder,
             detectoin_encoder, detection_classifier)