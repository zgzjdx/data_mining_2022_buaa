"""
author:yqtong@buaa.edu.cn
date:2022-05-03
"""
import os

import numpy as np
import sklearn
import augmentation
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


def create_recognition_dataset(training_image_path, labelEncoder, aug_flag):
    X, Y = [], []
    dirs = get_training_dirs(training_image_path)
    images = get_each_labels_files(training_image_path)
    labels = get_training_labels(training_image_path)
    for idx, item in enumerate(zip(dirs, images, labels)):
        label = labelEncoder.transform([item[-1]])[0]
        for file_name in item[1]:
            print('Processing {}...'.format(file_name))
            file_path = os.path.join(item[0], file_name)
            # height * width * channel
            image = cv2.imread(file_path)
            # height * width
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 数据增广对模型效果的提升是否明显？
            if aug_flag:
                aug_images = augmentation.transform(image)
                aug_grey_images = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in aug_images]
                # 413 * 295 标准一寸照片
                aug_grey_resize_images = [cv2.resize(x, (295, 413)) for x in aug_grey_images]
                # normalization, 也可以试试归一化 or not
                aug_grey_resize_norm_images = [image_normalization(x) for x in aug_grey_resize_images]
                for temp in aug_grey_resize_norm_images:
                    X.append(temp.flatten())
                    Y.append(label)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (295, 413))
                image = image_normalization(image)
                X.append(image.flatten())
                Y.append(label)

    return X, Y


def load_recognition_test_image(test_image_path):
    X = []
    Y = []
    test_dir = os.path.join(os.getcwd(), test_image_path)
    image_list = os.listdir(test_dir)
    for idx, file_name in enumerate(image_list):
        label = file_name.split('-')
        file_path = os.path.join(test_image_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (295, 413))
        image = image_normalization(image)
        X.append(image.flatten())
        Y.append(label[0])

    return X, Y


def load_detection_test_image(test_image_path):
    X = []
    Y = []
    test_dir = os.path.join(os.getcwd(), test_image_path)
    image_list = os.listdir(test_dir)
    for idx, file_name in enumerate(image_list):
        file_path = os.path.join(test_image_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50, 50))
        image = image_normalization(image)
        X.append(image.flatten())
        Y.append('face')

    return X, Y


def recognition_train():
    # preprocessing
    training_image_path = 'training_images'
    labels = get_training_labels(training_image_path)
    print('recognition labels:', labels)
    labelEncoder = get_label_encoder(labels)
    X, Y = create_recognition_dataset(training_image_path, labelEncoder, aug_flag=True)
    # model training, n_components must be between 0 and min(n_samples, n_features)
    pca = PCA(n_components=15).fit(X)
    X_train = pca.transform(X)
    SVM_classifier = SVC(C=0.6, probability=True)
    SVM_classifier.fit(X=X_train, y=Y)
    # 除了accuracy外, 还有哪些指标? 只用accuracy作为评判依据是否足够了?
    X_raw, Y_raw = create_recognition_dataset(training_image_path, labelEncoder, aug_flag=False)
    X_raw = pca.transform(X_raw)
    y_pred = SVM_classifier.predict(X_raw).tolist()
    training_accuracy = metrics.accuracy_score(y_true=Y_raw, y_pred=y_pred)
    print("Training recognition accuracy: {}%".format(training_accuracy * 100))
    # prediction
    test_image_path = 'test_images'
    X_test, y_test = load_recognition_test_image(test_image_path)
    X_test = pca.transform(X_test)
    prediction_results = SVM_classifier.predict(X_test)
    # 可以把每次实验的结果写到文件中进行记录, or some other ways
    prediction_results_transform = labelEncoder.inverse_transform(prediction_results)
    print('Face recognition results:', prediction_results_transform)
    test_accuracy = metrics.accuracy_score(y_true=labelEncoder.transform(y_test), y_pred=prediction_results.tolist())
    print("Test recognition accuracy: {}%".format(test_accuracy * 100))
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
        print('current_path:', current_path)
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
    # 为什么要做shuffle?为什么recognition没做shuffle?
    X_shuff, y_shuff = sklearn.utils.shuffle(X, y)
    pca = PCA(n_components=50).fit(X_shuff)
    X_train_shuff = pca.transform(X_shuff)
    SVM_classifier = SVC(C=0.9, probability=True)
    SVM_classifier.fit(X=X_train_shuff, y=y_shuff)
    y_train_pred = SVM_classifier.predict(X_train_shuff)
    train_accuracy = metrics.accuracy_score(y_true=y_shuff, y_pred=y_train_pred)
    print('Training detection accuracy: {}%'.format(train_accuracy*100))
    # prediction
    test_image_path = 'test_images'
    X_test, y_test = load_detection_test_image(test_image_path)
    X_test = pca.transform(X_test)
    prediction_results = SVM_classifier.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_true=labelEncoder.transform(y_test), y_pred=prediction_results)
    print("Test detection accuracy: {}%".format(test_accuracy * 100))
    prediction_results_transform = labelEncoder.inverse_transform(prediction_results)
    print("Face detection results:", prediction_results_transform)
    # 可以考虑把模型保存下来
    return labelEncoder, pca, SVM_classifier


def sliding(image):
    # 滑动窗口大小该如何设置？
    win_width = 200
    win_height = 200
    # 步长, 步长设置的大或设置的小各有什么优劣？
    step_size = 100
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield image[y:y + win_height, x:x + win_width]


def is_recognition(recognition_result):
    for idx, score in enumerate(recognition_result):
        if score >= 0.1:
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

    for idx, file_name in enumerate(file_list):
        results = []
        file_path = os.path.join(attendance_images_dir, file_name)
        image = cv2.imread(file_path)
        images = sliding(image)
        for idy, sub_image in enumerate(images):
            cv2.imwrite('segment/{}-{}.jpg'.format(idx, idy), sub_image)
            sub_image_detection = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
            sub_image_detection = cv2.resize(sub_image_detection, (50, 50))
            sub_image_detection = image_normalization(sub_image_detection).flatten().reshape(1, -1)
            sub_image_detection_features = detection_encoder.transform(sub_image_detection)
            detection_result = detection_classifier.predict(sub_image_detection_features)
            detection_flag = detection_label_encoder.inverse_transform(detection_result)[0]
            # detection_result = detection_classifier.predict_proba(sub_image_detection_features)[0]
            # _, detection_flag = is_recognition(detection_result)
            if detection_flag == 'face':
                sub_image_recognition = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
                sub_image_recognition = cv2.resize(sub_image_recognition, (295, 413))
                sub_image_recognition = image_normalization(sub_image_recognition).flatten().reshape(1, -1)
                sub_image_recognition_features = recognition_encoder.transform(sub_image_recognition)
                recognition = recognition_classifier.predict(sub_image_recognition_features)
                recognition = recognition_label_encoder.inverse_transform(recognition)
                results.extend(recognition)
                # cv2.imshow("input image", sub_image)
                # cv2.waitKey(0)
                # recognition_result = recognition_classifier.predict_proba(sub_image_recognition_features)[0]
                # label, recognition_flag = is_recognition(recognition_result)
                # if recognition_flag:
                #     results.extend(recognition_label_encoder.inverse_transform([label]).tolist())
            else:
                continue
        date = file_name.split('.')[0]
        print('Recognition results of {}: {}'.format(date, set(results)))


if __name__ == '__main__':
    # face detection model
    detectoin_label_encoder, detectoin_encoder, detection_classifier = detection_train()
    # face recognition model
    recognition_label_encoder, recognition_encoder, recognition_classifier = recognition_train()
    check_in(recognition_label_encoder, recognition_encoder, recognition_classifier, detectoin_label_encoder,
             detectoin_encoder, detection_classifier)