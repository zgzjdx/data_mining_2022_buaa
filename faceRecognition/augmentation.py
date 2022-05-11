"""
author:yqtong@buaa.edu.cn
date:2022-05-10
"""
import numpy as np
import random
import cv2


def h_rotate(image):
    """水平翻转"""
    return cv2.flip(image, 1)


def v_rotate(image):
    """垂直翻转"""
    return cv2.flip(image, 0)


def hv_rotate(image):
    """水平垂直翻转"""
    return cv2.flip(image, -1)


def gaussian_noise(image, mean=0, var=0.001):
    """添加高斯噪声"""
    img = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gaussian_image = np.clip(out, low_clip, 1.0)
    gaussian_image = np.uint8(gaussian_image*255)
    return gaussian_image


def high_brightness(image, val=30):
    """增加亮度"""
    high_bright_image = np.clip(cv2.add(image, val), 0, 255)
    return high_bright_image


def low_brightness(image, val=-30):
    """降低亮度"""
    low_bright_image = np.clip(cv2.add(image, val), 0, 255)
    return low_bright_image


def high_contrast(image, alpha=1.2, beta=0):
    """提升对比度"""
    high_contrast_image = np.uint8(np.clip(cv2.add(alpha * image, beta), 0, 255))
    return high_contrast_image


def low_contrast(image, alpha=0.8, beta=0):
    """降低对比度"""
    low_contrast_image = np.uint8(np.clip(cv2.add(alpha * image, beta), 0, 255))
    return low_contrast_image


def transform(image):
    """数据增广"""
    res = [image]
    h_image = h_rotate(image)
    res.append(h_image)
    v_image = v_rotate(image)
    res.append(v_image)
    hv_image = hv_rotate(image)
    res.append(hv_image)
    gaussian_image = gaussian_noise(image)
    res.append(gaussian_image)
    high_bright_image = high_brightness(image)
    res.append(high_bright_image)
    low_bright_image = low_brightness(image)
    res.append(low_bright_image)
    high_contrast_image = high_contrast(image)
    res.append(high_contrast_image)
    low_contrast_image = low_contrast(image)
    res.append(low_contrast_image)
    # 这些方法是不是还能组合进行变换?还有其他的数据增广吗?这些方法都有用吗?
    # for idx, img in enumerate(res):
    #     cv2.imshow("input image", img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res

if __name__ == '__main__':
    random.seed(2022)
    image = cv2.imread('test.jpg')
    transform(image)