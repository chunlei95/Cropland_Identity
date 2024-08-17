import os.path
from glob import glob
from shutil import copy

import cv2
import numpy as np
import tifffile


def file_match(source_root, target_root, target_ext, dest_root):
    source_paths = glob(source_root + '/*')
    for pa in source_paths:
        target_name = os.path.splitext(os.path.split(pa)[-1])[0]
        target_name = target_name + target_ext
        target_path = os.path.join(target_root, target_name)
        copy(str(target_path), dest_root)


def tifconvert2rgb(img_path):
    img_p, img_n = os.path.split(img_path)
    img_n, img_e = os.path.splitext(img_n)
    save_path = os.path.join(img_p, img_n + '_1' + img_e)
    img_arr = tifffile.imread(img_path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(str(save_path), img_arr)


def fill_holes(img_arr):
    """
    大块中的孔洞填充
    :param img_arr: 代表图像的numpy数组
    """
    img = cv2.GaussianBlur(img_arr, (13, 13), 0)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out_img = np.zeros_like(img)
    del img
    for i in range(len(contours)):
        cnt = contours[i]
        cv2.fillPoly(out_img, [cnt], (255, 255, 255))
    # out_img = cv2.GaussianBlur(out_img, (13, 13), 0)
    # ret, out_img = cv2.threshold(out_img, 127, 255, cv2.THRESH_BINARY)
    return out_img


def remove_small_area(img_arr):
    """
    去除面积较小的孤立块
    :param img_arr: 代表图像的numpy数组
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_arr, connectivity=8)
    img = np.zeros_like(img_arr, np.uint8)  # 创建个全0的黑背景
    for i in range(1, num_labels):
        mask = labels == i  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
        if stats[i][4] > 200:  # 200是面积，根据实际需要调整
            img[mask] = 255
        else:
            img[mask] = 0
    return img


def label_post_process(label_path):
    img = cv2.imread(label_path)
    rgb = False
    if img.shape[-1] == 3:
        rgb = True
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('origin', img)
    img = fill_holes(img)
    img = remove_small_area(img)
    cv2.imshow('new', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(img.shape)
    return img


if __name__ == '__main__':
    file_match('D:/datasets/Cropland_Small/ann_dir/train',
               'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/train_labeled',
               '.tif',
               'D:/datasets/Cropland_Small/img_dir/train')

    # tifconvert2rgb("D:/datasets/曹湘地块识别/曹湘地块识别/test_csx.tif")

    # label_post_process('/output/result0/pseudo_color_prediction/rgb_version_caoxiang.png')
