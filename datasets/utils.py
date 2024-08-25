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
    img = img_arr
    # img = cv2.edgePreservingFilter(img_arr, None, 1, 120, 0.4)
    # img = cv2.bilateralFilter(img_arr, 0, 100, 15)
    # img = cv2.GaussianBlur(img_arr, (3, 3), 0)
    # ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((11, 11), np.uint8)
    # img = cv2.erode(img, kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out_img = np.zeros_like(img)
    del img
    for i in range(len(contours)):
        cnt = contours[i]
        cv2.fillPoly(out_img, [cnt], (255, 255, 255))
    # out_img = cv2.morphologyEx(out_img, cv2.MORPH_OPEN, kernel)
    # out_img = cv2.dilate(out_img, kernel)
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
        if stats[i][4] > 300:  # 200是面积，根据实际需要调整
            img[mask] = 255
        else:
            img[mask] = 0
    return img


def label_post_process(label_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.isfile(label_path):
        label_path = [label_path]
    else:
        label_path = glob(label_path + '/*')
    for path in label_path:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # rgb = False
        label_filename = os.path.split(path)[-1]
        save_name = os.path.join(save_path, label_filename)
        # if img.shape[-1] == 3:
        #     rgb = True
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('origin', img)
        img = fill_holes(img)
        img = remove_small_area(img)
        # cv2.imshow('new', img)
        cv2.imwrite(str(save_name), img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # if rgb:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # print(img.shape)
        # return img


if __name__ == '__main__':
    # file_match('D:/datasets/Cropland_Small/ann_dir/train',
    #            'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/train_labeled',
    #            '.tif',
    #            'D:/datasets/Cropland_Small/img_dir/train')

    # tifconvert2rgb("D:/datasets/曹湘地块识别/曹湘地块识别/test_csx.tif")

    label_post_process('D:/PycharmProjects/Paddle_Seg/output/compose_mscmnet_segnext/pseudo_color_prediction',
                       'D:/PycharmProjects/Paddle_Seg/output/compose_mscmnet_segnext/post_processed_prediction_open')
