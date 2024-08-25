import argparse
import os.path
from glob import glob
from shutil import copy
from labelme_utils import main

import cv2
import numpy as np
import tifffile


def split_image(img_path, save_path):
    """
    将下载的大幅遥感图像切割成512 x 512的小幅图像
    """
    img_arr = tifffile.imread(img_path)
    img_height, img_width = img_arr.shape[0:-1]
    assert img_arr.shape[-1] == 4
    if img_arr.shape[-1] == 4:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
    file_path = os.path.split(img_path)[-1]
    filename, ext = os.path.splitext(file_path)
    save_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cols, rows = img_width // 512, img_height // 512
    for i in range(rows):
        for j in range(cols):
            sub_num = i * (cols) + j + 1
            img_save_path = os.path.join(str(save_path), filename + '_' + str(sub_num) + ext)
            sub_arr = img_arr[i * 512: (i + 1) * 512, j * 512: (j + 1) * 512, :]
            cv2.imwrite(str(img_save_path), sub_arr)


def extract_area_and_label(img_save_dir, img_dir, ann_save_dir=None, ann_dir=None, crop_size=[256, 256],
                           pixel_interval=128, unlabeled=True):
    """
    将512 x 512的图像裁剪成256 x 256，每隔pixel_interval个像素裁剪一次
    """
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if ann_save_dir is not None and not os.path.exists(ann_save_dir) and not unlabeled:
        os.makedirs(ann_save_dir)
    img_paths = glob(img_dir + '/*')
    for k in range(len(img_paths)):
        img_path = img_paths[k]
        img_name, img_ext = os.path.splitext(os.path.split(img_path)[-1])
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if not unlabeled:
            ann_path = os.path.join(ann_dir, img_name + '.png')
            if not os.path.exists(ann_path):
                raise FileNotFoundError('annotation file not found: ' + str(ann_path))
            ann_arr = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        im_h, im_w = img_arr.shape[0:-1]
        num_h, num_w = im_h // pixel_interval, im_w // pixel_interval
        for i in range(num_h):
            for j in range(num_w):
                sub_num = i * (num_w) + j + 1
                img_save_path = os.path.join(img_save_dir, img_name + '_' + str(sub_num) + img_ext)
                sub_img = img_arr[i * pixel_interval: i * pixel_interval + crop_size[0],
                          j * pixel_interval: j * pixel_interval + crop_size[1], :]
                cv2.imwrite(str(img_save_path), sub_img)
                if not unlabeled:
                    ann_save_path = os.path.join(ann_save_dir, img_name + '_' + str(sub_num) + '.png')
                    sub_ann = ann_arr[i * pixel_interval: i * pixel_interval + crop_size[0],
                              j * pixel_interval: j * pixel_interval + crop_size[1]]
                    cv2.imwrite(str(ann_save_path), sub_ann)


def split_dataset(root_path, test_ratio, labeled_ratio=1.0 / 16):
    """
    数据集分割，分为有标签图像、无标签图像、验证图像，用于半监督学习，此处为还未标注的图像
    """
    imgs = glob(root_path + '/*')
    counts = len(imgs)
    test_size = int(counts * test_ratio)
    # np.random.seed(42)
    np.random.shuffle(imgs)
    test_paths = imgs[:test_size]
    train_paths = imgs[test_size:]
    labeled_train_size = int(len(train_paths) * labeled_ratio)
    labeled_train_paths = train_paths[:labeled_train_size]
    unlabeled_train_paths = train_paths[labeled_train_size:]
    copy_file(test_paths, 'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/val')
    copy_file(labeled_train_paths, 'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/train_labeled')
    copy_file(unlabeled_train_paths, 'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/train_unlabeled')


def copy_file(file_paths, dest_folder):
    if os.path.isdir(file_paths):
        for i in range(len(file_paths)):
            copy(file_paths[i], dest_folder)
    else:
        copy(file_paths, dest_folder)


def extract_label(label_root, save_path):
    label_folders = glob(label_root + '/*')
    label_paths = [folder + '/label.png' for folder in label_folders if os.path.isdir(folder)]
    for p in label_paths:
        label_name = p.split('/')[-2]
        label_name = label_name.split('\\')[-1]
        save_name = label_name + '.png'
        label_save_path = os.path.join(save_path, save_name)
        copy_file(p, label_save_path)


def label_map(label_root):
    """
    将标签值为255的像素映射为像素值1，表示第一个类别，此处暂时只适用于二分类
    """
    if os.path.isfile(label_root):
        label_paths = [label_root]
    else:
        label_paths = glob(label_root + '/*')
    for p in label_paths:
        label_arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        label_value = label_arr.max()
        label_ = label_arr.copy()
        label_arr[label_ == label_value] = 1
        cv2.imwrite(p, label_arr)


def show_label(label_path):
    """
    查看单张表示为类别值的标签图，此处暂时只适用于二分类
    """
    ann_arr = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    ann_ = ann_arr.copy()
    ann_arr[ann_ == 1] = 255
    cv2.imshow('label', ann_arr)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 数据集划分
    # img_path = r'D:/datasets/Cropland_Identity/cropland_identity_datasource/Cropland_Identity/region6'
    # split_image(img_path, 'D:/datasets/cropland_identity/Cropland_Identity')
    # split_dataset(img_path, 0.2, 1.0 / 16)

    # 将labelme标注的json形式标签转换为png标签图像，同时进行类别转换
    # parser = argparse.ArgumentParser()
    # parser.add_argument("json_file")
    # parser.add_argument("-o", "--out", default=None)
    # args = parser.parse_args()
    # if os.path.isfile(args.json_file):
    #     json_paths = [args.json_file]
    # else:
    #     file_path = args.json_file
    #     json_paths = glob(file_path + '/*')
    # for path in json_paths:
    #     main(path, args)

    # labelme转换json到png时会是一个文件夹一个图像，此处用于从文件夹中提取标签的png图像到一个指定文件夹，并对标签映射为类别标签形式
    # json_label_path = 'D:/datasets/Cropland_Identity/label_json/part2/json_label'
    # save_label_path = 'D:/datasets/Cropland_Identity/label_json/part2/png_labels'
    # extract_label(json_label_path, save_label_path)
    # label_map(save_label_path)

    # 512 * 512切成 256 * 256的（在标注之后切，标签图也同步切割，无标签图像无需切标签）
    # img_path = 'D:/datasets/Cropland_Identity/Cropland_Identity_256/img_dir/val'
    # ann_path = 'D:/datasets/Cropland_Identity/Cropland_Identity_256/ann_dir/val'
    # img_save_path = 'D:/datasets/Cropland_Identity/new_data/data_source/images'
    # ann_save_path = 'D:/datasets/Cropland_Identity/new_data/data_source/labels'
    # extract_area_and_label(img_save_path, img_path, ann_save_path, ann_path,
    #                        crop_size=[256, 256],
    #                        pixel_interval=256,
    #                        unlabeled=False)

    # 查看标签图
    show_label('D:/PycharmProjects/Paddle_Seg/output/result/pseudo_color_prediction/edge/test_area_1.png')
