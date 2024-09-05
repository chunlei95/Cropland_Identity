import argparse
import os.path
from glob import glob
from shutil import copy
from datasets.labelme_utils import main

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
            if os.path.exists(img_save_path):
                continue
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
                if os.path.exists(img_save_path):
                    continue
                sub_img = img_arr[i * pixel_interval: i * pixel_interval + crop_size[0],
                          j * pixel_interval: j * pixel_interval + crop_size[1], :]
                cv2.imwrite(str(img_save_path), sub_img)
                if not unlabeled:
                    ann_save_path = os.path.join(ann_save_dir, img_name + '_' + str(sub_num) + '.png')
                    if os.path.exists(ann_save_path):
                        continue
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
        if os.path.exists(label_save_path):
            continue
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


def add_png_label(image_path, save_dir):
    """
    为全部是背景的图像生成标签图，这部分图像由于在标注的时候不存在目标区域，因此不会生成json形式的标注文件，需要手动为其生成
    """
    image_paths = []
    if os.path.isdir(image_path):
        # 目前仅支持一级目录，如果是多级目录可以使用递归读取
        images = glob(image_path + '/*')
        images = [image for image in images if os.path.splitext(image)[-1] == '.tif']
        image_paths.extend(images)
    else:
        # 单个图像
        image_paths = [image_path]
    for p in image_paths:
        im = cv2.imread(p)
        im_h, im_w = im.shape[0:2]
        filename = os.path.splitext(os.path.split(p)[-1])[0]
        save_path = save_dir + '/' + filename + '.png'
        if os.path.exists(save_path):
            continue
        ann = np.zeros((im_h, im_w), dtype=np.uint8)
        cv2.imwrite(save_path, ann)


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
    # img_path = "D:/datasets/长光卫星_高分辨率遥感影像耕地地块提取挑战赛/长光卫星_高分辨率遥感影像耕地地块提取挑战赛/初赛训练集/image/CGDZ_1_offset.tif"
    # tif_data = gdal.Open(img_path)
    # band_1 = tif_data.GetRasterBand(1)
    # band_2 = tif_data.GetRasterBand(2)
    # band_3 = tif_data.GetRasterBand(3)
    # x_size = tif_data.GetRasterXSize()
    # y_size = tif_data.GetRasterYSize()

    # ① 将大幅遥感图像切割成512 x 512的小幅图像
    # img_path = r'D:/datasets/Cropland_Identity/cropland_identity_datasource/Cropland_Identity/new_area_16.tif'
    # split_image(img_path, 'D:/datasets/Cropland_Identity/label_json/part3')

    # 划分数据集，目前不使用这种方式划分
    # split_dataset(img_path, 0.2, 1.0 / 16)

    # ② 将labelme标注的json形式标签转换为png标签图像，同时进行类别转换
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

    # ③ labelme转换json到png时会是一个文件夹一个图像，此处用于从文件夹中提取标签的png图像到一个指定文件夹，并对标签映射为类别标签形式
    # json_label_path = 'D:/datasets/Cropland_Identity/label_json/part3/json_label'
    # save_label_path = 'D:/datasets/Cropland_Identity/label_json/part3/png_label'
    # extract_label(json_label_path, save_label_path)
    # label_map(save_label_path)

    # ④ 为全部为背景的图像添加对应标签图像（如果需要使用这部分数据的话，否则跳过该步）
    # image_path = 'D:/datasets/Cropland_Identity/label_json/part3/new_area_13'
    # save_path = 'D:/datasets/Cropland_Identity/label_json/part3/new_area_13'
    # add_png_label(image_path, save_path)

    # ④ / ⑤ 512 * 512切成 256 * 256的（在标注之后切，标签图也同步切割，无标签图像无需切标签）
    # img_path = 'D:/datasets/Cropland_Identity/label_json/part3/images'
    # ann_path = 'D:/datasets/Cropland_Identity/label_json/part3/png_label'
    # img_save_path = 'D:/datasets/Cropland_Identity/new_data/data_source/images'
    # ann_save_path = 'D:/datasets/Cropland_Identity/new_data/data_source/labels'
    # extract_area_and_label(img_save_path, img_path, ann_save_path, ann_path,
    #                        crop_size=[256, 256],
    #                        pixel_interval=256,
    #                        unlabeled=False)

    # 查看标签图
    show_label('D:/datasets/Cropland_Small/ann_dir/train/region5_325.png')
