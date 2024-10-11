import os.path

import cv2
import numpy as np
import paddle
from PIL import Image
from osgeo import gdal
from paddleseg.cvlibs import manager
from sklearn.preprocessing import MinMaxScaler
from paddle.vision.transforms import normalize, to_tensor


@manager.TRANSFORMS.add_component
class MultiSpectralCompose:
    """
        Do transformation on input data with corresponding pre-processing and augmentation operations.
        The shape of input data to all operations is [height, width, channels].

        Args:
            transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
            to_rgb (bool, optional): If converting image to RGB color space. Default: True.
            img_channels (int, optional): The image channels used to check the loaded image. Default: 3.

        Raises:
            TypeError: When 'transforms' is not a list.
            ValueError: when the length of 'transforms' is less than 1.
        """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms

    def __call__(self, data):
        """
        Args:
            data: A dict to deal with. It may include keys: 'img', 'label', 'trans_info' and 'gt_fields'.
                'trans_info' reserve the image shape informating. And the 'gt_fields' save the key need to transforms
                together with 'img'

        Returns: A dict after process。
        """
        if 'img' not in data.keys():
            raise ValueError("`data` must include `img` key.")
        if isinstance(data['img'], str):
            raster = gdal.Open(data['img'])
            raster_arr = raster.ReadAsArray()  # (C, H, W)
            raster_arr = np.transpose(raster_arr, (1, 2, 0))  # 转为(H, W, C)，进行数据增强时需要这种顺序
            data['img'] = raster_arr.astype('float32')
        if data['img'] is None:
            raise ValueError('Can\'t read The image file {}!'.format(data['img']))
        if not isinstance(data['img'], np.ndarray):
            raise TypeError("Image type is not numpy.")

        if 'label' in data.keys() and isinstance(data['label'], str):
            data['label'] = np.asarray(Image.open(data['label']))

        # the `trans_info` will save the process of image shape, and will be used in evaluation and prediction.
        if 'trans_info' not in data.keys():
            data['trans_info'] = []

        for op in self.transforms:
            data = op(data)

        if data['img'].ndim == 2:
            data['img'] = data['img'][..., np.newaxis]
        return data


@manager.TRANSFORMS.add_component
class MultiSpectralToTensor:
    """
    标准化处理
    """
    def __init__(self, data_format='CHW'):
        self.data_format = data_format

    def __call__(self, data):
        data['img'] = to_tensor(data['img'], data_format=self.data_format)
        return data


@manager.TRANSFORMS.add_component
class MultiSpectralNormalize:
    """
    标准化处理
    """
    def __init__(self, mean=(0.5,), std=(0.5,)):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data['img'] = normalize(data['img'], mean=self.mean, std=self.std)
        return data


@manager.TRANSFORMS.add_component
class MultiSpectralMinMaxScale:
    """
    最大最小归一化到（0，1）之间
    """
    def __init__(self):
        # self.scaler = MinMaxScaler()
        pass


    def __call__(self, data):
        img_arr = data['img']
        min_ = paddle.min(img_arr)
        max_ = paddle.max(img_arr)
        scale_data = (img_arr - min_) / (max_ - min_)
        data['img'] = scale_data
        return data