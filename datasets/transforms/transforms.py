import os.path

import cv2
import numpy as np
from PIL import Image
from osgeo import gdal
from paddleseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class GeoCompose:
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

    def __init__(self, transforms, to_rgb=True, img_channels=3):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.img_channels = img_channels
        self.read_flag = cv2.IMREAD_GRAYSCALE if img_channels == 1 else cv2.IMREAD_COLOR

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
            if os.path.splitext(data['img'])[-1] == '.tif':
                raster = gdal.Open(data['img'])
                raster_arr = raster.ReadAsArray()
                raster_arr = raster_arr[:3]  # 只要RGB通道
                raster_arr = np.transpose(raster_arr, (1, 2, 0))
                data['img'] = raster_arr.astype('float32')
            else:
                data['img'] = cv2.imread(data['img'], self.read_flag).astype('float32')
        if data['img'] is None:
            raise ValueError('Can\'t read The image file {}!'.format(data['img']))
        if not isinstance(data['img'], np.ndarray):
            raise TypeError("Image type is not numpy.")

        img_channels = 1 if data['img'].ndim == 2 else data['img'].shape[2]
        if img_channels != self.img_channels:
            raise ValueError(
                'The img_channels ({}) is not equal to the channel of loaded image ({})'.
                format(self.img_channels, img_channels))
        if self.to_rgb and img_channels == 3:
            data['img'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)

        if 'label' in data.keys() and isinstance(data['label'], str):
            data['label'] = np.asarray(Image.open(data['label']))

        # the `trans_info` will save the process of image shape, and will be used in evaluation and prediction.
        if 'trans_info' not in data.keys():
            data['trans_info'] = []

        for op in self.transforms:
            data = op(data)

        if data['img'].ndim == 2:
            data['img'] = data['img'][..., np.newaxis]
        data['img'] = np.transpose(data['img'], (2, 0, 1))
        return data
