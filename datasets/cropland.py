import os.path
import os.path as osp
import shutil
from glob import glob
from uuid import uuid4

import cv2
import numpy as np
import paddle.io
import paddleseg.transforms.functional as F
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class CroplandDataset(paddle.io.Dataset):
    NUM_CLASSES = 2
    IMG_CHANNELS = 3

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 strong_transforms=None,
                 augment_train=False,
                 augment_transforms=None,
                 augment_times=1,
                 img_dir=None,
                 ann_dir=None,
                 edge=False):
        super().__init__()
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.strong_transforms = strong_transforms
        mode = mode.lower()
        self.mode = mode
        self.edge = edge
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255  # labels only have 1/0, thus ignore_index is not necessary

        if mode not in ['train', 'train_unlabeled', 'val', 'test']:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is not None:
            if not osp.isabs(img_dir):
                img_dir = osp.join(self.dataset_root, img_dir)
            if not (ann_dir is None or osp.isabs(ann_dir)):
                ann_dir = osp.join(self.dataset_root, ann_dir)
        image_paths = glob(img_dir + '/*')
        grt_paths = glob(ann_dir + '/*')
        assert len(image_paths) == len(grt_paths)
        image_paths.sort()
        grt_paths.sort()

        for image_path, grt_path in zip(image_paths, grt_paths):
            assert image_path.split(image_path)[-1] == grt_path.split(grt_path)[-1]
            self.file_list.append([image_path, grt_path])

        if augment_train:
            augment_list = []
            aug_img_path = osp.join(self.dataset_root, 'img_dir', 'augment_train')
            if not os.path.exists(aug_img_path):
                os.makedirs(aug_img_path)
            aug_ann_path = osp.join(self.dataset_root, 'ann_dir', 'augment_train')
            if not os.path.exists(aug_ann_path):
                os.makedirs(aug_ann_path)
            for _ in range(augment_times):
                for i in range(len(self.file_list)):
                    img_ext = osp.splitext(self.file_list[i][0])[-1]
                    ann_ext = osp.splitext(self.file_list[i][1])[-1]
                    img_i = cv2.imread(self.file_list[i][0])
                    ann_i = cv2.imread(self.file_list[i][1], cv2.IMREAD_GRAYSCALE)
                    data = dict(img=img_i, label=ann_i, gt_fields=['label'])
                    if augment_transforms is not None:
                        for t in augment_transforms:
                            if np.random.random() < 0.5:
                                data = t(data)
                    filename = uuid4().hex
                    item_img_path = osp.join(aug_img_path, filename + img_ext)
                    item_ann_path = osp.join(aug_ann_path, filename + ann_ext)
                    cv2.imwrite(item_img_path, data['img'])
                    cv2.imwrite(item_ann_path, data['label'])
                    augment_list.append([item_img_path, item_ann_path])
            for img_p, ann_p in self.file_list:
                img_name, ann_name = osp.split(img_p)[-1], osp.split(ann_p)[-1]
                im_p = osp.join(aug_img_path, img_name)
                an_p = osp.join(aug_ann_path, ann_name)
                shutil.copy(img_p, im_p)
                shutil.copy(ann_p, an_p)
                augment_list.append([im_p, ann_p])
            self.file_list = augment_list

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        image_path, label_path = self.file_list[idx]
        data['img'] = image_path
        data['label'] = label_path
        # If key in gt_fields, the data[key] have transforms synchronous.
        data['gt_fields'] = []
        if self.mode == 'val' or self.mode == 'test':
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis, :, :]
        elif self.mode == 'train_unlabeled':
            data = self.transforms(data)
        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    data['label'], radius=2, num_classes=self.num_classes)
                data['edge'] = edge_mask
        return data

    def __len__(self):
        return len(self.file_list)
