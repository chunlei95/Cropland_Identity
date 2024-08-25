import argparse

import paddle
from paddleseg.transforms import Compose
from paddleseg.utils import get_sys_env, logger

from core.predict import predict
from cvlibs.config import Config
from datasets import CroplandDataset
from models import ConvAttnUNet, RMMedNet
from utils.utils import get_image_list

__all__ = ['CroplandDataset', 'ConvAttnUNet', 'RMMedNet']


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", nargs='+', help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        nargs='+',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')

    # augment for prediction
    parser.add_argument(
        '--aug_pred',
        dest='aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        dest='scales',
        nargs='+',
        help='Scales for augment',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        dest='flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        dest='flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # sliding window prediction
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to prediction by sliding window',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)

    # custom color map
    parser.add_argument(
        '--custom_color',
        dest='custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int,
        default=[0, 0, 0, 255, 255, 255])

    # set device
    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be GPU, XPU, NPU, CPU',
        default='gpu',
        type=str)

    return parser.parse_args()


def get_test_config(cfg, args):
    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales

    if args.flip_horizontal:
        test_config['flip_horizontal'] = args.flip_horizontal

    if args.flip_vertical:
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    if args.custom_color:
        test_config['custom_color'] = args.custom_color

    return test_config


def main(args):
    env_info = get_sys_env()

    if args.device == 'gpu' and env_info[
        'Paddle compiled with cuda'] and env_info['GPUs used']:
        place = 'gpu'
    elif args.device == 'xpu' and paddle.is_compiled_with_xpu():
        place = 'xpu'
    elif args.device == 'npu' and paddle.is_compiled_with_npu():
        place = 'npu'
    else:
        place = 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    models = []
    if len(args.cfg) > 1:
        for c in args.cfg:
            cfg = Config(c)
            models.append(cfg.model)
    else:
        cfg = Config(args.cfg[0])
        models = cfg.model

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    # model = cfg.model
    transforms = Compose(cfg.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    test_config = get_test_config(cfg, args)

    predict(
        models,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
