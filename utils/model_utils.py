import contextlib
import functools
import os
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
# import SimpleITK as sitk
from urllib.parse import urlparse, unquote

import filelock
import paddle
import requests
from paddleseg.utils import logger
from paddleseg.utils import op_flops_funs


def calculate_flops_and_params(images, model, precision='fp32', amp_level='01'):
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if local_rank == 0 and not (precision == 'fp16' and amp_level == 'O2'):
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})


def calculate_inference_time(images, model, pretrain_params):
    model_params = paddle.load(pretrain_params)
    model.set_stage_dict(model_params)
    start = time.time()
    predict = model(images)
    end = time.time()
    return start - end


def _get_user_home():
    return os.path.expanduser('~')


def _get_seg_home():
    if 'SEG_HOME' in os.environ:
        home_path = os.environ['SEG_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                logger.warning('SEG_HOME {} is a file!'.format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddleseg')


def _get_sub_home(directory):
    home = os.path.join(_get_seg_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


USER_HOME = _get_user_home()
SEG_HOME = _get_seg_home()
DATA_HOME = _get_sub_home('dataset')
TMP_HOME = _get_sub_home('tmp')
PRETRAINED_MODEL_HOME = _get_sub_home('pretrained_model')


@contextlib.contextmanager
def generate_tempdir(directory: str = None, **kwargs):
    '''Generate a temporary directory'''
    directory = TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


lasttime = time.time()
FLUSH_INTERVAL = 0.1


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


def _download_file(url, savepath, print_progress):
    if print_progress:
        print("Connecting to {}".format(url))
    r = requests.get(url, stream=True, timeout=15)
    total_length = r.headers.get('content-length')

    if total_length is None:
        with open(savepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        with open(savepath, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            starttime = time.time()
            if print_progress:
                print("Downloading %s" % os.path.basename(savepath))
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if print_progress:
                    done = int(50 * dl / total_length)
                    progress("[%-50s] %.2f%%" %
                             ('=' * done, float(100 * dl) / total_length))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)


def _uncompress_file_zip(filepath, extrapath):
    files = zipfile.ZipFile(filepath, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    total_num = len(filelist)
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def _uncompress_file_tar(filepath, extrapath, mode="r:gz"):
    files = tarfile.open(filepath, mode)
    filelist = files.getnames()
    total_num = len(filelist)
    rootpath = filelist[0]
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def _uncompress_file(filepath, extrapath, delete_file, print_progress):
    if print_progress:
        print("Uncompress %s" % os.path.basename(filepath))

    if filepath.endswith("zip"):
        handler = _uncompress_file_zip
    elif filepath.endswith("tgz"):
        handler = functools.partial(_uncompress_file_tar, mode="r:*")
    else:
        handler = functools.partial(_uncompress_file_tar, mode="r")

    for total_num, index, rootpath in handler(filepath, extrapath):
        if print_progress:
            done = int(50 * float(index) / total_num)
            progress("[%-50s] %.2f%%" %
                     ('=' * done, float(100 * index) / total_num))
    if print_progress:
        progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

    if delete_file:
        os.remove(filepath)

    return rootpath


def download_file_and_uncompress(url,
                                 savepath=None,
                                 extrapath=None,
                                 extraname=None,
                                 print_progress=True,
                                 cover=True,
                                 delete_file=True):
    if savepath is None:
        savepath = "."

    if extrapath is None:
        extrapath = "."

    savename = url.split("/")[-1]
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savepath = os.path.join(savepath, savename)
    savename = ".".join(savename.split(".")[:-1])
    savename = os.path.join(extrapath, savename)
    extraname = savename if extraname is None else os.path.join(extrapath,
                                                                extraname)

    if cover:
        if os.path.exists(savepath):
            shutil.rmtree(savepath)
        if os.path.exists(savename):
            shutil.rmtree(savename)
        if os.path.exists(extraname):
            shutil.rmtree(extraname)

    if not os.path.exists(extraname):
        if not os.path.exists(savename):
            if not os.path.exists(savepath):
                _download_file(url, savepath, print_progress)

            if (not tarfile.is_tarfile(savepath)) and (
                    not zipfile.is_zipfile(savepath)):
                if not os.path.exists(extraname):
                    os.makedirs(extraname)
                shutil.move(savepath, extraname)
                return extraname

            savename = _uncompress_file(savepath, extrapath, delete_file,
                                        print_progress)
            savename = os.path.join(extrapath, savename)
        shutil.move(savename, extraname)
    return extraname


def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."

    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
    else:
        savename = savename.split('.')[0]

    with generate_tempdir() as _dir:
        with filelock.FileLock(os.path.join(TMP_HOME, savename)):
            pretrained_model = download_file_and_uncompress(
                pretrained_model,
                savepath=_dir,
                extrapath=PRETRAINED_MODEL_HOME,
                extraname=savename)
            pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    return pretrained_model


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        logger.info('Loading pretrained model from {}'.format(pretrained_model))

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                                    .shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), model.__class__.__name__))

        else:
            raise ValueError('The pretrained model directory is not Found: {}'.
                             format(pretrained_model))
    else:
        logger.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))

# def label_map(label_path, label_maps, save_path):
#     # label = imageio.v2.imread(label_path)
#     img = Image.open(label_path).convert('RGB')
#     im_arr = np.asarray(img)
#     label = im_arr.copy()
#     for key in label_maps.keys():
#         label[label == key] = label_maps[key]
#     label = label.astype(np.float32)
#     img = Image.fromarray(label)
#
#     img.save(save_path)
#
#     # imageio.imwrite(save_path, label * 255, format='RGB')
#
#
# if __name__ == '__main__':
# label_path = "D:/dataset/ACDC/patient001_frame01_slice5.png"
# label_maps = {
#     1: 100,
#     2: 80,
#     3: 128
# }
# save_path = 'D:/dataset/ACDC/1.jpg'
# label_map(label_path, label_maps, save_path)
