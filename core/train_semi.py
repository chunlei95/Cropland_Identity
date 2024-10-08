import os
import shutil
import time
from collections import deque

import paddle
from paddleseg.core import infer
from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, train_profiler, op_flops_funs)

from core.val import evaluate
from models.semi.corr_match.utils import corr_compute_loss


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits_list, labels, edges, losses):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]
        if loss_i.__class__.__name__ in ('BCELoss',) and loss_i.edge_label:
            # Use edges as labels According to loss type.
            loss_list.append(coef_i * loss_i(logits, edges))
        elif loss_i.__class__.__name__ == 'MixedLoss':
            mixed_loss_list = loss_i(logits, labels)
            for mixed_loss in mixed_loss_list:
                loss_list.append(coef_i * mixed_loss)
        elif loss_i.__class__.__name__ in ("KLLoss",):
            loss_list.append(coef_i *
                             loss_i(logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(coef_i * loss_i(logits, labels))
    return loss_list


# noinspection PyUnboundLocalVariable,PyTypeChecker
def train_semi(model,
               train_dataset,
               unlabeled_train_dataset,
               num_classes,
               thresh_init,
               val_dataset=None,
               optimizer=None,
               save_dir='output',
               iters=10000,
               batch_size=2,
               resume_model=None,
               start_eval_iter=0,
               save_interval=1000,
               log_iters=10,
               num_workers=0,
               use_vdl=False,
               losses=None,
               keep_checkpoint_max=5,
               test_config=None,
               precision='fp32',
               amp_level='O1',
               profiler_options=None,
               to_static_training=False):
    """
    Launch training.

    Args:
        model（nn.Layer): A semantic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict, optional): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
        test_config(dict, optional): Evaluation config.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the training is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision,
            the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators
            parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        profiler_options (str, optional): The option of train profiler.
        to_static_training (bool, optional): Whether to use @to_static for training.
    """
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # use amp
    if precision == 'fp16':
        logger.info('use AMP to train. AMP level = {}'.format(amp_level))
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        if amp_level == 'O2':
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level='O2',
                save_dtype='float32')

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    unlabeled_batch_sampler = paddle.io.DistributedBatchSampler(
        unlabeled_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn
    )

    unlabeled_loader = paddle.io.DataLoader(
        unlabeled_train_dataset,
        batch_sampler=unlabeled_batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn
    )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    if to_static_training:
        model = paddle.jit.to_static(model)
        logger.info("Successfully applied @to_static")

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(batch_sampler)
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()

    iter = start_iter
    while iter < iters:
        for data, unlabeled_data in zip(loader, unlabeled_loader):
            iter += 1
            if iter > iters:
                version = paddle.__version__
                if version == '2.1.2':
                    continue
                else:
                    break
            reader_cost_averager.record(time.time() - batch_start)
            images = data['img']
            labels = data['label'].astype('int64')

            img_u_w = unlabeled_data['img_w']
            img_u_s = unlabeled_data['img_s']

            edges = None
            if 'edge' in data.keys():
                edges = data['edge'].astype('int64')
            if hasattr(model, 'data_format') and model.data_format == 'NHWC':
                images = images.transpose((0, 2, 3, 1))

            model.train()

            b_l, b_ul = images.shape[0], img_u_w.shape[0]
            inputs = paddle.concat([images, img_u_w])

            if precision == 'fp16':
                with paddle.amp.auto_cast(
                        level=amp_level,
                        enable=True,
                        custom_white_list={
                            "elementwise_add", "batch_norm", "sync_batch_norm"
                        },
                        custom_black_list={'bilinear_interp_v2'}):
                    dict_result = ddp_model(inputs, use_corr=True) if nranks > 1 else model(inputs, use_corr=True)
                    u_s_result = ddp_model(img_u_s, use_corr=True) if nranks > 1 else model(img_u_s, use_corr=True)

                    loss_list = corr_compute_loss(dict_result,
                                                  u_s_result,
                                                  labels,
                                                  loss_computation,
                                                  edges,
                                                  losses,
                                                  b_l,
                                                  b_ul,
                                                  num_classes,
                                                  thresh_init)

                    # loss_list = loss_computation(
                    #     logits_list=logits_list,
                    #     labels=labels,
                    #     edges=edges,
                    #     losses=losses)
                    loss = sum(loss_list)

                scaled = scaler.scale(loss)  # scale the loss
                scaled.backward()  # do backward
                if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                    scaler.minimize(optimizer.user_defined_optimizer, scaled)
                else:
                    scaler.minimize(optimizer, scaled)  # update parameters
            else:
                dict_result = ddp_model(inputs, use_corr=True) if nranks > 1 else model(inputs, use_corr=True)
                u_s_result = ddp_model(img_u_s, use_corr=True) if nranks > 1 else model(img_u_s, use_corr=True)
                loss_list = corr_compute_loss(dict_result,
                                              u_s_result,
                                              labels,
                                              loss_computation,
                                              edges,
                                              losses,
                                              b_l,
                                              b_ul,
                                              num_classes,
                                              thresh_init)
                # loss_list = loss_computation(
                #     logits_list=logits_list,
                #     labels=labels,
                #     edges=edges,
                #     losses=losses)
                loss = sum(loss_list)
                loss.backward()
                # if the optimizer is ReduceOnPlateau, the loss is the one which has been pass into step.
                if isinstance(optimizer, paddle.optimizer.lr.ReduceOnPlateau):
                    optimizer.step(loss)
                else:
                    optimizer.step()

            lr = optimizer.get_lr()

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(profiler_options)

            model.clear_gradients()
            # avg_loss += loss.numpy()[0]
            avg_loss += loss.numpy()
            # avg_loss += loss
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                # avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                avg_loss_list = [l / log_iters for l in avg_loss_list]
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter - 1
                             ) // iters_per_epoch + 1, iter, iters, avg_loss,
                            lr, avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)

                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0 or
                iter == iters) and (val_dataset is not None) and (iter >= start_eval_iter):
                num_workers = 1 if num_workers > 0 else 0

                if test_config is None:
                    test_config = {}

                mean_iou, acc, _, _, _ = evaluate(
                    model,
                    val_dataset,
                    num_workers=num_workers,
                    precision=precision,
                    amp_level=amp_level,
                    **test_config)

                model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0 and (iter >= start_eval_iter):
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                        .format(best_mean_iou, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            batch_start = time.time()

    # Calculate flops.
    if local_rank == 0 and not (precision == 'fp16' and amp_level == 'O2'):
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()


def predict_label(model, data, precision, amp_level):
    with paddle.no_grad():
        if precision == 'fp16':
            with paddle.amp.auto_cast(
                    level=amp_level,
                    enable=True,
                    custom_white_list={
                        "elementwise_add", "batch_norm",
                        "sync_batch_norm"
                    },
                    custom_black_list={'bilinear_interp_v2'}):
                pred, logits = infer.inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'])
        else:
            pred, logits = infer.inference(
                model,
                data['img'],
                trans_info=data['trans_info'])
    return pred, logits
