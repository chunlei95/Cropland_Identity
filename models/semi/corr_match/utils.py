import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):
        self.thresh_global = paddle.to_tensor(thresh_init)
        self.momentum = momentum
        self.nclass = nclass

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        n, c, h, w = pred.shape
        pred_gather = paddle.zeros([n, c, h, w])
        pred = pred_gather
        if ignore_mask is not None:
            ignore_mask_gather = paddle.zeros([n, h, w]).cuda().long()
            ignore_mask = ignore_mask_gather
        mask_pred = paddle.argmax(pred, axis=1)
        pred_softmax = pred.softmax(axis=1)
        pred_conf = pred_softmax.max(axis=1)[0]
        unique_cls = paddle.unique(mask_pred)
        cls_num = len(unique_cls)
        new_global = 0.0
        for cls in unique_cls:
            cls_map = (mask_pred == cls)
            if ignore_mask is not None:
                cls_map *= (ignore_mask != 255)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            pred_conf_cls_all = pred_conf[cls_map]
            cls_max_conf = pred_conf_cls_all.max()
            new_global += cls_max_conf
        if cls_num > 0:
            return_dict['new_global'] = new_global / cls_num
        else:
            return_dict['new_global'] = None

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g and thresh['new_global'] is not None:
            self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * thresh['new_global']

    def get_thresh_global(self):
        return self.thresh_global


def corr_compute_loss(dict_result: dict,
                      u_s_results: dict,
                      label,
                      loss_fn,
                      edges,
                      losses,
                      b_l,
                      b_u_l,
                      num_classes,
                      thresh_init):
    thresh_controller = ThreshController(nclass=num_classes, momentum=0.999, thresh_init=thresh_init)
    predicts = dict_result['out']
    predicts_corr = dict_result['corr_out']

    l_pred, u_w_pred = paddle.split(predicts, [b_l, b_u_l])
    l_x_corr, u_w_corr = paddle.split(predicts_corr, [b_l, b_u_l])
    u_s_pred = u_s_results['out']
    u_s_corr = u_s_results['corr_out']

    u_w_pred = u_w_pred.detach()
    conf_u_w = u_w_pred.detach().softmax(axis=1).max(axis=1)[0]
    label_u = u_w_pred.detach().argmax(axis=1)

    thresh_controller.thresh_update(u_w_pred.detach(), update_g=True)
    thresh_global = thresh_controller.get_thresh_global()
    mask_indicator = (conf_u_w > thresh_global)

    loss_s = loss_fn(l_pred, label, edges, losses)
    loss_s_c = loss_fn(l_x_corr, label, edges, losses)

    softmax_pred_u_w = F.softmax(u_w_pred.detach(), axis=1)
    logsoftmax_pred_u_s1 = F.log_softmax(u_s_pred, axis=1)
    loss_u_s = nn.KLDivLoss()(logsoftmax_pred_u_s1, softmax_pred_u_w) * mask_indicator

    l_u_w = loss_fn(u_w_pred, label_u, edges, losses) * mask_indicator
    l_u_s = loss_fn(u_s_pred, label_u, edges, losses) * mask_indicator
    loss_u = (sum(l_u_w) + sum(l_u_s)) * 0.5

    loss_u_w_c = loss_fn(u_w_corr, label_u, edges, losses) * mask_indicator
    loss_u_s_c = loss_fn(u_s_corr, label_u, edges, losses) * mask_indicator
    loss_u_c = (loss_u_w_c + loss_u_s_c) * 0.5

    loss = (sum(loss_s) + sum(loss_s_c)) * 0.5 + (loss_u * 0.5 + loss_u_s * 0.25 + loss_u_c * 0.25)
    return loss
