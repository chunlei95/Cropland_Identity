import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CorrModule(nn.Layer):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv_1 = nn.Conv2D(in_channels, in_channels, 1)
        self.conv_2 = nn.Conv2D(in_channels, in_channels, 1)

    def forward(self, x, predict: paddle.Tensor):
        """
        计算相关性矩阵、标准化后的相关性矩阵，相关性矩阵增强后的预测标签图
        :param x: 编码器的输出
        :param predict: 模型预测的分割标签图
        :return:
        """
        h_in, w_in = x.shape[2:]
        h_out, w_out = predict.shape[2:]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x1 = paddle.flatten(x1, 2)
        x2 = paddle.flatten(x2, 2)

        predict = F.interpolate(predict.detach(), x.shape[2:], mode='bilinear', align_corners=True)
        pred_temp = paddle.flatten(predict, 2)

        # C = Softmax(w1⊤ · w2)/√D ****************************************************************************
        # 原论文中描述：We first extract features w1 and w2 ∈ R_D×HW through linear layers after the encoder of the network,
        # where D is the channel dimension and HW is the number of feature vectors. These extracted features
        # enable correlation matching to quantify the degree of pairwise similarity. Thus, we compute the
        # correlation map C by performing a matrix multiplication between all pairs of feature vectors
        corr_map = paddle.matmul(paddle.transpose(x1, (0, 2, 1)), x2) / paddle.sqrt(
            paddle.to_tensor(float(x1.shape[1])))  # (b, hw, hw)
        # 此处在axis=1和axis=-1的结果其实都是一样的
        corr_map = F.softmax(corr_map, axis=-1)  # (b, hw, hw)
        # ****************************************************************************************************

        # 原论文中描述：it becomes evident that involving every row of the correlation map in pseudo labels
        # optimization is redundant. Hence, we employed a random sampling approach within the correlation
        # map to expedite label propagation
        sampled_corr_map = self.sample(corr_map, h_in, w_in)

        # 原论文中描述：Specifically, we first normalize c and turn it into a binary map ˆc
        normalized_corr_map = self.normalize_corr_map(sampled_corr_map, h_in, w_in, h_out, w_out)

        interp_predict = paddle.matmul(pred_temp, corr_map)
        b, c = interp_predict.shape[:-1]
        interp_predict = paddle.reshape(interp_predict, (b, c, h_in, w_in))
        return corr_map, normalized_corr_map, interp_predict

    def sample(self, corr_map, h_in, w_in):
        index = paddle.randint(0, h_in * w_in - 1, [128])
        sampled_corr_map = corr_map[:, index, :]
        return sampled_corr_map

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, hw = corr_map.shape  # 其中 m = hw
        corr_map = paddle.reshape(corr_map, (n, m, 1, h_in, w_in))
        corr_map = paddle.flatten(corr_map, 0, 1)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)
        corr_map = paddle.flatten(corr_map, 2)
        range_ = paddle.max(corr_map, axis=1, keepdim=True)[0] - paddle.min(corr_map, axis=1, keepdim=True)[0]
        temp_map = ((- paddle.min(corr_map, axis=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        norm_corr_map = paddle.transpose(corr_map, (1, 0)).reshape((h_out * w_out, n, m))
        norm_corr_map = paddle.transpose(norm_corr_map, (1, 2, 0)).reshape((n, m, h_out, w_out))
        return norm_corr_map
