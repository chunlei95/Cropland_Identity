import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init
from paddleseg.cvlibs import manager
from paddleseg.models.backbones.transformer_utils import *

from models.layers.layers import OverlapPatchEmbed, BuildNorm, ConvStem, PatchDecompose, \
    ConditionalPositionEncoding, SkipLayer, SegmentationHead, PatchCombined
from utils.model_utils import calculate_flops_and_params


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# noinspection PyDefaultArgument,PyTypeChecker
@manager.MODELS.add_component
class LGGFormer(nn.Layer):
    def __init__(self, in_channels, num_classes, patch_size=3, stage_channels=[32, 64, 128, 256],
                 encoder_stage_blocks=[2, 2, 2, 2], decoder_stage_blocks=[2, 2, 2], num_heads=[2, 4, 8, 16],
                 trans_layers=2, attn_drop=0.0, drop_path_rate=0.5, norm_type=nn.LayerNorm, act_type=nn.GELU):

        super().__init__()
        self.feat_channels = stage_channels
        if norm_type is None:
            raise RuntimeWarning('norm type is not specified! there is no normalization in the model!')
        if type(norm_type) == str:
            norm_type = eval(norm_type)
        if type(act_type) == str:
            act_type = eval(act_type)
        block_list = encoder_stage_blocks + decoder_stage_blocks
        dpr = np.linspace(0, drop_path_rate, sum(block_list)).tolist()
        encoder_dpr = dpr[0:sum(encoder_stage_blocks)]
        decoder_dpr = dpr[sum(encoder_stage_blocks):]
        self.feat_channels = stage_channels
        self.stem = ConvStem(in_channels=in_channels, out_channels=stage_channels[0], norm_type=norm_type,
                             act_type=act_type)
        self.embedding = OverlapPatchEmbed(in_channels=stage_channels[0], out_channels=stage_channels[0],
                                           patch_size=patch_size, stride=2)
        self.encoder = L2GEncoder(stage_channels, encoder_stage_blocks, num_heads,
                                  trans_layers,
                                  norm_type, act_type,
                                  encoder_dpr)
        self.decoder = L2GDecoder(stage_channels, decoder_stage_blocks, num_heads,
                                  trans_layers,
                                  norm_type, act_type,
                                  decoder_dpr)
        self.final_expand = PatchDecompose(dim=stage_channels[0], base_channels=stage_channels[0], dim_scale=2,
                                           reduce_dim=False, norm_layer=norm_type)
        self.head = SegmentationHead(in_channels=stage_channels[0], num_classes=num_classes, norm_type=norm_type,
                                     act_type=act_type)

    def forward(self, x):
        x = self.stem(x)
        x = self.embedding(x)
        x, skip_x = self.encoder(x)
        skip_x.pop()
        skip_x.reverse()
        x = self.decoder(x, skip_x)
        x = self.final_expand(x)
        x = self.head(x)
        return [x]


class L2GEncoder(nn.Layer):
    def __init__(self, stage_channels, stage_blocks, num_heads, trans_layers,
                 norm_type,
                 act_type,
                 drop_path_rate):
        super().__init__()
        self.down_sample_list = nn.LayerList(
            [
                PatchCombined(dim=stage_channels[i], merge_size=3, norm_layer=norm_type)
                # PatchSplitSelectDown(channels=stage_channels[i], norm_type=norm_type)
                if i != len(stage_channels) - 1 else nn.Identity()
                for i in range(len(stage_channels))
            ]
        )
        self.l2g_layer_list = nn.LayerList(
            [
                BasicMSLLayer(stage_blocks[i], stage_channels[i], drop_path_rate[i], norm_type, act_type)
                if i < len(stage_channels) - trans_layers
                else
                BasicL2GLayer(stage_blocks[i], stage_channels[i], num_heads[i],
                              drop_path_rate[i], norm_type,
                              act_type)
                for i in range(len(stage_channels))
            ]
        )

    def forward(self, x):
        skip_x = []
        for down, l2g in zip(self.down_sample_list, self.l2g_layer_list):
            x = l2g(x)
            skip_x.append(x)
            if type(down) != nn.Identity:
                x = down(x)
        return x, skip_x


class L2GDecoder(nn.Layer):
    def __init__(self, stage_channels, stage_blocks, num_heads, trans_layers,
                 norm_type,
                 act_type,
                 drop_path_rate):
        super().__init__()
        self.stages = len(stage_channels) - 1
        self.up_sample_list = nn.LayerList(
            [
                PatchDecompose(dim=stage_channels[self.stages - i],
                               base_channels=stage_channels[0],
                               norm_layer=norm_type)
                for i in range(self.stages)
            ]
        )
        self.l2g_layer_list = nn.LayerList(
            [
                BasicMSLLayer(stage_blocks[self.stages - i - 1],
                              stage_channels[self.stages - i - 1],
                              drop_path_rate[i],
                              norm_type,
                              act_type)
                if i >= trans_layers - 1 else
                BasicL2GLayer(stage_blocks[self.stages - i - 1],
                              stage_channels[self.stages - i - 1],
                              num_heads[self.stages - i - 1],
                              drop_path_rate[i],
                              norm_type, act_type)
                for i in range(self.stages)
            ]
        )
        self.skip_layer_list = nn.LayerList(
            [
                # ModSkip(stage_channels[self.stages - i - 1]) for i in range(self.stages)
                SkipLayer(stage_channels[self.stages - i - 1]) for i in range(self.stages)
            ]
        )

    def forward(self, x, skip_xs):
        for up, skip, l2g, skip_x in zip(self.up_sample_list, self.skip_layer_list, self.l2g_layer_list, skip_xs):
            x = up(x)
            x = skip(x, skip_x)
            x = l2g(x)
        return x


class MSLBlock(nn.Layer):
    def __init__(self, channels, norm_type, act_type, attn_drop):
        super().__init__()
        self.norm = BuildNorm(channels, norm_type)
        self.norm_mlp = BuildNorm(channels, norm_type)
        scale_channels = channels // 1
        self.conv_1 = nn.Sequential(
            nn.Conv2D(scale_channels, scale_channels, 3, 1, 'same', groups=scale_channels),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2D(scale_channels, scale_channels, 3, 1, 'same', groups=scale_channels),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2D(scale_channels, scale_channels, 3, 1, 'same', groups=scale_channels),
        )
        self.proj = nn.Conv2D(channels, channels, 1)
        self.proj_1 = nn.Conv2D(channels, channels, 1)
        self.proj_2 = nn.Conv2D(channels, channels, 1)
        self.drop = nn.Dropout2D(attn_drop)
        self.mlp = Mlp(channels, channels * 4, channels, attn_drop, norm_type, act_type)
        # self.mlp = AMCMixer(channels, act_type)

    def forward(self, x):
        residual = x
        # x = self.proj(x)
        x = self.norm(x)
        xl_1 = self.conv_1(x)
        xl_2 = self.conv_2(xl_1)
        xl_3 = self.conv_3(xl_2)
        xl = xl_1 + xl_2 + xl_3
        xl = self.proj_1(xl)
        xl = self.drop(xl)
        x = x * xl
        x = self.proj_2(x)
        x = x + residual
        residual = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = x + residual
        return x


class L2GBlock(nn.Layer):
    def __init__(self, channels, norm_type, act_type, num_head, drop_path_rate):
        super().__init__()
        self.norm = BuildNorm(channels, norm_type)
        self.norm_mlp = BuildNorm(channels, norm_type)
        scale_channels = channels // 2
        self.conv_1 = nn.Sequential(
            nn.Conv2D(scale_channels, scale_channels, 3, 1, 'same', groups=scale_channels),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2D(scale_channels, scale_channels, 3, 1, 'same', groups=scale_channels),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2D(scale_channels, scale_channels, 3, 1, 'same', groups=scale_channels),
        )

        # self.gfem = GFEM(scale_channels, num_head, attn_drop)
        self.gfem = FocusedLinearAttention(scale_channels, num_head, drop_path_rate)
        # self.gfem = SwinTransformerBlock(scale_channels, input_resolution, num_head, drop_path=drop_path_rate)
        self.proj = nn.Conv2D(channels, channels, 1)
        self.proj_x_ = nn.Conv2D(channels, scale_channels, 1)
        self.proj_1 = nn.Conv2D(scale_channels, scale_channels, 1)
        self.proj_2 = nn.Conv2D(channels, channels, 1)
        self.proj_3 = nn.Conv2D(scale_channels, scale_channels, 1)
        self.proj_4 = nn.Conv2D(scale_channels, scale_channels, 1)
        self.mod_xl = nn.Conv2D(scale_channels, scale_channels, 1)
        self.mod_xg = nn.Conv2D(scale_channels, scale_channels, 1)
        self.drop = DropPath(drop_path_rate)
        self.mlp = Mlp(channels, channels * 4, channels, drop_path_rate, norm_type, act_type)
        # self.mlp = AMCMixer(channels, act_type)

    def forward(self, x, pre_attn):
        residual = x
        x = self.norm(x)
        x1, x2 = paddle.split(x, 2, 1)
        xl_1 = self.conv_1(x1)
        xl_2 = self.conv_2(xl_1)
        xl_3 = self.conv_3(xl_1)
        xl = xl_1 + xl_2 + xl_3
        xl = self.proj_1(xl)

        xg, pre_attn = self.gfem(x2, xl, pre_attn)

        mod_xl = self.mod_xl(xg)
        mod_xl = self.drop(mod_xl)
        mod_xg = self.mod_xg(xl)
        mod_xg = self.drop(mod_xg)
        xl = xl * mod_xl
        xl = self.proj_3(xl)
        xg = xg * mod_xg
        xg = self.proj_4(xg)
        x = paddle.concat([xl, xg], 1)
        x = self.proj_2(x)
        x = x + residual
        residual = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = x + residual
        return x, pre_attn


class Mlp(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.,
                 norm_type=nn.LayerNorm,
                 act_type=nn.GELU):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = BuildNorm(hidden_channels, norm_type=norm_type)
        self.conv1 = nn.Conv2D(in_channels, hidden_channels, 1, 1)
        self.act = act_type()
        self.conv2 = nn.Conv2D(hidden_channels, out_channels, 1, 1)
        self.drop = DropPath(drop_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class AMCMixer(nn.Layer):
    def __init__(self, chans, act_type):
        super().__init__()
        self.linear_1 = nn.Conv2D(chans, chans * 4, 1)
        self.aap = nn.AdaptiveAvgPool2D(1)
        self.amp = nn.AdaptiveMaxPool2D(1)
        self.aap_act_1 = act_type()
        self.aap_act_2 = nn.Sigmoid()
        self.linear_2 = nn.Conv2D(chans * 4, chans, 1)
        self.drop = nn.Dropout2D(0.5)
        self.proj = nn.Conv2D(chans, chans, 1)

    def forward(self, x):
        mod = x
        x1 = self.aap(x)
        x2 = self.amp(x)
        x = x1 + x2
        x = self.linear_1(x)
        x = self.aap_act_1(x)
        x = self.linear_2(x)
        x = self.aap_act_2(x)
        x = self.drop(x)
        x = x * mod
        x = self.proj(x)
        return x


class BasicL2GLayer(nn.Layer):
    def __init__(self, blocks, channels, num_head, drop_path_rate, norm_type,
                 act_type):
        super().__init__()

        self.block_list = nn.LayerList(
            [L2GBlock(channels, norm_type, act_type, num_head, drop_path_rate)
             for _ in range(blocks)])

    def forward(self, x):
        pre_attn = None
        for block in self.block_list:
            x, pre_attn = block(x, pre_attn)
        return x


class BasicMSLLayer(nn.Layer):
    def __init__(self, blocks, channels, drop_rate, norm_type, act_type):
        super().__init__()
        self.block_list = nn.LayerList(
            [MSLBlock(channels, norm_type, act_type, drop_rate) for _ in range(blocks)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x


class FocusedLinearAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., focusing_factor=3, kernel_size=5,
                 norm_type=nn.LayerNorm, act_type=nn.GELU):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        # self.q = nn.Conv2D(dim, dim, 1)
        self.q_down = nn.Sequential(
            nn.Conv2D(dim, dim, 3, 2, 1, groups=dim),
            # BuildNorm(dim, norm_type),
            # nn.Conv2D(dim, dim, 1)
        )
        self.k = nn.Sequential(
            nn.Conv2D(dim, dim, 3, 1, 1, groups=dim),
            # BuildNorm(dim, norm_type),
            # nn.Conv2D(dim, dim, 1)
        )
        self.v = nn.Sequential(
            nn.Conv2D(dim, dim, 3, 1, 1, groups=dim),
            # BuildNorm(dim, norm_type),
            # nn.Conv2D(dim, dim, 1)
        )
        self.cpe = ConditionalPositionEncoding(dim, 3)
        self.attn_drop = DropPath(attn_drop)
        self.proj = nn.Conv2D(dim, dim, 1)
        self.proj_drop = DropPath(proj_drop)
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2D(in_channels=dim, out_channels=dim, kernel_size=3, groups=dim, padding=1)
        self.scale = paddle.static.create_parameter([1, 1, dim], dtype='float32')

    def forward(self, x, Q, pre_attn):
        # x = self.cpe(x)
        residual = x
        B, C, H, W = x.shape
        q = self.q_down(Q)
        h, w = q.shape[2:]
        q = paddle.flatten(q, 2).transpose((0, 2, 1))
        k = self.k(x)
        v = self.v(x)
        v1 = v
        k = paddle.flatten(k, 2).transpose((0, 2, 1))
        v = paddle.flatten(v, 2).transpose((0, 2, 1))
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(axis=-1, keepdim=True)
        k_norm = k.norm(axis=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(axis=-1, keepdim=True)) * q_norm
        k = (k / k.norm(axis=-1, keepdim=True)) * k_norm
        q, k, v = (
            paddle.reshape(m, (B, -1, self.num_heads, self.dim // self.num_heads)).transpose((0, 2, 1, 3))
            for m in [q, k, v])
        attn = (k.transpose((0, 1, 3, 2)) @ v)
        if pre_attn is not None:
            attn = attn + pre_attn
        attn = self.attn_drop(attn)
        x = q @ attn
        x = paddle.transpose(x, (0, 2, 1, 3)).reshape((B, -1, C))
        x = paddle.transpose(x, (0, 2, 1)).reshape((B, C, h, w))
        if h != H:
            x = F.interpolate(x, [H, W])
        feature_map = self.dwc(v1)
        x = x + feature_map
        x = self.proj(x)
        x = x + residual
        return x, attn


# noinspection PyProtectedMember,PyMethodMayBeStatic
class GFEM(nn.Layer):
    def __init__(self, channels, num_head, attn_drop, norm_type=nn.LayerNorm, act_type=nn.GELU):
        super().__init__()
        assert channels % num_head == 0
        self.down = nn.Sequential(
            nn.Conv2D(channels, channels, kernel_size=3, stride=2, padding='same', groups=channels),
        )
        self.k = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        self.v = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        self.num_head = num_head
        self.head_dim = channels // num_head
        self.scale = self.head_dim ** -0.5
        self.attn_drop = DropPath(attn_drop)
        self.proj = nn.Conv2D(channels, channels, 1)
        # self.mlp = Mlp(channels, channels * 4, channels, attn_drop, norm_type, act_type)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, Q, pre_attn):
        ori_H, ori_W = x.shape[2:]
        residual = x
        Q = self.down(Q)
        B, C, H, W = Q.shape
        K = self.k(x)
        V = self.v(x)
        Q = paddle.flatten(Q, 2).transpose((0, 2, 1))
        K = paddle.flatten(K, 2).transpose((0, 2, 1))
        V = paddle.flatten(V, 2).transpose((0, 2, 1))
        B, N, C = Q.shape
        Q = paddle.reshape(Q, (B, N, self.num_head, self.head_dim)).transpose((0, 2, 1, 3))
        K = paddle.reshape(K, (B, -1, self.num_head, self.head_dim)).transpose((0, 2, 1, 3))
        V = paddle.reshape(V, (B, -1, self.num_head, self.head_dim)).transpose((0, 2, 1, 3))
        attn = (Q @ K.transpose([0, 1, 3, 2])) * self.scale
        if pre_attn is not None:
            attn = attn + pre_attn
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        xg = (attn @ V)
        xg = xg.transpose([0, 2, 1, 3]).flatten(2).transpose((0, 2, 1)).reshape((B, -1, H, W))
        xg = F.interpolate(xg, [ori_H, ori_W])
        x = self.proj(xg)
        x = x + residual
        return x, attn


if __name__ == '__main__':
    model = LGGFormer(in_channels=3, stage_channels=[96, 192, 384, 768], num_classes=2,
                      encoder_stage_blocks=[2, 2, 2, 2], patch_size=3)
    x = paddle.randn((2, 3, 256, 256))
    calculate_flops_and_params(x, model)
    # out = model(x)
    # print(out[0].shape)
