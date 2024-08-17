import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init
from paddleseg.models.backbones.transformer_utils import *

from models.layers.layers import BuildNorm, Mlp, ConditionalPositionEncoding


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 norm_type=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = BuildNorm(dim, norm_type=norm_type)
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

    def forward(self, x, q_x=None, mode='alone'):
        """

        :param x:
        :param q_x:
        :param mode: alone或者add，alone表示Q由q_x映射生成，add表示q_x由(x+q_x)映射生成
        :return:
        """
        assert mode in ['alone', 'add']
        # 输入形状为B, C, H, W
        B, C, H, W = x.shape
        x = paddle.flatten(x, 2).transpose((0, 2, 1))
        if q_x is not None:
            H_q, W_q = q_x.shape[2:]
            q = paddle.flatten(q_x, 2).transpose((0, 2, 1))
            # q = self.q(q_x)
        else:
            H_q, W_q = H, W
            q = self.q(x)
        q = q.reshape([B, -1, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])
        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(x_)
            x_ = self.norm(x_)
            x_ = x_.reshape([B, C, -1]).transpose([0, 2, 1])
            kv = self.kv(x_).reshape(
                [B, -1, 2, self.num_heads,
                 C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        else:
            kv = self.kv(x).reshape(
                [B, -1, 2, self.num_heads,
                 C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, H * W, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        # 输出形状为B, C, H, W
        x = paddle.transpose(x, (0, 2, 1)).reshape((B, C, H, W))
        if x.shape[2:] != [H, W]:
            x = F.interpolate(x, [H, W])
        return x


class TransformerBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 act_type=nn.GELU,
                 sr_ratio=1):
        super().__init__()
        self.cpe = ConditionalPositionEncoding(channels=dim, encode_size=3)
        self.norm1 = BuildNorm(dim, norm_type=norm_layer)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_channels=dim,
                       hidden_channels=mlp_hidden_dim,
                       out_channels=dim,
                       drop_rate=drop,
                       norm_type=norm_layer,
                       act_type=act_type)

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

    def forward(self, x, q=None):
        # 输入形状为B, C, H, W
        x = self.cpe(x)
        residual = x
        # x = paddle.transpose(x, (0, 2, 3, 1))
        x = self.norm1(x)
        x = self.drop_path(self.attn(x, q))
        x = x + residual

        residual = x
        x = self.drop_path(self.mlp(x))
        x = x + residual
        # 输出形状为B, C, H, W
        return x
