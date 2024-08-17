import paddle.nn as nn

from models.layers.layers import BuildNorm, PatchCombined
from models.layers.layers import Mlp
from models.layers.transformer_layers import Attention
import paddle.nn.functional as F


class DownSample(nn.Layer):
    def __init__(self, in_channels, out_channels=None, norm_type=nn.LayerNorm, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels * 2
        self.norm_type = norm_type

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


class ModulationDownSample(DownSample):
    def __init__(self, in_channels, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, **kwargs)
        self.norm = BuildNorm(self.in_channels, self.norm_type)
        self.max_pool = nn.MaxPool2D(2)
        self.avg_pool = nn.AvgPool2D(2)
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            BuildNorm(self.out_channels, self.norm_type),
            act_type()
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.conv_down_1 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 2, 1, groups=self.in_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        self.conv_down_2 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 3, 2, 1, groups=self.in_channels),
            BuildNorm(self.out_channels, norm_type=self.norm_type),
            act_type(),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x):
        residual = x
        x = self.proj_1(x)
        # x = self.norm(x)
        x_local = self.conv_down_1(x)
        x_mod = sum([self.avg_pool(x), self.max_pool(x)])
        x_local = self.dropout(x_local)
        x = x_local * x_mod
        x = self.proj_2(x)
        x = x + self.conv_down_2(residual)
        return x


class ModulationDownSampleV2(DownSample):
    def __init__(self, in_channels, base_channels, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, **kwargs)
        self.norm = BuildNorm(self.in_channels, self.norm_type)
        self.max_pool = nn.MaxPool2D(2)
        self.avg_pool = nn.AvgPool2D(2)
        self.max_pool_sn = nn.MaxPool2D(2)
        self.avg_pool_sn = nn.AvgPool2D(2)
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            BuildNorm(self.out_channels, self.norm_type),
            act_type()
        )
        self.proj_1_sn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            BuildNorm(self.out_channels, self.norm_type),
            act_type()
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.conv_down_1 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 2, 'same', groups=base_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        self.conv_down_2 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 3, 2, 'same', groups=base_channels),
            BuildNorm(self.out_channels, norm_type=self.norm_type),
            act_type(),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        # self.sn_proj = nn.Sequential(
        #     nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=base_channels),
        #     nn.Conv2D(self.out_channels, self.out_channels, 1)
        # )
        self.act = nn.GELU()
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x):
        residual = x
        x = self.proj_1(x)
        x_local = sum([self.avg_pool(x), self.max_pool(x)])
        # sn = self.proj_1_sn(sn)
        # sn = sum([self.avg_pool_sn(sn), self.max_pool_sn(sn)])

        x_mod = self.conv_down_1(x)

        # sn = sum([sn, x_mod])
        # sn = self.sn_proj(sn)
        # sn = self.act(sn)
        x_mod = self.dropout(x_mod)
        x = x_local * x_mod
        x = self.proj_2(x)
        x = x + self.conv_down_2(residual)
        return x


class ModulationDownSampleV2_NoSN(DownSample):
    def __init__(self, in_channels, base_channels, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, **kwargs)
        self.norm = BuildNorm(self.in_channels, self.norm_type)
        self.max_pool = nn.MaxPool2D(2)
        self.avg_pool = nn.AvgPool2D(2)
        # self.max_pool_sn = nn.MaxPool2D(2)
        # self.avg_pool_sn = nn.AvgPool2D(2)
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            BuildNorm(self.out_channels, self.norm_type),
            act_type()
        )
        # self.proj_1_sn = nn.Sequential(
        #     nn.Conv2D(self.in_channels, self.out_channels, 1),
        #     BuildNorm(self.out_channels, self.norm_type),
        #     act_type()
        # )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.conv_down_1 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 2, 'same', groups=base_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        self.conv_down_2 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 3, 2, 'same', groups=base_channels),
            BuildNorm(self.out_channels, norm_type=self.norm_type),
            act_type(),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        # self.sn_proj = nn.Sequential(
        #     nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=base_channels),
        #     nn.Conv2D(self.out_channels, self.out_channels, 1)
        # )

        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x):
        residual = x
        x = self.proj_1(x)
        x_local = sum([self.avg_pool(x), self.max_pool(x)])
        # sn = self.proj_1_sn(sn)
        # sn = sum([self.avg_pool_sn(sn), self.max_pool_sn(sn)])

        x_mod = self.conv_down_1(x)
        if x_mod.shape[2:] != x_local.shape[2:]:
            x_mod = F.interpolate(x_mod, x_local.shape[2:])
        # sn = sum([sn, x_mod])
        # sn = self.sn_proj(sn)
        x_mod = self.dropout(x_mod)
        x = x_local * x_mod
        x = self.proj_2(x)
        residual = self.conv_down_2(residual)
        if x.shape[2:] != residual.shape[2:]:
            x = F.interpolate(x, residual.shape[2:])
        x = x + residual
        return x


class ModulationDownSampleV3_NoSN(DownSample):
    def __init__(self, in_channels, base_channels, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, **kwargs)
        self.norm = BuildNorm(self.in_channels, self.norm_type)
        self.max_pool = nn.MaxPool2D(2)
        self.avg_pool = nn.AvgPool2D(2)
        self.channel_fix = nn.Conv2D(self.in_channels, self.out_channels, 1)
        self.act = act_type()
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels, 1),
            BuildNorm(self.in_channels, self.norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.patch_combined_mod = PatchCombined(dim=self.in_channels, norm_layer=self.norm_type)
        self.patch_combined_residual = PatchCombined(dim=self.in_channels, norm_layer=self.norm_type)

        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x):
        residual = x
        x = self.proj_1(x)
        x_mod = self.patch_combined_mod(x)
        x_mod = self.dropout(x_mod)
        x = sum([self.avg_pool(x), self.max_pool(x)])
        x = self.channel_fix(x)
        # fixme 考虑去掉激活函数
        x = self.act(x)
        x = x * x_mod
        x = self.proj_2(x)
        x = x + self.patch_combined_residual(residual)
        return x


class TransformerDownSample(DownSample):
    def __init__(self, in_channels, out_channels=None, num_heads=8, attn_drop=0.5, proj_drop=0.5, sr_ratio=1,
                 norm_type=nn.LayerNorm, act_type=nn.GELU):
        super().__init__(in_channels, out_channels)
        self.norm = BuildNorm(in_channels, norm_type)
        self.pool = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
        self.attn = Attention(in_channels, num_heads, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio,
                              norm_type=norm_type)
        self.mlp = Mlp(in_channels=in_channels,
                       hidden_channels=in_channels * 4,
                       out_channels=in_channels * 2,
                       drop_rate=proj_drop,
                       norm_type=norm_type,
                       act_type=act_type)
        self.proj = nn.Conv2D(in_channels, in_channels * 2, 1)

    def forward(self, x):
        x = self.pool(x)
        residual = x
        x = self.norm(x)
        x = self.attn(x)
        x = x + residual
        residual = x
        x = self.mlp(x)
        x = x + self.proj(residual)
        return x
