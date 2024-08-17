import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.layers.layers import BuildNorm


class UpSample(nn.Layer):
    def __init__(self, in_channels, out_channels=None, reduce_dim=True, norm_type=nn.LayerNorm, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.reduce_dim = reduce_dim
        if self.reduce_dim:
            self.out_channels = out_channels or self.in_channels // 2
        else:
            self.out_channels = self.in_channels
        self.norm_type = norm_type

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


class ModulationUpSample(UpSample):
    def __init__(self, in_channels, base_channels, out_channels=None, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        expand_scale = 4
        self.expand = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.proj = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, sn):
        x = self.expand(x)
        B, C, H, W = x.shape
        x = paddle.transpose(x, (0, 2, 3, 1))
        x = x.reshape((B, H, W, 2, 2, C // 4))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * 2, W * 2, C // 4))
        x = paddle.transpose(x, (0, 3, 1, 2))
        x = self.compress(x)
        residual = x
        mod = self.ci(x)
        mod = self.dropout(mod)
        x = x * mod
        x = self.proj(x)
        x = x + residual
        return x


class ModulationUpSampleV2(UpSample):
    def __init__(self, in_channels, base_channels, out_channels=None, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        expand_scale = 4
        self.expand = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.expand_sn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.compress_sn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.ci_sn = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.proj = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.mod = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=self.out_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        self.sn_proj = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=self.out_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, sn):
        x = self.expand(x)
        B, C, H, W = x.shape
        x = paddle.transpose(x, (0, 2, 3, 1))
        x = x.reshape((B, H, W, 2, 2, C // 4))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * 2, W * 2, C // 4))
        x = paddle.transpose(x, (0, 3, 1, 2))
        x = self.compress(x)
        residual = x
        x = self.ci(x)

        mod = self.mod(x)

        sn = self.expand_sn(sn)
        B, C, H, W = sn.shape
        sn = paddle.transpose(sn, (0, 2, 3, 1))
        sn = sn.reshape((B, H, W, 2, 2, C // 4))
        sn = sn.transpose((0, 1, 3, 2, 4, 5))
        sn = sn.reshape((B, H * 2, W * 2, C // 4))
        sn = paddle.transpose(sn, (0, 3, 1, 2))
        sn = self.compress_sn(sn)
        sn = self.ci_sn(sn)

        sn = sum([sn, mod])
        sn = self.sn_proj(sn)
        sn = self.dropout(sn)
        x = x * sn
        x = self.proj(x)
        x = x + residual
        return x, sn


class ModulationUpSampleV3(UpSample):
    def __init__(self, in_channels, base_channels, out_channels=None, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            BuildNorm(self.out_channels, self.norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        self.sn_proj = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=self.out_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )

        self.sn_up = nn.Conv2DTranspose(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=self.out_channels)
        self.sn_up_pw = nn.Sequential(
            BuildNorm(self.out_channels, self.norm_type),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )

        self.residual_up = nn.Conv2DTranspose(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              groups=self.out_channels)
        self.residual_up_pw = nn.Sequential(
            BuildNorm(self.out_channels, self.norm_type),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )

        self.mod_up = nn.Conv2DTranspose(in_channels=self.out_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=self.out_channels)
        self.mod_up_pw = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )

        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, sn):
        B, C, H, W = x.shape
        residual = x
        x = self.proj_1(x)
        mod = self.mod_up(x, output_size=[2 * H, 2 * W])
        mod = self.mod_up_pw(mod)
        sn = self.sn_up(sn, output_size=[2 * H, 2 * W])
        sn = self.sn_up_pw(sn)
        sn = sum([sn, mod])
        sn = self.sn_proj(sn)
        sn = self.dropout(sn)
        x = F.interpolate(x, size=(2 * H, 2 * W), mode='bilinear')
        x = x * sn
        x = self.proj_2(x)
        residual = self.residual_up(residual, output_size=[2 * H, 2 * W])
        residual = self.residual_up_pw(residual)
        x = x + residual
        return x, sn


class ModulationUpSampleV4(UpSample):
    def __init__(self, in_channels, base_channels, out_channels=None, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            BuildNorm(self.out_channels, self.norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        expand_scale = 4
        self.expand = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        self.expand_sn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress_sn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci_sn = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        self.expand_residual = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress_residual = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci_residual = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        # self.sn_proj = nn.Sequential(
        #     nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=self.out_channels),
        #     nn.Conv2D(self.out_channels, self.out_channels, 1)
        # )
        #
        # self.mod = nn.Sequential(
        #     nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=self.out_channels),
        #     nn.Conv2D(self.out_channels, self.out_channels, 1)
        # )

        self.sn_proj = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=base_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )

        self.mod = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=base_channels),
            nn.Conv2D(self.out_channels, self.out_channels, 1)
        )

        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, sn):
        residual = x

        x = self.expand(x)
        B, C, H, W = x.shape
        x = paddle.transpose(x, (0, 2, 3, 1))
        x = x.reshape((B, H, W, 2, 2, C // 4))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * 2, W * 2, C // 4))
        x = paddle.transpose(x, (0, 3, 1, 2))
        x = self.compress(x)
        x = self.ci(x)

        mod = self.mod(x)

        sn = self.expand_sn(sn)
        B, C, H, W = sn.shape
        sn = paddle.transpose(sn, (0, 2, 3, 1))
        sn = sn.reshape((B, H, W, 2, 2, C // 4))
        sn = sn.transpose((0, 1, 3, 2, 4, 5))
        sn = sn.reshape((B, H * 2, W * 2, C // 4))
        sn = paddle.transpose(sn, (0, 3, 1, 2))
        sn = self.compress_sn(sn)
        sn = self.ci_sn(sn)

        sn = sum([sn, mod])
        sn = self.sn_proj(sn)
        sn = self.dropout(sn)
        x = x * sn
        x = self.proj_2(x)

        residual = self.expand_residual(residual)
        B, C, H, W = residual.shape
        residual = paddle.transpose(residual, (0, 2, 3, 1))
        residual = residual.reshape((B, H, W, 2, 2, C // 4))
        residual = residual.transpose((0, 1, 3, 2, 4, 5))
        residual = residual.reshape((B, H * 2, W * 2, C // 4))
        residual = paddle.transpose(residual, (0, 3, 1, 2))
        residual = self.compress_residual(residual)
        residual = self.ci_residual(residual)

        x = x + residual
        return x, sn


class ModulationUpSampleV5(UpSample):
    def __init__(self, in_channels, base_channels, out_channels=None, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels, 1),
            BuildNorm(self.in_channels, self.norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        expand_scale = 4
        self.expand = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        self.expand_sn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress_sn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci_sn = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        self.expand_residual = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress_residual = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci_residual = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        # self.sn_proj = nn.Sequential(
        #     nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=base_channels),
        #     nn.Conv2D(self.out_channels, self.out_channels, 1)
        # )
        self.act = nn.GELU()
        self.channels_fix = nn.Conv2D(self.in_channels, self.out_channels, 1)
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x):
        residual = x

        x = self.proj_1(x)
        mod = self.expand(x)
        B, C, H, W = mod.shape
        mod = paddle.transpose(mod, (0, 2, 3, 1))
        mod = mod.reshape((B, H, W, 2, 2, C // 4))
        mod = mod.transpose((0, 1, 3, 2, 4, 5))
        mod = mod.reshape((B, H * 2, W * 2, C // 4))
        mod = paddle.transpose(mod, (0, 3, 1, 2))
        mod = self.compress(mod)
        mod = self.ci(mod)

        # sn = self.expand_sn(sn)
        # B, C, H, W = sn.shape
        # sn = paddle.transpose(sn, (0, 2, 3, 1))
        # sn = sn.reshape((B, H, W, 2, 2, C // 4))
        # sn = sn.transpose((0, 1, 3, 2, 4, 5))
        # sn = sn.reshape((B, H * 2, W * 2, C // 4))
        # sn = paddle.transpose(sn, (0, 3, 1, 2))
        # sn = self.compress_sn(sn)
        # sn = self.ci_sn(sn)
        #
        # sn = sum([sn, mod])
        # # sn = self.sn_proj(sn)
        # sn = self.act(sn)
        mod = self.dropout(mod)

        x = self.channels_fix(x)
        x = F.interpolate(x, scale_factor=2)

        x = x * mod
        x = self.proj_2(x)

        residual = self.expand_residual(residual)
        B, C, H, W = residual.shape
        residual = paddle.transpose(residual, (0, 2, 3, 1))
        residual = residual.reshape((B, H, W, 2, 2, C // 4))
        residual = residual.transpose((0, 1, 3, 2, 4, 5))
        residual = residual.reshape((B, H * 2, W * 2, C // 4))
        residual = paddle.transpose(residual, (0, 3, 1, 2))
        residual = self.compress_residual(residual)
        residual = self.ci_residual(residual)

        x = x + residual
        return x


class ModulationUpSampleV5_NoSN(UpSample):
    def __init__(self, in_channels, base_channels, out_channels=None, drop_rate=0.5, act_type=nn.GELU, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.proj_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels, 1),
            BuildNorm(self.in_channels, self.norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        expand_scale = 4
        self.expand = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

        # self.expand_sn = nn.Sequential(
        #     nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
        #     BuildNorm(self.in_channels * expand_scale, self.norm_type),
        #     act_type()
        # )
        # self.compress_sn = nn.Sequential(
        #     nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        # )
        # self.ci_sn = nn.Sequential(
        #     nn.Conv2D(self.out_channels, self.out_channels, 1),
        # )

        self.expand_residual = nn.Sequential(
            nn.Conv2D(self.in_channels, self.in_channels * expand_scale, 1, groups=base_channels),
            BuildNorm(self.in_channels * expand_scale, self.norm_type),
            act_type()
        )
        self.compress_residual = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1, groups=base_channels),
        )
        self.ci_residual = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )
        # self.sn_proj = nn.Sequential(
        #     nn.Conv2D(self.out_channels, self.out_channels, 3, 1, 'same', groups=base_channels),
        #     nn.Conv2D(self.out_channels, self.out_channels, 1)
        # )
        self.channels_fix = nn.Conv2D(self.in_channels, self.out_channels, 1)
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x):
        residual = x

        x = self.proj_1(x)
        mod = self.expand(x)
        B, C, H, W = mod.shape
        mod = paddle.transpose(mod, (0, 2, 3, 1))
        mod = mod.reshape((B, H, W, 2, 2, C // 4))
        mod = mod.transpose((0, 1, 3, 2, 4, 5))
        mod = mod.reshape((B, H * 2, W * 2, C // 4))
        mod = paddle.transpose(mod, (0, 3, 1, 2))
        mod = self.compress(mod)
        mod = self.ci(mod)

        # sn = self.expand_sn(sn)
        # B, C, H, W = sn.shape
        # sn = paddle.transpose(sn, (0, 2, 3, 1))
        # sn = sn.reshape((B, H, W, 2, 2, C // 4))
        # sn = sn.transpose((0, 1, 3, 2, 4, 5))
        # sn = sn.reshape((B, H * 2, W * 2, C // 4))
        # sn = paddle.transpose(sn, (0, 3, 1, 2))
        # sn = self.compress_sn(sn)
        # sn = self.ci_sn(sn)

        # sn = sum([sn, mod])
        # sn = self.sn_proj(sn)
        mod = self.dropout(mod)

        x = self.channels_fix(x)
        x = F.interpolate(x, scale_factor=2)

        x = x * mod
        x = self.proj_2(x)

        residual = self.expand_residual(residual)
        B, C, H, W = residual.shape
        residual = paddle.transpose(residual, (0, 2, 3, 1))
        residual = residual.reshape((B, H, W, 2, 2, C // 4))
        residual = residual.transpose((0, 1, 3, 2, 4, 5))
        residual = residual.reshape((B, H * 2, W * 2, C // 4))
        residual = paddle.transpose(residual, (0, 3, 1, 2))
        residual = self.compress_residual(residual)
        residual = self.ci_residual(residual)

        x = x + residual
        return x


class TransformerDownSample(UpSample):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x):
        pass
