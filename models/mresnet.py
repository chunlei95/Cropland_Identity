import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager

from models.layers.downsample import ModulationDownSampleV2_NoSN
from models.layers.layer_libs import Encoder, Decoder, BasicConvBlock
from models.layers.layers import OverlapPatchEmbed, SegmentationHead, BuildNorm, ConvStem, ModulationSkipLayerV2_NoSN
from models.layers.upsample import ModulationUpSampleV5_NoSN


# noinspection PyDefaultArgument
@manager.MODELS.add_component
class RMMedNet(nn.Layer):
    def __init__(self,
                 in_channels=1,
                 num_classes=2,
                 patch_size=3,
                 stage_channels=[32, 64, 128, 256],
                 encoder_depth=[2, 2, 2, 2],
                 decoder_depth=[2, 2, 2],
                 stage_kernels=[3, 3, 3, 3],
                 drop_rate=0.5,
                 norm_type=nn.LayerNorm,
                 act_type=nn.GELU):
        super().__init__()
        if norm_type is None:
            raise RuntimeWarning('norm type is not specified! there is no normalization in the model!')
        if type(norm_type) == str:
            norm_type = eval(norm_type)
        if type(act_type) == str:
            act_type = eval(act_type)
        self.stem = ConvStem(in_channels=in_channels,
                             out_channels=stage_channels[0],
                             norm_type=norm_type,
                             act_type=act_type)
        self.embedding = OverlapPatchEmbed(in_channels=stage_channels[0],
                                           out_channels=stage_channels[0],
                                           patch_size=patch_size,
                                           stride=2)
        self.encoder = RecurrentModulationEncoder(stage_channels=stage_channels,
                                                  kernel_sizes=stage_kernels,
                                                  depth=encoder_depth,
                                                  drop_rate=drop_rate,
                                                  norm_type=norm_type,
                                                  act_type=act_type)
        self.decoder = RecurrentModulationDecoder(stage_kernels=stage_kernels,
                                                  stage_channels=stage_channels,
                                                  depth=decoder_depth,
                                                  drop_rate=drop_rate,
                                                  norm_type=norm_type,
                                                  act_type=act_type)
        self.final_expand = ModulationUpSampleV5_NoSN(in_channels=stage_channels[0],
                                                      base_channels=stage_channels[0],
                                                      norm_type=norm_type,
                                                      reduce_dim=False)
        self.head = SegmentationHead(in_channels=stage_channels[0],
                                     num_classes=num_classes,
                                     norm_type=norm_type,
                                     act_type=act_type)

    def forward(self, x):
        x = self.stem(x)
        x = self.embedding(x)
        x, skip_xs = self.encoder(x)
        x = self.decoder(x, skip_xs)
        x = self.final_expand(x)
        x = self.head(x)
        return [x]


class RecurrentModulationEncoder(Encoder):
    def __init__(self, kernel_sizes, drop_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.stage_layers = nn.LayerList()
        base_channels = self.stage_channels[0]
        self.down_sample_layers = nn.LayerList(
            [
                # 最后一个阶段不需要进行下采样，使用nn.Identity()占位符
                ModulationDownSampleV2_NoSN(in_channels=self.stage_channels[i],
                                            base_channels=base_channels,
                                            drop_rate=drop_rate,
                                            **kwargs) if i != self.stage_num - 1 else nn.Identity()
                for i in range(self.stage_num)
            ]
        )
        for i in range(self.stage_num):
            stage_layer = nn.Sequential(*[
                RecurrentModulationModule(in_channels=self.stage_channels[i],
                                          base_channels=base_channels,
                                          kernel_size=kernel_sizes[i],
                                          norm_type=self.norm_type,
                                          drop_rate=drop_rate,
                                          sn_proj_layer=nn.Sequential(
                                              BuildNorm(self.stage_channels[i], self.norm_type),
                                              nn.Conv2D(self.stage_channels[i], self.stage_channels[i], 1),
                                          ))
                for _ in range(self.stage_depth[i])
            ])
            self.stage_layers.append(stage_layer)

    def forward(self, x):
        skip_xs = []
        for stage_layer, down_sample in zip(self.stage_layers, self.down_sample_layers):
            x = stage_layer(x)
            skip_xs.append(x)
            if type(down_sample) != nn.Identity:
                x = down_sample(x)
        return x, skip_xs


class RecurrentModulationDecoder(Decoder):
    def __init__(self, stage_kernels, drop_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.stage_layers = nn.LayerList()
        self.up_sample_layers = nn.LayerList(
            [
                # 最后一个阶段不需要进行下采样，使用nn.Identity()占位符
                ModulationUpSampleV5_NoSN(in_channels=self.stage_in_channels[i],
                                          base_channels=self.base_channels,
                                          drop_rate=drop_rate,
                                          **kwargs)
                for i in range(self.stage_num)
            ]
        )
        for i in range(self.stage_num):
            stage_layer = nn.Sequential(*[
                RecurrentModulationModule(in_channels=self.stage_channels[i],
                                          base_channels=self.base_channels,
                                          kernel_size=stage_kernels[i],
                                          norm_type=self.norm_type,
                                          drop_rate=drop_rate,
                                          sn_proj_layer=nn.Sequential(
                                              BuildNorm(self.stage_channels[i], self.norm_type),
                                              nn.Conv2D(self.stage_channels[i], self.stage_channels[i], 1),
                                          ))
                for _ in range(self.stage_depth[i])
            ])
            self.stage_layers.append(stage_layer)
        self.skip_layers = nn.LayerList([
            ModulationSkipLayerV2_NoSN(in_channels=self.stage_channels[i],
                                       base_channels=self.base_channels,
                                       norm_type=self.norm_type,
                                       drop_rate=drop_rate)
            for i in range(self.stage_num)
        ])

    def forward(self, x, skip_xs):
        skip_xs.pop()
        skip_xs.reverse()
        for i in range(self.stage_num):
            x = self.up_sample_layers[i](x)
            x = self.skip_layers[i](x, skip_xs[i])
            x = self.stage_layers[i](x)
        return x


class RecurrentModulationModule(nn.Layer):
    def __init__(self, in_channels, base_channels, kernel_size, norm_type=nn.LayerNorm, drop_rate=0.5,
                 sn_proj_layer=None):
        super().__init__()
        self.proj_1 = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, 1),
            BuildNorm(in_channels, norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, 1),
        )
        self.lf = LocalFeatureModule(in_channels=in_channels,
                                     kernel_size=kernel_size,
                                     norm_type=norm_type,
                                     base_channels=base_channels,
                                     drop_rate=drop_rate)
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x):
        residual = x
        x = self.proj_1(x)
        mod = self.lf(x)
        mod = self.dropout(mod)
        x = x * mod

        x = self.proj_2(x)
        x = x + residual
        return x


class LocalFeatureModule(BasicConvBlock):
    def __init__(self, base_channels, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, self.kernel_size, 1, 'same',
                      groups=self.out_channels if self.out_channels <= self.in_channels else self.in_channels),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, self.kernel_size, 1, 'same', groups=self.out_channels),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, self.kernel_size, 1, 'same', groups=self.out_channels),
        )
        self.proj = nn.Sequential(
            nn.Conv2D(self.out_channels, self.out_channels, 1),
        )

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x = sum([x1, x2, x3])
        x = self.proj(x)
        return x


if __name__ == '__main__':
    x = paddle.randn((2, 1, 224, 224))
    model = RMMedNet()
    out = model(x)
    print(out[0].shape)
