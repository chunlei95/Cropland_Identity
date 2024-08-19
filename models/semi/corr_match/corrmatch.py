import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager

from models.layers.layers import BuildNorm
from models.semi.corr_match.corr import CorrModule


@manager.MODELS.add_component
class CorrMatch(nn.Layer):
    def __init__(self, backbone, in_channels, corr_channels, num_classes, drop_rate, norm_type, act_type):
        super().__init__()
        if norm_type is None:
            raise RuntimeWarning('norm type is not specified! there is no normalization in the model!')
        if type(norm_type) == str:
            norm_type = eval(norm_type)
        if type(act_type) == str:
            act_type = eval(act_type)
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Conv2D(in_channels, corr_channels, kernel_size=3, stride=1, padding=1),
            BuildNorm(corr_channels, norm_type),
            act_type(),
            nn.Dropout2D(drop_rate),
        )
        self.corr = CorrModule(in_channels=corr_channels, num_classes=num_classes)

    def forward(self, x, use_corr=False):
        encoder_out, logit_list = self.backbone(x)
        if use_corr:
            h, w = x.shape[2:]
            dict_result = dict()
            proj_feats = self.proj(encoder_out)
            corr_map, normalized_corr_map, interp_predict = self.corr(proj_feats, logit_list[0])
            if interp_predict.shape[2:] != (h, w):
                interp_predict = F.interpolate(interp_predict, (h, w), mode='bilinear', align_corners=True)
            dict_result['corr_map'] = normalized_corr_map
            dict_result['corr_out'] = interp_predict
            dict_result['out'] = logit_list[0]
            return dict_result
        return logit_list
