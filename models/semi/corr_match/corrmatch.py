import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager

from models.layers.layers import BuildNorm
from models.semi.corr_match.corr import CorrModule


@manager.MODELS.add_component
class CorrMatch(nn.Layer):
    def __init__(self, model, in_channels, out_channels, num_classes, drop_rate, norm_type, act_type):
        super().__init__()
        self.model = model
        self.corr = CorrModule(in_channels=in_channels, num_classes=num_classes)
        self.proj = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BuildNorm(out_channels, norm_type),
            act_type(),
            nn.Dropout2D(drop_rate),
        )

    def forward(self, x, use_corr=False):
        dict_result = dict()
        encoder_x, logit_list = self.model(x)
        h, w = x.shape[2:]
        if use_corr:
            proj_feats = self.proj(encoder_x)
            corr_map, normalized_corr_map, interp_predict = self.corr(proj_feats, logit_list[0])
            if interp_predict.shape[2:] != (h, w):
                interp_predict = F.interpolate(interp_predict, (h, w), align_corners=True)
            dict_result['corr_map'] = normalized_corr_map
            dict_result['corr_out'] = interp_predict
        dict_result['out'] = logit_list
        return dict_result
