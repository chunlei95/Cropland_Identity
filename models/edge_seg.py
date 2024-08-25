import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.transforms.functional as TF
from paddleseg.cvlibs import manager


@manager.MODELS.add_component
class EdgeEnhanceModel(nn.Layer):
    def __init__(self, seg_model, num_classes):
        super().__init__()
        self.seg_model = seg_model
        self.num_classes = num_classes

    def forward(self, x):
        x = self.seg_model(x)
        pred = F.softmax(x[0], axis=1)
        pred = paddle.argmax(pred, axis=1)
        edges =[]
        for i in range(len(pred)):
            edges.append(TF.mask_to_binary_edge(pred[i], radius=2, num_classes=self.num_classes))
        edge = paddle.to_tensor(edges).astype(paddle.float32)
        # edge = paddle.concat(edges, 0)
        return [x[0], edge]
