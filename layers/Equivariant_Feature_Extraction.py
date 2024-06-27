from layers.VN_DGCNN import SharedVN_DGCNNLayer
import torch.nn as nn
from layers.VNTransformer import VecDGCNNAtten, GlobalContext


class EquivariantFeatureExtraction(nn.Module):
    def __init__(self, k=20):
        super(EquivariantFeatureExtraction, self).__init__()

        # Local Context!
        out_channel_ratio = 32
        out_put_dimension = 64 // out_channel_ratio  # 256 // out_channel_ratio + 128 // out_channel_ratio + 64 // out_channel_ratio * 2
        self.local_context = SharedVN_DGCNNLayer(out_channel_ratio=out_channel_ratio, k=k)

        # Cross Context!
        self.cross_context = VecDGCNNAtten(in_channels=out_put_dimension, atten_multi_head_c=2, k=k)

        # Global Context!
        self.global_context = GlobalContext(in_channels=out_put_dimension,
                                            out_channels=out_put_dimension)

    def forward(self, x, y):
        x1 = self.local_context(x)
        y1 = self.local_context(y)
        x2 = self.cross_context(x=x, y=y)
        y2 = self.cross_context(x=y, y=x)
        # # del x, y
        x = x1 + x2
        y = y1 + y2
        global_features_X = self.global_context(x)
        global_features_Y = self.global_context(y)
        return global_features_X, global_features_Y
