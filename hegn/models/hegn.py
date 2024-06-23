import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_dgcnn import VNDGCNN
from models.vn_layers import VNLinearLeakyReLU, VNStdFeature, VNMaxPool, mean_pool, VNLinearAndLeakyReLU
from hegn.utils.vn_dgcnn_util import get_graph_feature




class HEGN(nn.Module):
    def __init__(self, args, normal_channel=False):
        super(HEGN, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        # first feature extraction
        self.vn_dgcnn1 = VNDGCNN(args, num_class, normal_channel)
        
        
        

    def forward(self, x):
        batch_size = x.size(0)
        