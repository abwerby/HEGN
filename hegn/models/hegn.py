import torch.nn as nn
import torch.nn.functional as F

from hegn.models.vn_dgcnn import VNDGCNN
from hegn.models.vn_layers import VNLinearLeakyReLU, VNStdFeature, VNMaxPool, mean_pool, VNLinearAndLeakyReLU
from hegn.utils.vn_dgcnn_util import get_graph_feature




class HEGN(nn.Module):
    def __init__(self, args, normal_channel=False):
        super(HEGN, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        self.vn_dgcnn1 = VNDGCNN(args, normal_channel)
        
        
        
        

    def forward(self, x):
        batch_size = x.size(0)
        
        ## This block repeated M times ##
        # 1. knn + from spatial to feature
        x = self.vn_dgcnn1(x)
        # 2. local context aggregation
        
        # 3. cross context 
        
        # 4. global context aggregation
        
        # 5. invariant mapping
        
        # 6. softmax
        
        # 7.topk
        ## End of block repeated M times ##
        
        # 8. global pooling
        
        # 9. Hierarchical aggregation
        
        # 10. 9Dof Alignment
        
           
        
        
        return x
        