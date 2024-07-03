import sys
import logging
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


from hegn.models.vn_layers import VNLinearLeakyReLU, VNStdFeature, VNMaxPool, mean_pool
from hegn.utils.vn_dgcnn_util import get_graph_feature

class VNDGCNN(nn.Module):
    def __init__(self, args, normal_channel=False):
        super(VNDGCNN, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        
        self.conv1 = VNLinearLeakyReLU(2, 32)
        # self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        # self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        # self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)
        # self.conv5 = VNLinearLeakyReLU(338, 256)

        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(32)
            # self.pool2 = VNMaxPool(64//3)
            # self.pool3 = VNMaxPool(128//3)
            # self.pool4 = VNMaxPool(256//3)
            # self.pool5 = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            # self.pool2 = mean_pool
            # self.pool3 = mean_pool
            # self.pool4 = mean_pool
            # self.pool5 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.pool1(x) # (batch_size, 32, num_points, 1)
        
        # x = get_graph_feature(x1, k=self.n_knn) # (batch_size, 64, num_points, 20)
        # x = self.conv2(x)
        # x2 = self.pool2(x)
        
        # x = get_graph_feature(x2, k=self.n_knn)
        # x = self.conv3(x)
        # x3 = self.pool3(x)
        
        # x = get_graph_feature(x3, k=self.n_knn)
        # x = self.conv4(x)
        # x4 = self.pool4(x)
    
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = get_graph_feature(x, k=self.n_knn)
        # x = self.conv5(x)
        # f = self.pool5(x)
        return x
