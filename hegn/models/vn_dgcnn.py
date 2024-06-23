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
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        # self.linear3 = nn.Linear(256, num_class)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        trans_feat = None
        return x
