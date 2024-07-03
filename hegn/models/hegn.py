import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from hegn.models.vn_dgcnn import VNDGCNN
from hegn.models.vn_layers import VNLinearLeakyReLU, VNStdFeature, VNMaxPool, mean_pool
from hegn.utils.vn_dgcnn_util import get_graph_feature


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')


class local_context_aggregation(nn.Module):
    def __init__(self, k=20):
        super(local_context_aggregation, self).__init__()
        self.k_nn = k
        self.vn_mlp = VNLinearLeakyReLU(64, 32)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples]
        '''
        # get graph feature calc the difference between the point and its neighbors and concatenate them.
        x = get_graph_feature(x, k=self.k_nn)
        # B, C, 3, N, 20
        # apply equation (4) in the paper
        x = 1/self.k_nn * torch.sum(self.vn_mlp(x), dim=-1)
        return x       

class cross_context(nn.Module):
    """
        Cross context aggregation module
        apply VN-TRANSFORMER to the input features
    """
    def __init__(self, k=16):
        super(cross_context, self).__init__()
        self.k_nn = k
        self.vn_mlp_q = VNLinearLeakyReLU(32, 32, dim=3)
        self.vn_mlp_k = VNLinearLeakyReLU(64, 32)
        self.vn_mlp_v = VNLinearLeakyReLU(64, 32)
    
    def channel_equi_vec_normalize(self, x):
        # B,C,3,...
        assert x.ndim >= 3, "x shape [B,C,3,...]"
        x_dir = F.normalize(x, dim=2)
        x_norm = x.norm(dim=2, keepdim=True)
        x_normalized_norm = F.normalize(x_norm, dim=1)  # normalize across C
        y = x_dir * x_normalized_norm
        return y

    def forward(self, x, y):
        '''
        x: point features of shape [B, N_feat, 3, N_samples]
        '''
        # get graph feature calc the difference between the point and its neighbors and concatenate them.
        # x = get_graph_feature(x, k=self.k_nn)
        Qx = self.channel_equi_vec_normalize(self.vn_mlp_q(x))
        logging.info(f"Qx {Qx.size()}")
        y = get_graph_feature(y, k=self.k_nn)
        Ky = self.channel_equi_vec_normalize(self.vn_mlp_k(y))
        Vy = self.vn_mlp_v(y)
        logging.info(f"Ky {Ky.size()}")
        logging.info(f"Vy {Vy.size()}")
        Qx = Qx.view(Qx.size(0), Qx.size(1), 3, Qx.size(3), 1).repeat(1, 1, 1, 1, self.k_nn)
        logging.info(f"Qx {Qx.size()}")
        # attn_x = torch.einsum('bckij,bcklj->bckil', Qx, Ky) / torch.sqrt(torch.tensor(3*64).float())
        attn_x = torch.matmul(Qx, Ky.transpose(3, 4)) / torch.sqrt(torch.tensor(3*32).float())
        ax = F.softmax(attn_x, dim=-1)
        logging.info(f"ax {ax.size()}")
        x += torch.sum(torch.matmul(ax, Vy), dim=-1)
        return x



class HEGN(nn.Module):
    def __init__(self, args, normal_channel=False):
        super(HEGN, self).__init__()
        self.device = args.device
        self.args = args
        self.n_knn = args.n_knn
        self.num_blocks = args.num_blocks
        self.vn_dgcnn1 = VNDGCNN(args, normal_channel)
        # self.local_context_aggregation = local_context_aggregation()
        # self.cross_context = cross_context()
        # self.vn_mlp_global = VNLinearLeakyReLU(64, 32, dim=3)
        self.local_context_aggregation = [local_context_aggregation().to(self.device) for i in range(self.num_blocks)]
        self.cross_context = [cross_context().to(self.device) for i in range(self.num_blocks)]
        self.vn_mlp_global = [VNLinearLeakyReLU(64, 32, dim=3).to(self.device) for i in range(self.num_blocks)]
        self.vn_mlp_hierarchical = VNLinearLeakyReLU(32*self.num_blocks, 32, dim=3)

    def forward(self, x, y):
        logging.disable(logging.CRITICAL)
        batch_size = x.size(0)
        fx_block = []
        fy_block = []
        # 1. knn + from spatial to feature
        fx = self.vn_dgcnn1(x)
        fy = self.vn_dgcnn1(y)
        logging.info(f"fx {fx.size()}")
        logging.info(f"fy {fy.size()}")
        b, c, n, s = fx.size()
        for i in range(self.num_blocks):
            # 2. local context aggregation
            fx = self.local_context_aggregation[i](fx)
            fy = self.local_context_aggregation[i](fy)
            logging.info(f"local context x {fx.size()}")
            logging.info(f"local context y {fy.size()}")
            # 3. cross context 
            fx = self.cross_context[i](fx, fy)
            fy = self.cross_context[i](fy, fx)
            logging.info(f"cross context fx {fx.size()}")
            logging.info(f"cross context fy {fy.size()}")
            # 4. global context aggregation
            Fx = torch.mean(fx, dim=-1, keepdim=True).expand(fx.size())
            Fy = torch.mean(fy, dim=-1, keepdim=True).expand(fy.size())
            logging.info(f"global context Fx {Fx.size()}")
            logging.info(f"global context Fy {Fy.size()}")
            fx = self.vn_mlp_global[i](torch.cat((fx, Fy), dim=1))
            fy = self.vn_mlp_global[i](torch.cat((fy, Fx), dim=1))
            logging.info(f"global context fx {fx.size()}")
            logging.info(f"global context fy {fy.size()}")
            # 5. invariant mapping
            # inner product
            fx_par = torch.mean(fx, dim=1)/torch.norm(torch.mean(fx, dim=1))
            fy_par = torch.mean(fy, dim=1)/torch.norm(torch.mean(fy, dim=1))
            fx_par = fx_par.contiguous().view(batch_size, -1, 3).unsqueeze(2)
            fy_par = fy_par.contiguous().view(batch_size, -1, 3).unsqueeze(2)
            fx = fx.contiguous().view(batch_size, -1, 32, 3)
            fy = fy.contiguous().view(batch_size, -1, 32, 3)
            logging.info(f"fx_par {fx_par.size()}")
            logging.info(f"fy_par {fy_par.size()}")
            logging.info(f"fx {fx.size()}")
            logging.info(f"fy {fy.size()}")
            phi_x = torch.einsum('bnci,bncj->bnc', fx, fx_par)
            phi_y = torch.einsum('bnci,bncj->bnc', fy, fy_par)
            logging.info(f"phix {phi_x.size()}")
            logging.info(f"phiy {phi_y.size()}")
            # 6. softmax
            Sc = F.softmax(torch.einsum('bnc,bnc->bn', phi_x, phi_y), dim=-1)
            logging.info(f"Sc {Sc.size()}")
            # select top k correspondences from fx and fy based on Sc
            idx = torch.topk(Sc, 1024//8, dim=-1)[1]
            logging.info(f"idx {idx.size()}")
            # 8. global pooling
            fx = torch.gather(fx, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3))
            fy = torch.gather(fy, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3))
            logging.info(f"fx {fx.size()}")
            logging.info(f"fy {fy.size()}")
            fx = fx.view(b, c, 3, -1)
            fy = fy.view(b, c, 3, -1)
            fx_block.append(fx)
            fy_block.append(fy)
            logging.info(f"fx {fx.size()}")
            logging.info(f"fy {fy.size()}")
            logging.info(f"fx_block {len(fx_block)}")
        
        # enable logging
        # logging.disable(logging.NOTSET)
        logging.info(f"num blocks {len(fx_block)}")
        # 9. Hierarchical aggregation
        for i in range(self.num_blocks):
            fx_block[i] = torch.mean(fx_block[i], dim=-1)
            fy_block[i] = torch.mean(fy_block[i], dim=-1)
        logging.info(f"fx0 {fx_block[0].size()}")
        logging.info(f"fy0 {fy_block[0].size()}")
        Fx = self.vn_mlp_hierarchical(torch.cat(fx_block, dim=1))
        Fy = self.vn_mlp_hierarchical(torch.cat(fy_block, dim=1))
        logging.info(f"Fx {Fx.size()}")
        logging.info(f"Fy {Fy.size()}")
        # 10. 9Dof Alignment
        # use SVD to compute the rotation matrix between Fx and Fy
        H = torch.matmul(Fx.transpose(1, 2), Fy)
        u, s, v = torch.svd(H)
        logging.info(f"H {H.size()}")
        logging.info(f"s {s.size()}")
        logging.info(f"u {u.size()}")
        logging.info(f"v {v.size()}")
        R = torch.matmul(u, v)
        logging.info(f"R {R.size()}")
        S = torch.norm(Fy, dim=1)/torch.norm(Fx, dim=1)
        logging.info(f"S {S.size()}")
        return R, S



class HEGN_Loss(nn.Module):
    def __init__(self):
        super(HEGN_Loss, self).__init__()
    
    def forward(self, x_aligned, y, R, S, t, R_gt, S_gt, t_gt):
        t = t.squeeze()
        # compute registration loss 
        R_loss = torch.matmul(R, R_gt.transpose(1, 2)) - torch.eye(3).to(R.device)
        L_reg = torch.norm(R_loss)**2 + torch.norm(S - S_gt)**2 + torch.norm(t - t_gt)**2
        # compute chamfer distance
        L_chamfer, _ = chamfer_distance(x_aligned, y)
        return L_reg + L_chamfer