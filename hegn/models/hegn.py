import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from hegn.models.vn_dgcnn import VNDGCNN
from hegn.models.vn_layers import VNLinearLeakyReLU, VNBatchNorm, VNMaxPool, mean_pool
from hegn.utils.vn_dgcnn_util import get_graph_feature


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

     
class CrossContext(nn.Module):
    """
        Cross context aggregation module
        apply VN-TRANSFORMER to the input features
    """
    def __init__(self, feature_dim=32, atten_multi_head_c=16, k=16):
        super(CrossContext, self).__init__()
        self.k_nn = k
        self.atten_multi_head_c = atten_multi_head_c
        self.vn_mlp_q = VNLinearLeakyReLU(feature_dim, feature_dim, dim=3)
        self.chnorm = self.channel_equi_vec_normalize
        self.vn_mlp_k = VNLinearLeakyReLU(2*feature_dim, feature_dim)
        self.vn_mlp_v = VNLinearLeakyReLU(2*feature_dim, feature_dim)
    
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
        Qx = self.chnorm(self.vn_mlp_q(x))
        logging.info(f"Qx {Qx.size()}")
        y = get_graph_feature(y, k=self.k_nn)
        logging.info(f"y {y.size()}")
        Ky = self.chnorm(self.vn_mlp_k(y))
        Vy = self.vn_mlp_v(y)
        logging.info(f"Ky {Ky.size()}")
        logging.info(f"Vy {Vy.size()}")
        qk = (Ky * Qx[..., None]).sum(2)
        logging.info(f"qk {qk.size()}")
        B, C, N, K = qk.size()
        # N_head = C // self.atten_multi_head_c
        # qk = qk.view(B, N_head, self.atten_multi_head_c, N, K)
        atten = qk / torch.sqrt(torch.tensor(3 * C, dtype=torch.float32))
        atten = torch.softmax(atten, dim=-1)
        # atten = atten.expand(-1, -1, sel, -1, -1).contiguous()
        atten = atten.view(B, C, N, K).unsqueeze(2)
        logging.info(f"atten {atten.size()}")
        return x + (atten * Vy).sum(-1)


class GlobalContext(nn.Module):
    def __init__(self, mlp_in, mlp_out):
        super(GlobalContext, self).__init__()
        self.vn_mlp = VNLinearLeakyReLU(mlp_in, mlp_out, dim=3)
        self.pool = mean_pool
    
    def forward(self, fx):
        Fx = self.pool(fx, dim=-1, keepdim=True).expand(fx.size())
        return self.vn_mlp(torch.cat((fx, Fx), dim=1))
        
class InvariantMapping(nn.Module):
    def __init__(self, in_feat):
        super(InvariantMapping, self).__init__()
    
    def forward(self, fx, fy, topk):
        fx_mean = torch.mean(fx, dim=1)
        fy_mean = torch.mean(fy, dim=1)
        fx_par = fx_mean / (torch.norm(fx_mean, dim=1).unsqueeze(1).repeat(1, fx_mean.size(1), 1) + 1e-6)
        fy_par = fy_mean / (torch.norm(fy_mean, dim=1).unsqueeze(1).repeat(1, fy_mean.size(1), 1) + 1e-6)
        phi_x = torch.einsum('bcdn,bdn->bnc', fx, fx_par)
        phi_y = torch.einsum('bcdn,bdn->bnc', fy, fy_par)
        Sc = F.softmax(torch.einsum('bnc,bnc->bn', phi_x, phi_y), dim=-1)
        idx = torch.topk(Sc, Sc.shape[0]//topk, dim=-1)[1]
        b, c, n, s = fx.size()
        idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, c, 3, -1)
        fx = fx.gather(3, idx)
        fy = fy.gather(3, idx)
        return fx, fy

class HierarchicalAggregation(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(HierarchicalAggregation, self).__init__()
        self.vn_mlp = VNLinearLeakyReLU(in_feat, out_feat, dim=3)
    
    def forward(self, fx):
        return self.vn_mlp(torch.cat(fx))

class Alignment(nn.Module):
    def __init__(self):
        super(Alignment, self).__init__()
    
    def forward(self, fx, fy):
        H = torch.einsum('bcd,bce->bde', fx, fy)
        u, s, v = torch.svd(H)
        R = torch.matmul(v, u.transpose(1, 2))
        S = torch.norm(fy, dim=1)/torch.norm(fx, dim=1)
        return R, S


class HEGN(nn.Module):
    def __init__(self, args):
        super(HEGN, self).__init__()
        self.device = args.device
        self.n_knn = args.n_knn
        self.num_blocks = args.num_blocks
        self.topk = args.topk
        self.cross_context_feat = args.vngcnn_out
        self.local_context_feat = VNDGCNN(args.vngcnn_in[0], args.vngcnn_out[0], self.n_knn[0], pooling=args.pooling)
        self.cross_context = CrossContext(args.vngcnn_out[0], 16, self.n_knn[1])
        self.global_context = GlobalContext(args.vngcnn_out[0]*2, args.vngcnn_out[0]*2)
        self.invariant_mapping = InvariantMapping(args.vngcnn_out[0])
        self.hierarchical_aggregation = HierarchicalAggregation(args.vngcnn_out[0]*2, args.vngcnn_out[0])
        self.alignment = Alignment()
        self.pool = mean_pool

    def forward(self, x, y):
        logging.disable(logging.CRITICAL)
        fx_block = []
        fy_block = []
        # add feature dimension
        fx = x.unsqueeze(1)
        fy = y.unsqueeze(1)
        logging.info(f"fx {fx.size()}")
        logging.info(f"fy {fy.size()}")
        # 1. local context aggregation
        fx = self.local_context_feat(fx) # B, C, 3, N
        fy = self.local_context_feat(fy)
        logging.info(f"local context x {fx.size()}")
        logging.info(f"local context y {fy.size()}")
        # 3. cross context 
        fx = self.cross_context(fx, fy)
        fy = self.cross_context(fy, fx)
        logging.info(f"cross context fx {fx.size()}")
        logging.info(f"cross context fy {fy.size()}")
        # 4. global context aggregation
        fx = self.global_context(fx)
        fy = self.global_context(fy)
        logging.info(f"global context fx {fx.size()}")
        logging.info(f"global context fy {fy.size()}")
        # 5. invariant mapping
        fx, fy = self.invariant_mapping(fx, fy, 4)
        logging.info(f"inv fx {fx.size()}")
        logging.info(f"inv fy {fy.size()}")
        # 9. Hierarchical aggregation
        fx = self.pool(fx)
        fy = self.pool(fy)
        fx = self.hierarchical_aggregation([fx])
        fy = self.hierarchical_aggregation([fy])
        logging.info(f"fx {fx.size()}")
        logging.info(f"fy {fy.size()}")
        # 10. 9Dof Alignment
        R, S = self.alignment(fx, fy)
        logging.info(f"R {R.size()}")
        logging.info(f"S {S.size()}")
        return R, S

class HEGN_Loss(nn.Module):
    def __init__(self):
        super(HEGN_Loss, self).__init__()
    
    def forward(self, x_aligned, y, R, S, t, R_gt, S_gt, t_gt):
        t = t.squeeze()
        S = S.diagonal(dim1=-2, dim2=-1)
        S_gt = S_gt.diagonal(dim1=-2, dim2=-1)
        # compute registration loss 
        batch_size = R.shape[0]
        R_loss = torch.matmul(R_gt.transpose(1, 2), R) - torch.eye(3).to(R.device).unsqueeze(0).repeat(batch_size, 1, 1)
        L_reg = torch.norm(R_loss, dim=(1, 2))**2 + torch.norm(S - S_gt, dim=1)**2 + torch.norm(t - t_gt, dim=1)**2
        # compute chamfer distance
        # L_chamfer, _ = chamfer_distance(x_aligned, y)
        return L_reg.mean() #+ L_chamfer