import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from hegn.models.vn_dgcnn import VNDGCNN
from hegn.models.vn_layers import VNLinearLeakyReLU, VNBatchNorm, VNMaxPool, mean_pool, VNLinearAndLeakyReLU
from hegn.utils.vn_dgcnn_util import get_graph_feature


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

     
class cross_context(nn.Module):
    """
        Cross context aggregation module
        apply VN-TRANSFORMER to the input features
    """
    def __init__(self, feature_dim=32, atten_multi_head_c=16, k=16):
        super(cross_context, self).__init__()
        self.k_nn = k
        self.atten_multi_head_c = atten_multi_head_c
        self.vn_mlp_q = VNLinearAndLeakyReLU(feature_dim, feature_dim, dim=3)
        self.chnorm = self.channel_equi_vec_normalize
        self.vn_mlp_k = VNLinearAndLeakyReLU(2*feature_dim, feature_dim)
        self.vn_mlp_v = VNLinearAndLeakyReLU(2*feature_dim, feature_dim)
    
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
        N_head = C // self.atten_multi_head_c
        qk = qk.view(B, N_head, self.atten_multi_head_c, N, K)
        atten = qk.sum(2, keepdim=True) / torch.sqrt(torch.tensor(3 * self.atten_multi_head_c, dtype=torch.float32))
        atten = torch.softmax(atten, dim=-1)
        atten = atten.expand(-1, -1, self.atten_multi_head_c, -1, -1).contiguous()
        atten = atten.view(B, C, N, K).unsqueeze(2)
        logging.info(f"atten {atten.size()}")
        return x + (atten * Vy).sum(-1)
        return x 




class HEGN(nn.Module):
    def __init__(self, args):
        super(HEGN, self).__init__()
        self.device = args.device
        self.n_knn = args.n_knn
        self.num_blocks = args.num_blocks
        self.topk = args.topk
        self.cross_context_feat = args.vngcnn_out
        self.vn_dgcnn = [VNDGCNN(args.vngcnn_in[i], args.vngcnn_out[i], args.n_knn[i]).to(self.device) for i in range(self.num_blocks)]
        self.cross_context = [cross_context(self.cross_context_feat[i], 32, self.n_knn[i]).to(self.device) for i in range(self.num_blocks)]
        self.vn_mlp_global = [VNLinearAndLeakyReLU(self.cross_context_feat[i]*2, self.cross_context_feat[i], dim=3).to(self.device) for i in range(self.num_blocks)]
        self.vn_mlp_hierarchical = VNLinearAndLeakyReLU(np.sum(self.cross_context_feat), 32, dim=3).to(self.device)

    def forward(self, fx, fy):
        logging.disable(logging.CRITICAL)
        fx_block = []
        fy_block = []
        # add feature dimension
        fx = fx.unsqueeze(1)
        fy = fy.unsqueeze(1)
        logging.info(f"fx {fx.size()}")
        logging.info(f"fy {fy.size()}")
        for i in range(self.num_blocks):
            # 1. local context aggregation
            fx = self.vn_dgcnn[i](fx)
            fy = self.vn_dgcnn[i](fy) # B, C, 3, N
            b, c, n, s = fx.size()
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
            fx = self.vn_mlp_global[i](torch.cat((fx, Fx), dim=1))
            fy = self.vn_mlp_global[i](torch.cat((fy, Fy), dim=1))
            logging.info(f"global context fx {fx.size()}")
            logging.info(f"global context fy {fy.size()}")
            # 5. invariant mapping
            fx_mean = torch.mean(fx, dim=1, keepdim=True)
            fy_mean = torch.mean(fy, dim=1, keepdim=True)
            logging.info(f"fx_mean {fx_mean.size()}")
            logging.info(f"fy_mean {fy_mean.size()}")
            fx_par = fx_mean / torch.norm(fx_mean)
            fy_par = fy_mean / torch.norm(fy_mean)
            logging.info(f"fx_par {fx_par.size()}")
            logging.info(f"fy_par {fy_par.size()}")
            phi_x = torch.einsum('bcdn,bcdn->bnc', fx, fx_par)
            phi_y = torch.einsum('bcdn,bcdn->bnc', fy, fy_par)
            logging.info(f"phi_x {phi_x.size()}")
            logging.info(f"phi_y {phi_y.size()}")
            # 6. softmax
            Sc = F.softmax(torch.einsum('bnc,bnc->bn', phi_x, phi_y), dim=-1)
            logging.info(f"Sc {Sc.size()}")
            # select top k correspondences from fx and fy based on Sc
            logging.info(f"topk {Sc.size(1)//self.topk[i]}")
            idx = torch.topk(Sc, Sc.size(1)//self.topk[i], dim=-1)[1]
            logging.info(f"idx {idx.size()}")
            idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, c, 3, -1)
            fx = torch.gather(fx, -1, idx)
            fy = torch.gather(fy, -1, idx)
            fx_block.append(fx)
            fy_block.append(fy)
            logging.info(f"fx {fx.size()}")
            logging.info(f"fy {fy.size()}")
            logging.info(f"fx_block {len(fx_block)}")
        logging.info(f"num blocks {len(fx_block)}")
        logging.info(f"fx0 {fx_block[0].size()}")
        logging.info(f"fy0 {fy_block[0].size()}")
        # 9. Hierarchical aggregation
        for i in range(len(fx_block)):
            fx_block[i] = mean_pool(fx_block[i], dim=-1)
            fy_block[i] = mean_pool(fy_block[i], dim=-1)
        logging.info(f"fx cat {torch.cat(fx_block, dim=1).size()}")
        Fx = self.vn_mlp_hierarchical(torch.cat(fx_block, dim=1))
        Fy = self.vn_mlp_hierarchical(torch.cat(fy_block, dim=1))
        logging.info(f"Fx {Fx.size()}")
        logging.info(f"Fy {Fy.size()}")
        # 10. 9Dof Alignment
        # H = torch.einsum('bcd,bce->bde', Fx, Fy)
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
        S = torch.diagonal(S)
        S_gt = torch.diagonal(S_gt)
        # compute registration loss 
        batch_size = R.shape[0]
        R_loss = torch.matmul(R_gt.transpose(1, 2), R) - torch.eye(3).to(R.device).unsqueeze(0).repeat(batch_size, 1, 1)
        L_reg = torch.norm(R_loss)**2 + torch.norm(S - S_gt)**2 + torch.norm(t - t_gt)**2
        # compute chamfer distance
        L_chamfer, _ = chamfer_distance(x_aligned, y)
        return L_reg + L_chamfer