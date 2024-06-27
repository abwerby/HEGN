import torch
import torch.nn as nn


class InvariantMapping(nn.Module):
    def __init__(self, in_channels):
        super(InvariantMapping, self).__init__()
        self.linear = nn.Linear(in_channels, in_channels)

    def forward(self, f):
        # Step 1: Compute the average vector feature along dimension C
        f_X_avg = torch.mean(f, dim=1, keepdim=True)  # Resulting shape B*1*D*N

        # Step 2: Normalize the average vector feature
        f_X_avg = f_X_avg / (torch.norm(f_X_avg, dim=2, keepdim=True) + 1e-6)  # Normalize along D

        # Step 3: Compute phi_X
        f = torch.einsum('bcdn,bcdn->bcn', f, f_X_avg)  # B*C*N*D

        return f


class InvariantNodePooling(nn.Module):
    def __init__(self, in_channels, K=256):
        super(InvariantNodePooling, self).__init__()
        self.K = K
        self.invariant_mapping_x = InvariantMapping(in_channels)
        self.invariant_mapping_y = InvariantMapping(in_channels)

    def forward(self, f_X, f_Y):
        # Compute invariant mappings
        B, C, D, N = f_X.shape  # B*C*3*N
        phi_X = self.invariant_mapping_x(f_X)  # shape: B*C*N
        phi_Y = self.invariant_mapping_y(f_Y)  # shape: B*C*N

        # Compute the inner product between phi_X and phi_Y along the last dimension
        # resulting in B*N. Use einsum to perform batched inner product
        s_C = torch.einsum('bcn,bcn->bn', phi_X, phi_Y)  # B*N

        del phi_X, phi_Y

        # Apply softmax to obtain the importance scores
        s_C = torch.nn.functional.softmax(s_C, dim=-1)  # Softmax along the N dimension

        # Rank nodes and select top K nodes
        _, idx = torch.topk(s_C, self.K, dim=-1, largest=True, sorted=True)  # Indices of top K nodes, B*K

        # Use idx to select top-ranking nodes for f_X and f_Y
        # idx is of shape B*K, we need to expand it to B*K*1*1 to use for advanced indexing
        idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, C, D, -1)  # B*K*D*N

        f_X = torch.gather(f_X, -1, idx)  # B*C*D*K
        f_Y = torch.gather(f_Y, -1, idx)  # B*C*D*K

        return f_X, f_Y
