from __future__ import print_function

import model.mesh_utils as utils
import torch.nn as nn
import torch.nn.functional as F

import math
# A number of functions/classes are adopted from: https://github.com/jingraham/neurips19-graph-protein-design

from torch.autograd import Variable
from model.utils import loss_smoothed
from model.primitives import Linear

import torch
import numpy as np
import random
import time
import timeit
from ..chems import add_chem_features,light_PositionalEncoding

class trans_LKD_TO_LLL():
    def __init__(self,B, L, K, D):
        self.B = B
        self.L = L
        self.K = K
        self.D = D

    # 生成一个第一个数字为行号的，其余数值的范围为除了行号外的[0,K)的L*K随机矩阵
    def creat_K(self):
        x = torch.rand(self.L, self.K)
        # samples=random.sample(range(0,L),48)
        for i in range(0, self.L):
            samples = []
            samples.append(i)
            samples1 = random.sample(range(0, self.L), self.K - 1)
            samples += samples1
            samples = torch.tensor(samples, dtype=torch.long)
            x[i][:] = samples
        return x

    # 生成一个随机L*K*D+1的矩阵,这里我们设置D+1维度，是因为我不知道D维具体内容是什么，因此让L*K的index矩阵的
    # 信息存放在第0维度，剩下D维是正常的L*K*D的M1矩阵
    def creat_LKD(self):
        matrix = torch.zeros(self.L, self.K, self.D + 1)
        x = self.creat_K()
        # 生成D维的随机信息
        y = torch.rand(self.L, self.K, self.D + 1)
        y[:, :, 0] = x
        matrix = y
        return matrix

    # 生成一个L*L*D的空矩阵
    def creat_LLL(self):
        LLL = torch.ones(self.B,self.L, self.L, self.D)*-1
        return LLL

    # 将L*K*D填写回L*L*D中
    def write_back(self,index):

        LLL = self.creat_LLL()
        for i in range(self.L):
            for j in range(self.K):
                node_information = index[:,i, j]
                # temp = matrix[:,i, j, :]
                for b in range(self.B):
                    LLL[b,i, node_information[b], :] = j
        mask=LLL!=-1
        return LLL*mask,mask





def _interp(x, scale=4):
    """
    X:b*l,4,dim
    return;X:,B,L,4*nums-3,dim  #最后一个Ca 并为向后延伸插值
    """
    O = x[:, -1, :]
    a = x[:, :-1, :]  # [B*L,nums-1,:]
    b = x[:, 1:, :]  # [B*L,nums-1,:]
    diff = b - a  # [B*L,nums-1,:]
    sigma = diff / scale  ##[B*L,nums-1,:]
    new = []
    for i in range(scale):
        new.append(a + sigma * i)

    new = torch.stack(new, 2)  ##[B*L,4，(nums-1),:]
    new = torch.flatten(new, 1, 2)
    new = torch.cat((new, O.unsqueeze(1)), dim=-2)
    # lastca=x[:,-1,:].unsqueeze(1)
    # new=torch.cat([new,lastca],dim=-2)
    # cc=new[0,:,:].detach().cpu().numpy()
    # ccc=x[0, :, :].detach().cpu().numpy()
    return new


def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def _S_to_seq(S, mask):
    alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


class meshProteinFeaturesold(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=15, augment_eps=0., E_SSE=False):
        """ Extract protein features """
        super(meshProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.use_ESSE = E_SSE

        self.use_rbf_ture_and_indexdistance = True

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        edge_in_tri = num_positional_embeddings + int(num_rbf / 2) * 12 * 10

        self.edge_embedding_tri = nn.Linear(edge_in_tri, edge_features, bias=False)

        if self.use_ESSE:
            rbf_node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25 + 32 + 32
            self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        else:
            rbf_node_in, edge_ins = 6, num_positional_embeddings + num_rbf * 25
            self.edge_embeddings = nn.Linear(edge_ins, edge_features, bias=False)

        self.norm_edges_tri = nn.LayerNorm(edge_features)
        self.norm_edges = nn.LayerNorm(edge_features)

        self.SSE3_embedding = nn.Embedding(num_embeddings=4, embedding_dim=32)
        self.SSE8_embedding = nn.Embedding(num_embeddings=9, embedding_dim=32)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _tri_dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(int(self.top_k / 2), X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def mesh_rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 0., 15., int(self.num_rbf / 2)
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        neg_dis = torch.exp(-D_A_B_neighbors)
        return RBF_A_B  # ,neg_dis

    def forward(self, X, mask, residue_idx, SSE3_seq, SSE8_seq, use_tri=False, use_rbf=False, use_sse=False, ):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        if use_tri:
            # fake_ca = residue_idx.unsqueeze(-1)
            # zeros = torch.zeros_like(fake_ca).repeat(1, 1, 2)
            # fake_ca = torch.cat((fake_ca, zeros), -1)
            D_neighbors, E_idx_fake = self._tri_dist(Ca, mask)
            # E_idx=torch.ones(E_idx)*-100
            RBF_all = []
            RBF_all.append(self._get_rbf(N, N, E_idx_fake))  # N-N

            tri_data = []
            NCaC = torch.cat((N.unsqueeze(-2), Ca.unsqueeze(-2), C.unsqueeze(-2)), dim=-2)
            NCbCa = torch.cat((N.unsqueeze(-2), Cb.unsqueeze(-2), Ca.unsqueeze(-2)), dim=-2)
            NOCa = torch.cat((N.unsqueeze(-2), O.unsqueeze(-2), Ca.unsqueeze(-2)), dim=-2)

            NCO = torch.cat((N.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)), dim=-2)
            NCbC = torch.cat((N.unsqueeze(-2), Cb.unsqueeze(-2), C.unsqueeze(-2),), dim=-2)
            NOCb = torch.cat((N.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)

            CaOCb = torch.cat((Ca.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)
            CaCbC = torch.cat((Ca.unsqueeze(-2), Cb.unsqueeze(-2), C.unsqueeze(-2)), dim=-2)
            CaCO = torch.cat((Ca.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)), dim=-2)

            COCb = torch.cat((C.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)

            tri_data.append(NCaC)
            tri_data.append(NCbCa)
            tri_data.append(NOCa)
            tri_data.append(NCO)
            tri_data.append(NCbC)
            tri_data.append(NOCb)
            tri_data.append(CaOCb)
            tri_data.append(CaCbC)
            tri_data.append(CaCO)
            tri_data.append(COCb)

            tri_feature = []
            for tri in tri_data:
                mesh_batch, base_batch = _generate_mesh(tri, E_idx_fake)
                tri_feature_ = self._tri_feature(base_batch, mesh_batch)

                tri_feature_ = self.mesh_rbf(tri_feature_).flatten(-2, -1)
                # mask_tri = mask.view(mask.shape[0], mask.shape[1], 1, 1).expand(tri_feature_.shape)
                tri_feature.append(tri_feature_)

            tri_feature = torch.cat(tri_feature, dim=-1)

            offset = residue_idx[:, :, None] - residue_idx[:, None, :]
            offset = gather_edges(offset[:, :, :, None], E_idx_fake)[:, :, :, 0]  # [B, L, K]

            E_positional = self.embeddings(offset.long())
            E_tri = torch.cat((E_positional, tri_feature), -1)
            E_tri = self.edge_embedding_tri(E_tri)
            E_tri = self.norm_edges_tri(E_tri)

        if use_rbf is not False:
            D_neighbors, E_idx = self._dist(Ca, mask)
            RBF_all = []
            RBF_scale = []
            RBF_all.append(self._get_rbf(Ca, Ca, E_idx))  # Ca-Ca

            RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
            RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
            RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
            RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
            RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
            RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
            RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
            RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
            RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
            RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
            RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
            RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
            RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
            RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
            RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
            RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
            RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
            RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
            RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
            RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
            RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
            RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
            RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
            RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O

            # RBF_embedding = []
            # for rbf in RBF_all:
            #     RBF_embedding.append(rbf[0])
            #     RBF_scale.append(torch.as_tensor(rbf[1], device=X.device).unsqueeze(-1))

            RBF_all = torch.cat(tuple(RBF_all), dim=-1)

            offset = residue_idx[:, :, None] - residue_idx[:, None, :]
            offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

            E_positional = self.embeddings(offset.long())

            # SSE feature

            sse3 = self.SSE3_embedding(SSE3_seq)
            sse8 = self.SSE8_embedding(SSE8_seq)
            if use_sse == False:
                sse3 = torch.zeros_like(sse3)
                sse8 = torch.zeros_like(sse8)

            if self.use_ESSE:
                E_SSE3 = gather_nodes(sse3, E_idx)
                E_SSE8 = gather_nodes(sse8, E_idx)
                E = torch.cat((E_positional, RBF_all, E_SSE3, E_SSE8), -1)
                E = self.edge_embedding(E)
            else:
                E = torch.cat((E_positional, RBF_all), -1)
                E = self.edge_embeddings(E)

            E = self.norm_edges(E)

        else:
            # SSE feature
            sse3 = self.SSE3_embedding(SSE3_seq)
            sse8 = self.SSE8_embedding(SSE8_seq)
            if use_sse == False:
                sse3 = torch.zeros_like(sse3)
                sse8 = torch.zeros_like(sse8)
            E_SSE3 = gather_nodes(sse3, E_idx_fake)
            E_SSE8 = gather_nodes(sse8, E_idx_fake)
            E = torch.zeros_like(E_tri)
            E_idx = E_idx_fake

        if use_tri:
            B, L, EDGE, D = E_tri.shape
            padingzero = torch.zeros(size=(B, L, E.size(-2) - EDGE, D), device=E_tri.device)
            E_tri = torch.cat((E_tri, padingzero), dim=-2)
            E = E + E_tri

        hv = torch.cat((sse3, sse8), dim=-1)

        return E, E_idx, hv

    def _tri_feature(self, query_triangle_pos, face_neighbor_pos):
        tri_feature = encode_points_and_triangles(query_triangle_pos, face_neighbor_pos
                                                  )  # L,1,F,13
        return tri_feature


class meshProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=15, augment_eps=0., aatypes=21, E_SSE=False, E_aatye=False):
        """ Extract protein features """
        super(meshProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.use_ESSE = E_SSE

        self.use_rbf_ture_and_indexdistance = True

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        edge_in_tri = num_positional_embeddings + int(num_rbf / 2) * 12 * 10

        self.edge_embedding_tri = nn.Linear(edge_in_tri, edge_features, bias=False)

        if self.use_ESSE:
            rbf_node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25 + 32 + 32
            self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        else:
            rbf_node_in, edge_ins = 6, num_positional_embeddings + num_rbf * 25
            self.edge_embeddings = nn.Linear(edge_ins, edge_features, bias=False)

        self.E_aatye = E_aatye
        if self.E_aatye:
            self.aatype_embeddings = nn.Linear(num_positional_embeddings + aatypes, edge_features)
            self.norm_aatypes = nn.LayerNorm(edge_features)




        self.norm_edges_tri = nn.LayerNorm(edge_features)
        self.norm_edges = nn.LayerNorm(edge_features)

        self.SSE3_embedding = nn.Embedding(num_embeddings=4, embedding_dim=32)
        self.SSE8_embedding = nn.Embedding(num_embeddings=9, embedding_dim=32)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _tri_dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(int(self.top_k / 2), X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def mesh_rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 0., 15., int(self.num_rbf / 2)
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        neg_dis = torch.exp(-D_A_B_neighbors)
        return RBF_A_B  # ,neg_dis

    def forward(self, S, X, mask, residue_idx, SSE3_seq, SSE8_seq, use_tri=False, use_rbf=False, use_sse=False, ):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        if use_tri:
            # fake_ca = residue_idx.unsqueeze(-1)
            # zeros = torch.zeros_like(fake_ca).repeat(1, 1, 2)
            # fake_ca = torch.cat((fake_ca, zeros), -1)
            D_neighbors, E_idx_fake = self._tri_dist(Ca, mask)
            # E_idx=torch.ones(E_idx)*-100
            RBF_all = []
            RBF_all.append(self._get_rbf(N, N, E_idx_fake))  # N-N

            tri_data = []
            NCaC = torch.cat((N.unsqueeze(-2), Ca.unsqueeze(-2), C.unsqueeze(-2)), dim=-2)
            NCbCa = torch.cat((N.unsqueeze(-2), Cb.unsqueeze(-2), Ca.unsqueeze(-2)), dim=-2)
            NOCa = torch.cat((N.unsqueeze(-2), O.unsqueeze(-2), Ca.unsqueeze(-2)), dim=-2)

            NCO = torch.cat((N.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)), dim=-2)
            NCbC = torch.cat((N.unsqueeze(-2), Cb.unsqueeze(-2), C.unsqueeze(-2),), dim=-2)
            NOCb = torch.cat((N.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)

            CaOCb = torch.cat((Ca.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)
            CaCbC = torch.cat((Ca.unsqueeze(-2), Cb.unsqueeze(-2), C.unsqueeze(-2)), dim=-2)
            CaCO = torch.cat((Ca.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)), dim=-2)

            COCb = torch.cat((C.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)

            tri_data.append(NCaC)
            tri_data.append(NCbCa)
            tri_data.append(NOCa)
            tri_data.append(NCO)
            tri_data.append(NCbC)
            tri_data.append(NOCb)
            tri_data.append(CaOCb)
            tri_data.append(CaCbC)
            tri_data.append(CaCO)
            tri_data.append(COCb)

            tri_feature = []
            for tri in tri_data:
                mesh_batch, base_batch = _generate_mesh(tri, E_idx_fake)
                tri_feature_ = self._tri_feature(base_batch, mesh_batch)

                tri_feature_ = self.mesh_rbf(tri_feature_).flatten(-2, -1)
                # mask_tri = mask.view(mask.shape[0], mask.shape[1], 1, 1).expand(tri_feature_.shape)
                tri_feature.append(tri_feature_)

            tri_feature = torch.cat(tri_feature, dim=-1)

            offset = residue_idx[:, :, None] - residue_idx[:, None, :]
            offset = gather_edges(offset[:, :, :, None], E_idx_fake)[:, :, :, 0]  # [B, L, K]

            E_positional = self.embeddings(offset.long())
            E_tri = torch.cat((E_positional, tri_feature), -1)
            E_tri = self.edge_embedding_tri(E_tri)
            E_tri = self.norm_edges_tri(E_tri)

        if use_rbf is not False:
            D_neighbors, E_idx = self._dist(Ca, mask)
            RBF_all = []
            RBF_scale = []
            RBF_all.append(self._get_rbf(Ca, Ca, E_idx))  # Ca-Ca

            RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
            RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
            RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
            RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
            RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
            RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
            RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
            RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
            RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
            RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
            RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
            RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
            RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
            RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
            RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
            RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
            RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
            RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
            RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
            RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
            RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
            RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
            RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
            RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O

            # RBF_embedding = []
            # for rbf in RBF_all:
            #     RBF_embedding.append(rbf[0])
            #     RBF_scale.append(torch.as_tensor(rbf[1], device=X.device).unsqueeze(-1))

            RBF_all = torch.cat(tuple(RBF_all), dim=-1)

            offset = residue_idx[:, :, None] - residue_idx[:, None, :]
            offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

            E_positional = self.embeddings(offset.long())

            # SSE feature

            sse3 = self.SSE3_embedding(SSE3_seq)
            sse8 = self.SSE8_embedding(SSE8_seq)
            if use_sse == False:
                sse3 = torch.zeros_like(sse3)
                sse8 = torch.zeros_like(sse8)
            if self.E_aatye:
                # sequence featires
                S = torch.nn.functional.one_hot(S, 21).type(torch.float)
                E_S = gather_nodes(S, E_idx)
                E_S = torch.cat((E_positional, E_S), -1)
                E_S = self.aatype_embeddings(E_S)
                E_S = self.norm_aatypes(E_S)

            if self.use_ESSE:
                E_SSE3 = gather_nodes(sse3, E_idx)
                E_SSE8 = gather_nodes(sse8, E_idx)

                E = torch.cat((E_positional, RBF_all, E_SSE3, E_SSE8), -1)
                E = self.edge_embedding(E)
            else:
                E = torch.cat((E_positional, RBF_all), -1)
                E = self.edge_embeddings(E)

            E = self.norm_edges(E)

        else:
            # SSE feature
            sse3 = self.SSE3_embedding(SSE3_seq)
            sse8 = self.SSE8_embedding(SSE8_seq)
            if use_sse == False:
                sse3 = torch.zeros_like(sse3)
                sse8 = torch.zeros_like(sse8)
            E_SSE3 = gather_nodes(sse3, E_idx_fake)
            E_SSE8 = gather_nodes(sse8, E_idx_fake)
            E = torch.zeros_like(E_tri)
            E_idx = E_idx_fake

        if use_tri:
            B, L, EDGE, D = E_tri.shape
            padingzero = torch.zeros(size=(B, L, E.size(-2) - EDGE, D), device=E_tri.device)
            E_tri = torch.cat((E_tri, padingzero), dim=-2)
            if self.E_aatye:
                E = (E + E_tri + E_S) / 3
            else:
                E = (E + E_tri) / 2

        hv = torch.cat((sse3, sse8), dim=-1)

        return E, E_idx, hv

    def _tri_feature(self, query_triangle_pos, face_neighbor_pos):
        tri_feature = encode_points_and_triangles(query_triangle_pos, face_neighbor_pos
                                                  )  # L,1,F,13
        return tri_feature


class meshProteinFeatures_withchem(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=15, augment_eps=0., aatypes=21, E_SSE=False, E_aatye=False,usechem=True):
        """ Extract protein features """
        super(meshProteinFeatures_withchem, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.usechem=usechem
        self.use_ESSE = E_SSE

        self.use_rbf_ture_and_indexdistance = True

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        edge_in_tri = num_positional_embeddings + int(num_rbf / 2) * 12 * 10

        self.edge_embedding_tri = nn.Linear(edge_in_tri, edge_features, bias=False)

        if self.use_ESSE:
            rbf_node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25 + 32 + 32
            self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        else:
            rbf_node_in, edge_ins = 6, num_positional_embeddings + num_rbf * 25
            self.edge_embeddings = nn.Linear(edge_ins, edge_features, bias=False)

        self.E_aatye = E_aatye
        if self.E_aatye:
            self.aatype_embeddings = nn.Linear(num_positional_embeddings + aatypes, edge_features)
            self.norm_aatypes = nn.LayerNorm(edge_features)

        if self.usechem:
            self.chem_features=add_chem_features(chem_dim=node_features-aatypes)
            self.msa_s = nn.Linear(node_features , node_features, bias=False)
            self.chemlayernorm = nn.LayerNorm(node_features)

        self.norm_edges_tri = nn.LayerNorm(edge_features)
        self.norm_edges = nn.LayerNorm(edge_features)

        self.SSE3_embedding = nn.Embedding(num_embeddings=4, embedding_dim=32)
        self.SSE8_embedding = nn.Embedding(num_embeddings=9, embedding_dim=32)

    def _dist(self, X, mask,eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max

        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _tri_dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(int(self.top_k / 2), X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def mesh_rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 0., 15., int(self.num_rbf / 2)
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        neg_dis = torch.exp(-D_A_B_neighbors)
        return RBF_A_B  # ,neg_dis

    def forward(self, S, X, mask, residue_idx, SSE3_seq, SSE8_seq, use_tri=False, use_rbf=False, use_sse=False, ):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        if use_tri:
            # fake_ca = residue_idx.unsqueeze(-1)
            # zeros = torch.zeros_like(fake_ca).repeat(1, 1, 2)
            # fake_ca = torch.cat((fake_ca, zeros), -1)
            D_neighbors, E_idx_fake = self._tri_dist(Ca, mask)
            # E_idx=torch.ones(E_idx)*-100
            RBF_all = []
            RBF_all.append(self._get_rbf(N, N, E_idx_fake))  # N-N

            tri_data = []
            NCaC = torch.cat((N.unsqueeze(-2), Ca.unsqueeze(-2), C.unsqueeze(-2)), dim=-2)
            NCbCa = torch.cat((N.unsqueeze(-2), Cb.unsqueeze(-2), Ca.unsqueeze(-2)), dim=-2)
            NOCa = torch.cat((N.unsqueeze(-2), O.unsqueeze(-2), Ca.unsqueeze(-2)), dim=-2)

            NCO = torch.cat((N.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)), dim=-2)
            NCbC = torch.cat((N.unsqueeze(-2), Cb.unsqueeze(-2), C.unsqueeze(-2),), dim=-2)
            NOCb = torch.cat((N.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)

            CaOCb = torch.cat((Ca.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)
            CaCbC = torch.cat((Ca.unsqueeze(-2), Cb.unsqueeze(-2), C.unsqueeze(-2)), dim=-2)
            CaCO = torch.cat((Ca.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)), dim=-2)

            COCb = torch.cat((C.unsqueeze(-2), O.unsqueeze(-2), Cb.unsqueeze(-2)), dim=-2)

            tri_data.append(NCaC)
            tri_data.append(NCbCa)
            tri_data.append(NOCa)
            tri_data.append(NCO)
            tri_data.append(NCbC)
            tri_data.append(NOCb)
            tri_data.append(CaOCb)
            tri_data.append(CaCbC)
            tri_data.append(CaCO)
            tri_data.append(COCb)

            tri_feature = []
            for tri in tri_data:
                mesh_batch, base_batch = _generate_mesh(tri, E_idx_fake)
                tri_feature_ = self._tri_feature(base_batch, mesh_batch)

                tri_feature_ = self.mesh_rbf(tri_feature_).flatten(-2, -1)
                # mask_tri = mask.view(mask.shape[0], mask.shape[1], 1, 1).expand(tri_feature_.shape)
                tri_feature.append(tri_feature_)

            tri_feature = torch.cat(tri_feature, dim=-1)

            offset = residue_idx[:, :, None] - residue_idx[:, None, :]
            offset = gather_edges(offset[:, :, :, None], E_idx_fake)[:, :, :, 0]  # [B, L, K]

            E_positional = self.embeddings(offset.long())
            E_tri = torch.cat((E_positional, tri_feature), -1)
            E_tri = self.edge_embedding_tri(E_tri)
            E_tri = self.norm_edges_tri(E_tri)

        if use_rbf is not False:
            D_neighbors, E_idx = self._dist(Ca, mask)
            RBF_all = []
            RBF_scale = []
            RBF_all.append(self._get_rbf(Ca, Ca, E_idx))  # Ca-Ca

            RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
            RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
            RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
            RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
            RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
            RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
            RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
            RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
            RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
            RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
            RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
            RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
            RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
            RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
            RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
            RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
            RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
            RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
            RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
            RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
            RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
            RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
            RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
            RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O

            # RBF_embedding = []
            # for rbf in RBF_all:
            #     RBF_embedding.append(rbf[0])
            #     RBF_scale.append(torch.as_tensor(rbf[1], device=X.device).unsqueeze(-1))

            RBF_all = torch.cat(tuple(RBF_all), dim=-1)

            offset = residue_idx[:, :, None] - residue_idx[:, None, :]
            offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

            E_positional = self.embeddings(offset.long())

            # SSE feature

            sse3 = self.SSE3_embedding(SSE3_seq)
            sse8 = self.SSE8_embedding(SSE8_seq)
            if use_sse == False:
                sse3 = torch.zeros_like(sse3)
                sse8 = torch.zeros_like(sse8)
            if self.E_aatye:
                # sequence featires
                S = torch.nn.functional.one_hot(S, 21).type(torch.float)
                E_S = gather_nodes(S, E_idx)
                E_S = torch.cat((E_positional, E_S), -1)
                E_S = self.aatype_embeddings(E_S)
                E_S = self.norm_aatypes(E_S)

            if self.use_ESSE:
                E_SSE3 = gather_nodes(sse3, E_idx)
                E_SSE8 = gather_nodes(sse8, E_idx)

                E = torch.cat((E_positional, RBF_all, E_SSE3, E_SSE8), -1)
                E = self.edge_embedding(E)
            else:
                E = torch.cat((E_positional, RBF_all), -1)
                E = self.edge_embeddings(E)

            E = self.norm_edges(E)

        else:
            # SSE feature
            sse3 = self.SSE3_embedding(SSE3_seq)
            sse8 = self.SSE8_embedding(SSE8_seq)
            if use_sse == False:
                sse3 = torch.zeros_like(sse3)
                sse8 = torch.zeros_like(sse8)
            E_SSE3 = gather_nodes(sse3, E_idx_fake)
            E_SSE8 = gather_nodes(sse8, E_idx_fake)
            E = torch.zeros_like(E_tri)
            E_idx = E_idx_fake

        if use_tri:
            B, L, EDGE, D = E_tri.shape
            padingzero = torch.zeros(size=(B, L, E.size(-2) - EDGE, D), device=E_tri.device)
            E_tri = torch.cat((E_tri, padingzero), dim=-2)
            if self.E_aatye:
                E = (E + E_tri + E_S) / 3
            else:
                E = (E + E_tri) / 2


        if self.usechem:

            chem_s = self.chem_features(S)
            msa = torch.cat((S, chem_s), dim=-1)  # [*,2*msa_dim]

            # get padding mask
            padding_mask = (residue_idx != -100).type(torch.float)[:,:,None]

            pos = light_PositionalEncoding(residue_idx, msa.shape[-1])
            #pos = pos.unsqueeze(1)#.repeat(1, msa.shape[1], 1, 1)
            msa = self.msa_s(msa + pos)  #
            msa = msa * padding_mask
            msa=self.chemlayernorm(msa)
        #hv = torch.cat((sse3, sse8), dim=-1)
        hv=msa

        return E, E_idx, hv

    def _tri_feature(self, query_triangle_pos, face_neighbor_pos):
        tri_feature = encode_points_and_triangles(query_triangle_pos, face_neighbor_pos
                                                  )  # L,1,F,13
        return tri_feature


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncoderLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=48):
        super(EncoderLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        # self.HE_up=nn.Linear(num_hidden, int(num_hidden-delat_dim), bias=True)

        # Encoder layers
        self.Trans_layers = nn.ModuleList([
            Transformerlayer(num_hidden, dropout=dropout)
            for _ in range(1)
        ])

    def forward(self, h_V, h_E, E_idx, mask_V, mask_attend):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
            mask_V = mask_V.squeeze(-1)

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))

        for layer in self.Trans_layers:
            h_V = layer(h_V, mask_V)
            # x=h_V.detach()[0][:,:,0].cpu().numpy()#*255
            # pyplot.imshow(x)
            # pyplot.show()

        return h_V, h_E


class Transformerlayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.1):
        super(Transformerlayer, self).__init__()

        self.num_heads = 4

        feature_dim = num_hidden
        feature_dim_t = feature_dim
        self.transformer_layer = nn.MultiheadAttention(embed_dim=feature_dim_t, num_heads=self.num_heads, dropout=0.1,
                                                       batch_first=True)
        self.res_attn_layer_norm = torch.nn.LayerNorm(feature_dim_t)
        self.res_outer_attn_layer_norm = torch.nn.LayerNorm(feature_dim_t)

        self.node_fc1 = self.build_fc1(feature_dim_t, feature_dim_t * 2)
        self.node_fc2 = self.build_fc2(feature_dim_t * 2, feature_dim_t)
        self.activation_fn = nn.GELU()
        self.dropout_module = nn.Dropout(dropout)
        self.res_out = nn.Linear(feature_dim_t, feature_dim_t, bias=True)

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def residual_connection(self, x, residual):
        return residual + x

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def forward(self, h_V, mask_V):
        """ Parallel computation of full transformer layer """

        residual = h_V
        paddingmask = torch.logical_not(mask_V.squeeze(-1))
        h_V, atten = self.transformer_layer(query=h_V, key=h_V, value=h_V, key_padding_mask=paddingmask)
        # attennp=atten.detach().cpu().numpy()[0]
        assert not torch.any(torch.isnan(h_V))

        h_V = self.dropout_module(h_V)
        h_V = self.residual_connection(h_V, residual)
        h_V = self.res_attn_layer_norm(h_V)

        # node  FFN
        res_features_residual = h_V
        h_V = self.activation_fn(self.node_fc1(h_V))
        h_V = self.node_fc2(h_V)
        h_V = self.dropout_module(h_V)
        h_V = self.residual_connection(h_V, res_features_residual)
        # Transformer 模块的res输出
        h_V = self.res_attn_layer_norm(h_V)
        assert not torch.any(torch.isnan(h_V))

        h_V = self.activation_fn(self.res_out(h_V))

        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


def _generate_mesh(points_batchs, E_idx, l_shift=1):
    """

    Args:
        points: batch,L,3
        l_shift:

    Returns:
       batch， L,2*l_shift+1,3,3
    """
    mesh_batch = []
    base_batch = []

    for i in range(len(points_batchs)):
        points = points_batchs[i]
        Eix = E_idx[i]
        mp = torch.index_select(points, 0, Eix.flatten())
        mesh = mp.view(Eix.shape[0], Eix.shape[1], mp.shape[-2], mp.shape[-1])[:, :, :3, :]
        mesh_data = mesh - points[:, 1, :].unsqueeze(1).unsqueeze(1)

        base = points[:, :3, :] - points[:, 1, :].unsqueeze(1)
        mesh_batch.append(mesh_data)
        base_batch.append(base)

    mesh_batch = torch.stack(mesh_batch).to(points.device)
    base_batch = torch.stack(base_batch).to(points.device)

    return mesh_batch, base_batch  # ,dihedral_batch


def generate_coords(points_pos, query_triangles_pos):
    EPS = 1e-6

    # First, compute and remove the normal component
    area_normals = 0.5 * torch.cross(
        query_triangles_pos[:, :, 1, :] - query_triangles_pos[:, :, 0, :],
        query_triangles_pos[:, :, 2, :] - query_triangles_pos[:, :, 0, :], dim=-1)

    areas = utils.norm(area_normals) + EPS  # (B, Q)
    normals = area_normals / areas.unsqueeze(-1)  # (B, Q, 3)
    barycenters = torch.mean(query_triangles_pos, dim=2)  # (B, Q, 3)
    centered_neighborhood = points_pos - barycenters.unsqueeze(2)
    normal_comp = utils.dot(normals.unsqueeze(2), centered_neighborhood)
    neighborhood_planar = points_pos - normals.unsqueeze(2) * normal_comp.unsqueeze(-1)  # 点在平面上的位置

    # Compute barycentric coordinates in plane
    def coords_i(i):
        point_area = 0.5 * utils.dot(
            normals.unsqueeze(2),
            torch.cross(
                query_triangles_pos[:, :, (i + 1) % 3, :].unsqueeze(2) - neighborhood_planar,
                query_triangles_pos[:, :, (i + 2) % 3, :].unsqueeze(2) - neighborhood_planar,
                dim=-1)
        )

        area_frac = (point_area + EPS / 3.) / areas.unsqueeze(-1)
        return area_frac

    BARY_MAX = 5.
    u = torch.clamp(coords_i(0), -BARY_MAX, BARY_MAX)
    v = torch.clamp(coords_i(1), -BARY_MAX, BARY_MAX)
    w = torch.clamp(coords_i(2), -BARY_MAX, BARY_MAX)

    # Compute cartesian coordinates with the x-axis along the i --> j edge
    basisX = utils.normalize(query_triangles_pos[:, :, 1, :] - query_triangles_pos[:, :, 0, :])
    basisY = utils.normalize(torch.cross(normals, basisX))
    x_comp = utils.dot(basisX.unsqueeze(2), centered_neighborhood)
    y_comp = utils.dot(basisY.unsqueeze(2), centered_neighborhood)

    coords = torch.stack((x_comp, y_comp, normal_comp, u, v, w), dim=-1)

    return coords


def encode_points_and_triangles(query_triangles_pos,
                                nearby_triangles_pos=None, nearby_triangle_probs=None):
    B = query_triangles_pos.shape[0]
    Q = query_triangles_pos.shape[1]
    # K = nearby_points_pos.shape[2]

    have_triangles = (nearby_triangles_pos is not None)

    if have_triangles:
        K_T = nearby_triangles_pos.shape[2]

    # Normalize neighborhood (translation won't matter, but unit scale is nice)
    # note that we normalize vs. the triangle, not vs. the points
    neigh_centers = torch.mean(query_triangles_pos, dim=2)  # (B, Q, 3) zhongxin de x y z
    neigh_scales = torch.mean(utils.norm(query_triangles_pos - neigh_centers.unsqueeze(2)), dim=-1) + 1e-5  # (B, Q)

    # nearby_points_pos = nearby_points_pos.clone() / neigh_scales.unsqueeze(-1).unsqueeze(-1)
    query_triangles_pos = query_triangles_pos / neigh_scales.unsqueeze(-1).unsqueeze(-1)
    if have_triangles:
        nearby_triangles_pos = nearby_triangles_pos / neigh_scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Encode the nearby points
    # point_coords = generate_coords(nearby_points_pos, query_triangles_pos)

    # Encode the nearby triangles
    if have_triangles:
        try:
            tri_coords = generate_coords(nearby_triangles_pos.view(B, Q, K_T * 3, 3), query_triangles_pos).view(B, Q,
                                                                                                                K_T,
                                                                                                                3, 6)
        except:
            print("-----------")
        max_vals = torch.max(tri_coords, dim=3).values  # (B, Q, K_T, 6)
        min_vals = torch.min(tri_coords, dim=3).values  # (B, Q, K_T, 6)
        triangle_coords = torch.cat((min_vals, max_vals), dim=-1)
        # triangle_coords=triangle_coords.flatten(-2,-1)

    # if have_triangles:
    return triangle_coords
    # else:
    #     return point_coords


class Generator(nn.Module):
    def __init__(self, node_features, edge_features,
                 hidden_dim, num_encoder_layers=6,
                 k_neighbors=48, augment_eps=0.0, dropout=0.1, vocab=21, use_ESSE=False, use_Eaatype=False, **kwargs):
        super(Generator, self).__init__()

        # Hyperparameters
        self.k_neighbors = k_neighbors

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = meshProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps,
                                            E_SSE=use_ESSE, E_aatye=use_Eaatype)  #

        self.W_h = nn.Linear(64, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.linearouts = nn.Sequential(
            nn.Linear(node_features, node_features),
            nn.ReLU(),
            nn.Linear(node_features, vocab),
        )
        # self.linearout = nn.Sequential(nn.Linear(node_features, vocab), )

        self.W_ss = nn.Linear(21, 256)

        self.ce = nn.CrossEntropyLoss()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, S, X, mask, residue_idx, SSE3_seq=None, SSE8_seq=None, use_tri=True, use_rbf=True, use_sse=False,log_softmax=True,TrainG=False
                ,**kwargs):
        """ Graph-conditioned sequence model """
        device = X.device
        batch, L = X.shape[0], X.shape[1]
        # Prepare node and edge embeddings

        if SSE3_seq == None or SSE8_seq == None:
            SSE3_seq = torch.zeros((batch, L), device=device).type(torch.long)
            SSE8_seq = torch.zeros((batch, L), device=device).type(torch.long)
        E, E_idx, h_V = self.features(None, X, mask, residue_idx, SSE3_seq, SSE8_seq, use_tri=use_tri, use_rbf=use_rbf,
                                      use_sse=use_sse)  # ,


        h_V = self.W_h(h_V)

        h_E = self.W_e(E)

        # # # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        if log_softmax:
            result = self.linearouts(h_V)
            logtis =F.log_softmax(result,dim=-1)
        else:
            logtis = self.linearouts(h_V)
        loss_seq, seq_recovery_rate_g = self._get_ce_loss(S, logtis, mask)
        if TrainG:
            return loss_seq, seq_recovery_rate_g
        msa_num = kwargs['msas']
        samples = []
        for i in range(msa_num):
            xx = F.gumbel_softmax(logtis, tau=1, hard=False)  # .type(torch.float)
            #aa=xx[-1,:].type(torch.float).detach().cpu().numpy()
            # print(torch.argmax(xx,-1))
            samples.append(xx)

        return loss_seq, seq_recovery_rate_g, samples, h_V,E_idx  #F.gumbel_softmax(logtis, tau=1, hard=False)   [logtis]



    def design(self, S, X, mask, residue_idx, SSE3_seq=None, SSE8_seq=None, use_tri=True, use_rbf=True, use_sse=False,log_softmax=False
                ,**kwargs):
        """ Graph-conditioned sequence model """
        device = X.device
        batch, L = X.shape[0], X.shape[1]
        # Prepare node and edge embeddings

        if SSE3_seq == None or SSE8_seq == None:
            SSE3_seq = torch.zeros((batch, L), device=device).type(torch.long)
            SSE8_seq = torch.zeros((batch, L), device=device).type(torch.long)
        E, E_idx, h_V = self.features(None, X, mask, residue_idx, SSE3_seq, SSE8_seq, use_tri=use_tri, use_rbf=use_rbf,
                                      use_sse=use_sse)


        h_V = self.W_h(h_V)

        h_E = self.W_e(E)

        # # # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        if log_softmax:
            result = self.linearouts(h_V)
            logtis =F.log_softmax(result,dim=-1)
        else:
            logtis = self.linearouts(h_V)
        loss_seq, seq_recovery_rate_g = self._get_ce_loss(S, logtis, mask)

        aatype=torch.argmax(logtis,dim=-1)
        result={
            'aatype':aatype,
            'recovery':seq_recovery_rate_g.detach().cpu().numpy()
        }

        return result


    def _get_ce_loss(self, S, log_probs, mask):

        scores, loss_seq = loss_smoothed(S, log_probs, mask, 0.1)

        pred = (torch.argmax(log_probs, dim=-1) * mask).detach().type(torch.int)
        true = (S * mask).detach().type(torch.int)

        this_correct = ((pred == true).sum() - (1 - mask.detach()).sum())
        thisnods = torch.sum(mask)
        seq_recovery_rate = 100 * this_correct / thisnods

        return loss_seq, seq_recovery_rate

    def _positional_embeddings(self, pos,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings

        pse = []
        for idx in pos:
            d = idx

            frequency = torch.exp(
                torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=pos.device)
                * -(np.log(10000.0) / num_embeddings)
            )
            angles = d.unsqueeze(-1) * frequency
            E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
            pse.append(E)
        pse = torch.stack(pse)
        return pse


class Graph_MSA(nn.Module):
    def __init__(self, node_features, edge_features,
                 hidden_dim, num_encoder_layers=3,
                 k_neighbors=48,  dropout=0.1,):
        super(Graph_MSA, self).__init__()

        # Hyperparameters
        self.k_neighbors = k_neighbors

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim


        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

    def _dist(self, X, mask,eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max

        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum( self.k_neighbors , X.shape[1]), dim=-1, largest=False)
        return  E_idx

    def forward(self,  msa, mask, Ca
                ):
        """ Graph-conditioned sequence model """
        device = msa.device
        batch, L = msa.shape[0], msa.shape[1]
        # Prepare node and edge embeddings
        msa=msa.squeeze(1)
        E_idx = self._dist(Ca, mask)

        E = gather_nodes(msa, E_idx)


        h_V = self.W_h(msa)

        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)



        return  h_V



from model.network.feats import batched_gather
def edgek_to_all(h_E,E_idx,batch,L):

    K=E_idx.shape[-1]
    t = trans_LKD_TO_LLL(B=batch, L=L, K=K, D=1)
    newindex, ebmask = t.write_back(E_idx)
    newindex = torch.as_tensor(newindex, dtype=torch.long).squeeze(-1)
    edgeall = batched_gather(
        h_E,
        newindex,
        dim=-2,
        no_batch_dims=len(h_E.shape[:-2]),
    )

    return edgeall
class Repacker_Str_Encoder(nn.Module):
    def __init__(self, node_features, edge_features,
                 hidden_dim, num_encoder_layers=6,
                 k_neighbors=48, augment_eps=0.0, dropout=0.1, vocab=21, use_ESSE=False, use_Eaatype=True, **kwargs):
        super(Repacker_Str_Encoder, self).__init__()
        print(use_Eaatype)

        # Hyperparameters
        self.k_neighbors = k_neighbors

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = meshProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps,
                                            E_SSE=use_ESSE, E_aatye=use_Eaatype)  #

        self.W_h = nn.Linear(64, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.linearouts = nn.Sequential(
            nn.Linear(node_features, node_features),
            nn.ReLU(),
            nn.Linear(node_features, node_features),
        )
        self.chinorm = nn.LayerNorm(node_features)

        self.W_ss = nn.Linear(vocab, hidden_dim)

        #self.W_es = nn.Linear(hidden_dim, 32)



    def forward(self, S, X, mask, residue_idx, SSE3_seq=None, SSE8_seq=None, use_tri=True, use_rbf=True, use_sse=False,
                **kwargs):
        """ Graph-conditioned sequence model """
        device = X.device
        batch, L = X.shape[0], X.shape[1]
        # Prepare node and edge embeddings

        if SSE3_seq == None or SSE8_seq == None:
            SSE3_seq = torch.zeros((batch, L), device=device).type(torch.long)
            SSE8_seq = torch.zeros((batch, L), device=device).type(torch.long)
        E, E_idx, h_V = self.features(S, X, mask, residue_idx, SSE3_seq, SSE8_seq, use_tri=use_tri, use_rbf=use_rbf,
                                      use_sse=use_sse)  # ,



        h_V = self.W_h(h_V)

        h_E = self.W_e(E)

        # # # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        features = self.chinorm(self.linearouts(h_V))


        return features,h_E

    def _positional_embeddings(self, pos,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings

        pse = []
        for idx in pos:
            d = idx

            frequency = torch.exp(
                torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=pos.device)
                * -(np.log(10000.0) / num_embeddings)
            )
            angles = d.unsqueeze(-1) * frequency
            E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
            pse.append(E)
        pse = torch.stack(pse)
        return pse


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 230000
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        if self._step == 1:
            print("\n initing lr is %5f " % (float(rate)))
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(256, 0.7, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def probability_distribution(data, ):
    data = data.type(torch.int).flatten().numpy()
    plt.hist(data)
    # cc =data.tolist()
    # dict = {}
    # for key in cc:
    #     dict[key] = dict.get(key, 0) + 1
    #
    # sns.set_palette("hls")
    # # sns.set_style("whitegrid")
    # plt.figure(dpi=120)
    # sns.set(style='dark')
    # sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    # sns.distplot(cc,hist=True,
    #                  kde=True, )

    plt.savefig('softmax.jpg')


def polar(v):
    """
    v:边的单位方向向量，length x 1*3
    return：phi，psi两个角度
    https://blog.csdn.net/u010087338/article/details/118553249
    由于torch版本比较低，没有arctan2 函数，所以要跳到numpy处理
    return : L,1,2  [0~2pi]
    """
    x = v[:, :, :, 0]
    y = v[:, :, :, 1]
    z = v[:, :, :, 2]
    device = v.device
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2).unsqueeze(-1)

    xy = torch.sqrt(x ** 2 + y ** 2)  # sqrt(x² + y²)
    # theta = torch.arctan2(y, x).unsqueeze(-1)+torch.as_tensor(math.pi)
    # phi = torch.arctan2(xy, z).unsqueeze(-1)+torch.as_tensor(math.pi)

    yn = y
    xn = x

    theta_n = torch.arctan2(yn, xn).unsqueeze(-1)
    phi_n = torch.arctan2(xy, z).unsqueeze(-1)

    a = torch.cat([theta_n, phi_n, r], dim=-1)

    return a


class dis_MaskedDirectMultiheadAttention(nn.Module):
    def __init__(self, d_in, dropout=0.1):
        super(dis_MaskedDirectMultiheadAttention, self).__init__()
        heads = 4

        d_k = int(d_in / heads)
        d_out = d_in
        self.heads = heads
        self.scaling = 1 / math.sqrt(d_k)
        self.to_query = nn.Linear(d_in, heads * d_k)
        self.to_key = nn.Linear(d_in, heads * d_k)
        self.to_value = nn.Linear(d_out, d_out)
        self.to_out = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, atten, value, mask, padding_mask):
        value = value.unsqueeze(1)
        batch, N, L = value.shape[:3]
        #
        q = self.to_query(value).view(batch, L, self.heads, -1).permute(0, 2, 1, 3)  # (B, h, L, -1)
        v = self.to_value(value).view(batch, N, L, self.heads, -1).permute(0, 3, 1, 2, 4)  # (B, h, N, L, -1)
        #

        attention = atten.unsqueeze(1)  # (B, h, L, L)
        attention = attention.masked_fill(mask < 0.5, torch.finfo(q.dtype).min)

        padding_mask = padding_mask.unsqueeze(1).expand(attention.shape)
        attention = attention.masked_fill(padding_mask == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)  # (B, h, L1, L2)

        attention = self.dropout(attention)  # (B, h, 1, L, L)
        #
        # out = torch.matmul(attention, v) # (B, h, N, L, d_out//h)
        out = torch.einsum('bhij,bhnjk->bhnik', attention, v)  # (B, h, N, L, d_out//h)
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(batch, N, L, -1)
        #
        out = self.to_out(out)
        return out


if __name__ == '__main__':
    index = 'XACDEFGHIKLMNPQRSTVWY'
    seq = 'RPALPDQAEMRLVFIDGDADEWLAGIEAARLDAMALSIHRYIRE'
    s = []
    for i in seq:
        s.append(index.index(i))
    print(s)
    print("===> testing polar ...")

    # out = polar(data)
    # print(out.shape)