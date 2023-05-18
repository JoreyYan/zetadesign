from __future__ import print_function


import model.mesh_utils as utils
import torch.nn as nn
import torch.nn.functional as F
from  ..primitives import Linear

import math
# A number of functions/classes are adopted from: https://github.com/jingraham/neurips19-graph-protein-design

from torch.autograd import Variable
from model.AngelResnet import AngleResnet

import torch
import numpy as np
from model.network.invariant_point_attention import IPA_Stack
from ..Rigid import Rigid,Rotation
from model.network.feats import torsion_angles_to_frames,frames_and_literature_positions_to_atom14_pos
from model.network.loss_old import Repacker_Aux_loss,Repackerloss
from model.np.residue_constants import  (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)

class meshProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=15, augment_eps=0., aatypes=21,E_SSE=False):
        """ Extract protein features """
        super(meshProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.use_ESSE=E_SSE

        self.use_rbf_ture_and_indexdistance = True

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        edge_in_tri=num_positional_embeddings + int(num_rbf/2)*12 * 10

        self.edge_embedding_tri = nn.Linear(edge_in_tri, edge_features, bias=False)

        if self.use_ESSE:
            rbf_node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25 + 32 + 32
            self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        else:
            rbf_node_in, edge_ins = 6, num_positional_embeddings + num_rbf * 25
            self.edge_embeddings = nn.Linear(edge_ins, edge_features, bias=False)


        self.aatype_embeddings=nn.Linear(num_positional_embeddings+aatypes,edge_features)


        self.norm_edges_tri = nn.LayerNorm(edge_features)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.norm_aatypes=nn.LayerNorm(edge_features)

        self.SSE3_embedding=nn.Embedding(num_embeddings=4,embedding_dim=32)
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
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(int(self.top_k/2), X.shape[1]), dim=-1, largest=False)
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
        D_min, D_max, D_count = 0., 15., int(self.num_rbf/2)
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
        neg_dis=torch.exp(-D_A_B_neighbors)
        return RBF_A_B#,neg_dis

    def forward(self,S, X, mask, residue_idx, SSE3_seq, SSE8_seq, use_tri=False, use_rbf=False, use_sse=False, ):
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

            # sequence featires
            S=torch.nn.functional.one_hot(S,21).type(torch.float)
            E_S = gather_nodes(S, E_idx)
            E_S = torch.cat((E_positional,E_S), -1)
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
            E = (E + E_tri+E_S)/3

        hv = torch.cat((sse3, sse8), dim=-1)

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
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features





def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn



class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class RepackerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1,  scale=48):
        super(RepackerLayer, self).__init__()
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
        #self.HE_up=nn.Linear(num_hidden, int(num_hidden-delat_dim), bias=True)




    def forward(self, h_V,h_E, E_idx, mask_V, mask_attend):
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
            mask_V=mask_V.squeeze(-1)



        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))



        return h_V, h_E



class Relax_Repacker_Module(nn.Module):
    def __init__(self, hidden_dim,IPA_layer, d_ipa ,n_head_ipa,num_ff,dropout=0.1):
        super(Relax_Repacker_Module, self).__init__()

        self.RepackerLayer=RepackerLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
        self.Relaxlayer=IPA_Stack(dim=d_ipa,depth=IPA_layer,heads=n_head_ipa,detach_rotations=False)


        self.chi2bb1=nn.Linear(hidden_dim,d_ipa)
        self.chi2bb=PositionWiseFeedForward(  d_ipa, num_ff)

        self.bb2chi1=nn.Linear(d_ipa,hidden_dim)
        self.bb2chi=PositionWiseFeedForward(  hidden_dim, num_ff)




    def forward(self,rigid, h_V,h_E, E_idx, mask, mask_attend):
        """ Parallel computation of full transformer layer """

        h_V, h_E =  self.RepackerLayer(h_V, h_E, E_idx, mask, mask_attend)
        chi_init=h_V

        state=self.chi2bb(self.chi2bb1(h_V))
        state,rigid=self.Relaxlayer(state, rigid,mask.type(torch.bool))

        h_V=self.bb2chi(h_V+self.bb2chi1(state))


        return rigid,h_V, h_E,chi_init



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






def _generate_mesh(points_batchs,E_idx, l_shift=1):
    """

    Args:
        points: batch,L,3
        l_shift:

    Returns:
       batch， L,2*l_shift+1,3,3
    """
    mesh_batch=[]
    base_batch=[]

    for i in range(len(points_batchs)):
        points = points_batchs[i]
        Eix=E_idx[i]
        mp=torch.index_select(points,0,Eix.flatten())
        mesh=mp.view(Eix.shape[0],Eix.shape[1],mp.shape[-2],mp.shape[-1])[:,:,:3,:]
        mesh_data=mesh-points[:,1,:].unsqueeze(1).unsqueeze(1)



        base=points[:,:3,:]-points[:,1,:].unsqueeze(1)
        mesh_batch.append(mesh_data)
        base_batch.append(base)



    mesh_batch=torch.stack(mesh_batch).to(points.device)
    base_batch=torch.stack(base_batch).to(points.device)

    return mesh_batch,base_batch#,dihedral_batch


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
    neighborhood_planar = points_pos - normals.unsqueeze(2) * normal_comp.unsqueeze(-1) # 点在平面上的位置
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
            tri_coords = generate_coords(nearby_triangles_pos.view(B, Q, K_T * 3, 3), query_triangles_pos).view(B, Q, K_T,
                                                                                                        3, 6)
        except:
            print("-----------")
        max_vals = torch.max(tri_coords, dim=3).values  # (B, Q, K_T, 6)
        min_vals = torch.min(tri_coords, dim=3).values  # (B, Q, K_T, 6)
        triangle_coords = torch.cat((min_vals, max_vals), dim=-1)
        #triangle_coords=triangle_coords.flatten(-2,-1)

    # if have_triangles:
    return triangle_coords
    # else:
    #     return point_coords

class Pred_angle_Points(nn.Module):
    def __init__(self,c_s,c_resnet,no_resnet_blocks,no_angles,trans_scale_factor,a_epsilon,**kwargs
                ):
        super(Pred_angle_Points, self).__init__()
        self.c_s=c_s
        self.c_resnet=c_resnet
        self.no_resnet_blocks=no_resnet_blocks
        self.no_angles=no_angles
        self.epsilon=a_epsilon

        self.trans_scale_factor=trans_scale_factor
        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(self, s_init,s, rigid,aatype,pred_point=False):  #


        # [*, N, 7, 2]
        unnormalized_angles, angles = self.angle_resnet(s_init,s)

        if pred_point:
            backb_to_global = Rigid(
                Rotation(
                    rot_mats=rigid.get_rots().get_rot_mats(),
                    quats=None
                ),
                rigid.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                self.trans_scale_factor
            )
            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )
        else:
            pred_xyz=None


        return unnormalized_angles, angles,pred_xyz

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        '''


        :param r: gobal rigid
        :param alpha:  angels
        :param f: aatype , sequenceof [B,L] eg,34645671
        :return:
        '''
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

class Layers(nn.Module):
    def __init__(self,  node_features, edge_features,
                 hidden_dim,  IPA_layer, d_ipa ,n_head_ipa,num_ff,modules,
                  k_neighbors=48, augment_eps=0.0, dropout=0.1,vocab=21,use_ESSE=False,**kwargs):
        super(Layers, self).__init__()

        # Hyperparameters
        self.k_neighbors=k_neighbors

        self.trans_scale_factor=kwargs['trans_scale_factor']

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = meshProteinFeatures(node_features, edge_features,aatypes=vocab, top_k=k_neighbors, augment_eps=augment_eps,E_SSE=use_ESSE) #

        self.W_h=nn.Linear(64, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)


        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            Relax_Repacker_Module(hidden_dim, IPA_layer, d_ipa ,n_head_ipa,num_ff, dropout=dropout)  #hidden_dim,IPA_layer, d_ipa ,n_head_ipa,num_ff,dropout
            for _ in range(modules)
        ])


        self.angles = Pred_angle_Points(**kwargs)





        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,input,bbrelaxdistance,r_epsilon,SSE3_seq=None,SSE8_seq=None,use_tri=True,use_rbf=True,use_sse=False,**kwargs):
        """ Graph-conditioned sequence model """


        Seqs = input['S']
        residue_idx=input['residue_idx']
        mask=input['mask']
        # add noise and build rigid
        points = input['X']

        device = points.device
        batch, L = points.shape[0], points.shape[1]
        xyz = points + bbrelaxdistance * torch.randn_like(points)

        # transfer \AA to nm
        xyz = xyz / self.trans_scale_factor

        # get noised rigid
        rigid = Rigid.from_3_points(p_neg_x_axis=xyz[..., 0, :], origin=xyz[..., 1, :], p_xy_plane=xyz[..., 2, :],
                                    eps=r_epsilon, )

        # Prepare node and edge embeddings

        if SSE3_seq==None or SSE8_seq==None:
            SSE3_seq=torch.zeros((batch, L),device=device).type(torch.long)
            SSE8_seq = torch.zeros((batch, L),device=device).type(torch.long)
        E, E_idx,h_V= self.features(Seqs,points, mask, residue_idx,SSE3_seq,SSE8_seq,use_tri=use_tri,use_rbf=use_rbf,use_sse=use_sse)  #,

        h_V=self.W_h(h_V)

        h_E = self.W_e(E)

        # # # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend


        for layer in self.encoder_layers:
            rigid,h_V,h_E,chi_init = layer(rigid,h_V,h_E, E_idx,mask, mask_attend)


        unnormalized_angles, angles,pred_xyz=self.angles(chi_init,h_V,rigid,Seqs)


        scaled_rigids = rigid.scale_translation(self.trans_scale_factor)

        #pred
        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "unnormalized_angles_sin_cos": unnormalized_angles,
            "angles_sin_cos": angles,
            "aatype": Seqs,
        }



        return preds







    def _positional_embeddings(self, pos,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings

        pse=[]
        for idx in pos:

            d = idx

            frequency = torch.exp(
                torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=pos.device)
                * -(np.log(10000.0) / num_embeddings)
            )
            angles = d.unsqueeze(-1) * frequency
            E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
            pse.append(E)
        pse=torch.stack(pse )
        return pse

class RelaxRepacker(nn.Module):
    def __init__(self,**kwargs,
                 ):
        super(RelaxRepacker, self).__init__()



        self.iter_block = Layers(**kwargs)





    def forward(self, input:dict,
                gt_batchs:dict,
                bbrelaxdistance:float,
                r_epsilon:float,
                loss_factor:dict
                ,**kwargs):
        # input:
        #   Seqs: initial MSA token (B, N, L, 21)
        #   bbrelaxdistance: the scale of noise added to original XYZ



        preds= self.iter_block(input,bbrelaxdistance,r_epsilon)



        loss,result=self.get_loss(preds,gt_batchs,loss_factor)




        result.update({
            'finial_loss':loss.detach().cpu(),

            })


        return loss,result

    def get_aux_loss(self, preds:dict, gt_batchs:dict,loss_factor:dict):
        a_loss=Repacker_Aux_loss(preds, gt_batchs,**loss_factor)

        return a_loss

    def get_loss(self,preds:dict, gt_batchs:dict,loss_factor:dict):
        floss=Repackerloss(preds, gt_batchs,**loss_factor)
        return floss

    def get_recovery(self, pred, S, mask):


        pred=pred*mask
        true = (S * mask).detach().type(torch.int)

        this_correct = ((pred == true).sum() - (1 - mask.detach()).sum())
        thisnods = torch.sum(mask)
        seq_recovery_rate = 100 * this_correct / thisnods


        return seq_recovery_rate

    def _get_ce_loss(self,S, log_probs, mask):
        scores, loss_seq = loss_smoothed(S, log_probs, mask,0.1)

        pred = (torch.argmax(log_probs, dim=-1) * mask).detach().type(torch.int)
        true = (S * mask).detach().type(torch.int)

        this_correct = ((pred == true).sum() - (1 - mask.detach()).sum())
        thisnods = torch.sum(mask)
        seq_recovery_rate = 100 * this_correct / thisnods



        return loss_seq,seq_recovery_rate

    def checkoverlap(self,seqs,mask):
        print('overlap between sequences')
        target=torch.argmax(seqs[0],-1)
        #target=self.seq_gt
        for i in range(1,len(seqs)):
            ref=torch.argmax(seqs[i],-1)
            over=self.get_recovery(ref,target,mask)
            kl=torch.exp(F.kl_div(seqs[i],seqs[0]))
            print(over)

        print('overlap between gt')
        target=self.seq_gt
        for i in range(len(seqs)):
            ref=torch.argmax(seqs[i],-1)
            over=self.get_recovery(ref,target,mask)

            print(over)


    def design(self, input:dict,
                gt_batchs:dict,
                bbrelaxdistance:float,
                r_epsilon:float,
                loss_factor:dict
                ,**kwargs):
        # input:
        #   Seqs: initial MSA token (B, N, L, 21)
        #   noiseFactor: the scale of noise added to original XYZ


        loss_seq, seq_recovery_rate_g, SSams=self.G(**input,Index_embed=True, **self.G_param)

        Seqs = torch.stack(SSams, dim=1)
        # self.seq_gt=gt_batchs['gt_aatype'][-1]
        # self.checkoverlap(Seqs[-1],input['mask'][-1])
        # Seqs = torch.stack(SSams, dim=1)
        # showmat(Seqs[-1])


        # add noise and build rigid
        points=input['X']
        xyz = points + bbrelaxdistance * torch.randn_like(points)
        #transfer \AA to nm
        xyz=xyz/self.trans

        #get noised rigid
        rigid=Rigid.from_3_points(p_neg_x_axis=xyz[..., 0, :],origin=xyz[..., 1, :],p_xy_plane=xyz[..., 2, :],eps=r_epsilon,)
        B,N,L,_=Seqs.shape





        residue_indexs=input['residue_idx']
        seq_masks=input['mask']


        # add chem features
        msa=self.MSA_add(Seqs,residue_indexs)


        Aux_loss_list=[]

        # iteration between sequences and str
        for i_m in range(self.n_module_str):
            last= False if i_m<(self.n_module_str-1) else True
            msa,rigid,preds= self.iter_block[i_m](msa,residue_indexs, seq_masks,rigid,last)

            Auxloss=self.get_aux_loss(preds,gt_batchs,loss_factor)
            Aux_loss_list.append(Auxloss)

        # preds['angles_sin_cos']=gt_batchs['gt_angles_sin_cos']+torch.rand_like(gt_batchs['gt_angles_sin_cos'])*0.5
        loss,result=self.get_loss(preds,gt_batchs,loss_factor)

        Auxloss=torch.mean(torch.stack(Aux_loss_list,0))
        loss=loss+Auxloss*loss_factor['aux_f']+0.1*loss_seq
        recovery=self.get_recovery(preds['aatype'], S=gt_batchs['gt_aatype'].detach(), mask=gt_batchs['gtframes_mask'].detach())  #preds['aatype'].detach()
        result.update({
            'finial_loss':loss.detach().cpu(),
            'aux_loss': Auxloss.detach().cpu(),
            'recovery':recovery.cpu(),})

        # logits=preds['logits']
        # loss,recovery=self._get_ce_loss(S=gt_batchs['gt_aatype'],log_probs=logits,mask=gt_batchs['gtframes_mask'])
        mask=torch.ones_like(preds['aatype'])
        print(_S_to_seq(preds['aatype'][0],seq_masks[0]))
        print(result)
        print(60*'-')


        return loss,seq_recovery_rate_g,result#seq_recovery_rate_g,recovery