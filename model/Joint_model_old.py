import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network.Transformer import *
import torch_cluster
from einops import repeat
from model.network.invariant_point_attention import IPABlock,IPATransformer,IPA_Stack
from  model.primitives import Linear


from model.Rigid import Rigid
from model.chems import add_chem_features,light_PositionalEncoding
from model.network.Generator import loss_smoothed

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot


# Attention module based on AlphaFold2's idea written by Minkyung Baek
#  - Iterative MSA feature extraction
#    - 1) MSA2Pair: extract pairwise feature from MSA --> added to previous residue-pair features
#                   architecture design inspired by CopulaNet paper
#    - 2) MSA2MSA:  process MSA features using Transformer (or Performer) encoder. (Attention over L first followed by attention over N)
#    - 3) Pair2MSA: Update MSA features using pair feature
#    - 4) Pair2Pair: process pair features using Transformer (or Performer) encoder.

def _S_to_seq(S, mask):
    alphabet = 'XACDEFGHIKLMNPQRSTVWY'
    # seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])

    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) ])
    return seq

def get_bonded_neigh(idx):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - neighbor: bonded neighbor information with sign (B, L, L, 1)
    '''
    neighbor = idx[:, None, :] - idx[:, :, None]
    neighbor = neighbor.float()
    sign = torch.sign(neighbor)  # (B, L, L)
    neighbor = torch.abs(neighbor)
    neighbor[neighbor > 1] = 0.0
    neighbor = sign * neighbor
    return neighbor.unsqueeze(-1)


def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 22., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None, :]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class MSA_features(nn.Module):
    """
    to add chem and pos .etc features
    we use:
    onehot
    chem
    position embedding
    logits

    """


    def __init__(self,msa_dim,chem_dim=127):

        super(MSA_features, self).__init__()

        self.chem_features=add_chem_features(chem_dim=chem_dim)
        self.msa_s=Linear(21+chem_dim,msa_dim)

    def forward(self,msa,idx,mask):
        """
        msa: soft onehot of generated sequences

        """
        chem_s=self.chem_features(msa)
        msa=torch.cat((msa,chem_s),dim=-1)

        pos=light_PositionalEncoding(idx,msa.shape[-1])

        msa=self.msa_s(msa+pos.unsqueeze(1).repeat(1,msa.shape[1],1,1))

        return msa

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
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s

class Output_sequence_Layer(nn.Module):
    def __init__(self, c,out):
        super(Output_sequence_Layer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c,out, init="final")


        self.relu = nn.ReLU()

    def forward(self, s):

        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s =F.log_softmax(s,dim=-1)


        return s


class CoevolExtractor(nn.Module):
    def __init__(self, n_feat_proj, n_feat_out, p_drop=0.1):
        super(CoevolExtractor, self).__init__()

        self.norm_2d = LayerNorm(n_feat_proj * n_feat_proj)
        # project down to output dimension (pair feature dimension)
        self.proj_2 = nn.Linear(n_feat_proj ** 2, n_feat_out)

    def forward(self, x_down, x_down_w):
        B, N, L = x_down.shape[:3]

        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down_w)  # outer-product & average pool
        pair = pair.reshape(B, L, L, -1)
        pair = self.norm_2d(pair)
        pair = self.proj_2(pair)  # (B, L, L, n_feat_out) # project down to pair dimension
        return pair


class MSA2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=4, n_feat=256, r_ff=4, p_drop=0.1,
                 performer_N_opts=None, performer_L_opts=None, ):
        super(MSA2MSA, self).__init__()
        # attention along L
        enc_layer_1 = EncoderLayer(d_model=n_feat, d_ff=n_feat * r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   use_tied=True,
                                   performer_opts=performer_L_opts)
        self.encoder_1 = Encoder(enc_layer_1, n_layer)

        # attention along N
        enc_layer_2 = EncoderLayer(d_model=n_feat, d_ff=n_feat * r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_N_opts)
        self.encoder_2 = Encoder(enc_layer_2, n_layer)

    def forward(self, x, ):
        # Input: MSA embeddings (B, N, L, K)
        # Output: updated MSA embeddings (B, N, L, K)
        B, N, L, _ = x.shape
        # attention along L
        x = self.encoder_1(x, return_att=False)

        # if breaks:
        #     return x

        # attention along N
        if N > 1:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = self.encoder_2(x)
            x = x.permute(0, 2, 1, 3).contiguous()
            return x
        else:
            return x


class MSA2SINGLE(nn.Module):
    def __init__(self, d_msa=64,
                p_drop=0.1,
                ):
        super(MSA2SINGLE, self).__init__()
        self.StructureModuleTransition= StructureModuleTransition(c=d_msa, num_layers=1, dropout_rate=p_drop)




    def forward(self, msa ):
        """
        this msa could be N msa generated, also could be just one.
        we want to make sure that the final sequence is suitable to str, after add all sequences, the output is not the
        one sequence. the error in one sequnce will be made up by others.
        so just use one is ok.

        for better str, should use much sequnces
        for better sequence, should just use the target one

        """
        if len(msa.shape)==4:
            msa=self.StructureModuleTransition(msa)
            s = torch.sum(msa, dim=1) / msa.shape[1]
            s=s.squeeze(1)
        elif len(msa.shape)==3:
            s=self.StructureModuleTransition(msa)

        return s


class Str2MSA(nn.Module):
    def __init__(self, d_msa=64, d_state=32, inner_dim=32, r_ff=4,
                 distbin=[2.5, 4.0, 8.0, 20.0], p_drop=0.1):
        super(Str2MSA, self).__init__()
        self.distbin = distbin
        n_att_head = len(distbin)

        self.norm_state = LayerNorm(d_state)
        self.norm1 = LayerNorm(d_msa)
        self.attn = MaskedDirectMultiheadAttention(d_state, d_msa, n_att_head, d_k=inner_dim, dropout=p_drop)
        self.dropout1 = nn.Dropout(p_drop,inplace=True)

        self.norm2 = LayerNorm(d_msa)
        self.ff = FeedForwardLayer(d_msa, d_msa*r_ff, p_drop=p_drop)
        self.dropout2 = nn.Dropout(p_drop,inplace=True)

    def forward(self, msa, xyz, state):
        dist = torch.cdist(xyz[:,:,1], xyz[:,:,1]) # (B, L, L)

        mask_s = list()
        for distbin in self.distbin:
            mask_s.append(1.0 - torch.sigmoid(dist-distbin))
        mask_s = torch.stack(mask_s, dim=1) # (B, h, L, L)

        state = self.norm_state(state)
        msa2 = self.norm1(msa)
        msa2 = self.attn(state, state, msa2, mask_s)
        msa = msa + self.dropout1(msa2)

        msa2 = self.norm2(msa)
        msa2 = self.ff(msa2)
        msa = msa + self.dropout2(msa2)

        return msa

from model.network.Joint_model import MSA_Stack,MSA2SINGLE

class Joint_layer(nn.Module):
    def __init__(self, msa_layer,IPA_layer, d_msa,d_ipa,  n_head_msa, n_head_ipa,  r_ff,
                p_drop,performer_L_opts=None, performer_N_opts=None,
                ):
        super(Joint_layer, self).__init__()

        self.d_state=d_ipa

        # self.msa2msa = MSA2MSA(n_layer=msa_layer, n_att_head=n_head_msa, n_feat=d_msa,
        #                        r_ff=r_ff, p_drop=p_drop,
        #                        performer_N_opts=performer_N_opts,
        #                        performer_L_opts=None)  # tied and performer just could choose one
        #
        self.msa2Single= MSA2SINGLE(d_msa=d_msa, p_drop=p_drop )
        self.msa_stack=MSA_Stack(msa_layer=msa_layer, d_msa=d_msa, d_ipa=d_ipa, n_head_msa=n_head_msa, r_ff=r_ff,
                 p_drop=p_drop,trans_scale_factor=10)
            #StructureModuleTransition(c=d_msa, num_layers=1, dropout_rate=p_drop)
        #MSA2SINGLE(d_msa=d_msa, p_drop=p_drop, MSAs=MSAs,)

        #self.structure_refine=IPATransformer(dim=d_ipa,depth=IPA_layer,heads=n_head_ipa,detach_rotations=False,predict_points=True)
        self.structure_refine=IPA_Stack(dim=d_ipa,depth=IPA_layer,heads=n_head_ipa,detach_rotations=False)
        self.str2msa=Str2MSA(d_msa=d_msa, d_state=d_ipa)

        #3. get sequences
        self.get_sequences=Output_sequence_Layer(d_msa,out=21)




    def forward(self, msa,rigid ,residue_indexs,XYZ,mask): #
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)


        # if i ==0:
        #     # 0. optimze msa by noised xyz
        #     state=msa
        #     # state=torch.zeros((msa.shape[0],msa.shape[2],self.d_state),device=msa.device)
        #     # msa = self.str2msa(msa, XYZ, state)
        # Ca=translations
        # msa=self.msa_stack( msa, Ca, state,residue_indexs)
        # # 3. process MSA features
        # msa = self.msa2msa(msa)

        # # 1. refine noised structures
        # single_R=self.msa2Single(msa[:,0,:,:])  # take the first one  or conatact them
        # XYZ,state,rotations,translations=self.structure_refine(single_R,translations=translations,quaternions=quaternions)

        # # 2. structure update msa
        # msa=self.str2msa(msa,XYZ,state)

        # 1. Structure refine
        state_init=self.msa2Single(msa[:,0,:,:])
        state, rigid = self.structure_refine(state_init, rigid,mask.type(torch.bool))

        Ca=rigid.get_trans()
        msa=self.msa_stack( msa, Ca, msa,residue_indexs)


        # # 3. Get sequences
        logits=self.get_sequences(msa)
        aatype=torch.argmax(logits[:,0,:,:],1)





        return XYZ,msa,rigid,logits

from model.network.Joint_model import MSA_add

class Joint_module(nn.Module):
    def __init__(self, n_module_str=4, n_layer=3,IPA_layer=3, d_msa=128,d_ipa=128,
                 n_head_msa=8, n_head_ipa=8, r_ff=2,
                  p_drop=0.1,
                 performer_L_opts={"nb_features": 128}, performer_N_opts={"nb_features": 128},
                 ):
        super(Joint_module, self).__init__()

        # add chem
        self.MSA_features=MSA_features(msa_dim=d_msa,chem_dim=127)
        # 0. MSA add
        self.MSA_add=MSA_add( d_msa=d_msa, p_drop=p_drop)

        self.n_module_str = n_module_str



        if self.n_module_str > 0:
            self.iter_block = nn.ModuleList(Joint_layer(msa_layer=n_layer,
                                                        IPA_layer=IPA_layer,
                                                            d_msa=d_msa,
                                                            d_ipa=d_ipa,
                                                            n_head_msa=n_head_msa,
                                                            n_head_ipa=n_head_ipa,

                                                            r_ff=r_ff,

                                                            p_drop=p_drop,

                                                            performer_N_opts=performer_N_opts,
                                                            performer_L_opts=performer_L_opts,

                                                            ) for _ in range(n_module_str))

        self.CE = nn.CrossEntropyLoss()
        self.output_sequence = Output_sequence_Layer(c=d_msa,out=21)



    def forward(self, msa,S_True,points,mask,idx,noiseFactor=0.1,auxk=0.02,inference=False):
        # input:
        #   msa: initial MSA token (N, L, 21)
        #   points: initial atoms (N, L, 4,3)
        # noiseFactor: the scale of noise added to original XYZ

        RMSD_loss=0
        Msa_loss=0

        # add chem features
        msa=self.MSA_add(msa,None,idx)



        # add noise and build rigid
        xyz = points + noiseFactor * torch.randn_like(points)
        xyz=xyz/10
        Init_Rigid=Rigid.from_3_points(xyz[:,:,0,:],xyz[:,:,1,:],xyz[:,:,2,:])
        rots,translations=Init_Rigid._rots,Init_Rigid._trans
        quaternions=rots.get_quats()

        rigid=Init_Rigid
        # iteration between sequences and str
        for i_m in range(self.n_module_str):

            pXYZ,msa,  rigid,logits= self.iter_block[i_m](msa, rigid,idx,xyz,mask)
            RMSD_i = torch.mean(torch.abs(
                pXYZ * mask.unsqueeze(-1).unsqueeze(-1).expand(pXYZ.shape) - points * mask.unsqueeze(-1).unsqueeze(
                    -1).expand(pXYZ.shape)))

            #logit = self.output_sequence(msa[:, 0, :, :])
            # lgs_i, _, pred_i = self.get_S_loss(logit, S_True, mask)
            # seq = _S_to_seq(pred_i[0].type(torch.long), mask[0])
            # print(seq)

            if not inference:
                RMSD_loss=RMSD_loss+auxk*RMSD_i

        # to get nll of sequences
        logit = logits[:, 0, :, :]
        # logit=self.output_sequence(msa[:, 0, :, :])

            # # aux msa loss we dont use
            # lgs_i, _, pred_i = self.get_S_loss(logit, S_True, mask)
            # # Msa_loss=Msa_loss#+auxk*lgs_i
            #
            # seq = _S_to_seq(pred_i[0].type(torch.long), mask[0])
            # print(seq)


        # loss
        lgs, seq_recovery_rate, pred=self.get_S_loss(logit,S_True,mask)
        Msa_loss=lgs

        RMSD = torch.mean(torch.abs(
            pXYZ * mask.unsqueeze(-1).unsqueeze(-1).expand(pXYZ.shape) - points * mask.unsqueeze(-1).unsqueeze(
                -1).expand(pXYZ.shape)))

        return Msa_loss,seq_recovery_rate,RMSD_loss+(1-auxk)*RMSD,pred

    def get_S_loss(self, logits, S, mask):


        score,lgs=loss_smoothed(S,logits,mask,0.0,kind=21)

        pred = (torch.argmax(logits, dim=-1) * mask).detach().type(torch.int)
        true = (S * mask).detach().type(torch.int)

        this_correct = ((pred == true).sum() - (1 - mask.detach()).sum())
        thisnods = torch.sum(mask)
        seq_recovery_rate = 100 * this_correct / thisnods
        #cm=confusion_matrix(true.flatten(0,1).detach().cpu().numpy(), pred.flatten(0,1).detach().cpu().numpy(), labels=np.arange(0,21))

        return lgs, seq_recovery_rate, pred







if __name__ == "__main__":
    model = Joint_module()
    B = 1
    N = 8
    L = 500
    d_msa = 64
    d_pair = 128

    seq = torch.randint(0, 20, (B, L))
    idx = torch.arange(0, L).unsqueeze(0).expand((B, L))
    mask = torch.randint(0, 2, (B, L))
    Points = torch.randn(B, L, 4, 3)
    token = torch.randn(B, L, 768)

    msa = torch.randint(0, 20,size=(B, N, L))
    seq1hot = F.one_hot(seq)
    pair = torch.randn(size=(B, L, L, d_pair))

    x = model(msa.cuda(),seq.cuda(),Points.cuda(),mask,idx )



