import numpy as np

import torch
from typing import Tuple, Sequence, Optional
from .Transformer import *
from .invariant_point_attention import IPA_Stack
from  ..primitives import Linear,GlobalAttention,Attention
from model.network.Generator import Repacker_Str_Encoder
from model.network.resnet import ResidualNetwork
from ..Rigid import Rigid,Rotation
from ..chems import add_chem_features,light_PositionalEncoding


from model.network.Embedding.Embeddings import Pair_emb_wo_templ
from model.network.Embedding.outer_product_mean import OuterProductMean
from model.network.Embedding.pair_trans import PairTransition

from model.AngelResnet import AngleResnet,litResnet
from .feats import torsion_angles_to_frames,frames_and_literature_positions_to_atom14_pos
from model.network.loss import Repacker_Aux_loss,Repackerloss
from model.np.residue_constants import  (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    chi_angles_mask
)


from data.data_module import tied_features
from model.np import residue_constants

# Attention module based on AlphaFold2's idea written by Minkyung Baek
#  - Iterative MSA feature extraction
#    - 1) MSA2Pair: extract pairwise feature from MSA --> added to previous residue-pair features
#                   architecture design inspired by CopulaNet paper
#    - 2) MSA2MSA:  process MSA features using Transformer (or Performer) encoder. (Attention over L first followed by attention over N)
#    - 3) Pair2MSA: Update MSA features using pair feature
#    - 4) Pair2Pair: process pair features using Transformer (or Performer) encoder.

def show_numpy(matrix,i):
    im = Image.fromarray(matrix)  # numpy 转 image类
    im.show()
    # im.save('/home/junyu/图片/strmask/'+i+'.jpg')
    # matrix=matrix.type(np.int)
    # plt.imshow(matrix)


def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if(not inplace):
        m1 = m1 + m2
    else:
        m1 += m2

    return m1

class Pair2Pair(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8, n_feat=32, r_ff=4, p_drop=0.1,
                 performer_L_opts=None):
        super(Pair2Pair, self).__init__()
        enc_layer = AxialEncoderLayer(d_model=n_feat, d_ff=n_feat * r_ff,
                                      heads=n_att_head, p_drop=p_drop,
                                      performer_opts=performer_L_opts)
        self.encoder = Encoder(enc_layer, n_layer)

    def forward(self, x):
        return self.encoder(x)


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.
    Implements Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class MSA2Pair(nn.Module):
    def __init__(self, n_feat=128, n_feat_out=32, n_feat_proj=32,
                 n_resblock=1, p_drop=0.1, n_att_head=8):
        super(MSA2Pair, self).__init__()
        # project down embedding dimension (n_feat --> n_feat_proj)
        self.norm_1 = LayerNorm(n_feat)
        self.proj_1 = nn.Linear(n_feat, n_feat_proj)

        self.encoder = SequenceWeight(n_feat_proj, 1, dropout=p_drop)
        self.coevol = CoevolExtractor(n_feat_proj, n_feat_out)

        # ResNet to update pair features
        self.norm_down = LayerNorm(n_feat_proj)
        self.norm_orig = LayerNorm(n_feat_out)
        self.norm_new = LayerNorm(n_feat_out)
        self.update = ResidualNetwork(n_resblock, n_feat_out  + n_att_head, n_feat_out, n_feat_out,
                                      p_drop=p_drop)  #+ n_feat_proj * 4

    def forward(self, pair_orig, att):
        # Input: MSA embeddings (B, N, L, K), original pair embeddings (B, L, L, C)
        # Output: updated pair info (B, L, L, C)

        # average pooling over N of given MSA info
        #feat_1d = feat_1d.sum(1)

        # query sequence info
        #query = x_down[:, 0]  # (B,L,K)
        #feat_1d = torch.cat((feat_1d, query), dim=-1)  # additional 1D features
        # # tile 1D features
        # left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)
        # right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)
        # update original pair features through convolutions after concat
        pair_orig = self.norm_orig(pair_orig)

        pair = torch.cat((pair_orig, att), -1)
        pair = pair.permute(0, 3, 1, 2).contiguous()  # prep for convolution layer
        pair = self.update(pair)
        pair = pair.permute(0, 2, 3, 1).contiguous()  # (B, L, L, C)

        return pair

def _S_to_seq(S, mask):
    alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    # seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])

    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
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
                                   use_tied=False,use_Causal=True,
                                   performer_opts=performer_L_opts)
        self.encoder_1 = Encoder(enc_layer_1, n_layer)

        # attention along N
        enc_layer_2 = EncoderLayer(d_model=n_feat, d_ff=n_feat * r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_N_opts)
        self.encoder_2 = Encoder(enc_layer_2, n_layer)

    def forward(self, x, mask,return_att=False):
        # Input: MSA embeddings (B, N, L, K)
        # Output: updated MSA embeddings (B, N, L, K)
        B, N, L, _ = x.shape
        # attention along L
        if return_att:
            x,atten = self.encoder_1(x, return_att=True,mask=mask)
        else:
            x = self.encoder_1(x, return_att=False,mask=mask)
            atten=None

        # if breaks:
        #     return x

        # attention along N don use mask in coloum
        if N > 1:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = self.encoder_2(x)
            x = x.permute(0, 2, 1, 3).contiguous()
            return x,atten
        else:
            return x,atten


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
                 distbin=[ 4.0, 8.0, 12.0, 16.0], p_drop=0.1,trans_scale_factor=10):
        super(Str2MSA, self).__init__()
        self.trans_scale_factor=trans_scale_factor
        self.distbin = distbin
        n_att_head = len(distbin)

        self.norm_state = LayerNorm(d_state)
        self.norm1 = LayerNorm(d_msa)
        self.attn = MaskedDirectMultiheadAttention(d_state, d_msa, n_att_head, d_k=inner_dim, dropout=p_drop)
        self.dropout1 = nn.Dropout(p_drop,inplace=True)

        self.norm2 = LayerNorm(d_msa)
        self.ff = FeedForwardLayer(d_msa, d_msa*r_ff, p_drop=p_drop)
        self.dropout2 = nn.Dropout(p_drop,inplace=True)
        self.norm3 = LayerNorm(d_msa)

    def forward(self, msa, Ca, state):
        if len(msa.shape)==3:
            msa=msa.unsqueeze(1)
        dist = torch.cdist(Ca, Ca) # (B, L, L)
        # #check
        # lookdis=dist[0].detach().cpu().numpy()

        mask_s = list()
        for i in range(len(self.distbin)):
            distbin=self.distbin[i]
            distbin=distbin/self.trans_scale_factor
            mask_s.append(1.0 - torch.sigmoid(dist-distbin))
            # look=np.around(mask_s[i][0,:,:].detach().cpu().numpy(),0)
            # show_numpy(look*255,str(i))
        mask_s = torch.stack(mask_s, dim=1) # (B, h, L, L)

        state = self.norm_state(state)
        msa2 = self.norm1(msa)
        msa2 = self.attn(state, state, msa2, mask_s)
        msa = msa + self.dropout1(msa2)

        msa2 = self.norm2(msa)
        msa2 = self.ff(msa2)
        msa = msa + self.dropout2(msa2)
        msa=self.norm3(msa)

        if len(msa.shape)==4:
            msa=msa.squeeze(1)

        return msa

class MSA_features(nn.Module):
    """
    to add chem and pos .etc features
    we use:
    onehot
    chem
    position embedding
    logits

    """


    def __init__(self,msa_dim,onehotdim=21,chem_dim=127):

        super(MSA_features, self).__init__()
        chem_dim=2*msa_dim-onehotdim
        self.chem_features=add_chem_features(chem_dim=chem_dim)
        self.chem_dim=chem_dim
        self.msa_dim=msa_dim

        # type embedding
        self.onehotdim=onehotdim
        self.msa_emb=nn.Linear(onehotdim,msa_dim,bias=False)
        self.msa_s=nn.Linear(msa_dim*2,msa_dim,bias=False )

        self.layernorm=nn.LayerNorm(msa_dim)

    def forward(self,msa_onehot,residue_indexs):
        """
        msa: soft onehot of generated sequences

        return: representation of msa features

        """
        msa_nums=msa_onehot.shape[1]
        chem_s=self.chem_features(msa_onehot)
        #msa=self.msa_emb(msa_onehot)
        msa=torch.cat((msa_onehot,chem_s),dim=-1)  #[*,2*msa_dim]

        #get padding mask
        padding_mask=(residue_indexs!=-100).type(torch.float)[:,None,:,None].repeat(1,msa_nums,1,2*self.msa_dim)


        pos=light_PositionalEncoding(residue_indexs,msa.shape[-1])
        pos=pos.unsqueeze(1).repeat(1, msa.shape[1], 1, 1)
        msa=msa*padding_mask
        msa=self.msa_s(msa+pos)  #self.layernorm(

        return msa
class MSA_features_emb(nn.Module):
    """
    to add chem and pos .etc features
    we use:
    onehot
    chem
    position embedding
    logits

    """


    def __init__(self,msa_dim,onehotdim=21,):

        super(MSA_features_emb, self).__init__()
        chem_dim=msa_dim
        self.chem_features=add_chem_features(chem_dim=chem_dim)
        self.chem_dim=chem_dim
        self.msa_dim=msa_dim

        #str encoder
        self.str_layernorm=nn.LayerNorm(msa_dim)

        # type embedding
        self.onehotdim=onehotdim
        self.msa_embss=nn.Linear(onehotdim+4+3*8,msa_dim,)  # the 4 is b_factors
        self.msa_trans=nn.Linear(msa_dim*3,msa_dim,bias=False )



    def forward(self,msa_onehot,residue_indexs,b_factors,sse3_emb,sse8_emb,cen_emb,Str_features):
        """
        msa: soft onehot of generated sequences

        return: representation of msa features

        """

        chem_s=self.chem_features(msa_onehot)
        msa=self.msa_embss(torch.cat((msa_onehot,b_factors,sse3_emb,sse8_emb,cen_emb,),dim=-1))


        #
        Str_features=self.str_layernorm(Str_features)
        msa=torch.cat((msa,chem_s,Str_features),dim=-1)  #[*,2*msa_dim]




        pos=light_PositionalEncoding(residue_indexs,msa.shape[-1])


        # get padding mask
        padding_mask = (residue_indexs != -100).type(torch.float)[:,  :, None].expand(*msa.shape)
        msa=msa*padding_mask
        msa=self.msa_trans(msa+pos)

        return msa
class MSA_add(nn.Module):
    def __init__(self,  d_msa,
                p_drop,
                ):
        super(MSA_add, self).__init__()


        # 0. add chem
        self.MSA_features_emb=MSA_features_emb(msa_dim=d_msa)

        # 0.1 MSA Transation
        self.MSA_Trans=StructureModuleTransition(c=d_msa,num_layers=1,dropout_rate=p_drop)

        self.Linear_1=Linear(d_msa,d_msa,bias=False)
        self.Linear_2=Linear(d_msa,d_msa,bias=False)




    def forward(self, msa_onehot,residue_indexs,b_factors,sse3_emb,sse8_emb,cen_emb,Str_features): #
        # input:
        #   msa: initial MSA onehot token (B, N, L, 21)
        #  residue_indexs: if value =-100. is a padding mask

        msa_f=self.MSA_features_emb(msa_onehot,residue_indexs,b_factors,sse3_emb,sse8_emb,cen_emb,Str_features)

        msa=self.MSA_Trans(msa_f)


        return msa


class MSA_Stack(nn.Module):
    def __init__(self, msa_layer, d_msa, d_ipa, n_head_msa, r_ff,
                 p_drop,trans_scale_factor,performer_L_opts,
                 ):
        super(MSA_Stack, self).__init__()



        self.d_state = d_ipa
        self.d_msa = d_msa

        self.MSA2MSA=MSA2MSA(n_layer=msa_layer, n_att_head=n_head_msa, n_feat=d_msa, r_ff=r_ff, p_drop=p_drop,
                 performer_N_opts=None, performer_L_opts=performer_L_opts, )

        # self.MSA2MSAs1=nn.MultiheadAttention(embed_dim=d_msa,num_heads=n_head_msa,dropout=p_drop,batch_first=True) #Attention( d_msa, d_msa,d_msa,d_msa, n_head_msa)
        # self.MSA2MSAs2 = nn.MultiheadAttention(embed_dim=d_msa, num_heads=n_head_msa, dropout=p_drop, batch_first=True)

        self.str2msa = Str2MSA(d_msa=d_msa, d_state=d_ipa,trans_scale_factor=trans_scale_factor)

    def forward(self,  msa, Ca, state,residue_indexs,mask,MSA2MSA=True,return_att=False):
        '''

        input:
        :param msa: [B,L,C] msa features
        :param Ca:  [B,L,3]translation
        :param state:  [B,L,C]output of sttucture
        :param residue_indexs : use to get pading mask
        :return: msa updated under curent str(Ca)
        '''
        if MSA2MSA:

            # get padding mask
            # padding_mask = (residue_indexs != -100)[:,  :]

            # 0. msa update
            if len(msa.shape)==3:
                # pos = light_PositionalEncoding(residue_indexs, msa.shape[-1])
                # msa = msa + pos
                msa=msa.unsqueeze(1)

            msa,atten = self.MSA2MSA(msa,mask.type(torch.bool),return_att=return_att)  #
            msa=msa.squeeze(1)

            # # 0. global attention update
            # msa,A = self.MSA2MSAs1(msa,msa,msa, key_padding_mask=~mask.type(torch.bool),average_attn_weights=False)
            # msa,A  = self.MSA2MSAs2(msa,msa,msa, key_padding_mask=~mask.type(torch.bool),average_attn_weights=False)
            # # atten=atten.permute(0,2,3,1)
            return msa

        else:
            assert state!=None
            # 0. structure update msa
            msa=self.str2msa(msa,Ca,state)
            return msa


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
        self.lit_scale = litResnet(
            self.c_s,
            self.c_resnet,
            2,
            14,
            self.epsilon,
        )


    def forward(self, s_init,s, rigid,aatype,pred_point=True):  #


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
            litscale=self.lit_scale(s_init,s)
            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
                litscale=litscale,
            )
        else:
            pred_xyz=None
            all_frames_to_global=None


        return unnormalized_angles, angles,pred_xyz,all_frames_to_global

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
                    requires_grad=True,
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
        self, r, f,litscale  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        aatype=f
        lit_positions = self.lit_positions[aatype, ...]*litscale
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            lit_positions,
        )

class Output_Layer(nn.Module):
    def __init__(self, c,out,sacle=10):
        super(Output_Layer, self).__init__()

        self.c = c
        self.sacle=sacle
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
        s=s*self.sacle


        return s

class Pair2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=4, n_feat_in=64, n_feat_out=128, r_ff=4, p_drop=0.1):
        super(Pair2MSA, self).__init__()
        enc_layer = DirectEncoderLayer(heads=n_att_head, \
                                       d_in=n_feat_in, d_out=n_feat_out,\
                                       d_ff=n_feat_out*r_ff,\
                                       p_drop=p_drop)
        self.encoder = CrossEncoder(enc_layer, n_layer)

    def forward(self, pair, msa):
        out = self.encoder(pair, msa) # (B, N, L, K)
        return out


class Joint_layer(nn.Module):
    def __init__(self,msa_layer,IPA_layer, d_msa,d_ipa,  n_head_msa, n_head_ipa,  r_ff,
                p_drop,performer_L_opts= None, performer_N_opts=None,**kwargs
                ):
        super(Joint_layer, self).__init__()

        self.d_state=d_ipa
        self.d_msa=d_msa
        self.trans_scale_factor=kwargs['trans_scale_factor']
        self.bf_scale_factor=kwargs['bf_scale_factor']

        # #0. PAIRWMSA
        # self.pair2msa=Pair2MSA( n_feat_in=32,)


        #1. MSA Stack
        self.MSASTACK=MSA_Stack(msa_layer, d_msa, d_ipa, n_head_msa, r_ff,p_drop, self.trans_scale_factor,performer_L_opts=performer_L_opts)
        self.msa2Single= MSA2SINGLE(d_msa=d_msa, p_drop=p_drop, )  #
        # self.msa2_trans=MSATransition(d_msa,n_head_msa)   #




        # 3. Structure refine
        self.structure_refine=IPA_Stack(dim=d_ipa,depth=IPA_layer,heads=n_head_ipa,detach_rotations=False,**kwargs)


        #3. bf
        self.bf_out=Output_Layer(d_msa, 10,  self.bf_scale_factor)
        #4. get angle and xyz
        self.get_angle_points=Pred_angle_Points(**kwargs)

    def forward(self,aatype,msa,residue_indexs,mask,rigid ,last): #
        # input:
        #   msa: initial MSA onehot token (B, N, L, d)


        # 0. MSA refine
        msa=self.MSASTACK(msa,Ca=None,state=None,residue_indexs=residue_indexs,mask=mask,MSA2MSA=True,return_att=False)




        # attennp=atten.detach().cpu().numpy()[0][:,:,-1]
        # 1. Structure refine
        state_init=self.msa2Single(msa,)  # mask
        state, rigid = self.structure_refine(state_init,None, rigid,mask.type(torch.bool))

        # 2. MSA Stack
        Ca=rigid.get_trans()
        msa=self.MSASTACK(msa,Ca,state,residue_indexs=residue_indexs,mask=mask,MSA2MSA=False)


        # 4. to get exact angles
        state_out=msa
        unnormalized_angles, angles,pred_xyz,all_frames_to_global=self.get_angle_points(state_init,state_out,rigid,aatype,)
        pred_b_factors=self.bf_out(state_out)
        scaled_rigids = rigid.scale_translation(self.trans_scale_factor)



        #pred
        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames":all_frames_to_global.to_tensor_4x4() ,
            "unnormalized_angles_sin_cos": unnormalized_angles,
            "angles_sin_cos": angles,
            "aatype": aatype,
            "pred_b_factors": pred_b_factors,
            "positions": pred_xyz,
        }



        if not last:
            rigid = rigid.stop_rot_gradient()

        return msa,rigid,preds


class Joint_encoder_layer(nn.Module):
    def __init__(self, msa_layer, IPA_layer, d_msa, d_ipa, n_head_msa, n_head_ipa, r_ff,
                 p_drop, performer_L_opts=None, performer_N_opts=None, **kwargs
                 ):
        super(Joint_encoder_layer, self).__init__()

        self.d_state = d_ipa
        self.d_msa = d_msa
        self.trans_scale_factor = kwargs['trans_scale_factor']
        self.bf_scale_factor = kwargs['bf_scale_factor']

        # 0. PAIRWMSA
        self.pair2msa = Pair2MSA(n_feat_in=32, )
        # outer mean
        self.out_p_mean = OuterProductMean(c_m=d_msa, c_z=32, c_hidden=32)

        # 1. MSA Stack
        self.MSASTACK = MSA_Stack(msa_layer, d_msa, d_ipa, n_head_msa, r_ff, p_drop, self.trans_scale_factor)
        self.msa2Single = MSATransition(d_msa, n_head_msa)  # MSA2SINGLE(d_msa=d_msa, p_drop=p_drop, )

        # 2. MSA pair
        self.msa2pairs = MSA2Pair()





    def forward(self,  msa, pair, residue_indexs, mask):  #
        # input:
        #   msa: initial MSA onehot token (B, N, L, d)

        msa = self.pair2msa(pair, msa.unsqueeze(1)).squeeze(1)
        # 0. MSA refine
        msa, atten = self.MSASTACK(msa, Ca=None, state=None, residue_indexs=residue_indexs, MSA2MSA=True,
                                   return_att=True)
        state_init = self.msa2Single(msa, mask)  #


        # 1. msa 2 pair
        opm = self.out_p_mean(state_init, mask)
        pair = add(pair, opm, inplace=False)
        pair=self.msa2pairs(pair,atten)






        return state_init, pair

class Str_layer(nn.Module):
    def __init__(self,msa_layer,IPA_layer, d_msa,d_ipa,  n_head_msa, n_head_ipa,  r_ff,
                p_drop,performer_L_opts=None, performer_N_opts=None,**kwargs
                ):
        super(Str_layer, self).__init__()

        self.d_state=d_ipa
        self.d_msa=d_msa
        self.trans_scale_factor=kwargs['trans_scale_factor']
        self.bf_scale_factor=kwargs['bf_scale_factor']


        self.MSASTACK = MSA_Stack(msa_layer, d_msa, d_ipa, n_head_msa, r_ff, p_drop, self.trans_scale_factor)
        self.structure_refine=IPA_Stack(dim=d_ipa,depth=IPA_layer,heads=n_head_ipa,detach_rotations=False,**kwargs)



        #3. bf
        self.bf_out=Output_Layer(d_msa, 10,  self.bf_scale_factor)
        #4. get angle and xyz
        self.get_angle_points=Pred_angle_Points(**kwargs)

    def forward(self,aatype,msa,pair,mask,residue_indexs,rigid ,last): #
        # input:
        #   msa: initial MSA onehot token (B, N, L, d)


        msa_state, rigid = self.structure_refine(msa,pair, rigid,mask.type(torch.bool))
        Ca=rigid.get_trans()
        msa_state=self.MSASTACK(msa,Ca,msa_state,residue_indexs,MSA2MSA=False)


        # 4. to get exact angles
        state_out=msa_state
        unnormalized_angles, angles,pred_xyz,all_frames_to_global=self.get_angle_points(msa,state_out,rigid,aatype,)
        pred_b_factors=self.bf_out(msa_state)
        scaled_rigids = rigid.scale_translation(self.trans_scale_factor)

        #pred
        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames":all_frames_to_global.to_tensor_4x4() ,
            "unnormalized_angles_sin_cos": unnormalized_angles,
            "angles_sin_cos": angles,
            "aatype": aatype,
            "pred_b_factors": pred_b_factors,
            "positions": pred_xyz,
        }


        if not last:
            rigid = rigid.stop_rot_gradient()

        return msa,rigid,preds


class Repacker_iter(nn.Module):
    def __init__(self, n_module_str, Str_encoder_param, **kwargs,
                 ):
        super(Repacker_iter, self).__init__()

        self.n_module_str = n_module_str
        self.Str_encoder_param = Str_encoder_param
        self.Str_Encoder = Repacker_Str_Encoder(**self.Str_encoder_param)

        self.trans = kwargs['trans_scale_factor']
        self.bf_sacle = kwargs['bf_scale_factor']
        self.device = kwargs['device']

        self.loss_factor = kwargs['loss_factor']

        ## 0. MSA add

        d_msa = kwargs['d_msa']
        p_drop = kwargs['p_drop']
        self.chi2ipa = nn.Linear(Str_encoder_param['node_features'], d_msa)
        self.MSA_add = MSA_add(d_msa=d_msa, p_drop=p_drop)

        # bf cen features
        self.sse3_emb = nn.Embedding(4, 8)
        self.sse8_emb = nn.Embedding(9, 8)
        self.cen_emb = nn.Embedding(64, 8)

        # pair init
        # self.init_pair = Pair_emb_wo_templ(d_model=32, d_seq=256, p_drop=0.1)

        # if self.n_module_str > 0:
        #     self.iter_block = nn.ModuleList(Joint_layer(**kwargs) for _ in range(n_module_str))

        if self.n_module_str > 0:
            self.iter_block = nn.ModuleList(Joint_encoder_layer(**kwargs) for _ in range(n_module_str))

        if self.n_module_str > 0:
            self.iter_block_str = nn.ModuleList(Str_layer(**kwargs) for _ in range(n_module_str))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input: dict,
                gt_batchs: dict,

                ):

        # input:
        #   Seqs: initial MSA token (B, N, L, 21)
        #   bbrelaxdistance: the scale of noise added to original XYZ

        loss_factor = self.loss_factor
        bbrelaxdistance = 0
        r_epsilon = 1e-8

        # get data
        residue_indexs = input['residue_idx']
        seq_masks = input['mask']

        # encoder
        CHI_features, HE = self.Str_Encoder(**input, **self.Str_encoder_param)

        # str pair
        pair = self.init_pair(CHI_features, residue_indexs)

        CHI_features = self.chi2ipa(CHI_features)

        Seqs = input['S']

        # add noise and build rigid
        points = input['X']
        xyz = points + bbrelaxdistance * torch.randn_like(points)

        # transfer \AA to nm
        xyz = xyz / self.trans

        # get noised rigid
        rigid = Rigid.from_3_points(p_neg_x_axis=xyz[..., 0, :], origin=xyz[..., 1, :], p_xy_plane=xyz[..., 2, :],
                                    eps=r_epsilon, )
        B, L = Seqs.shape

        # add chem features
        Seqs_onehot = torch.nn.functional.one_hot(Seqs, loss_factor['kind']).type(torch.float)

        usechem = True
        bb_b_factors = gt_batchs['b_factors'][:, :, [0, 1, 2, 4]] / self.bf_sacle
        # print(torch.max(gt_batchs['sse3']))
        # print(torch.max(gt_batchs['sse8']))
        sse3_emb = self.sse3_emb(gt_batchs['sse3'])
        sse8_emb = self.sse8_emb(gt_batchs['sse8'])
        cen_emb = self.cen_emb(gt_batchs['cens'])

        if usechem:
            msa = self.MSA_add(Seqs_onehot, residue_indexs, bb_b_factors, sse3_emb, sse8_emb, cen_emb, CHI_features)
        else:
            msa = CHI_features

        # msa=torch.cat((msa.unsqueeze(1),HE.transpose(1,2)),dim=1)

        Aux_loss_list = []

        # iteration between sequences and str
        for i_m in range(self.n_module_str):
            msa, pair = self.iter_block[i_m]( msa, pair, residue_indexs, seq_masks)


        # iteration between sequences and str
        for i_m in range(self.n_module_str):
            last = False if i_m < (self.n_module_str - 1) else True
            msa, rigid, preds = self.iter_block_str[i_m](Seqs, msa, pair, seq_masks,residue_indexs, rigid, last)

            Auxloss = self.get_aux_loss(preds, gt_batchs, loss_factor)
            Aux_loss_list.append(Auxloss)

        # preds['angles_sin_cos']=gt_batchs['gt_angles_sin_cos']+torch.rand_like(gt_batchs['gt_angles_sin_cos'])*0.5
        loss, result = self.get_loss(preds, gt_batchs, loss_factor)

        Auxloss = torch.mean(torch.stack(Aux_loss_list, 0))
        loss = loss + Auxloss * loss_factor['aux_f']


        result.update({
            'finial_loss': loss.detach(),
            'aux_loss': Auxloss.detach(),
        })

        return loss, result

    def get_aux_loss(self, preds: dict, gt_batchs: dict, loss_factor: dict):
        a_loss = Repacker_Aux_loss(preds, gt_batchs, **loss_factor)

        return a_loss

    def get_loss(self, preds: dict, gt_batchs: dict, loss_factor: dict):
        floss = Repackerloss(preds, gt_batchs, **loss_factor)
        return floss

    def get_recovery(self, pred, S, mask):

        pred = pred * mask
        true = (S * mask).detach().type(torch.int)

        this_correct = ((pred == true).sum() - (1 - mask.detach()).sum())
        thisnods = torch.sum(mask)
        seq_recovery_rate = 100 * this_correct / thisnods

        return seq_recovery_rate



    def checkoverlap(self, seqs, mask):
        print('overlap between sequences')
        target = torch.argmax(seqs[0], -1)
        # target=self.seq_gt
        for i in range(1, len(seqs)):
            ref = torch.argmax(seqs[i], -1)
            over = self.get_recovery(ref, target, mask)
            kl = torch.exp(F.kl_div(seqs[i], seqs[0]))
            print(over)

        print('overlap between gt')
        target = self.seq_gt
        for i in range(len(seqs)):
            ref = torch.argmax(seqs[i], -1)
            over = self.get_recovery(ref, target, mask)

            print(over)



class Repacker(nn.Module):
    def __init__(self,n_module_str,Str_encoder_param,**kwargs,
                 ):
        super(Repacker, self).__init__()

        self.n_module_str = n_module_str
        self.Str_encoder_param=Str_encoder_param
        self.Str_Encoder=Repacker_Str_Encoder(**self.Str_encoder_param)

        self.trans=kwargs['trans_scale_factor']
        self.bf_sacle=kwargs['bf_scale_factor']
        self.device=kwargs['device']

        self.loss_factor=kwargs['loss_factor']

        ## 0. MSA add

        d_msa=kwargs['d_msa']
        p_drop=kwargs['p_drop']
        self.chi2ipa=nn.Linear(Str_encoder_param['node_features'],d_msa)
        self.MSA_add=MSA_add( d_msa=d_msa, p_drop=p_drop)

        #bf cen features
        self.sse3_emb=nn.Embedding(4,8)
        self.sse8_emb=nn.Embedding(9,8)
        self.cen_emb=nn.Embedding(64,8)

        # # pair init
        # self.init_pair=Pair_emb_wo_templ(d_model=32, d_seq=256, p_drop=0.1)




        if self.n_module_str > 0:
            self.iter_block = nn.ModuleList(Joint_layer(**kwargs) for _ in range(n_module_str))



        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self,input: dict,
                gt_batchs: dict,
                ):



        # input:
        #   Seqs: initial MSA token (B, N, L, 21)
        #   bbrelaxdistance: the scale of noise added to original XYZ

        loss_factor=self.loss_factor
        bbrelaxdistance=0
        r_epsilon=1e-8

        #get data
        residue_indexs=input['residue_idx']
        seq_masks=input['mask']

        # encoder
        CHI_features,HE=self.Str_Encoder(**input, **self.Str_encoder_param)

        # #str pair
        # pair=self.init_pair(CHI_features,residue_indexs)

        CHI_features = self.chi2ipa(CHI_features)

        Seqs = input['S']

        # add noise and build rigid
        points=input['X']
        xyz = points + bbrelaxdistance * torch.randn_like(points)

        #transfer \AA to nm
        xyz=xyz/self.trans

        #get noised rigid
        rigid=Rigid.from_3_points(p_neg_x_axis=xyz[..., 0, :],origin=xyz[..., 1, :],p_xy_plane=xyz[..., 2, :],eps=r_epsilon,)
        B,L=Seqs.shape


        # add chem features
        Seqs_onehot=torch.nn.functional.one_hot(Seqs,loss_factor['kind']).type(torch.float)


        usechem=True
        bb_b_factors=gt_batchs['b_factors'][:,:,[0,1,2,4]]/self.bf_sacle
        sse3_emb=self.sse3_emb(gt_batchs['sse3'])
        sse8_emb=self.sse8_emb(gt_batchs['sse8'])
        cen_emb=self.cen_emb(gt_batchs['cens'])




        if usechem:
            msa=self.MSA_add(Seqs_onehot,residue_indexs,bb_b_factors,sse3_emb,sse8_emb,cen_emb,CHI_features)
        else:
            msa=CHI_features

        # msa=torch.cat((msa.unsqueeze(1),HE.transpose(1,2)),dim=1)

        Aux_loss_list=[]


        # iteration between sequences and str

        # iteration between sequences and str
        for i_m in range(self.n_module_str):
            last= False if i_m<(self.n_module_str-1) else True
            msa,rigid,preds= self.iter_block[i_m](Seqs,msa,residue_indexs, seq_masks,rigid,last)

            Auxloss=self.get_aux_loss(preds,gt_batchs,loss_factor)
            Aux_loss_list.append(Auxloss)

        # preds['angles_sin_cos']=gt_batchs['gt_angles_sin_cos']+torch.rand_like(gt_batchs['gt_angles_sin_cos'])*0.5
        loss,result=self.get_loss(preds,gt_batchs,loss_factor)

        Auxloss=torch.mean(torch.stack(Aux_loss_list,0))
        loss=loss+Auxloss*loss_factor['aux_f']

        result.update({
            'finial_loss':loss.detach(),
            'aux_loss': Auxloss.detach(),
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


    def design(self, input:dict,gt_batchs:dict,
            ):

        #

        # input:
        #   Seqs: initial MSA token (B, N, L, 21)
        #   bbrelaxdistance: the scale of noise added to original XYZ

        loss_factor=self.loss_factor
        bbrelaxdistance=0
        r_epsilon=1e-8


        # encoder
        CHI_features,_ = self.Str_Encoder(**input, **self.Str_encoder_param)
        CHI_features = self.chi2ipa(CHI_features)

        Seqs = input['S']

        # add noise and build rigid
        points = input['X']
        xyz = points + bbrelaxdistance * torch.randn_like(points)

        # transfer \AA to nm
        xyz = xyz / self.trans

        # get noised rigid
        rigid = Rigid.from_3_points(p_neg_x_axis=xyz[..., 0, :], origin=xyz[..., 1, :], p_xy_plane=xyz[..., 2, :],
                                    eps=r_epsilon, )
        B, L = Seqs.shape

        residue_indexs = input['residue_idx']
        seq_masks = input['mask']

        # add chem features
        Seqs_onehot = torch.nn.functional.one_hot(Seqs, loss_factor['kind']).type(torch.float)

        # usechem = True
        # if usechem:
        #     msa = self.MSA_add(Seqs_onehot, residue_indexs, CHI_features)
        # else:
        #     msa = CHI_features

        usechem=True
        bb_b_factors=gt_batchs['b_factors'][:,:,[0,1,2,4]]/self.bf_sacle
        # print(torch.max(gt_batchs['sse3']))
        # print(torch.max(gt_batchs['sse8']))
        sse3_emb=self.sse3_emb(gt_batchs['sse3'])
        sse8_emb=self.sse8_emb(gt_batchs['sse8'])
        cen_emb=self.cen_emb(gt_batchs['cens'])


        if usechem:
            msa=self.MSA_add(Seqs_onehot,residue_indexs,bb_b_factors,sse3_emb,sse8_emb,cen_emb,CHI_features)
        else:
            msa=CHI_features

        Aux_loss_list = []

        # iteration between sequences and str
        for i_m in range(self.n_module_str):
            last = False if i_m < (self.n_module_str - 1) else True
            msa, rigid, preds = self.iter_block[i_m](Seqs, msa, residue_indexs, seq_masks, rigid, last)

            Auxloss = self.get_aux_loss(preds, gt_batchs, loss_factor)
            Aux_loss_list.append(Auxloss)

        chis=self.get_chi(preds['angles_sin_cos'],gt_batchs['aatype'])
        loss, result = self.get_loss(preds, gt_batchs, loss_factor)


        pred_b_factors=torch.concat((bb_b_factors[:,:,:3]*self.bf_sacle,preds['pred_b_factors'][:,:,0].unsqueeze(-1),bb_b_factors[:,:,-1].unsqueeze(-1)*self.bf_sacle,preds['pred_b_factors'][:,:,1:]),dim=-1)
        preb=pred_b_factors.squeeze(0).cpu().numpy()
        Auxloss = torch.mean(torch.stack(Aux_loss_list, 0))
        loss = loss + Auxloss * loss_factor['aux_f']

        result.update({
            'finial_loss': loss.detach().cpu(),
            'aux_loss': Auxloss.detach().cpu(),
            'final_atom_positions':preds['positions'],
            'chis':chis,
            'angles_sin_cos':preds['angles_sin_cos'][...,3:,:],
            'pred_b_factors':preb
        })

        return loss, result

    def get_chi(self,angles_sin_cos,aatype,eps=1e-4):
        #angles_tan=angles_sin_cos[...,:,:,0]/(angles_sin_cos[...,:,:,1])
        # angles_tan=torch.clamp()
       # a=gt_angles_mask*180*torch.arctan(angles_tan)/torch.pi

        chi_angles_mask = list(residue_constants.chi_angles_mask)
        chi_angles_mask = torch.tensor(chi_angles_mask,device=aatype.device)

        chis_mask = chi_angles_mask[aatype, :]

        angles=180*torch.arctan2(angles_sin_cos[...,:,:,0],angles_sin_cos[...,:,:,1])/torch.pi
        chis=(chis_mask*angles[...,:,3:]).squeeze(0)
        return chis.detach().cpu().numpy()



def showmat(seqs):

    mat=torch.mean(seqs,dim=0)
    # mat=F.layer_norm(mat,(21,)).transpose(0,1)
    # mat=F.normalize(mat,-1).transpose(0,1)
    mat=mat.transpose(0,1).detach().cpu().numpy()
    plt.matshow(mat, cmap=plt.cm.Blues) #plt.matshow(mat, cmap=plt.cm.Blues)
    plt.show()
    #cax = plt.imshow(mat,  origin='lower') #cmap='viridis',













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



