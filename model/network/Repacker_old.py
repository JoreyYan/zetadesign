import numpy as np

import torch

from .Transformer import *
from .invariant_point_attention import IPA_Stack
from  ..primitives import Linear
from model.network.Generator import Repacker_Str_Encoder

from ..Rigid import Rigid,Rotation
from ..chems import add_chem_features,light_PositionalEncoding

from model.AngelResnet import AngleResnet
from .feats import torsion_angles_to_frames,frames_and_literature_positions_to_atom14_pos
from model.network.loss_old import Repackerloss,Repacker_Aux_loss
# from model.network.repackerloss import Repackerloss,Repackerloss_aux
from model.np.residue_constants import  (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)


# Attention module based on AlphaFold2's idea written by Minkyung Baek
#  - Iterative MSA feature extraction
#    - 1) MSA2Pair: extract pairwise feature from MSA --> added to previous residue-pair features
#                   architecture design inspired by CopulaNet paper
#    - 2) MSA2MSA:  process MSA features using Transformer (or Performer) encoder. (Attention over L first followed by attention over N)
#    - 3) Pair2MSA: Update MSA features using pair feature
#    - 4) Pair2Pair: process pair features using Transformer (or Performer) encoder.
from PIL import Image
import matplotlib.pyplot as plt
def show_numpy(matrix,i):
    im = Image.fromarray(matrix)  # numpy 转 image类
    im.show()
    # im.save('/home/junyu/图片/strmask/'+i+'.jpg')
    # matrix=matrix.type(np.int)
    # plt.imshow(matrix)



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
                                   use_tied=True,
                                   performer_opts=performer_L_opts)
        self.encoder_1 = Encoder(enc_layer_1, n_layer)

        # attention along N
        enc_layer_2 = EncoderLayer(d_model=n_feat, d_ff=n_feat * r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_N_opts)
        self.encoder_2 = Encoder(enc_layer_2, n_layer)

    def forward(self, x, mask):
        # Input: MSA embeddings (B, N, L, K)
        # Output: updated MSA embeddings (B, N, L, K)
        B, N, L, _ = x.shape
        # attention along L
        x = self.encoder_1(x, return_att=False,mask=mask)

        # if breaks:
        #     return x

        # attention along N don use mask in coloum
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
        self.msa_emb=nn.Linear(onehotdim,msa_dim,)
        self.msa_trans=nn.Linear(msa_dim*3,msa_dim,bias=False )



    def forward(self,msa_onehot,residue_indexs,Str_features):
        """
        msa: soft onehot of generated sequences

        return: representation of msa features

        """

        chem_s=self.chem_features(msa_onehot)
        msa=self.msa_emb(msa_onehot)


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




    def forward(self, msa_onehot,residue_indexs,Str_features): #
        # input:
        #   msa: initial MSA onehot token (B, N, L, 21)
        #  residue_indexs: if value =-100. is a padding mask

        msa_f=self.MSA_features_emb(msa_onehot,residue_indexs,Str_features)

        msa=self.MSA_Trans(msa_f)


        return msa


class MSA_Stack(nn.Module):
    def __init__(self, msa_layer, d_msa, d_ipa, n_head_msa, r_ff,
                 p_drop,trans_scale_factor,
                 ):
        super(MSA_Stack, self).__init__()



        self.d_state = d_ipa
        self.d_msa = d_msa

        self.MSA2MSA=MSA2MSA(n_layer=msa_layer, n_att_head=n_head_msa, n_feat=d_msa, r_ff=r_ff, p_drop=p_drop,
                 performer_N_opts=None, performer_L_opts=None, )
        self.str2msa = Str2MSA(d_msa=d_msa, d_state=d_ipa,trans_scale_factor=trans_scale_factor)

    def forward(self,  msa, Ca, state,residue_indexs,MSA2MSA=True):
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
            padding_mask = (residue_indexs != -100)[:,  :]

            # 0. msa update
            if len(msa.shape)==3:
                msa=msa.unsqueeze(1)
            msa = self.MSA2MSA(msa, padding_mask)
            msa=msa.squeeze(1)

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


class Joint_layer(nn.Module):
    def __init__(self,msa_layer,IPA_layer, d_msa,d_ipa,  n_head_msa, n_head_ipa,  r_ff,
                p_drop,**kwargs
                ):
        super(Joint_layer, self).__init__()

        self.d_state=d_ipa
        self.d_msa=d_msa
        self.trans_scale_factor=kwargs['trans_scale_factor']




        # 1. Structure refine
        self.msa2Single= MSA2SINGLE(d_msa=d_msa, p_drop=p_drop, )

        self.structure_refine=IPA_Stack(dim=d_ipa,depth=IPA_layer,heads=n_head_ipa,detach_rotations=False)

        #2. MSA Stack
        self.MSASTACK=MSA_Stack(msa_layer, d_msa, d_ipa, n_head_msa, r_ff,p_drop, self.trans_scale_factor)




        #4. get angle and xyz
        self.get_angle_points=Pred_angle_Points(**kwargs)

    def forward(self,aatype,msa, residue_indexs,mask,rigid ,last): #
        # input:
        #   msa: initial MSA onehot token (B, N, L, d)
        # 0. MSA refine
        msa=self.MSASTACK(msa,Ca=None,state=None,residue_indexs=residue_indexs,MSA2MSA=True)

        # 1. Structure refine
        state_init=self.msa2Single(msa)  #msa[:,0,:,:]
        state, rigid = self.structure_refine(state_init, rigid,mask.type(torch.bool))

        # 2. MSA Stack
        Ca=rigid.get_trans()
        msa=self.MSASTACK(msa,Ca,state,residue_indexs,MSA2MSA=False)



        # 4. to get exact angles
        state_out=msa
        unnormalized_angles, angles,pred_xyz,all_frames_to_global=self.get_angle_points(state_init,state_out,rigid,aatype,pred_point=True)

        scaled_rigids = rigid.scale_translation(self.trans_scale_factor)

        #pred

        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles_sin_cos": unnormalized_angles,
            "angles_sin_cos": angles,
            "aatype": aatype,
            "positions": pred_xyz,

        }


        if not last:
            rigid = rigid.stop_rot_gradient()

        return msa,rigid,preds





class Repacker(nn.Module):
    def __init__(self,n_module_str,Str_encoder_param,**kwargs,
                 ):
        super(Repacker, self).__init__()

        self.n_module_str = n_module_str
        self.Str_encoder_param=Str_encoder_param
        self.Str_Encoder=Repacker_Str_Encoder(**self.Str_encoder_param)

        self.trans=kwargs['trans_scale_factor']

        ## 0. MSA add
        d_msa=kwargs['d_msa']
        p_drop=kwargs['p_drop']
        self.MSA_add=MSA_add( d_msa=d_msa, p_drop=p_drop)
        self.chi2ipa=nn.Linear(Str_encoder_param['node_features'],d_msa)


        if self.n_module_str > 0:
            self.iter_block = nn.ModuleList(Joint_layer(**kwargs) for _ in range(n_module_str))





    def forward(self, input:dict,
                gt_batchs:dict,
                # confgfape,
                bbrelaxdistance:float,
                r_epsilon:float,
                loss_factor:dict
                ,**kwargs):
        # input:
        #   Seqs: initial MSA token (B, N, L, 21)
        #   bbrelaxdistance: the scale of noise added to original XYZ

        # encoder
        CHI_features=self.Str_Encoder(**input, **self.Str_encoder_param)
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


        residue_indexs=input['residue_idx']
        seq_masks=input['mask']


        # add chem features
        Seqs_onehot=torch.nn.functional.one_hot(Seqs,loss_factor['kind']).type(torch.float)


        usechem=True
        if usechem:
            msa=self.MSA_add(Seqs_onehot,residue_indexs,CHI_features)
        else:
            msa=CHI_features


        Aux_loss_list=[]

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
            'finial_loss':loss.detach().cpu(),
            'aux_loss': Auxloss.detach().cpu(),
            })



        return loss,result

    def get_aux_loss(self, preds:dict, gt_batchs:dict,lossfactor):


        a_loss=Repacker_Aux_loss(preds, gt_batchs,**lossfactor)


        return a_loss

    def get_loss(self,preds:dict, gt_batchs:dict,lossfactor):
        floss,result=Repackerloss(preds, gt_batchs,**lossfactor)
        return floss,result

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



