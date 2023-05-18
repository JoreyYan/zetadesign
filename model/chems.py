import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import logging
import model.np.residue_constants as rc
import os

logger = logging.getLogger(__name__)
CE_LOSS = nn.CrossEntropyLoss()
KL_LOSS=nn.KLDivLoss()


def aa_psy_che(indexfile='/AAindex1',lookup = None):

    title='A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V'
    oldindex=title.split()
    index=[]
    for i in oldindex:
        index.append(i[0])
    for i in oldindex:
        index.append(i[2])


    with open(indexfile) as f:
        contents = f.readlines()
    n=0
    data=[]
    for i in range(len(contents)):
        this_proper=[]

        if contents[i][:1]=='I':
            n=n+1
            for a in list(contents[i + 1].split('\n')[0].split()):
                if a == 'NA':
                    a = 0.
                this_proper.append(float(a) )
            for a in list(contents[i + 2].split('\n')[0].split()):
                if a=='NA':
                    a=0.

                this_proper.append(float(a))

            data.append(this_proper)

    npdata=torch.as_tensor(np.asarray(data),dtype=torch.float)
    npx=torch.mean(npdata,-1).unsqueeze(-1)
    digits=torch.transpose(torch.cat((npdata,npx),-1),0,1)
    index.append('X')

    new_list=[]
    for i in list(lookup):
        aa=index.index(i)
        new_list.append(digits[aa])

    digits=torch.stack((new_list),0)
    return digits
##Sequence Feature From https://github.com/phbradley/tcr-dist/blob/master/amino_acids.py
HP = {'I': 0.73, 'F': 0.61, 'V': 0.54, 'L': 0.53, 'W': 0.37,
      'M': 0.26, 'A': 0.25, 'G': 0.16, 'C': 0.04, 'Y': 0.02,
      'P': -0.07, 'T': -0.18, 'S': -0.26, 'H': -0.40, 'E': -0.62,
      'N': -0.64, 'Q': -0.69, 'D': -0.72, 'K': -1.10, 'R': -1.76}

HP['X'] = sum(HP.values())/20.



GES = {'F': -3.7, 'M': -3.4, 'I': -3.1, 'L': -2.8, 'V': -2.6,
       'C': -2.0, 'W': -1.9, 'A': -1.6, 'T': -1.2, 'G': -1.0,
       'S': -0.6, 'P': 0.2,  'Y': 0.7,  'H': 3.0,  'Q': 4.1,
       'N': 4.8,  'E': 8.2,  'K': 8.8,  'D': 9.2,  'R': 12.3}

GES['X'] = sum(GES.values())/20.

## KD values (Kyte-Doolittle) taken from http://web.expasy.org/protscale/pscale/Hphob.Doolittle.html

KD = {'A': 1.8, 'C': 2.5, 'E': -3.5, 'D': -3.5, 'G': -0.4, 'F': 2.8, 'I': 4.5, 'H': -3.2, 'K': -3.9, 'M': 1.9, 'L': 3.8, 'N': -3.5, 'Q': -3.5, 'P': -1.6, 'S': -0.8, 'R': -4.5, 'T': -0.7, 'W': -0.9, 'V': 4.2, 'Y': -1.3}

assert len(KD) == 20
KD['X'] = sum(KD.values())/20.

###from https://github.com/s-andrews/python_course_answers/blob/b0f6a904a8bd122c64d9ded79e74990179d56bb7/Exercise%202/amino_acid_properties.py
amino_acids_Weight = {}
amino_acids_Weight["A"] = {"name":"Alanine",       "weight":89.1}
amino_acids_Weight["R"] = {"name":"Arginine",      "weight":174.2}
amino_acids_Weight["N"] = {"name":"Asparigine",    "weight":132.1}
amino_acids_Weight["D"] = {"name":"Aspartate",     "weight":133.1}
amino_acids_Weight["C"] = {"name":"Cysteine",      "weight":121.2}
amino_acids_Weight["E"] = {"name":"Glutamate",     "weight":147.1}
amino_acids_Weight["Q"] = {"name":"Glutamine",     "weight":146.2}
amino_acids_Weight["G"] = {"name":"Glycine",       "weight":75.1}
amino_acids_Weight["H"] = {"name":"Histidine",     "weight":155.2}
amino_acids_Weight["I"] = {"name":"Isoleucine",    "weight":131.2}
amino_acids_Weight["L"] = {"name":"Leucine",       "weight":131.2}
amino_acids_Weight["K"] = {"name":"Lysine",        "weight":146.2}
amino_acids_Weight["M"] = {"name":"Methionine",    "weight":149.2}
amino_acids_Weight["F"] = {"name":"Phenylalanine", "weight":165.2}
amino_acids_Weight["P"] = {"name":"Proline",       "weight":115.1}
amino_acids_Weight["S"] = {"name":"Serine",        "weight":105.1}
amino_acids_Weight["T"] = {"name":"Threonine",     "weight":119.1}
amino_acids_Weight["W"] = {"name":"Tryptophan",    "weight":204.2}
amino_acids_Weight["Y"] = {"name":"Tyrosine",      "weight":181.2}
amino_acids_Weight["V"] = {"name":"Valine",        "weight":117.1}
amino_acids_Weight["X"] = {"name":"X",        "weight":136.90}



class Posi_Features(nn.Module):
    def __init__(self, H_features):
        """ Extract protein features """
        super(Posi_Features, self).__init__()
        self.H_features=H_features

    def forward(self,S,residua_index,mask,):

        B,M,L=S.shape
        mask_expand=mask.unsqueeze(-1).repeat(1,1,self.H_features)

        POS_EMBD = light_PositionalEncoding(residua_index, num_embeddings=self.H_features) * mask_expand

        return POS_EMBD

def light_PositionalEncoding(idx_s,num_embeddings=None):
    '''
    idx_s:B,L
    '''
    if num_embeddings is not None:
        num_embeddings=num_embeddings
    else:
        print("please enter num_embedding")

    # From https://github.com/jingraham/neurips19-graph-protein-design

    d = idx_s

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float, device=idx_s.device)
        * -(math.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)


    return E

class add_chem_features(nn.Module):
    def __init__(self,  chem_dim=256):
        super().__init__()

        alphabet=''.join(rc.restypes_with_x)
        alphabets=rc.restype_order_with_x


        self.s2n = np.asarray([alphabet.index(a) for a in alphabet], dtype=np.int32)
        self.s2n_HP = []
        self.s2n_KD = []
        self.s2n_GES = []
        self.s2n_Weight = []
        for i in range(21):
            self.s2n_HP.append(HP[alphabet[i]])
            self.s2n_KD.append(KD[alphabet[i]])
            self.s2n_GES.append(GES[alphabet[i]])
            self.s2n_Weight.append(amino_acids_Weight[alphabet[i]]['weight'])

        # chempath=os.getcwd()
        self.chem_f=aa_psy_che(indexfile='./model/AAindex1',lookup=alphabet)  #/home/omnisky/data/everyOne/yjy/pdhs
        self.s2n_HP = torch.as_tensor(self.s2n_HP)
        self.s2n_KD = torch.as_tensor(self.s2n_KD)
        self.s2n_GES = torch.as_tensor(self.s2n_GES)
        self.s2n_Weight = torch.as_tensor(self.s2n_Weight)


        self.aaFeatures=torch.cat((self.s2n_HP.unsqueeze(-1),self.s2n_KD.unsqueeze(-1),self.s2n_Weight.unsqueeze(-1),self.s2n_GES.unsqueeze(-1),self.chem_f),dim=-1)
        self.aaFeatures=F.normalize(self.aaFeatures, p=2, dim=0)
        #torch.save(self.aaFeatures.detach().cpu(),'/home/junyu/桌面/code/ProteinCLIP/CLIP/data/570aaf.pth')

        self.chem_dims = nn.Linear(570, chem_dim,bias=False )  #+566
        self.chem_norm=nn.LayerNorm(chem_dim)


    def _chemfea_add(self,S_onehot,):
        # S： b,m,l,21


        S_onehot = S_onehot.type(torch.float)

        fullchem_F=torch.tensordot(S_onehot,self.aaFeatures.to(S_onehot.device),dims=([-1],[0]))#*mask.unsqueeze(-1) #570  this mask is unnessary
        seq_chem=self.chem_dims(fullchem_F)
        seq_chem=self.chem_norm(seq_chem)

        return seq_chem


    def forward(self, S_onehot):
        '''
            input: B N 4,3
            mask:B,L 0 for mask ,and 1 for true value
            ---------------------------

        '''

        msa_chem=self._chemfea_add(S_onehot)
        return msa_chem









class args():
    def __init__(self):
        self.group_size = 12
        self.num_group = 64
        self.encoder_dims = 128
        self.tokens_dims = 64

        self.decoder_dims =128
        self.num_tokens = 1024


if __name__ == "__main__":

    # model=Group(num_group=8,group_size=6,top_k=48).cuda()
    # encoder=Encoder().cuda()

    B=2

    L=500
    d_msa = 768

    seq_embed=nn.Embedding(20,20).cuda()

    seq=torch.randint(0,20,(B,L)).cuda()
    #seq=seq_embed(seq)
    idx=torch.arange(0,L).unsqueeze(0).expand((B,L)).cuda()
    mask=torch.randint(0,2,(B,L)).cuda()
    Points=torch.randn(B,L,4,3).cuda()
    token=torch.randn(B,L,768).cuda()
    # nei,center=model(Ca,mask)
    # logits=encoder(nei)

    # dgcnn=DGCNN().cuda()
    # logits=torch.randn(B*L,8,128).cuda()
    # center = torch.randn(B * L, 8, 3).cuda()
    # x=dgcnn(logits,center)
    #
    # print(x)

    config=args()
    model=SEQCLIP(config).cuda()
    x,soft_one_hot=model(Points,token,idx,mask)
    model.get_loss(x)